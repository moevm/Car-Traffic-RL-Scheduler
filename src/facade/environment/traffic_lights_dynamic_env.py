import random

import gymnasium as gym
import traci
import numpy as np

from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box, MultiBinary, MultiDiscrete
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator


class TrafficLightsDynamicEnv(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
                 checkpoint_file: str,
                 sumo_config: str,
                 traffic_lights_groups: list[list[str]],
                 n_lanes: int,
                 gui: bool = False,
                 train_mode: bool = True):
        self.__gui = gui
        self.__train_mode = train_mode
        self.__n_lanes = n_lanes
        self.__traffic_lights_groups = traffic_lights_groups
        self.__schedule = []
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__start_global_step = step
        self.__global_step = step
        self.__traffic_lights_ids = traffic_lights_ids
        self.__n_traffic_lights = len(self.__traffic_lights_ids)
        self.__local_step = 0
        self.__SUMO_CONFIG = sumo_config
        self.__checkpoint_file = checkpoint_file
        if gui:
            sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        else:
            sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__checkpoint_file)
        self.__vehicle_size = self._average_vehicle_size()
        self.__group_size = len(self.__traffic_lights_groups[0])
        self.__min_duration = 15
        self.__max_duration = 90
        self.__critical_duration = 180
        self.__max_phases = 4
        self.__i_window = 0
        observation_space = self.__make_observation_space()
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(self.__group_size)
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.__traffic_lights_ids}
        self.__vehicles_on_lanes_before = {i: [] for i in range(len(self.__traffic_lights_groups))}

    @staticmethod
    def _average_vehicle_size():
        sum_length = 0
        vehicles = traci.vehicle.getIDList()
        for vehicle in vehicles:
            sum_length += traci.vehicle.getLength(vehicle)
        return sum_length / len(vehicles)

    def __make_observation_space(self):
        observation_space = Dict({
            "density": Box(low=0, high=100, shape=(self.__group_size, self.__n_lanes), dtype=np.float32),
            "waiting": Box(low=0, high=1, shape=(self.__group_size, self.__n_lanes), dtype=np.float32),
            "phase": MultiDiscrete([self.__max_phases] * self.__group_size),
            "time": Box(low=0, high=self.__critical_duration, shape=(self.__group_size,), dtype=np.float32),
            "bounds": MultiDiscrete([3] * self.__group_size),
            "accumulated_capacity": Box(low=0, high=10 * self.__critical_duration ** 0.75, shape=(self.__group_size,),
                                        dtype=np.float32)
        })
        return observation_space

    def __get_observation(self, i_window):
        tls_group = self.__traffic_lights_groups[i_window]
        tls_id = tls_group[0]
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        density = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        waiting = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        phase = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        time = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        bounds = np.zeros(shape=(self.__group_size,), dtype=np.int8)
        accumulated_capacity = np.zeros(shape=(self.__group_size,), dtype=np.float32)
        for i, tls_id in enumerate(tls_group):
            for j, lane in enumerate(lanes):
                waiting[i, j] = traci.lane.getLastStepHaltingNumber(lane) / (
                        traci.lane.getLength(lane) / self.__vehicle_size)
                density[i, j] = traci.lane.getLastStepOccupancy(lane)
            phase[i] = traci.trafficlight.getPhase(tls_id)
            time[i] = min(traci.trafficlight.getSpentDuration(tls_id), self.__critical_duration)
            accumulated_capacity[i] = self.__n_steps_capacity[tls_id]
            if traci.trafficlight.getSpentDuration(tls_id) < self.__min_duration:
                bounds[i] = 0
            elif self.__min_duration <= traci.trafficlight.getSpentDuration(tls_id) <= self.__max_duration:
                bounds[i] = 1
            elif traci.trafficlight.getSpentDuration(tls_id) > self.__max_duration:
                bounds[i] = 2
        observation = {
            "density": density,
            "waiting": waiting,
            "phase": phase,
            "time": time,
            "bounds": bounds,
            "accumulated_capacity": accumulated_capacity
        }
        return observation

    @staticmethod
    def _normalize_reward(x: float, x_min: float, x_max: float, a: float, b: float) -> float:
        return a + (x - x_min) * (b - a) / (x_max - x_min)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.__traffic_lights_ids}
        traci.simulation.loadState(self.__checkpoint_file)
        traci.simulationStep()
        for tls_id in self.__traffic_lights_ids:
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            current_phase = traci.trafficlight.getPhase(tls_id)
            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        self.__global_step = self.__start_global_step + 1
        self.__schedule = self.__transport_generator.generate_schedule_for_poisson_flow(self.__global_step + 1)
        self.__update_vehicles_on_lanes_before()
        self.__i_window, self.__local_step = 0, 0
        observation = self.__get_observation(self.__i_window)
        group_rewards = {
            "sum_reward": 0.0,
            "step_capacity": 0.0,
            "phase_capacity": 0.0,
        }
        return observation, group_rewards

    def __change_phase(self, action: ActType, i_window: int) -> tuple[list[bool], list[float]]:
        tls_group = self.__traffic_lights_groups[i_window]
        switched_tls = [False] * len(tls_group)
        spent_duration_tls = []
        for i, tls_id in enumerate(tls_group):
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            spent_duration_tls.append(traci.trafficlight.getSpentDuration(tls_id))
            spent = traci.trafficlight.getSpentDuration(tls_id)
            if 'y' not in state:
                if self.__train_mode:
                    if action[i] == 1:
                        traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
                        switched_tls[i] = True
                    elif action[i] == 0:
                        traci.trafficlight.setPhase(tls_id, current_phase)
                else:
                    if action[i] == 1:
                        if spent >= self.__min_duration:
                            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
                        else:
                            traci.trafficlight.setPhase(tls_id, current_phase)
                        switched_tls[i] = True
                    elif action[i] == 0:
                        if spent > self.__max_duration:
                            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
                        else:
                            traci.trafficlight.setPhase(tls_id, current_phase)
        return switched_tls, spent_duration_tls

    def __get_vehicles_on_lanes(self, i_window: int) -> list[list[list[str]]]:
        tls_group = self.__traffic_lights_groups[i_window]
        vehicles_on_lanes = []
        for tls_id in tls_group:
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            tls_lanes = []
            for lane in lanes:
                tls_lanes.append(traci.lane.getLastStepVehicleIDs(lane))
            vehicles_on_lanes.append(tls_lanes)
        return vehicles_on_lanes

    def __get_context_reward(self, spent: float, action: int, reward: float, max_reward: float):
        if action == 1 and spent < self.__min_duration:
            context_reward = -max_reward * (1 - spent / self.__min_duration)
        elif action == 0 and spent > self.__max_duration:
            context_reward = max_reward * (1 - spent / self.__max_duration)
        else:
            context_reward = reward
        return context_reward

    def __calculate_phase_capacity(self,
                                   switched_tls: list[bool],
                                   spent_duration_tls: list[float],
                                   action: ActType,
                                   i_window: int) -> float:
        sum_phase_capacity = 0.0
        tls_group = self.__traffic_lights_groups[i_window]
        for i in range(len(switched_tls)):
            phase_capacity = 0
            tls_id = tls_group[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if switched_tls[i] and 'y' in state:
                max_spent = min(max(spent_duration_tls[i], 1), self.__critical_duration)
                reward = self.__n_steps_capacity[tls_id]
                if (action[i] == 1 and max_spent < self.__min_duration) or (
                        action[i] == 0 and max_spent > self.__max_duration):
                    phase_capacity = 0
                else:
                    phase_capacity = reward
                self.__n_steps_capacity[tls_id] = 0
            sum_phase_capacity += phase_capacity
        return sum_phase_capacity / 10

    def __calculate_step_capacity(self,
                                  vehicles_on_tls_after: list[list[list[str]]],
                                  switched_tls: list[bool],
                                  spent_duration_tls: list[float],
                                  action: ActType,
                                  i_window: int) -> float:
        vehicles_on_tls_before = self.__vehicles_on_lanes_before[i_window]
        tls_group = self.__traffic_lights_groups[i_window]
        sum_step_capacity = 0.0
        for i in range(len(vehicles_on_tls_before)):
            reward = 0
            tls_id = tls_group[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if ('y' not in state) or (switched_tls[i] and 'y' in state):
                reward = 0.1
                vehicles_on_lanes_before = vehicles_on_tls_before[i]
                vehicles_on_lanes_after = vehicles_on_tls_after[i]
                for j, vehicles_before in enumerate(vehicles_on_lanes_before):
                    for vehicle_before in vehicles_before:
                        if (vehicle_before not in vehicles_on_lanes_after[j]) and (
                                vehicle_before not in traci.simulation.getArrivedIDList()):
                            reward += 1
                sum_step_capacity += self.__get_context_reward(spent_duration_tls[i], action[i], reward, 5)
            self.__n_steps_capacity[tls_id] = reward
        return sum_step_capacity * 0.1

    def __update_vehicles_on_lanes_before(self) -> None:
        for i in range(len(self.__traffic_lights_groups)):
            vehicles_on_lanes_before = self.__get_vehicles_on_lanes(i)
            self.__vehicles_on_lanes_before[i] = vehicles_on_lanes_before

    def __next_timestep(self) -> None:
        self.__update_vehicles_on_lanes_before()
        traci.simulationStep()
        self.__local_step += 1
        self.__global_step += 1
        self.__transport_generator.clean_vehicles_data()
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        switched_tls, spent_duration_tls = self.__change_phase(action, self.__i_window)
        vehicles_on_tls_after = self.__get_vehicles_on_lanes(self.__i_window)
        step_capacity = self.__calculate_step_capacity(vehicles_on_tls_after, switched_tls, spent_duration_tls, action,
                                                       self.__i_window)
        phase_capacity = self.__calculate_phase_capacity(switched_tls, spent_duration_tls, action, self.__i_window)
        group_rewards = {
            "step_capacity": step_capacity,
            "phase_capacity": phase_capacity,
        }
        reward = step_capacity + phase_capacity
        group_rewards["sum_reward"] = reward
        truncated = (self.__local_step == 5999) and (self.__i_window == len(self.__traffic_lights_groups) - 1)
        terminated = False
        if self.__i_window == len(self.__traffic_lights_groups) - 1:
            self.__next_timestep()
        self.__i_window = (self.__i_window + 1) % len(self.__traffic_lights_groups)
        observation = self.__get_observation(self.__i_window)
        return observation, reward, terminated, truncated, group_rewards

    def close(self):
        traci.close()

    def get_schedule(self):
        return self.__schedule
