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
                 gui: bool = False):
        self.__n_lanes = n_lanes
        self.__traffic_lights_groups = traffic_lights_groups
        self.__schedule = transport_generator.generate_schedule_for_poisson_flow(step + 1)
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
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
        self.__vehicle_size = traci.vehicle.getLength(traci.vehicle.getIDList()[0])
        self.__group_size = len(self.__traffic_lights_groups[0])
        self.__min_duration = 15
        self.__max_duration = 90
        self.__max_phases = 4
        self.__i_window = 0
        observation_space = self.__make_observation_space()
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(self.__group_size)
        (self.__total_reward, self.__total_tls_reward, self.__total_step_capacity, self.__total_phase_capacity,
         self.__total_halting_reward, self.__reward, self.__tls_reward, self.__step_capacity,
         self.__phase_capacity, self.__halting_reward) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.__traffic_lights_ids}
        self.__vehicles_on_lanes_before = {i: [] for i in range(len(self.__traffic_lights_groups))}

    def __make_observation_space(self):
        observation_space = Dict({
            "density": Box(low=0, high=100, shape=(self.__group_size, self.__n_lanes), dtype=np.float32),
            "waiting": Box(low=0, high=1, shape=(self.__group_size, self.__n_lanes), dtype=np.float32),
            "phase": MultiDiscrete([self.__max_phases] * self.__group_size),
            "time": Box(low=0, high=600, shape=(self.__group_size,), dtype=np.float32),
            "bounds": MultiDiscrete([3] * self.__group_size)
        })
        return observation_space

    @staticmethod
    def __get_info(reward: float,
                   tls_reward: float,
                   step_capacity: float,
                   phase_capacity: float,
                   halting_reward: float,
                   group_reward: float,
                   group_tls_reward: float,
                   group_step_capacity: float,
                   group_phase_capacity: float,
                   group_halting_reward: float) -> dict[str, float]:
        info = {
            "reward": reward,
            "tls_reward": tls_reward,
            "step_capacity": step_capacity,
            "phase_capacity": phase_capacity,
            "halting_reward": halting_reward,
            "group_reward": group_reward,
            "group_tls_reward": group_tls_reward,
            "group_step_capacity": group_step_capacity,
            "group_phase_capacity": group_phase_capacity,
            "group_halting_reward": group_halting_reward
        }
        return info

    def __get_observation(self, i_window):
        tls_group = self.__traffic_lights_groups[i_window]
        tls_id = tls_group[0]
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        density = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        waiting = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        phase = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        time = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        bounds = np.zeros(shape=(self.__group_size,), dtype=np.int8)
        for i, tls_id in enumerate(tls_group):
            for j, lane in enumerate(lanes):
                waiting[i, j] = traci.lane.getLastStepHaltingNumber(lane) / (
                        traci.lane.getLength(lane) / self.__vehicle_size)
                density[i, j] = traci.lane.getLastStepOccupancy(lane)
            phase[i] = traci.trafficlight.getPhase(tls_id)
            time[i] = traci.trafficlight.getSpentDuration(tls_id)
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
            "bounds": bounds
        }
        return observation

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__update_vehicles_on_lanes_before()
        self.__i_window = 0
        observation = self.__get_observation(self.__i_window)
        self.__local_step, self.__i_window = 0, 0
        self.__update_total_rewards(self.__reward, self.__tls_reward, self.__step_capacity, self.__phase_capacity,
                                    self.__halting_reward)
        (self.__reward, self.__tls_reward, self.__step_capacity, self.__phase_capacity,
         self.__halting_reward) = 0.0, 0.0, 0.0, 0.0, 0.0
        info = self.__get_info(self.__total_reward,
                               self.__total_tls_reward,
                               self.__total_step_capacity,
                               self.__total_phase_capacity,
                               self.__total_halting_reward,
                               0.0,
                               0.0,
                               0.0,
                               0.0,
                               0.0)
        return observation, info

    def __change_phase(self, action: ActType, i_window: int) -> tuple[float, list[bool], list[float]]:
        tls_reward = 0.0
        tls_group = self.__traffic_lights_groups[i_window]
        switched_tls = [False] * len(tls_group)
        spent_duration_tls = []
        for i, tls_id in enumerate(tls_group):
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            n_lanes = len(set(traci.trafficlight.getControlledLanes(tls_id)))
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            spent_duration_tls.append(traci.trafficlight.getSpentDuration(tls_id))
            if 'y' not in state:
                if action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) > self.__max_duration:
                    tls_reward += -n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / self.__max_duration)
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) < self.__min_duration:
                    tls_reward += -n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / self.__min_duration)
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) > self.__max_duration:
                    tls_reward += n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / self.__max_duration)
                elif action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) < self.__min_duration:
                    tls_reward += n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / self.__min_duration) ** 0.1
                elif self.__min_duration <= traci.trafficlight.getSpentDuration(tls_id) <= self.__max_duration:
                    tls_reward += n_lanes
                if action[i] == 0:
                    traci.trafficlight.setPhase(tls_id, current_phase)
                else:
                    traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
                    switched_tls[i] = True
        return tls_reward * 0.2, switched_tls, spent_duration_tls

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

    def __calculate_halting_reward(self, switched_tls: list[bool], i_window: int) -> float:
        tls_group = self.__traffic_lights_groups[i_window]
        sum_halting_number = 0.0
        for i, tls_id in enumerate(tls_group):
            tls_id = tls_group[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            if ('y' not in state) or (('y' in state) and switched_tls[i]):
                for j, lane in enumerate(lanes):
                    sum_halting_number += traci.lane.getLastStepHaltingNumber(lane) / (
                            traci.lane.getLength(lane) / self.__vehicle_size)
        return -sum_halting_number

    def __calculate_phase_capacity(self,
                                   switched_tls: list[bool],
                                   spent_duration_tls: list[float],
                                   i_window: int,) -> float:
        phase_capacity = 0.0
        tls_group = self.__traffic_lights_groups[i_window]
        for i in range(len(switched_tls)):
            tls_id = tls_group[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if switched_tls[i] and 'y' in state:
                phase_capacity += self.__n_steps_capacity[tls_id] * min(
                    (spent_duration_tls[i] / self.__min_duration) ** 0.1, 1)
                self.__n_steps_capacity[tls_id] = 0
        return phase_capacity * 0.1

    def __calculate_step_capacity(self,
                                  vehicles_on_tls_after: list[list[list[str]]],
                                  switched_tls: list[bool],
                                  spent_duration_tls: list[float],
                                  i_window: int) -> float:
        vehicles_on_tls_before = self.__vehicles_on_lanes_before[i_window]
        tls_group = self.__traffic_lights_groups[i_window]
        local_reward = {tls_id: 0 for tls_id in tls_group}
        for i in range(len(vehicles_on_tls_before)):
            tls_id = tls_group[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if ('y' not in state) or (switched_tls[i] and 'y' in state):
                reward = 0.0
                vehicles_on_lanes_before = vehicles_on_tls_before[i]
                vehicles_on_lanes_after = vehicles_on_tls_after[i]
                for j, vehicles_before in enumerate(vehicles_on_lanes_before):
                    for vehicle_before in vehicles_before:
                        if (vehicle_before not in vehicles_on_lanes_after[j]) and (
                                vehicle_before not in traci.simulation.getArrivedIDList()):
                            reward += 1
                local_reward[tls_id] = reward * min(
                    (spent_duration_tls[i] / self.__min_duration) ** 0.1, 1)
        step_capacity = 0.0
        for tls_id in tls_group:
            step_capacity += local_reward[tls_id]
            self.__n_steps_capacity[tls_id] += local_reward[tls_id]
        # print(
        #     f"step_capacity = {step_capacity}\nvehicles_before = {vehicles_on_tls_before}\nvehicles_after = {vehicles_on_tls_after}\ntls_group = {tls_group}\n")
        return step_capacity * 0.8

    def __update_rewards(self,
                         reward: float,
                         tls_reward: float,
                         step_capacity: float,
                         phase_capacity: float,
                         halting_reward: float) -> None:
        self.__reward += reward
        self.__tls_reward += tls_reward
        self.__step_capacity += step_capacity
        self.__phase_capacity += phase_capacity
        self.__halting_reward += halting_reward

    def __update_total_rewards(self,
                               reward: float,
                               tls_reward: float,
                               step_capacity: float,
                               phase_capacity: float,
                               halting_reward: float) -> None:
        self.__total_reward = reward
        self.__total_tls_reward = tls_reward
        self.__total_step_capacity = step_capacity
        self.__total_phase_capacity = phase_capacity
        self.__total_halting_reward = halting_reward

    def __update_vehicles_on_lanes_before(self) -> None:
        for i in range(len(self.__traffic_lights_groups)):
            vehicles_on_lanes_before = self.__get_vehicles_on_lanes(i)
            self.__vehicles_on_lanes_before[i] = vehicles_on_lanes_before

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.__i_window == 0:
            self.__update_total_rewards(self.__reward, self.__tls_reward, self.__step_capacity, self.__phase_capacity,
                                        self.__halting_reward)
            (self.__reward, self.__tls_reward, self.__step_capacity, self.__phase_capacity,
             self.__halting_reward) = 0, 0, 0, 0, 0
            self.__update_vehicles_on_lanes_before()
            traci.simulationStep()
            self.__local_step += 1
            self.__global_step += 1
            self.__transport_generator.clean_vehicles_data()
        if self.__global_step in self.__schedule and self.__i_window == 0:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)
        tls_reward, switched_tls, spent_duration_tls = self.__change_phase(action, self.__i_window)
        vehicles_on_tls_after = self.__get_vehicles_on_lanes(self.__i_window)
        step_capacity = self.__calculate_step_capacity(vehicles_on_tls_after,  switched_tls, spent_duration_tls,
                                                       self.__i_window)
        phase_capacity = self.__calculate_phase_capacity(switched_tls, spent_duration_tls, self.__i_window)
        halting_reward = self.__calculate_halting_reward(switched_tls, self.__i_window)
        reward = tls_reward + step_capacity + phase_capacity + halting_reward
        self.__update_rewards(reward, tls_reward, step_capacity, phase_capacity, halting_reward)
        observation = self.__get_observation((self.__i_window + 1) % len(self.__traffic_lights_groups))
        info = self.__get_info(self.__total_reward,
                               self.__total_tls_reward,
                               self.__total_step_capacity,
                               self.__total_phase_capacity,
                               self.__total_halting_reward,
                               reward,
                               tls_reward,
                               step_capacity,
                               phase_capacity,
                               halting_reward)
        truncated = (self.__local_step == 3000) and (self.__i_window == len(self.__traffic_lights_groups))
        terminated = False
        self.__i_window = (self.__i_window + 1) % len(self.__traffic_lights_groups)
        return observation, reward, terminated, truncated, info

    def close(self):
        traci.close()

    def get_schedule(self):
        return self.__schedule
