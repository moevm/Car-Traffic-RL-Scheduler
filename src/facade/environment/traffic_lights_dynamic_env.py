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
                 remain_tls: list[str],
                 edges: list[str],
                 cycle_time: int,
                 truncated_time: int,
                 n_lanes: int,
                 gui: bool = False,
                 train_mode: bool = True):
        self.__cycle_time = cycle_time
        self.__edges = edges
        self.__truncated_time = truncated_time
        self.__gui = gui
        self.__train_mode = train_mode
        self.__n_lanes = n_lanes
        self.__traffic_lights_groups = traffic_lights_groups
        self.__remain_tls = remain_tls
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
        self.__max_duration = 60
        self.__max_tls_neighbors = 4
        self.__critical_duration = 90
        self.__max_phases = 6
        self.__i_window = 0
        observation_space = self.__make_observation_space()
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(self.__group_size)
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.__traffic_lights_ids}
        self.__vehicles_on_lanes_before = {i: [] for i in range(len(self.__traffic_lights_groups))}
        self.__statistics = {
            "mean_halting_number": [],
            "mean_waiting_time": [],
            "mean_speed": [],
            "arrived_number": [],
            "step": []
        }

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
            "accumulated_capacity": Box(low=0, high=10 * self.__critical_duration, shape=(self.__group_size,),
                                        dtype=np.float32),
            "mean_distance": Box(low=0, high=20, shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        })
        return observation_space

    @staticmethod
    def __get_mean_distance_from_tls(tls_id: str, lane_id: str):
        vehicles_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        total_distance = 0
        for vehicle_id in vehicles_ids:
            x_tls, y_tls = traci.junction.getPosition(junctionID=tls_id)
            x_vehicle, y_vehicle = traci.vehicle.getPosition(vehicle_id)
            total_distance += np.sqrt((x_tls - x_vehicle) ** 2 + (y_tls - y_vehicle) ** 2) / traci.lane.getLength(lane_id)
        return total_distance / max(len(vehicles_ids), 1)

    def __get_observation(self, i_window):
        tls_group = self.__traffic_lights_groups[i_window]
        density = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        waiting = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        phase = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        time = np.zeros(shape=(self.__group_size,), dtype=np.int32)
        bounds = np.zeros(shape=(self.__group_size,), dtype=np.int8)
        accumulated_capacity = np.zeros(shape=(self.__group_size,), dtype=np.float32)
        mean_distance = np.zeros(shape=(self.__group_size, self.__n_lanes), dtype=np.float32)
        for i, tls_id in enumerate(tls_group):
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
            for j, lane_id in enumerate(lanes):
                waiting[i, j] = traci.lane.getLastStepHaltingNumber(lane_id) / (
                        traci.lane.getLength(lane_id) / self.__vehicle_size)
                density[i, j] = traci.lane.getLastStepOccupancy(lane_id)
                mean_distance[i, j] = self.__get_mean_distance_from_tls(tls_id, lane_id)
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
            "accumulated_capacity": accumulated_capacity,
            "mean_distance": mean_distance
        }
        return observation

    def __assign_cycle_time_for_remain_tls(self):
        sum_yellow_duration = {tls_id: 0 for tls_id in self.__remain_tls}
        n_yellow_states = {tls_id: 0 for tls_id in self.__remain_tls}
        for tls_id in self.__remain_tls:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = logic.phases
            for phase in phases:
                if 'y' in phase.state:
                    sum_yellow_duration[tls_id] += phase.duration
                    n_yellow_states[tls_id] += 1
        for tls_id in self.__remain_tls:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = logic.phases
            for phase in phases:
                if 'y' not in phase.state:
                    phase.duration = (self.__cycle_time - sum_yellow_duration[tls_id]) // (
                            len(phases) - n_yellow_states[tls_id])
                phase.minDur = phase.duration
                phase.maxDur = phase.duration
            traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.__traffic_lights_ids}
        traci.simulation.loadState(self.__checkpoint_file)
        traci.simulationStep()
        self.__assign_cycle_time_for_remain_tls()
        for tls_id in self.__traffic_lights_ids:
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            current_phase = traci.trafficlight.getPhase(tls_id)
            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        self.__global_step = self.__start_global_step + 1
        self.__schedule = self.__transport_generator.generate_schedule_for_poisson_flow(self.__global_step + 1,
                                                                                        self.__truncated_time)
        self.__update_vehicles_on_lanes_before()
        self.__i_window, self.__local_step = 0, 0
        observation = self.__get_observation(self.__i_window)
        group_rewards = {
            "sum_reward": 0.0,
            "step_capacity": 0.0,
            "tls_reward": 0.0,
            "halting_reward": 0.0
        }
        return observation, group_rewards

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

    def __change_phase(self, action: ActType, i_window: int) -> tuple[float, list[bool], list[float]]:
        tls_group = self.__traffic_lights_groups[i_window]
        switched_tls = [False] * len(tls_group)
        spent_duration_tls = []
        sum_reward = 0
        for i, tls_id in enumerate(tls_group):
            reward = 0
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
                    reward = self.__get_context_reward(max(spent, 1), action[i], -10)
                else:
                    if action[i] == 1:
                        print(f"duration {spent} | tls_id {tls_id}")
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
                    reward = self.__get_context_reward(max(spent, 1), action[i], -10)
            sum_reward += reward
        return sum_reward, switched_tls, spent_duration_tls

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

    def __get_context_reward(self, spent: float, action: int, min_reward: float):
        if action == 1 and spent < self.__min_duration:
            context_reward = min_reward * (1 - spent / self.__min_duration)
        elif action == 0 and spent > self.__max_duration:
            context_reward = min_reward * spent / self.__max_duration
        else:
            context_reward = 0
        return context_reward

    def __calculate_step_capacity(self,
                                  vehicles_on_tls_after: list[list[list[str]]],
                                  switched_tls: list[bool],
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
                vehicles_on_lanes_before = vehicles_on_tls_before[i]
                vehicles_on_lanes_after = vehicles_on_tls_after[i]
                for j, vehicles_before in enumerate(vehicles_on_lanes_before):
                    for vehicle_before in vehicles_before:
                        if (vehicle_before not in vehicles_on_lanes_after[j]) and (
                                vehicle_before not in traci.simulation.getArrivedIDList()):
                            reward += 1
                self.__n_steps_capacity[tls_id] += reward
            elif ('y' in state) and (not switched_tls[i]):
                self.__n_steps_capacity[tls_id] = 0
            sum_step_capacity += reward
        return sum_step_capacity

    def __update_vehicles_on_lanes_before(self) -> None:
        for i in range(len(self.__traffic_lights_groups)):
            vehicles_on_lanes_before = self.__get_vehicles_on_lanes(i)
            self.__vehicles_on_lanes_before[i] = vehicles_on_lanes_before

    def __next_timestep(self) -> None:
        if not self.__train_mode:
            self.__collect_statistics()
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
        tls_reward, switched_tls, spent_duration_tls = self.__change_phase(action, self.__i_window)
        vehicles_on_tls_after = self.__get_vehicles_on_lanes(self.__i_window)
        step_capacity = self.__calculate_step_capacity(vehicles_on_tls_after, switched_tls, self.__i_window)
        halting_reward = self.__calculate_halting_reward(switched_tls, self.__i_window)
        group_rewards = {
            "step_capacity": step_capacity,
            "tls_reward": tls_reward,
            "halting_reward": halting_reward
        }
        group_rewards["sum_reward"] = sum(group_rewards.values())
        reward = sum(group_rewards.values())
        truncated = (self.__local_step == self.__truncated_time) and (
                self.__i_window == len(self.__traffic_lights_groups) - 1)
        terminated = False
        if self.__i_window == len(self.__traffic_lights_groups) - 1:
            self.__next_timestep()
        self.__i_window = (self.__i_window + 1) % len(self.__traffic_lights_groups)
        observation = self.__get_observation(self.__i_window)
        return observation, reward, terminated, truncated, group_rewards

    def __collect_statistics(self):
        halting_number = 0
        waiting_time = 0
        speed = 0
        for edge in self.__edges:
            halting_number += traci.edge.getLastStepHaltingNumber(edge)
            waiting_time += traci.edge.getWaitingTime(edge)
        vehicles = traci.vehicle.getIDList()
        for vehicle in vehicles:
            speed += traci.vehicle.getSpeed(vehicle)
        self.__statistics["mean_halting_number"].append(halting_number / len(self.__edges))
        self.__statistics["mean_waiting_time"].append(waiting_time / len(self.__edges))
        self.__statistics["mean_speed"].append(speed / len(vehicles))
        self.__statistics["arrived_number"].append(traci.simulation.getArrivedNumber())
        self.__statistics["step"].append(self.__local_step + 1)

    def get_statistics(self) -> dict[str, list[float]]:
        return self.__statistics

    def close(self):
        traci.close()

    def get_schedule(self):
        return self.__schedule
