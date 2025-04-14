import gymnasium as gym
import traci
import numpy as np

from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box, MultiBinary, Discrete, MultiDiscrete

from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from stable_baselines3.common.callbacks import BaseCallback


class TrafficLightsStaticEnv(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
                 unique_phases: dict[str, int],
                 checkpoint_file: str,
                 sumo_config: str,
                 gui: bool = False):
        self.__schedule = transport_generator.generate_schedule_for_poisson_flow(step + 1)
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__global_step = step
        self.traffic_lights_ids = traffic_lights_ids
        self.n_traffic_lights = len(self.traffic_lights_ids)
        self.local_step = 0
        self.__SUMO_CONFIG = sumo_config
        self.__checkpoint_file = checkpoint_file
        if gui:
            sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        else:
            sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__checkpoint_file)
        observation_space = self.__make_observation_space()
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(self.n_traffic_lights)
        self.__total_capacity = 0
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.traffic_lights_ids}
        self.__i_window = 0

    def __make_observation_space(self):
        observation_space = {}
        for tls_id in self.traffic_lights_ids:
            n_lanes = len(set(traci.trafficlight.getControlledLanes(tls_id)))
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            tls_info = Dict({
                "density": Box(low=0, high=100, shape=(n_lanes,), dtype=np.float32),
                "waiting": Box(low=0, high=1, shape=(n_lanes,), dtype=np.float32),
                "phase": Discrete(n_phases),
                "time": Box(low=0, high=600, dtype=np.float32),
            })
            observation_space[tls_id] = tls_info
        return observation_space

    @staticmethod
    def __get_info(tls_reward, step_capacity, phase_capacity, halting_reward, phase_duration):
        info = {
            "capacity": traci.simulation.getArrivedNumber(),
            "tls_reward": tls_reward,
            "step_capacity": step_capacity,
            "phase_capacity": phase_capacity,
            "halting_reward": halting_reward,
            "phase_duration": phase_duration
        }
        return info

    def __get_observation(self):
        observation = {}
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            n_lanes = len(lanes)
            density = np.zeros(shape=(n_lanes,), dtype=np.float32)
            waiting = np.zeros(shape=(n_lanes,), dtype=np.float32)
            speed = np.zeros(shape=(n_lanes,), dtype=np.int16)
            for j, lane in enumerate(lanes):
                waiting[j] = traci.lane.getLastStepHaltingNumber(lane) / traci.lane.getLength(lane)
                density[j] = traci.lane.getLastStepOccupancy(lane)
                speed[j] = traci.lane.getLastStepMeanSpeed(lane)
            tls_info = {
                "density": density,
                "waiting": waiting,
                "phase": traci.trafficlight.getPhase(tls_id),
                "time": traci.trafficlight.getSpentDuration(tls_id),
            }
            observation[tls_id] = tls_info
        return observation

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self.__get_observation()
        self.local_step = 0
        self.__total_capacity = 0
        info = self.__get_info(0, 0, 0, 0, 0)
        return observation, info

    def __change_phase(self, action) -> int:
        tls_reward = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            n_lanes = len(set(traci.trafficlight.getControlledLanes(tls_id)))
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if 'y' not in state and 'Y' not in state:
                if action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) > 120:
                    tls_reward += -n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / 120) ** 0.5
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) < 25:
                    tls_reward += -n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / 25)
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) > 120:
                    tls_reward += n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / 120)
                elif action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) < 25:
                    tls_reward += n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.1
                elif 25 <= traci.trafficlight.getSpentDuration(tls_id) <= 120:
                    tls_reward += n_lanes
                if action[i] == 0:
                    traci.trafficlight.setPhase(tls_id, current_phase)
                else:
                    traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        return tls_reward

    def __get_vehicles_on_lanes(self) -> dict[str, list[list[str]]]:
        vehicles_on_lanes = {}
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            vehicles_on_lanes[tls_id] = []
            for lane in lanes:
                vehicles_on_lanes[tls_id].append(traci.lane.getLastStepVehicleIDs(lane))
        return vehicles_on_lanes

    def __calculate_halting_reward(self):
        sum_halting_number = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            for lane in lanes:
                sum_halting_number += traci.lane.getLastStepHaltingNumber(lane) / traci.lane.getLength(lane)
        return -sum_halting_number

    def __calculate_phase_capacity(self, action):
        phase_capacity = 0
        for i in range(len(action)):
            tls_id = self.traffic_lights_ids[i]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if action[i] == 1 and 'y' not in state and 'Y' not in state:
                phase_capacity += self.__n_steps_capacity[tls_id] * min(
                    (traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.5, 1)
                self.__n_steps_capacity[self.traffic_lights_ids[i]] = 0
        return phase_capacity

    def __calculate_step_capacity(self, vehicles_on_tls_before: dict[str, list[list[str]]],
                                  vehicles_on_tls_after: dict[str, list[list[str]]]) -> int:
        local_reward = {tls_id: 0 for tls_id in self.traffic_lights_ids}
        for tls_id in vehicles_on_tls_before.keys():
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if 'y' not in state and 'Y' not in state:
                reward = 0
                vehicles_on_lanes_before = vehicles_on_tls_before[tls_id]
                vehicles_on_lanes_after = vehicles_on_tls_after[tls_id]
                for i, vehicles_before in enumerate(vehicles_on_lanes_before):
                    for vehicle_before in vehicles_before:
                        if (vehicle_before not in vehicles_on_lanes_after[i]) and (
                                vehicle_before not in traci.simulation.getArrivedIDList()):
                            reward += 1
                local_reward[tls_id] = reward * min((traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.5, 1)
        step_capacity = 0
        for tls_id in self.traffic_lights_ids:
            step_capacity += local_reward[tls_id]
            self.__n_steps_capacity[tls_id] += local_reward[tls_id]
        return step_capacity

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)
        tls_id = self.traffic_lights_ids[0]
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        current_phase = traci.trafficlight.getPhase(tls_id)
        state = current_logic.phases[current_phase].state
        phase_duration = traci.trafficlight.getSpentDuration(tls_id)
        tls_reward = self.__change_phase(action)
        if action[0] == 0 or 'y' in state or 'Y' in state:
            phase_duration = -1
        vehicles_on_tls_before = self.__get_vehicles_on_lanes()
        traci.simulationStep()
        vehicles_on_tls_after = self.__get_vehicles_on_lanes()
        step_capacity = self.__calculate_step_capacity(vehicles_on_tls_before, vehicles_on_tls_after)
        phase_capacity = self.__calculate_phase_capacity(action)
        halting_reward = self.__calculate_halting_reward()
        reward = tls_reward + step_capacity + phase_capacity + halting_reward
        self.local_step += 1
        self.__global_step += 1
        observation = self.__get_observation()
        info = self.__get_info(tls_reward, step_capacity, phase_capacity, halting_reward, phase_duration)
        truncated = self.local_step == 6000
        terminated = False
        self.__transport_generator.clean_vehicles_data()
        return observation, reward, terminated, truncated, info

    def close(self):
        traci.close()

    def get_schedule(self):
        return self.__schedule


class TrafficLightsDynamicEnv(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
                 unique_phases: dict[str, int],
                 checkpoint_file: str,
                 sumo_config: str,
                 traffic_lights_groups: list[list[str]],
                 n_lanes: int,
                 gui: bool = False):
        self.n_lanes = n_lanes
        self.__traffic_lights_groups = traffic_lights_groups
        self.__unique_phases = unique_phases
        self.__schedule = transport_generator.generate_schedule_for_poisson_flow(step + 1)
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__global_step = step
        self.traffic_lights_ids = traffic_lights_ids
        self.n_traffic_lights = len(self.traffic_lights_ids)
        self.local_step = 0
        self.__SUMO_CONFIG = sumo_config
        self.__checkpoint_file = checkpoint_file
        if gui:
            sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        else:
            sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__checkpoint_file)
        observation_space = self.__make_observation_space()
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(4)
        self.__total_capacity = 0
        self.__n_steps_capacity = {tls_id: 0 for tls_id in self.traffic_lights_ids}
        self.__i_window = 0

    def __make_observation_space(self):
        observation_space = Dict({
            "density": Box(low=0, high=100, shape=(4, self.n_lanes * 4), dtype=np.float32),
            "waiting": Box(low=0, high=1, shape=(4, self.n_lanes * 4), dtype=np.float32),
            # "phase": MultiDiscrete([len(self.__unique_phases)] * 4),
            "phase": MultiDiscrete([4] * 4),
            "time": Box(low=0, high=600, shape=(4,), dtype=np.float32)
        })
        return observation_space

    @staticmethod
    def __get_info(tls_reward, step_capacity, phase_capacity, halting_reward, phase_duration):
        info = {
            "capacity": traci.simulation.getArrivedNumber(),
            "tls_reward": tls_reward,
            "step_capacity": step_capacity,
            "phase_capacity": phase_capacity,
            "halting_reward": halting_reward,
            "phase_duration": phase_duration
        }
        return info

    def __get_observation(self, i):
        tls_group = self.__traffic_lights_groups[i]
        tls_id = tls_group[0]
        lanes = set(traci.trafficlight.getControlledLanes(tls_id))
        density = np.zeros(shape=(4, self.n_lanes * 4), dtype=np.float32)
        waiting = np.zeros(shape=(4, self.n_lanes * 4), dtype=np.float32)
        phase = np.zeros(shape=(4,), dtype=np.int32)
        time = np.zeros(shape=(4,), dtype=np.int32)
        for i, tls_id in enumerate(tls_group):
            for j, lane in enumerate(lanes):
                waiting[i, j] = traci.lane.getLastStepHaltingNumber(lane) / traci.lane.getLength(lane)
                density[i, j] = traci.lane.getLastStepOccupancy(lane)
            # phase[i] = self.__unique_phases[traci.trafficlight.getRedYellowGreenState(tls_id)]
            phase[i] = traci.trafficlight.getPhase(tls_id)
            time[i] = traci.trafficlight.getSpentDuration(tls_id)
        observation = {
            "density": density,
            "waiting": waiting,
            "phase": phase,
            "time": time,
        }
        return observation

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self.__get_observation(0)
        self.local_step = 0
        self.__total_capacity = 0
        info = self.__get_info(0, 0, 0, 0, 0)
        return observation, info

    def __change_phase(self, action, i) -> int:
        tls_reward = 0
        tls_group = self.__traffic_lights_groups[i]
        for i, tls_id in enumerate(tls_group):
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            n_lanes = len(set(traci.trafficlight.getControlledLanes(tls_id)))
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if 'y' not in state and 'Y' not in state:
                if action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) > 120:
                    tls_reward += -n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / 120) ** 0.5
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) < 25:
                    tls_reward += -n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / 25)
                elif action[i] == 1 and traci.trafficlight.getSpentDuration(tls_id) > 120:
                    tls_reward += n_lanes * (1 - traci.trafficlight.getSpentDuration(tls_id) / 120)
                elif action[i] == 0 and traci.trafficlight.getSpentDuration(tls_id) < 25:
                    tls_reward += n_lanes * (traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.1
                elif 25 <= traci.trafficlight.getSpentDuration(tls_id) <= 120:
                    tls_reward += n_lanes
                if action[i] == 0:
                    traci.trafficlight.setPhase(tls_id, current_phase)
                else:
                    traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        return tls_reward

    def __get_vehicles_on_lanes(self, i) -> dict[str, list[list[str]]]:
        tls_group = self.__traffic_lights_groups[i]
        vehicles_on_lanes = {}
        for i, tls_id in enumerate(tls_group):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            vehicles_on_lanes[tls_id] = []
            for lane in lanes:
                vehicles_on_lanes[tls_id].append(traci.lane.getLastStepVehicleIDs(lane))
        return vehicles_on_lanes

    def __calculate_halting_reward(self, i):
        tls_group = self.__traffic_lights_groups[i]
        sum_halting_number = 0
        for i, tls_id in enumerate(tls_group):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            for lane in lanes:
                sum_halting_number += traci.lane.getLastStepHaltingNumber(lane) / traci.lane.getLength(lane)
        return -sum_halting_number

    def __calculate_phase_capacity(self, action, i):
        phase_capacity = 0
        tls_group = self.__traffic_lights_groups[i]
        for j in range(len(action)):
            tls_id = tls_group[j]
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if action[j] == 1 and 'y' not in state and 'Y' not in state:
                phase_capacity += self.__n_steps_capacity[tls_id] * min(
                    (traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.5, 1)
                self.__n_steps_capacity[tls_id] = 0
        return phase_capacity

    def __calculate_step_capacity(self, vehicles_on_tls_before: dict[str, list[list[str]]],
                                  vehicles_on_tls_after: dict[str, list[list[str]]], i) -> int:
        tls_group = self.__traffic_lights_groups[i]
        local_reward = {tls_id: 0 for tls_id in tls_group}
        for tls_id in vehicles_on_tls_before.keys():
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            current_phase = traci.trafficlight.getPhase(tls_id)
            state = current_logic.phases[current_phase].state
            if 'y' not in state and 'Y' not in state:
                reward = 0
                vehicles_on_lanes_before = vehicles_on_tls_before[tls_id]
                vehicles_on_lanes_after = vehicles_on_tls_after[tls_id]
                for i, vehicles_before in enumerate(vehicles_on_lanes_before):
                    for vehicle_before in vehicles_before:
                        if (vehicle_before not in vehicles_on_lanes_after[i]) and (
                                vehicle_before not in traci.simulation.getArrivedIDList()):
                            reward += 1
                local_reward[tls_id] = reward * min((traci.trafficlight.getSpentDuration(tls_id) / 25) ** 0.5, 1)
        step_capacity = 0
        for tls_id in tls_group:
            step_capacity += local_reward[tls_id]
            self.__n_steps_capacity[tls_id] += local_reward[tls_id]
        return step_capacity

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)
        tls_id = self.traffic_lights_ids[0]
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        current_phase = traci.trafficlight.getPhase(tls_id)
        state = current_logic.phases[current_phase].state
        phase_duration = traci.trafficlight.getSpentDuration(tls_id)
        tls_reward = self.__change_phase(action, self.__i_window)
        if action[0] == 0 or 'y' in state or 'Y' in state:
            phase_duration = -1
        vehicles_on_tls_before = self.__get_vehicles_on_lanes(self.__i_window)
        traci.simulationStep()
        vehicles_on_tls_after = self.__get_vehicles_on_lanes(self.__i_window)
        step_capacity = self.__calculate_step_capacity(vehicles_on_tls_before, vehicles_on_tls_after, self.__i_window)
        phase_capacity = self.__calculate_phase_capacity(action, self.__i_window)
        halting_reward = self.__calculate_halting_reward(self.__i_window)
        reward = tls_reward + step_capacity + phase_capacity + halting_reward
        self.__i_window = (self.__i_window + 1) % len(self.__traffic_lights_groups)
        observation = self.__get_observation(self.__i_window)
        self.local_step += 1
        self.__global_step += 1
        info = self.__get_info(tls_reward, step_capacity, phase_capacity, halting_reward, phase_duration)
        truncated = self.local_step == 30_000
        terminated = False
        self.__transport_generator.clean_vehicles_data()
        return observation, reward, terminated, truncated, info

    def close(self):
        traci.close()

    def get_schedule(self):
        return self.__schedule


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.__rollout_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.int32)
        self.__rollout_tls_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_step_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_phase_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)
        self.__rollout_halting_reward = np.zeros(shape=(self.training_env.num_envs,), dtype=np.float32)

    def _on_step(self) -> bool:
        self.__rollout_rewards = self.__rollout_rewards + (1 / (self.locals["n_steps"] + 1)) * (
                self.locals["rewards"] - self.__rollout_rewards)
        infos = self.locals["infos"]
        tls_reward, step_capacity, phase_capacity, halting_reward = 0, 0, 0, 0
        for i, info in enumerate(infos):
            self.__rollout_capacity[i] += info["capacity"]
            self.__rollout_tls_rewards[i] = self.__rollout_tls_rewards[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["tls_reward"] - self.__rollout_tls_rewards[i])
            self.__rollout_step_capacity[i] = self.__rollout_step_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["step_capacity"] - self.__rollout_step_capacity[i])
            self.__rollout_phase_capacity[i] = self.__rollout_phase_capacity[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["phase_capacity"] - self.__rollout_phase_capacity[i])
            self.__rollout_halting_reward[i] = self.__rollout_halting_reward[i] + (1 / (self.locals["n_steps"] + 1)) * (
                    info["halting_reward"] - self.__rollout_halting_reward[i])
            tls_reward = tls_reward + (1 / len(infos)) * (info["tls_reward"] - tls_reward)
            step_capacity = step_capacity + (1 / len(infos)) * (info["step_capacity"] - step_capacity)
            phase_capacity = phase_capacity + (1 / len(infos)) * (info["phase_capacity"] - phase_capacity)
            halting_reward = halting_reward + (1 / len(infos)) * (info["halting_reward"] - halting_reward)
        self.logger.record("step/tls_reward", tls_reward)
        self.logger.record("step/step_capacity", step_capacity)
        self.logger.record("step/phase_capacity", phase_capacity)
        self.logger.record("step/halting_reward", halting_reward)
        total_policy_norm = 0.0
        for p in self.model.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_policy_norm += param_norm.item() ** 2
        total_policy_norm = total_policy_norm ** 0.5
        self.logger.record("step/policy_norm", total_policy_norm)
        total_value_norm = 0.0
        for p in self.model.policy.value_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_value_norm += param_norm.item() ** 2
        total_value_norm = total_value_norm ** 0.5
        self.logger.record("step/value norm", total_value_norm)
        if infos[0]["phase_duration"] != -1:
            self.logger.record("step/phase_duration", infos[0]["phase_duration"])
        return True

    def _on_rollout_end(self) -> None:
        mean_rollout_reward = np.mean(self.__rollout_rewards)
        mean_rollout_tls_reward = np.mean(self.__rollout_tls_rewards)
        mean_rollout_step_capacity = np.mean(self.__rollout_step_capacity)
        mean_rollout_phase_capacity = np.mean(self.__rollout_phase_capacity)
        mean_capacity = np.mean(self.__rollout_capacity)
        mean_rollout_halting_reward = np.mean(self.__rollout_halting_reward)
        self.logger.record("rollout/mean_reward", mean_rollout_reward)
        self.logger.record("rollout/mean_tls_reward", mean_rollout_tls_reward)
        self.logger.record("rollout/mean_step_capacity", mean_rollout_step_capacity)
        self.logger.record("rollout/mean_phase_capacity", mean_rollout_phase_capacity)
        self.logger.record("rollout/mean_halting_reward", mean_rollout_halting_reward)
        self.logger.record("rollout/mean_capacity", mean_capacity)
