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
        observation_space = {}
        for tls_id in self.traffic_lights_ids:
            n_lanes = len(set(traci.trafficlight.getControlledLanes(tls_id)))
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            tls_info = Dict({
                "density": Box(low=0, high=100, shape=(n_lanes,), dtype=np.float32),
                "waiting": MultiDiscrete([500] * n_lanes),
                # можно сделать масштабируемость в зависимости от длины полосы
                "speed": Box(low=0, high=200, shape=(n_lanes,), dtype=np.float32),
                "phase": Discrete(n_phases),
                "time": Box(low=0, high=600, dtype=np.float32)
            })
            observation_space[tls_id] = tls_info
        self.observation_space = Dict(observation_space)
        self.action_space = MultiBinary(self.n_traffic_lights)
        self.__total_capacity = 0

    def __get_observation(self):
        observation = {}
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            n_lanes = len(lanes)
            density = np.zeros(shape=(n_lanes,), dtype=np.float32)
            waiting = np.zeros(shape=(n_lanes,), dtype=np.int16)
            speed = np.zeros(shape=(n_lanes,), dtype=np.int16)
            for j, lane in enumerate(lanes):
                waiting[j] = traci.lane.getLastStepVehicleNumber(lane)
                density[j] = traci.lane.getLastStepOccupancy(lane)
                speed[j] = traci.lane.getLastStepMeanSpeed(lane)
            tls_info = {
                "density": density,
                "waiting": waiting,
                "speed": speed,
                "phase": self.__unique_phases[traci.trafficlight.getRedYellowGreenState(tls_id)],
                "time": traci.trafficlight.getSpentDuration(tls_id)
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
        info = {"capacity": self.__total_capacity}
        return observation, info

    @staticmethod
    def __change_phase(tls_id, action) -> int:
        penalty = 0
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        n_phases = len(current_logic.phases)
        current_phase = traci.trafficlight.getPhase(tls_id)
        state = current_logic.phases[current_phase].state
        if 'y' not in state and 'Y' not in state:
            if ((action == 0 and traci.trafficlight.getSpentDuration(tls_id) > 120) or
                    (action == 1 and traci.trafficlight.getSpentDuration(tls_id) < 20)):
                penalty = -10
            else:
                penalty = 10
            if action == 0:
                traci.trafficlight.setPhase(tls_id, current_phase)
            else:
                traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        return penalty

    def __calculate_sum_vehicles(self) -> int:
        sum_vehicles = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            sum_vehicles += np.sum([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes])
        return sum_vehicles

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)
        total_penalty = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            total_penalty += self.__change_phase(tls_id, action[i])
        sum_vehicles_before = self.__calculate_sum_vehicles()
        traci.simulationStep()
        passed_vehicles = 10 * (sum_vehicles_before - self.__calculate_sum_vehicles())
        reward = passed_vehicles + total_penalty
        self.local_step += 1
        self.__global_step += 1
        observation = self.__get_observation()
        info = {"capacity": traci.simulation.getArrivedNumber()}
        truncated = self.local_step == 4800
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
        self.__rollout_rewards = np.zeros(shape=(self.training_env.num_envs,), dtype=np.int32)
        self.__rollout_capacity = np.zeros(shape=(self.training_env.num_envs,), dtype=np.int32)

    def _on_step(self) -> bool:
        self.__rollout_rewards = self.__rollout_rewards + (1 / (self.locals["n_steps"] + 1)) * (
                self.locals["rewards"] - self.__rollout_rewards)
        infos = self.locals["infos"]
        for i, info in enumerate(infos):
            self.__rollout_capacity[i] += info["capacity"]
        return True

    def _on_rollout_end(self) -> None:
        mean_rollout_reward = np.mean(self.__rollout_rewards)
        mean_capacity = np.mean(self.__rollout_capacity)
        self.logger.record("rollout/mean_reward", mean_rollout_reward)
        self.logger.record("rollout/mean_capacity", mean_capacity)
