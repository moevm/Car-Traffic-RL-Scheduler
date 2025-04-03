import gymnasium as gym
import sumolib
import traci
import numpy as np

from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box, MultiDiscrete
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from stable_baselines3.common.callbacks import BaseCallback


class TrafficLightsEnv(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
                 unique_phases: dict[str, int],
                 checkpoint_file: str,
                 sumo_config: str):
        self.__unique_phases = unique_phases
        self.__schedule = transport_generator.generate_schedule_for_poisson_flow(step + 1)
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__global_step = step
        self.traffic_lights_ids = traffic_lights_ids
        self.n_traffic_lights = len(self.traffic_lights_ids)
        self.observation_space = Dict({
            "density": Box(low=0, high=500, shape=(self.n_traffic_lights,), dtype=np.float32),
            "waiting": Box(low=0, high=500, shape=(self.n_traffic_lights,), dtype=np.float32),
            "phase": MultiDiscrete([len(unique_phases)] * self.n_traffic_lights)
        })
        self.action_space = Box(low=-1, high=1, shape=(self.n_traffic_lights,), dtype=np.float32)
        self.local_step = 0
        self.__SUMO_CONFIG = sumo_config
        self.__checkpoint_file = checkpoint_file
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__checkpoint_file)
        self.__total_capacity = 0

    def __get_observation(self):
        waiting = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        densities = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        phases = np.zeros(shape=(self.n_traffic_lights,), dtype=np.int8)
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            waiting[i] = np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes], dtype=np.float32)
            densities[i] = np.mean([traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane)
                                    for lane in lanes], dtype=np.float32)
            phases[i] = self.__unique_phases[traci.trafficlight.getRedYellowGreenState(tls_id)]
        observation = {
            "density": densities,
            "waiting": waiting,
            "phase": phases
        }
        return observation

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self.__get_observation()
        self.local_step = 0
        self.__total_capacity += traci.simulation.getArrivedNumber()
        info = {"capacity": traci.simulation.getArrivedNumber()}
        return observation, info

    @staticmethod
    def __change_phase_duration(tls_id, new_duration):
        new_duration = 59 * new_duration + 61  # rescale
        penalty = 0
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        n_phases = len(current_logic.phases)
        current_phase = traci.trafficlight.getPhase(tls_id)
        state = current_logic.phases[current_phase].state
        if 'y' not in state and 'Y' not in state:
            if new_duration > 120 or new_duration < 20:
                penalty = -1000
            traci.trafficlight.setPhaseDuration(tls_id, new_duration)
        return penalty
        # current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        # current_phase = traci.trafficlight.getPhase(tls_id)
        # new_phases = []
        # for i, phase in enumerate(current_logic.phases):
        #     if i == current_phase and 'y' not in phase.state and 'Y' not in phase.state:
        #         new_phase = sumolib.net.Phase(new_duration, phase.state)
        #     else:
        #         new_phase = sumolib.net.Phase(phase.duration, phase.state)
        #     new_phases.append(new_phase)
        # new_logic = traci.trafficlight.Logic(
        #     programID=current_logic.programID,
        #     type=current_logic.type,
        #     currentPhaseIndex=current_logic.currentPhaseIndex,
        #     phases=new_phases
        # )
        # traci.trafficlight.setProgramLogic(tls_id, new_logic)

    def __calculate_reward(self) -> int:
        sum_waiting = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            sum_waiting += np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes], dtype=np.float32)
        return traci.simulation.getArrivedNumber() - sum_waiting

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate_transport(last_target_nodes_data)
        for i, tls_id in enumerate(self.traffic_lights_ids):
            self.__change_phase_duration(tls_id, action[i])
        reward = self.__calculate_reward()
        observation = self.__get_observation()
        info = {"capacity": traci.simulation.getArrivedNumber()}
        truncated = self.local_step >= 99
        traci.simulationStep()
        self.local_step += 1
        self.__global_step += 1
        self.__transport_generator.clean_vehicles_data()
        return observation, reward, False, truncated, info

    def close(self):
        traci.close()


class TrafficLightsEnv1(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
                 unique_phases: dict[str, int],
                 checkpoint_file: str,
                 sumo_config: str,
                 gui: bool = False):
        super().__init__()
        self.__unique_phases = unique_phases
        self.__schedule = transport_generator.generate_schedule_for_poisson_flow(step + 1)
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__global_step = step
        self.traffic_lights_ids = traffic_lights_ids
        self.n_traffic_lights = len(self.traffic_lights_ids)
        self.observation_space = Dict({
            "density": Box(low=0, high=500, shape=(self.n_traffic_lights,), dtype=np.float32),
            "waiting": Box(low=0, high=500, shape=(self.n_traffic_lights,), dtype=np.float32),
            "phase": MultiDiscrete([len(unique_phases)] * self.n_traffic_lights)
        })
        self.action_space = MultiDiscrete([2] * self.n_traffic_lights)
        self.local_step = 0
        self.__SUMO_CONFIG = sumo_config
        self.__checkpoint_file = checkpoint_file
        if gui:
            sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        else:
            sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__checkpoint_file)
        self.__departed_vehicles = set()
        self.__remain_vehicles = set()
        self.__total_capacity = 0

    def __get_observation(self):
        waiting = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        densities = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        phases = np.zeros(shape=(self.n_traffic_lights,), dtype=np.int8)
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            waiting[i] = np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes], dtype=np.float32)
            densities[i] = np.mean([traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane)
                                    for lane in lanes], dtype=np.float32)
            phases[i] = self.__unique_phases[traci.trafficlight.getRedYellowGreenState(tls_id)]
        observation = {
            "density": densities,
            "waiting": waiting,
            "phase": phases
        }
        return observation

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self.__get_observation()
        self.local_step = 0
        self.__total_capacity += traci.simulation.getArrivedNumber()
        info = {"capacity": traci.simulation.getArrivedNumber()}
        self.__departed_vehicles = set(traci.simulation.getDepartedIDList())
        self.__remain_vehicles = set(traci.simulation.getDepartedIDList())
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
                penalty = -10000
            if action == 0:
                traci.trafficlight.setPhase(tls_id, current_phase)
            else:
                traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
        return penalty

    def __calculate_reward(self) -> int:
        sum_waiting = 0
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            sum_waiting += np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes], dtype=np.float32)
        return traci.simulation.getArrivedNumber() - sum_waiting

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
        traci.simulationStep()
        self.local_step += 1
        self.__global_step += 1
        reward = self.__calculate_reward() + total_penalty
        observation = self.__get_observation()
        info = {"capacity": traci.simulation.getArrivedNumber()}
        truncated = self.local_step == 1000
        if truncated:
            reward -= 100
        # for vehicle_id in traci.simulation.getArrivedIDList():
        #     if vehicle_id in self.__departed_vehicles:
        #         self.__remain_vehicles.remove(vehicle_id)
        terminated = False
        self.__transport_generator.clean_vehicles_data()
        return observation, reward, terminated, truncated, info

    def close(self):
        traci.close()


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
