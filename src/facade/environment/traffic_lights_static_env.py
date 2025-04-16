import gymnasium as gym
import traci
import numpy as np

from typing import Any, SupportsFloat
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box, MultiBinary, Discrete

from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator

class TrafficLightsStaticEnv(gym.Env):
    def __init__(self,
                 traffic_lights_ids: list[str],
                 route_generator: RouteGenerator,
                 transport_generator: TransportGenerator,
                 step: int,
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