from typing import Any, SupportsFloat

import gymnasium as gym
import sumolib
import traci
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict, Box, MultiDiscrete
import numpy as np

from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator


class TrafficLightsEnv(gym.Env):
    def __init__(self, traffic_lights_ids: list[str], schedule: dict[int, list[str]],
                 route_generator: RouteGenerator, transport_generator: TransportGenerator, step):
        print()
        self.__schedule = schedule
        self.__route_generator = route_generator
        self.__transport_generator = transport_generator
        self.__global_step = step
        self.traffic_lights_ids = traffic_lights_ids
        self.n_traffic_lights = len(self.traffic_lights_ids)
        self.observation_space = Dict({
            "speed": Box(low=0, high=100, shape=(self.n_traffic_lights,), dtype=np.float32),
            "count": Box(low=0, high=500, shape=(self.n_traffic_lights,), dtype=np.float32),
            "density": Box(low=0, high=100, shape=(self.n_traffic_lights,), dtype=np.float32),
            "phase": MultiDiscrete([8] * self.n_traffic_lights)
        })
        self.action_space = Box(low=2, high=120, shape=(self.n_traffic_lights,), dtype=np.float32)
        self.local_step = 0

    def __get_observation(self):
        speeds = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        counts = np.zeros(shape=(self.n_traffic_lights,), dtype=np.int32)
        densities = np.zeros(shape=(self.n_traffic_lights,), dtype=np.float32)
        phases = np.zeros(shape=(self.n_traffic_lights,), dtype=np.int8)
        for i, tls_id in enumerate(self.traffic_lights_ids):
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            speeds[i] = np.mean([traci.lane.getLastStepMeanSpeed(lane) for lane in lanes], dtype=np.float32)
            counts[i] = np.mean([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes], dtype=np.float32)
            densities[i] = np.mean([traci.lane.getLastStepOccupancy(lane) for lane in lanes], dtype=np.float32)
            phases[i] = traci.trafficlight.getPhase(tls_id)
        observation = {
            "speed": speeds,
            "count": counts,
            "density": densities,
            "phase": phases
        }

        return observation

    @staticmethod
    def __get_info():
        info = {
            "learning:": "Processing"
        }
        return info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self.__get_observation()
        info = self.__get_info()
        self.local_step = 0
        return observation, info

    @staticmethod
    def __change_phase_duration(tls_id, new_duration):
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        current_phase = traci.trafficlight.getPhase(tls_id)
        new_phases = []
        for i, phase in enumerate(current_logic.phases):
            if i == current_phase:
                new_phase = sumolib.net.Phase(new_duration, phase.state)
            else:
                new_phase = sumolib.net.Phase(phase.duration, phase.state)
            new_phases.append(new_phase)
        new_logic = traci.trafficlight.Logic(
            programID=current_logic.programID,
            type=current_logic.type,
            currentPhaseIndex=current_logic.currentPhaseIndex,
            phases=new_phases
        )
        traci.trafficlight.setProgramLogic(tls_id, new_logic)

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        traci.simulationStep()
        self.local_step += 1
        if self.__global_step in self.__schedule:
            self.__route_generator.make_routes(self.__schedule[self.__global_step])
            last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
            self.__transport_generator.generate(last_target_nodes_data)
        for i, tls_id in enumerate(self.traffic_lights_ids):
            self.__change_phase_duration(tls_id, action[i])
        reward = 0
        observation = self.__get_observation()
        info = self.__get_info()
        for tls_id in self.traffic_lights_ids:
            lanes = set(traci.trafficlight.getControlledLanes(tls_id))
            reward += -np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        truncated = self.local_step >= 50
        self.__global_step += 1
        return observation, reward, False, truncated, info