import json

import sumolib
import gymnasium as gym
import traci
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from facade.environment.traffic_lights_env import TrafficLightsEnv, TensorboardCallback, TrafficLightsEnv1
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.logger.logger import *
from facade.net import Net
from facade.structures import SimulationParams
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv


class TrafficScheduler:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__SUMO_CONFIG = sumo_config
        self.__NET_CONFIG = self.__extract_net_path()
        self.__CHECKPOINT_CONFIG = self.__extract_checkpoint_path()
        self.__simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.__net = Net(self.__NET_CONFIG,
                         self.__simulation_params.poisson_generators_edges,
                         self.__simulation_params.CPU_SCALE
                         )
        self.__net.parallel_make_restore_path_matrix()
        self.__net.parallel_find_way_back()
        self.__net.parallel_find_routes()
        self.__route_generator = RouteGenerator(self.__net)
        self.__last_target_nodes_data = []
        self.__transport_generator = TransportGenerator(self.__simulation_params.intensities,
                                                        self.__simulation_params.poisson_generators_edges,
                                                        self.__simulation_params.DURATION, self.__net.get_edges(),
                                                        self.__simulation_params.PART_OF_THE_PATH)
        self.__step = 0
        self.__traffic_logger = Logger("[TrafficInfo]")
        self.__learning_logger = Logger("[LearningInfo]")
        self.__num_envs = 6

    def __extract_net_path(self) -> str:
        slash_position = self.__SUMO_CONFIG.rfind('/')
        extension_position = self.__SUMO_CONFIG.rfind('.sumocfg')
        net_name = f"{self.__SUMO_CONFIG[slash_position + 1:extension_position]}.net.xml"
        return self.__SUMO_CONFIG[:slash_position + 1] + net_name

    def __extract_checkpoint_path(self):
        slash_position = self.__SUMO_CONFIG.rfind('/')
        extension_position = self.__SUMO_CONFIG.rfind('.sumocfg')
        net_name = f"{self.__SUMO_CONFIG[slash_position + 1:extension_position]}_checkpoint.xml"
        return self.__SUMO_CONFIG[:slash_position + 1] + net_name

    def __get_simulation_params_from_file(self) -> SimulationParams:
        with open(self.__simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self) -> None:
        while self.__step < self.__simulation_params.ITERATIONS * self.__simulation_params.INIT_DELAY:
            self.__transport_generator.clean_vehicles_data()
            if self.__step % self.__simulation_params.INIT_DELAY == 0:
                self.__route_generator.make_routes()
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate_transport(self.__last_target_nodes_data)
            self.__step += 1
            traci.simulationStep()
        self.__route_generator.print_all_routes_data_info()
        self.__traffic_logger.print_info(Message.stabilization_of_initial_traffic)
        while self.__step < self.__simulation_params.DURATION:
            self.__transport_generator.clean_vehicles_data()
            if (self.__step % self.__simulation_params.CHECK_TIME == 0 and
                    self.__transport_generator.have_vehicles_passed_part_of_path_in_total()):
                break
            self.__step += 1
            traci.simulationStep()

    @staticmethod
    def __get_unique_phases_dict(turned_on_traffic_lights_ids: list[str]) -> dict[str, int]:
        unique_phases_states = set()
        for tls_id in turned_on_traffic_lights_ids:
            all_programs = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = all_programs.getPhases()
            for phase in phases:
                if phase.state not in unique_phases_states:
                    unique_phases_states.add(phase.state)
        unique_phases_states = list(unique_phases_states)
        unique_phases_states_dict = {unique_phases_states[i]: i for i in range(len(unique_phases_states))}
        return unique_phases_states_dict

    @staticmethod
    def _make_env(turned_on_traffic_lights_ids: list[str],
                  route_generator: RouteGenerator,
                  transport_generator: TransportGenerator,
                  step: int,
                  unique_phases_states: dict[str, int],
                  checkpoint_file: str,
                  sumo_config: str):
        def _init():
            env = TrafficLightsEnv1(
                turned_on_traffic_lights_ids,
                route_generator,
                transport_generator,
                step,
                unique_phases_states,
                checkpoint_file,
                sumo_config)
            return env

        return _init

    def learn(self):
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        self.__net.turn_off_traffic_lights(self.__simulation_params.turned_off_traffic_lights)
        self.__generate_initial_traffic()
        turned_on_traffic_lights_ids = self.__net.get_turned_on_traffic_lights_ids()
        unique_phases_states = self.__get_unique_phases_dict(turned_on_traffic_lights_ids)
        self.__learning_logger.print_info(Message.training_started)
        traci.simulation.saveState(self.__CHECKPOINT_CONFIG)
        vec_env = SubprocVecEnv([self._make_env(turned_on_traffic_lights_ids,
                                                self.__route_generator,
                                                self.__transport_generator,
                                                self.__step,
                                                unique_phases_states,
                                                self.__CHECKPOINT_CONFIG,
                                                self.__SUMO_CONFIG) for _ in range(self.__num_envs)])
        model = A2C(policy='MultiInputPolicy',
                    env=vec_env,
                    device='cuda',
                    tensorboard_log='./a2c_traffic_lights_tensorboard',
                    n_steps=5)
        model.learn(total_timesteps=self.__simulation_params.DURATION,
                    progress_bar=True,
                    callback=TensorboardCallback(),
                    log_interval=10)
        model.save('a2c_crossroad_3')
        traci.close()

    def predict(self, path):
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        self.__net.turn_off_traffic_lights(self.__simulation_params.turned_off_traffic_lights)
        self.__generate_initial_traffic()
        turned_on_traffic_lights_ids = self.__net.get_turned_on_traffic_lights_ids()
        tls_id = turned_on_traffic_lights_ids[0]
        unique_phases_states = self.__get_unique_phases_dict(turned_on_traffic_lights_ids)
        self.__learning_logger.print_info(Message.training_started)
        traci.simulation.saveState(self.__CHECKPOINT_CONFIG)
        traci.close()

        env = TrafficLightsEnv1(
            turned_on_traffic_lights_ids,
            self.__route_generator,
            self.__transport_generator,
            self.__step,
            unique_phases_states,
            self.__CHECKPOINT_CONFIG,
            self.__SUMO_CONFIG,
            gui=True)
        model = A2C.load(path, env=env)
        current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        n_phases = len(current_logic.phases)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(self.__simulation_params.DURATION):
            action, _states = model.predict(obs, deterministic=True)
            if action == [[1]]:
                print("CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            current_phase = traci.trafficlight.getPhase(tls_id)
            if action == [[0]]:
                traci.trafficlight.setPhase(tls_id, current_phase)
            else:
                traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)
            obs, rewards, dones, info = vec_env.step(action)
