import json

import numpy as np
import traci
import os
from sb3_contrib import RecurrentPPO
from facade.environment.tensorboard_callback import TensorboardCallback
from facade.environment.traffic_lights_dynamic_env import TrafficLightsDynamicEnv
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.logger.logger import *
from facade.net import Net
from facade.structures import SimulationParams
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3 import PPO
import pandas as pd


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
        self.__num_envs = 4
        self.__turned_on_traffic_lights = []
        self.__traffic_lights_groups = []
        self.__n_lanes: int
        self.__edges = self.__net.get_edges()

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
    def _make_env_dynamic(turned_on_traffic_lights_ids: list[str],
                          route_generator: RouteGenerator,
                          transport_generator: TransportGenerator,
                          step: int,
                          checkpoint_file: str,
                          sumo_config: str,
                          traffic_lights_groups: list[list[str]],
                          n_lanes: int,
                          edges: list[str],
                          truncated_time: int,
                          gui: bool,
                          train_mode: bool):
        def _init():
            env = TrafficLightsDynamicEnv(
                turned_on_traffic_lights_ids,
                route_generator,
                transport_generator,
                step,
                checkpoint_file,
                sumo_config,
                traffic_lights_groups,
                edges,
                truncated_time,
                n_lanes,
                gui,
                train_mode)
            return env

        return _init

    def __save_statistics(self, statistics):
        statistics['sumocfg-file'] = self.__SUMO_CONFIG
        filename = 'statistics.json'
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        data.update(statistics)
        with open(filename, 'w') as json_file:
            json.dump(statistics, json_file, indent=4)

    def __setup_start_simulation_state(self):
        sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        self.__net.turn_off_traffic_lights(self.__simulation_params.turned_off_traffic_lights)
        self.__net.make_traffic_lights_groups()
        print(f"GROUPS: {self.__net.get_traffic_lights_groups()}")
        print(f"len_groups = {len(self.__net.get_traffic_lights_groups())}")
        self.__generate_initial_traffic()
        self.__turned_on_traffic_lights = self.__net.get_turned_on_traffic_lights_ids()
        self.__traffic_lights_groups = self.__net.get_traffic_lights_groups()
        self.__n_lanes = self.__net.get_number_of_lanes() * 4
        self.__learning_logger.print_info(Message.training_started)
        self.__edges = self.__net.get_edges()
        traci.simulation.saveState(self.__CHECKPOINT_CONFIG)
        traci.close()

    def learn(self):
        self.__setup_start_simulation_state()
        n_steps = 60 * len(self.__traffic_lights_groups)
        vec_env = SubprocVecEnv([self._make_env_dynamic(self.__turned_on_traffic_lights,
                                                        self.__route_generator,
                                                        self.__transport_generator,
                                                        self.__step,
                                                        self.__CHECKPOINT_CONFIG,
                                                        self.__SUMO_CONFIG,
                                                        self.__traffic_lights_groups,
                                                        self.__n_lanes,
                                                        self.__edges,
                                                        truncated_time=n_steps * 10,
                                                        gui=i == 0,
                                                        train_mode=True) for i in range(self.__num_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               norm_obs_keys=["density", "waiting", "time"])
        model = PPO(policy='MultiInputPolicy',
                    env=vec_env,
                    tensorboard_log='./ppo_traffic_lights_tensorboard',
                    learning_rate=get_linear_fn(start=0.0003, end=0.000012, end_fraction=0.5),
                    n_steps=n_steps,
                    batch_size=n_steps // 4,
                    max_grad_norm=0.8,
                    normalize_advantage=True,
                    gae_lambda=0.95,
                    ent_coef=0.005,
                    vf_coef=0.5,
                    device='cuda')
        model.learn(total_timesteps=self.__simulation_params.DURATION,
                    progress_bar=True,
                    callback=TensorboardCallback(len(self.__traffic_lights_groups[0])),
                    log_interval=1)
        vec_env.save('vec_normalize.pkl')
        vec_env.close()
        model.save('trained_model')

    def additional_learning(self):
        self.__setup_start_simulation_state()
        vec_env = SubprocVecEnv([self._make_env_dynamic(self.__turned_on_traffic_lights,
                                                        self.__route_generator,
                                                        self.__transport_generator,
                                                        self.__step,
                                                        self.__CHECKPOINT_CONFIG,
                                                        self.__SUMO_CONFIG,
                                                        self.__traffic_lights_groups,
                                                        self.__n_lanes,
                                                        self.__edges,
                                                        truncated_time=5999,
                                                        gui=i == 0,
                                                        train_mode=True) for i in range(self.__num_envs)])
        vec_env = VecNormalize.load('pre_vec_normalize_6731200.pkl', vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        vec_env.norm_obs = True
        custom_params = {'learning_rate': get_linear_fn(start=0.000266, end=0.000012, end_fraction=0.5)}
        model = RecurrentPPO.load('pre_trained_model_1200000', env=vec_env, device='cuda', custom_objects=custom_params)
        model.learn(total_timesteps=self.__simulation_params.DURATION,
                    progress_bar=True,
                    callback=TensorboardCallback(len(self.__traffic_lights_groups[0])),
                    log_interval=1,
                    reset_num_timesteps=True)
        vec_env.save('addition_vec_normalize.pkl')
        vec_env.close()
        model.save('additional_trained_model')

    def trained_model_evaluation(self) -> None:
        self.__setup_start_simulation_state()
        vec_env = DummyVecEnv([self._make_env_dynamic(self.__turned_on_traffic_lights,
                                                      self.__route_generator,
                                                      self.__transport_generator,
                                                      self.__step,
                                                      self.__CHECKPOINT_CONFIG,
                                                      self.__SUMO_CONFIG,
                                                      self.__traffic_lights_groups,
                                                      self.__n_lanes,
                                                      self.__edges,
                                                      truncated_time=5999,
                                                      gui=True,
                                                      train_mode=False)])
        vec_env = VecNormalize.load('vec_normalize.pkl', vec_env)
        vec_env.training, vec_env.norm_reward = False, False
        model = PPO.load('trained_model', env=vec_env, device='cpu')
        model_env = model.get_env()
        obs = model_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = model_env.step(action)
            if all(dones):
                break
        agent_statistics = vec_env.venv.envs[0].unwrapped.get_statistics()
        model_env.close()
        self.__save_statistics(agent_statistics)

    def default_agent_evaluation(self) -> None:
        self.__setup_start_simulation_state()
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulation.loadState(self.__CHECKPOINT_CONFIG)
        traci.simulationStep()
        default_agent_statistics = {
            "mean_halting_number": [],
            "running_cars": [],
            "mean_waiting_time": [],
            "mean_speed": [],
            "arrived_number": [],
            "step": []
        }
        duration = 6000
        schedule = self.__transport_generator.generate_schedule_for_poisson_flow(self.__step + 1, duration)
        edges = self.__net.get_edges()
        for i in range(duration):
            halting_number = 0
            waiting_time = 0
            speed = 0
            for edge in edges:
                halting_number += traci.edge.getLastStepHaltingNumber(edge)
                waiting_time += traci.edge.getWaitingTime(edge)
            vehicles = traci.vehicle.getIDList()
            for vehicle in vehicles:
                speed += traci.vehicle.getSpeed(vehicle)
            default_agent_statistics["mean_halting_number"].append(halting_number / len(edges))
            default_agent_statistics["running_cars"].append(len(vehicles))
            default_agent_statistics["mean_waiting_time"].append(waiting_time / len(edges))
            default_agent_statistics["mean_speed"].append(speed / len(vehicles))
            default_agent_statistics["arrived_number"].append(traci.simulation.getArrivedNumber())
            default_agent_statistics["step"].append(i + 1)
            traci.simulationStep()
            self.__step += 1
            if self.__step in schedule:
                self.__route_generator.make_routes(schedule[self.__step])
                last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate_transport(last_target_nodes_data)
            self.__transport_generator.clean_vehicles_data()
        traci.close()
        self.__save_statistics(default_agent_statistics)
