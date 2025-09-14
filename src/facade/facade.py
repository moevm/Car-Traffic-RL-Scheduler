import json
from typing import Callable

import numpy as np
import traci
import os
from sb3_contrib import RecurrentPPO
from triton.language import dtype

from facade.learning_algorithm.policy import MaskableRecurrentActorCriticPolicy
from facade.learning_algorithm.maskable_recurrent_ppo import MaskableRecurrentPPO
from facade.environment.tensorboard_callback import TensorboardCallback
from facade.environment.traffic_lights_dynamic_env import TrafficLightsDynamicEnv
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.logger.logger import *
from facade.net import Net
from facade.structures import SimulationParams
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn
import pandas as pd


class TrafficScheduler:
    def __init__(self, sumo_config: str, simulation_parameters_file: str, new_checkpoint: str, enable_gui: bool,
                 cycle_time: int):
        self.__cycle_time = cycle_time
        self.__enable_gui = enable_gui
        self.__new_checkpoint = new_checkpoint
        self.__SUMO_CONFIG = sumo_config
        self.__NET_CONFIG = self.__extract_net_path()
        self.__CHECKPOINT_CONFIG = self.__extract_checkpoint_path()
        self.__simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.__net = Net(self.__NET_CONFIG,
                         self.__simulation_params.poisson_generators_edges,
                         self.__simulation_params.CPU_SCALE,
                         )
        self.__net.parallel_make_restore_path_matrix()
        self.__net.parallel_find_way_back()
        self.__net.parallel_find_routes()
        self.__route_generator = RouteGenerator(self.__net)
        self.__last_target_nodes_data = []
        self.__eval_duration: int
        self.__transport_generator = TransportGenerator(self.__simulation_params.intensities,
                                                        self.__simulation_params.poisson_generators_edges,
                                                        self.__simulation_params.DURATION, self.__net.get_edges(),
                                                        self.__simulation_params.PART_OF_THE_PATH)
        self.__step = 0
        self.__traffic_logger = Logger("[TrafficInfo]")
        self.__learning_logger = Logger("[LearningInfo]")
        self.__num_envs = 8
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
        return f"configs/checkpoints/{net_name}"

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
                          remain_tls: list[str],
                          n_lanes: int,
                          edges: list[str],
                          cycle_time: int,
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
                remain_tls,
                edges,
                cycle_time,
                truncated_time,
                n_lanes,
                gui,
                train_mode)
            return env

        return _init

    def __save_statistics(self, statistics: dict):
        name = self.__SUMO_CONFIG.replace('configs/', '')
        name = name.replace('/', '_')
        name = name.replace('.sumocfg', '')
        new_data = {}
        json_path = f"statistics/runs_{name}"
        for key, value in statistics.items():
            if key != 'step' and key != 'arrived_number':
                new_data[f"mean_{key}"] = [sum(value)]
            elif key == 'arrived_number':
                new_data[f"sum_{key}"] = [sum(value)]
        if os.path.isfile(json_path):
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                keys = new_data.keys()
                for key in keys:
                    data[key].extend(new_data[key])
        else:
            data = new_data
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        df = pd.DataFrame(statistics)
        os.makedirs('statistics', exist_ok=True)
        df.to_csv(f'statistics/{name}')

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
        self.__remain_tls = self.__net.get_remain_tls()
        self.__n_lanes = self.__net.get_number_of_lanes() * 4
        self.__learning_logger.print_info(Message.training_started)
        self.__edges = self.__net.get_edges()
        if self.__new_checkpoint:
            traci.simulation.saveState(self.__CHECKPOINT_CONFIG)
        traci.close()

    @staticmethod
    def __exp_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return initial_value * np.exp(-7 * (1 - progress_remaining))
        return func

    def learn(self):
        self.__setup_start_simulation_state()
        if not self.__new_checkpoint:
            n_steps = 15 * len(self.__traffic_lights_groups)
            vec_env = SubprocVecEnv([self._make_env_dynamic(self.__turned_on_traffic_lights,
                                                            self.__route_generator,
                                                            self.__transport_generator,
                                                            self.__step,
                                                            self.__CHECKPOINT_CONFIG,
                                                            self.__SUMO_CONFIG,
                                                            self.__traffic_lights_groups,
                                                            self.__remain_tls,
                                                            self.__n_lanes,
                                                            self.__edges,
                                                            self.__cycle_time,
                                                            truncated_time=n_steps * 50,
                                                            gui=(i == 0) and self.__enable_gui,
                                                            train_mode=True) for i in range(self.__num_envs)])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                                   norm_obs_keys=["density", "waiting", "time"])
            os.makedirs('metrics_logs', exist_ok=True)
            os.makedirs('pretrained_info', exist_ok=True)
            model = MaskableRecurrentPPO(policy=MaskableRecurrentActorCriticPolicy,
                                         env=vec_env,
                                         tensorboard_log='./metrics_logs',
                                         learning_rate=self.__exp_schedule(0.001),
                                         n_steps=n_steps,
                                         batch_size=n_steps,
                                         max_grad_norm=2.0,
                                         normalize_advantage=True,
                                         gae_lambda=0.95,
                                         ent_coef=0.01,
                                         vf_coef=0.5,
                                         device='cuda')
            model.learn(total_timesteps=self.__simulation_params.DURATION,
                        progress_bar=True,
                        callback=TensorboardCallback(len(self.__traffic_lights_groups[0])),
                        log_interval=1)
            vec_env.save('vec_normalized.pkl')
            vec_env.close()
            model.save('trained_model')

    def trained_model_evaluation(self, normalized_env: str, model_weights: str, duration: int) -> None:
        self.__eval_duration = duration
        self.__setup_start_simulation_state()
        if not self.__new_checkpoint:
            vec_env = DummyVecEnv([self._make_env_dynamic(self.__turned_on_traffic_lights,
                                                          self.__route_generator,
                                                          self.__transport_generator,
                                                          self.__step,
                                                          self.__CHECKPOINT_CONFIG,
                                                          self.__SUMO_CONFIG,
                                                          self.__traffic_lights_groups,
                                                          self.__remain_tls,
                                                          self.__n_lanes,
                                                          self.__edges,
                                                          self.__cycle_time,
                                                          truncated_time=duration,
                                                          gui=self.__enable_gui,
                                                          train_mode=False)])
            vec_env = VecNormalize.load(normalized_env, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

            model = MaskableRecurrentPPO.load(model_weights, env=vec_env, device='cpu')
            model_env = model.get_env()
            obs = model_env.reset()
            lstm_states = None
            episode_starts = np.ones((model_env.num_envs,), dtype=np.float32)
            action_mask = np.zeros((model_env.num_envs, len(self.__traffic_lights_groups[0])), dtype=np.float32)
            while True:
                action, lstm_states = model.predict_maskable(
                    obs,
                    action_mask,
                    state=lstm_states,
                    deterministic=True,
                    episode_start=episode_starts
                )
                obs, rewards, dones, infos = model_env.step(action)
                episode_starts = dones.astype(np.float32)
                for i, info in enumerate(infos):
                    action_mask[i] = info["action_mask"]
                if np.all(dones):
                    break
            agent_statistics = vec_env.venv.envs[0].unwrapped.get_statistics()
            model_env.close()
            self.__save_statistics(agent_statistics)

    def __reset_tls_after_loading(self):
        traci.simulationStep()
        for tls_id in self.__turned_on_traffic_lights:
            current_logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            n_phases = len(current_logic.phases)
            current_phase = traci.trafficlight.getPhase(tls_id)
            traci.trafficlight.setPhase(tls_id, (current_phase + 1) % n_phases)

    def default_agent_evaluation(self, duration: int) -> None:
        self.__setup_start_simulation_state()
        if not self.__new_checkpoint:
            if self.__enable_gui:
                sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
            else:
                sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
            traci.start(sumo_cmd)
            traci.simulation.loadState(self.__CHECKPOINT_CONFIG)
            self.__reset_tls_after_loading()
            default_agent_statistics = {
                "mean_halting_number": [],
                "mean_waiting_time": [],
                "mean_speed": [],
                "arrived_number": [],
                "step": []
            }
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
