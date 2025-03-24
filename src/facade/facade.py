import json

import traci
import torch
from facade.agents.traffic_lights_env import TrafficLightsEnv
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.net import Net
from facade.structures import SimulationParams
from facade.traffic_control import TrafficControl
from stable_baselines3 import A2C


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__SUMO_CONFIG = sumo_config
        self.__NET_CONFIG = self.__extract_net_config()
        self.__simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.__net = Net(self.__NET_CONFIG)
        self.__net.init_poisson_generators(self.__simulation_params.poisson_generators_edges)
        self.__net.parallel_make_restore_path_matrix()
        self.__net.parallel_find_way_back()
        self.__route_generator = RouteGenerator(self.__net)
        self.__transport_generator = TransportGenerator()
        self.__last_target_nodes_data = []
        self.__traffic_control = TrafficControl(self.__simulation_params.intensities,
                                                self.__simulation_params.poisson_generators_edges,
                                                self.__simulation_params.DURATION, self.__net.get_edges(),
                                                self.__simulation_params.PART_OF_THE_PATH)
        self.__step = 0

    def __extract_net_config(self):
        slash_position = self.__SUMO_CONFIG.rfind('/')
        extension_position = self.__SUMO_CONFIG.rfind('.sumocfg')
        net_name = f"{self.__SUMO_CONFIG[slash_position + 1:extension_position]}.net.xml"
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
                self.__transport_generator.generate(self.__last_target_nodes_data)
                # self.__route_generator.print_all_routes_data_info()
            self.__step += 1
            traci.simulationStep()
        self.__traffic_control.init_vehicles_data(self.__transport_generator.get_vehicles_data())
        while self.__step < self.__simulation_params.DURATION:
            traci.simulationStep()
            self.__step += 1
            if (self.__step % self.__simulation_params.CHECK_TIME == 0 and
                    self.__traffic_control.have_vehicles_passed_part_of_path_in_total()):
                break

    def __generate_main_traffic(self) -> None:
        schedule = self.__traffic_control.generate_schedule_for_poisson_flow(self.__step + 1)
        turned_on_traffic_lights_ids = self.__net.get_turned_on_traffic_lights_ids()
        env = TrafficLightsEnv(turned_on_traffic_lights_ids, schedule, self.__route_generator,
                               self.__transport_generator, self.__step)
        model = A2C(policy='MultiInputPolicy', env=env, device='cuda')
        model.learn(total_timesteps=10000)
        for tls_id in turned_on_traffic_lights_ids:
            all_programs = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            print(tls_id, all_programs)

    def execute(self) -> None:
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulationStep()
        self.__net.turn_off_traffic_lights(self.__simulation_params.turned_off_traffic_lights)
        self.__generate_initial_traffic()
        self.__generate_main_traffic()
        traci.close()
