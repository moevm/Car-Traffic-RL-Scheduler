import random
import json
import numpy as np
import traci

from facade.generation.route_generator import RouteGenerator, StartNodeType
from facade.generation.transport_generator import TransportGenerator
from facade.net import Net
from facade.structures import SimulationParams


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__start_node_type = StartNodeType
        self.sumo_config = sumo_config
        self.simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.net = Net(self.sumo_config, self.__simulation_params)
        self.__routes = RouteGenerator(self.net)
        self.__transport = TransportGenerator()
        self.__last_target_nodes_data = []

    def __get_simulation_params_from_file(self):
        with open(self.simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self):
        step = 0
        traci.simulationStep()
        self.__routes.make_routes(self.__start_node_type.extreme_node)
        self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()
        self.__transport.generate(self.__last_target_nodes_data)
        self.__routes.print_all_routes_data_info()
        while step < self.__simulation_params.duration:
            traci.simulationStep()
            '''
            здесь надо брать время отправления автомобилей из экспоненциалього распределения
            также надо сделать возможность построения статистики для проверки, является ли поток реально пуассоновским
            '''
            if step % 10 == 0:
                self.__routes.make_routes(self.__start_node_type.poisson_generator)
                self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()
                self.__transport.generate(self.__last_target_nodes_data)
                self.__routes.print_all_routes_data_info()
            step += 1

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", self.sumo_config]
        traci.start(sumo_cmd)
        self.__generate_initial_traffic()
        traci.close()
