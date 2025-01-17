import random
import numpy as np
import traci

from Facade.Generation.RouteGeneration import RouteGeneration, StartNodeType
from Facade.Generation.TransportGeneration import TransportGeneration
from Facade.Net import Net


class Facade:
    def __init__(self, config_file: str):
        self.__start_node_type = StartNodeType
        self.__steps_of_simulation = 1000
        self.config_file = config_file
        self.net = Net(self.config_file)
        self.__poisson_generators = self.net.get_poisson_generators()
        self.__traffic_intensity = [random.uniform(0, 0.5) for _ in range(len(self.__poisson_generators))]

        self.__routes = RouteGeneration(self.net)
        self.__transport = TransportGeneration()
        self.__last_target_nodes_data = []

    def __generate_initial_traffic(self):
        step = 0
        traci.simulationStep()
        self.__routes.make_routes(self.__start_node_type.extreme_node)
        self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()
        self.__transport.generate(self.__last_target_nodes_data)
        self.__routes.print_all_routes_data_info()
        while step < self.__steps_of_simulation:
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
        sumo_cmd = ["sumo-gui", "-c", self.config_file]
        traci.start(sumo_cmd)
        self.__generate_initial_traffic()
        traci.close()
