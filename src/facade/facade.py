import random
import json
import numpy as np
import traci

from facade.generation.route_generator import RouteGenerator, StartNodeType
from facade.generation.transport_generator import TransportGenerator
from facade.net import Net
from facade.structures import SimulationParams
from facade.poisson_flow_control import PoissonFlowControl


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__start_node_type = StartNodeType
        self.sumo_config = sumo_config
        self.simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.net = Net(self.sumo_config, self.__simulation_params)
        self.__route_generator = RouteGenerator(self.net)
        self.__transport_generator = TransportGenerator()
        self.__last_target_nodes_data = []
        self.__poisson_flow_control = PoissonFlowControl(self.__simulation_params.intensities,
                                                         self.__simulation_params.poisson_generators_edges,
                                                         self.__simulation_params.duration)

    def __get_simulation_params_from_file(self):
        with open(self.simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self):
        # step = 0
        # traci.simulationStep()
        # self.__route_generator.make_routes(self.__start_node_type.extreme_node)
        # self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
        # self.__transport_generator.generate(self.__last_target_nodes_data)
        # self.__route_generator.print_all_routes_data_info()
        # while step < self.__simulation_params.duration:
        #     traci.simulationStep()
        #     if step % 10 == 0:
        #         self.__route_generator.make_routes(self.__start_node_type.poisson_generator)
        #         self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
        #         self.__transport_generator.generate(self.__last_target_nodes_data)
        #         self.__route_generator.print_all_routes_data_info()
        #     step += 1
        step = 0
        print(self.__simulation_params.initialization_delay)
        while step < self.__simulation_params.iterations * self.__simulation_params.initialization_delay:
            traci.simulationStep()
            if step % self.__simulation_params.initialization_delay == 0:
                self.__route_generator.make_routes()
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate(self.__last_target_nodes_data)
                self.__route_generator.print_all_routes_data_info()
            step += 1
        schedule = self.__poisson_flow_control.generate_schedule(step+1) #немного неверно но в целом ок
        while step < self.__simulation_params.duration:
            traci.simulationStep()
            if step in schedule:
                self.__route_generator.make_routes(schedule[step])

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", self.sumo_config]
        traci.start(sumo_cmd)
        self.__generate_initial_traffic()
        traci.close()
