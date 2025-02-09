import json
import traci
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.net import Net
from facade.structures import SimulationParams
from facade.traffic_control import TrafficControl


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__SUMO_CONFIG = sumo_config
        sumo_cmd = ["sumo", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        traci.simulationStep()
        self.__simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.__net = Net()
        self.__net.init_poisson_generators(self.__simulation_params.poisson_generators_edges)
        traci.close()
        self.__net.parallel_make_restore_path_matrix()
        self.__route_generator = RouteGenerator(self.__net)
        self.__transport_generator = TransportGenerator()
        self.__last_target_nodes_data = []
        self.__traffic_control = TrafficControl(self.__simulation_params.intensities,
                                                self.__simulation_params.poisson_generators_edges,
                                                self.__simulation_params.DURATION, self.__net.get_clear_edges())
        self.__step = 0

    def __get_simulation_params_from_file(self):
        with open(self.__simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self):
        while self.__step < self.__simulation_params.ITERATIONS * self.__simulation_params.INIT_DELAY:
            traci.simulationStep()
            if self.__step % self.__simulation_params.INIT_DELAY == 0:
                self.__route_generator.make_routes()
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate(self.__last_target_nodes_data)
                self.__route_generator.print_all_routes_data_info()
            self.__step += 1
        self.__traffic_control.init_vehicles_data(self.__transport_generator.get_vehicles_data())
        while self.__step < self.__simulation_params.DURATION:
            traci.simulationStep()
            self.__step += 1
            if self.__traffic_control.have_vehicles_passed_halfway_in_total():
                break

    def __generate_main_traffic(self):
        schedule = self.__traffic_control.generate_schedule_for_poisson_flow(self.__step + 1)
        while self.__step < self.__simulation_params.DURATION:
            traci.simulationStep()
            if self.__step in schedule:
                self.__route_generator.make_routes(schedule[self.__step])
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate(self.__last_target_nodes_data)
                #self.__route_generator.print_all_routes_data_info()
            self.__step += 1

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        self.__generate_initial_traffic()
        self.__generate_main_traffic()
        traci.close()
