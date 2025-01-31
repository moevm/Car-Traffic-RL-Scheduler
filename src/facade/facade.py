import json
import traci
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.net import Net
from facade.structures import SimulationParams
from facade.traffic_control import TrafficControl


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.sumo_config = sumo_config
        sumo_cmd = ["sumo", "-c", self.sumo_config]
        traci.start(sumo_cmd)
        traci.simulationStep()
        self.simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.net = Net()
        self.net.init_poisson_generators(self.__simulation_params.poisson_generators_edges)
        traci.close()
        self.net.parallel_make_restore_path_matrix()
        self.__route_generator = RouteGenerator(self.net)
        self.__transport_generator = TransportGenerator()
        self.__last_target_nodes_data = []
        self.__traffic_control = TrafficControl(self.__simulation_params.intensities,
                                                self.__simulation_params.poisson_generators_edges,
                                                self.__simulation_params.duration, self.net.get_clear_edges())
        self.__dataset = []
        self.__step = 0

    def __get_simulation_params_from_file(self):
        with open(self.simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self):
        while self.__step < self.__simulation_params.iterations * self.__simulation_params.initialization_delay:
            traci.simulationStep()
            if self.__step % self.__simulation_params.initialization_delay == 0:
                self.__route_generator.make_routes()
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate(self.__last_target_nodes_data)
                self.__route_generator.print_all_routes_data_info()
            self.__step += 1
        self.__traffic_control.init_vehicles_data(self.__transport_generator.get_vehicles_data())
        while self.__step < self.__simulation_params.duration:
            traci.simulationStep()
            self.__step += 1
            if self.__traffic_control.have_vehicles_passed_halfway_in_total():
                break

    def __generate_main_traffic(self):
        schedule = self.__traffic_control.generate_schedule_for_poisson_flow(self.__step + 1)
        while self.__step < self.__simulation_params.duration:
            traci.simulationStep()
            if self.__step in schedule:
                self.__route_generator.make_routes(schedule[self.__step])
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate(self.__last_target_nodes_data)
                self.__route_generator.print_all_routes_data_info()
            self.__step += 1

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", self.sumo_config]
        traci.start(sumo_cmd)
        # self.__generate_initial_traffic()
        self.__generate_main_traffic()
        traci.close()
