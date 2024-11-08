import traci
from Facade.Generation.TransportGeneration import *
from Facade.Generation.RouteGeneration import *
from Facade.Net import *

class Facade:
    def __init__(self, config_file: str):
        self.config_file = config_file
        sumo_cmd = ["sumo-gui", "-c", self.config_file]
        traci.start(sumo_cmd)
        traci.simulationStep()
        self.net = Net()
        self.__routes = RouteGeneration(self.net)
        self.__transport = TransportGeneration()
        self.__last_target_nodes_data = []

    def __generate_transport(self):
        self.__transport.generate(self.__last_target_nodes_data)

    def __generate_routes(self):
        self.__routes.uniform_distribution_for_target_nodes_test()
        self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()

    def execute(self):
        self.__generate_routes()
        self.__generate_transport()
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0 and step < 1000:
            traci.simulationStep()
            if step % 10 == 0:
                self.__generate_routes()
                self.__generate_transport()
            step += 1
        traci.close()

    def make_statistic(self):
        target_nodes_data = self.__routes.get_target_nodes_data()
        for target_node_data in target_nodes_data:
            path_length_meters_counter = {}
            print(f"node_id = {target_node_data.node_id}")
            print(f"path_len = {len(target_node_data.paths)}")
            for path_length_in_meters in target_node_data.path_length_meters:
                try:
                    path_length_meters_counter[path_length_in_meters] += 1
                except KeyError:
                    path_length_meters_counter[path_length_in_meters] = 1
            print(f"{path_length_meters_counter}\n")
        print(f"\n {self.__routes.get_start_nodes_counter()}")