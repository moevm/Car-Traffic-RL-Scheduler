import traci, time
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

    def __generate_initial_traffic(self):
        step = 0
        while step < 1000:
            traci.simulationStep()
            if step % 50 == 0:
                self.__routes.uniform_distribution_for_target_nodes()
                self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()
                self.__transport.generate(self.__last_target_nodes_data)
            step += 1
        self.__routes.print_all_routes_data_info()

    def execute(self):
        self.__generate_initial_traffic()
        traci.close()
