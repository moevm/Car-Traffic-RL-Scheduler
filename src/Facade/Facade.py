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
        self.__target_nodes_data = []

    def __generate_transport(self):
        pass

    def __generate_routes(self):
        self.__routes.uniform_distribution_for_target_edges()
        self.__target_nodes_data = self.__routes.get_target_nodes_data()

    def execute(self):
        self.__generate_routes()
        self.__generate_transport()
        traci.close()