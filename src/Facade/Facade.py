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
        self.__routes.uniform_distribution_for_target_edges()
        self.__last_target_nodes_data = self.__routes.get_last_target_nodes_data()

    def execute(self):
        self.__generate_routes()
        self.__generate_transport()
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if step % 10 == 0:
                self.__generate_routes()
                self.__generate_transport()
            step += 1
        traci.close()