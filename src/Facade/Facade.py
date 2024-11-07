import traci
from Facade.Generation.TransportGeneration import *
from Facade.Generation.RouteGeneration import *


class Facade:
    def __init__(self, config_file):
        self.config_file = config_file
        self.transport = []
        self.routes = []

    def __generate_transport(self):
        traci.simulationStep()
        self.transport = TransportGeneration()
        self.transport.generate()

    def __generate_routes(self):
        pass

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", self.config_file]
        traci.start(sumo_cmd)
        self.__generate_transport()
        self.__generate_routes()
        traci.close()