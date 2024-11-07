from Facade.Generation.TransportGeneration import *
from Facade.Generation.RouteGeneration import *


class Facade:
    def __init__(self):
        pass

    def execute(self):
        sumo_cmd = ["sumo-gui", "-c", "test.sumocfg"]
        traci.start(sumo_cmd)
        while traci.simulation.getMinExpectedNumber() > 0:
            transport = TransportGeneration()
            transport.generate()
        