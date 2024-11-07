import sys
from Facade.Generation.AbstractGeneration import *


class TransportGeneration(AbstractGeneration):
    def __init__(self):
        super().__init__()

    def __get_extreme_edges(self):
        print(f"lanes_ids = {traci.lane.getIDList()}")
        print(f"edges_ids = {traci.edge.getIDList()}")

    def generate(self):
        self.__get_extreme_edges()