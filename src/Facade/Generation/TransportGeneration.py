import sys

import traci

from Facade.Generation.AbstractGeneration import *


class TransportGeneration(AbstractGeneration):
    def __init__(self):
        super().__init__()

    def __get_clear_nodes(self):
        nodes = traci.junction.getIDList()
        clear_nodes = []
        for node in nodes:
            if ":" not in node:
                clear_nodes.append(node)
        nodes = clear_nodes
        return nodes

    def __get_extreme_edges(self) -> list:
        nodes = self.__get_clear_nodes()
        extreme_edges = []
        for node in nodes:
            if len(traci.junction.getOutgoingEdges(node)) == 2:
                for edge in traci.junction.getOutgoingEdges(node):
                    if ":" not in edge:
                        extreme_edges.append(edge)
        return extreme_edges

    def generate(self):
        extreme_edges = self.__get_extreme_edges()
        