from Logger.Logger import *


class NetworkLogger(Logger):
    def __init__(self):
        super().__init__("[NetworkInfo]")

    def print_graph_info(self, number_of_edges, number_of_nodes):
        print(f"{self._logger_type} Number of edges: {number_of_edges}")
        print(f"{self._logger_type} Number of nodes: {number_of_nodes}")



