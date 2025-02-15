from facade.logger.logger import *


class NetworkLogger(Logger):
    def __init__(self):
        super().__init__("[NetworkInfo]")

    def print_graph_info(self, number_of_edges: int, number_of_nodes: int) -> None:
        print(f"{self._logger_type} Number of edges: {number_of_edges}")
        print(f"{self._logger_type} Number of nodes: {number_of_nodes}")



