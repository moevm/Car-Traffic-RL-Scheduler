import traci

from Facade.NodeData import NodeData
from Facade.Net import Net
import random

class RouteGeneration:
    def __init__(self, net):
        self.__route_counter = 0
        self.net = net
        self.__extreme_nodes = self.net.get_extreme_nodes()
        self.__target_nodes = self.net.get_clear_nodes()
        self.__target_nodes_data = []
        for node_id in self.__target_nodes:
            target_node_data = NodeData(node_id=node_id,
                                        paths=[],
                                        start_nodes_ids=[],
                                        path_length_meters=[],
                                        path_length_edges=[],
                                        routes_ids=[]
                                        )

            self.__target_nodes_data.append(target_node_data)
    def generate(self):
        pass

    def uniform_distribution_for_target_edges(self):
        self.__target_nodes_data.sort(key=lambda x: len(x.start_nodes_ids))
        n_routes = random.randint(1, len(self.__extreme_nodes))
        extreme_nodes = self.__extreme_nodes.copy()
        for i in range(n_routes):
            extreme_nodes_tmp = extreme_nodes.copy()
            if self.__target_nodes_data[i].node_id in extreme_nodes_tmp:
                extreme_nodes_tmp.remove(self.__target_nodes_data[i].node_id)
            start_node = random.choice(extreme_nodes_tmp)
            extreme_nodes_tmp.remove(start_node)
            path = self.net.find_shortest_path(start_node, self.__target_nodes_data[i].node_id)
            path_length_in_meters = self.net.find_path_length_meters(path)
            path_length_in_edges = self.net.find_path_length_edges(path)
            self.__target_nodes_data[i].start_nodes_ids.append(start_node)
            self.__target_nodes_data[i].paths.append(path)
            self.__target_nodes_data[i].path_length_meters.append(path_length_in_meters)
            self.__target_nodes_data[i].path_length_edges.append(path_length_in_edges)
            self
    def get_target_nodes_data(self):
        return self.__target_nodes_data