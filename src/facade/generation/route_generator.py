import random
import traci

from facade.structures import NodeData
from facade.logger.logger import Message
from facade.logger.route_logger import RouteLogger
from facade.net import Net


class RouteGenerator:
    def __init__(self, net: Net):
        self.__route_logger = RouteLogger()
        self.__route_counter = 0
        self.__net = net
        self.__extreme_nodes = self.__net.get_extreme_nodes()
        self.__target_nodes = self.__net.get_nodes()
        self.__poisson_generators_edges = self.__net.get_poisson_generators_edges()
        self.__poisson_generators_from_nodes = self.__net.get_poisson_generators_from_nodes()
        self.__poisson_generators_to_nodes = self.__net.get_poisson_generators_to_nodes()
        self.__start_node_counter = {node: 0 for node in (self.__extreme_nodes + self.__poisson_generators_to_nodes)}
        self.__target_nodes_data = self.__init_target_nodes_data()
        self.__last_indices = []

    def __init_target_nodes_data(self) -> list[NodeData]:
        target_nodes_data = []
        for node_id in self.__target_nodes:
            target_node_data = NodeData(node_id=node_id)
            target_nodes_data.append(target_node_data)
        return target_nodes_data

    @staticmethod
    def __get_filtered_target_node_data(target_node_data: NodeData, start_nodes: list[str]) -> NodeData:
        n = len(target_node_data.path_length_meters)
        filtered_target_node_data = NodeData(node_id=target_node_data.node_id)
        for i in range(n):
            for start_node in start_nodes:
                if target_node_data.start_nodes_ids[i] == start_node:
                    filtered_target_node_data.start_nodes_ids.append(target_node_data.start_nodes_ids[i])
                    if start_node not in filtered_target_node_data.start_nodes_ids_counter.keys():
                        filtered_target_node_data.start_nodes_ids_counter[start_node] = 1
                    else:
                        filtered_target_node_data.start_nodes_ids_counter[start_node] += 1
        return filtered_target_node_data

    def __set_start_node(self, start_nodes: list[str], target_node_data: NodeData) -> str:
        filtered_target_node_data = self.__get_filtered_target_node_data(target_node_data, start_nodes)
        for start_node in start_nodes:
            if start_node not in filtered_target_node_data.start_nodes_ids_counter:
                return start_node
        start_node = filtered_target_node_data.start_nodes_ids[0]
        min_start_nodes_ids_counter = min(filtered_target_node_data.start_nodes_ids_counter.items(),
                                          key=lambda item: item[1])[0]
        for i in range(len(filtered_target_node_data.start_nodes_ids)):
            if filtered_target_node_data.start_nodes_ids[i] == min_start_nodes_ids_counter:
                start_node = filtered_target_node_data.start_nodes_ids[i]
                break
        return start_node

    def __add_target_node_data(self, i: int, start_node: str, path: list[str], path_length_in_meters: int,
                               path_length_in_edges: int, route_counter: int) -> None:
        self.__target_nodes_data[i].start_nodes_ids.append(start_node)
        self.__target_nodes_data[i].last_paths.append(path)
        self.__target_nodes_data[i].path_length_meters.append(path_length_in_meters)
        self.__target_nodes_data[i].path_length_edges.append(path_length_in_edges)
        self.__target_nodes_data[i].last_routes_ids.append(route_counter)
        if start_node not in self.__target_nodes_data[i].start_nodes_ids_counter.keys():
            self.__target_nodes_data[i].start_nodes_ids_counter[start_node] = 1
        else:
            self.__target_nodes_data[i].start_nodes_ids_counter[start_node] += 1

    def make_routes(self, poisson_generators_edges=None) -> None:
        if len(self.__target_nodes_data[0].start_nodes_ids) == len(self.__extreme_nodes +
                                                                   self.__poisson_generators_to_nodes):
            self.__target_nodes_data = self.__init_target_nodes_data()
        if poisson_generators_edges is None:
            start_nodes = self.__extreme_nodes.copy()
        else:
            start_nodes = [traci.edge.getToJunction(edge) for edge in poisson_generators_edges]
            prev_nodes = [traci.edge.getFromJunction(edge) for edge in poisson_generators_edges]
        n = len(start_nodes)
        i = 0
        self.__last_indices = []
        while i < n:
            j = random.randint(0, len(self.__target_nodes_data) - 1)
            target_node_data = self.__target_nodes_data[j]
            start_node = self.__set_start_node(start_nodes, target_node_data)
            if start_node != target_node_data.node_id:
                node_id = start_nodes.index(start_node)
                start_nodes.remove(start_node)
                if poisson_generators_edges is not None:
                    prev_node = prev_nodes.pop(node_id)
                else:
                    prev_node = None
                path = self.__net.get_shortest_path(start_node, prev_node, target_node_data.node_id)
                path_length_in_meters = self.__net.get_path_length_in_meters(path)
                path_length_in_edges = self.__net.get_path_length_in_edges(path)
                self.__add_target_node_data(j, start_node, path, path_length_in_meters, path_length_in_edges,
                                            self.__route_counter)
                self.__route_counter += 1
                self.__last_indices.append(j)
                i += 1

    def get_last_target_nodes_data(self) -> list[NodeData]:
        last_target_nodes_data = []
        for i in self.__last_indices:
            n = len(self.__target_nodes_data[i].last_paths)
            last_target_node_data = NodeData(
                node_id=self.__target_nodes_data[i].node_id,
                last_paths=self.__target_nodes_data[i].last_paths,
                start_nodes_ids=[self.__target_nodes_data[i].start_nodes_ids[-n]],
                path_length_meters=[self.__target_nodes_data[i].path_length_meters[-n]],
                path_length_edges=[self.__target_nodes_data[i].path_length_meters[-n]],
                last_routes_ids=self.__target_nodes_data[i].last_routes_ids,
                start_nodes_ids_counter=self.__target_nodes_data[i].start_nodes_ids_counter
            )
            self.__target_nodes_data[i].last_paths = []
            self.__target_nodes_data[i].last_routes_ids = []
            last_target_nodes_data.append(last_target_node_data)
        return last_target_nodes_data

    def get_target_nodes_data(self) -> list[NodeData]:
        return self.__target_nodes_data

    def get_start_nodes_counter(self) -> dict[str, int]:
        return self.__start_node_counter

    def print_all_routes_data_info(self) -> None:
        self.__route_logger.print_routes_data_info(Message.target_nodes_data, self.__target_nodes_data)
