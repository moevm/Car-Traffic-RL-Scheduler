from facade.structures import NodeData
import traci

from facade.logger.logger import Message
from facade.logger.route_logger import RouteLogger


class RouteGenerator:
    def __init__(self, net):
        self.__route_logger = RouteLogger()
        self.__COEFFICIENT = 10
        self.__route_counter = 0
        self.__last_n_routes = 0
        self.__net = net
        self.__extreme_nodes = self.__net.get_extreme_nodes()
        self.__target_nodes = self.__net.get_clear_nodes()
        self.__poisson_generators_edges = self.__net.get_poisson_generators_edges()
        self.__poisson_generators_from_nodes = self.__net.get_poisson_generators_from_nodes()
        self.__poisson_generators_to_nodes = self.__net.get_poisson_generators_to_nodes()
        self.__start_node_counter = {node: 0 for node in (self.__extreme_nodes + self.__poisson_generators_to_nodes)}
        self.__target_nodes_data = self.__init_target_nodes_data()
        self.__last_indices = []

    def __init_target_nodes_data(self):
        target_nodes_data = []
        for node_id in self.__target_nodes:
            target_node_data = NodeData(node_id=node_id,
                                        last_path=[],
                                        start_nodes_ids=[],
                                        path_length_meters=[],
                                        path_length_edges=[],
                                        last_route_id=0
                                        )

            target_nodes_data.append(target_node_data)
        return target_nodes_data

    """
    если start_nodes -- это пуассоновские генераторы, то нужно к path добавить edge
    1) проверить код для случая, если работаем с пачками для инициализации
    2) можно отправляться только из тех узлов, которые указаны в start_nodes если речь о пуассоновских потоках
    3) очистка
    """

    """
    
    """
    def __set_start_node_from_extreme_nodes(self, start_nodes, target_node_data):
        path_length_meters_counter = {}
        for path_length_in_meters in target_node_data.path_length_meters:
            if path_length_in_meters in path_length_meters_counter:
                path_length_meters_counter[path_length_in_meters] += 1
            else:
                path_length_meters_counter[path_length_in_meters] = 1
        if path_length_meters_counter:
            min_path_length_meters = min(path_length_meters_counter.items(), key=lambda item: item[1])[0]
        else:
            min_path_length_meters = -1
        for start_node in start_nodes:
            path = self.__net.get_shortest_path(start_node, target_node_data.node_id)
            path_length_in_meters = self.__net.get_path_length_in_meters(path)
            if path_length_in_meters not in path_length_meters_counter:
                return start_node
        for i in range(len(target_node_data.path_length_meters)):
            if target_node_data.path_length_meters[i] == min_path_length_meters:
                return target_node_data.start_nodes_ids[i]

    def __set_start_node_from_poisson_generators(self, start_nodes, target_node_data):
        path_length_meters_counter = {}
        n = len(target_node_data.path_length_meters)
        filtered_target_node_data = NodeData(node_id=target_node_data.node_id,
                                             last_path=[],
                                             start_nodes_ids=[],
                                             path_length_meters=[],
                                             path_length_edges=[],
                                             last_route_id=0
                                             )
        for i in range(n):
            for start_node in start_nodes:
                if target_node_data.start_nodes_ids[i] == start_node:
                    filtered_target_node_data.path_length_meters.append(target_node_data.path_length_meters[i])
        target_node_data = filtered_target_node_data
        for path_length_in_meters in target_node_data.path_length_meters:
            if path_length_in_meters in path_length_meters_counter:
                path_length_meters_counter[path_length_in_meters] += 1
            else:
                path_length_meters_counter[path_length_in_meters] = 1
        if path_length_meters_counter:
            min_path_length_meters = min(path_length_meters_counter.items(), key=lambda item: item[1])[0]
        else:
            min_path_length_meters = -1
        for start_node in start_nodes:
            path = self.__net.get_shortest_path(start_node, target_node_data.node_id)
            path_length_in_meters = self.__net.get_path_length_in_meters(path)
            if path_length_in_meters not in path_length_meters_counter:
                return start_node
        for i in range(len(target_node_data.path_length_meters)):
            if target_node_data.path_length_meters[i] == min_path_length_meters:
                """
                данный сценарий нужно протестировать на маленьких картах
                """
                return target_node_data.start_nodes_ids[i]

    def __add_target_node_data(self, i, start_node, path, path_length_in_meters, path_length_in_edges):
        self.__target_nodes_data[i].start_nodes_ids.append(start_node)
        self.__target_nodes_data[i].last_path = path
        self.__target_nodes_data[i].path_length_meters.append(path_length_in_meters)
        self.__target_nodes_data[i].path_length_edges.append(path_length_in_edges)
        self.__target_nodes_data[i].last_route_id = self.__route_counter

    def __find_indices_of_suitable_target_nodes(self, start_nodes):
        indices = []
        for i in range(len(self.__target_nodes_data)):
            included = False
            for start_node in start_nodes:
                if start_node == self.__target_nodes_data[i].node_id:
                    included = True
                    break
            if not included:
                indices.append(i)
            if len(indices) == len(start_nodes):
                break
        return indices

    def make_routes(self, poisson_generators_edges=None):
        if len(self.__target_nodes_data[0].start_nodes_ids) == self.__COEFFICIENT * len(self.__extreme_nodes +
                                                                                        self.__poisson_generators_to_nodes):
            self.__target_nodes_data = self.__init_target_nodes_data()
        self.__target_nodes_data.sort(key=lambda x: len(x.path_length_meters))
        if poisson_generators_edges is None:
            start_nodes = self.__extreme_nodes.copy()
        else:
            start_nodes = [traci.edge.getToJunction(edge) for edge in poisson_generators_edges]
        indices = self.__find_indices_of_suitable_target_nodes(start_nodes)
        self.__last_indices = indices
        n_routes = len(start_nodes)
        for i in indices:
            if poisson_generators_edges is None:
                start_node = self.__set_start_node_from_extreme_nodes(start_nodes, self.__target_nodes_data[i])
            else:
                start_node = self.__set_start_node_from_poisson_generators(start_nodes, self.__target_nodes_data[i])
            if start_node in start_nodes:
                index = start_nodes.index(start_node)
                start_nodes.remove(start_node)
            path = self.__net.get_shortest_path(start_node, self.__target_nodes_data[i].node_id)
            if poisson_generators_edges is not None:
                edge = poisson_generators_edges.pop(index)
                path.insert(0, edge)
            path_length_in_meters = self.__net.get_path_length_in_meters(path)
            path_length_in_edges = self.__net.get_path_length_in_edges(path)
            self.__add_target_node_data(i, start_node, path, path_length_in_meters, path_length_in_edges)
            self.__route_counter += 1
            self.__start_node_counter[start_node] += 1
        self.__last_n_routes = n_routes
        self.__route_logger.print_routes_data_info(Message.last_target_nodes_data,
                                                   self.__target_nodes_data[:self.__last_n_routes])

    def get_last_target_nodes_data(self):
        last_target_nodes_data = []
        for i in self.__last_indices:
            last_target_node_data = NodeData(
                node_id=self.__target_nodes_data[i].node_id,
                last_path=self.__target_nodes_data[i].last_path,
                start_nodes_ids=[self.__target_nodes_data[i].start_nodes_ids[-1]],
                path_length_meters=[self.__target_nodes_data[i].path_length_meters[-1]],
                path_length_edges=[self.__target_nodes_data[i].path_length_meters[-1]],
                last_route_id=self.__target_nodes_data[i].last_route_id
            )
            last_target_nodes_data.append(last_target_node_data)
        return last_target_nodes_data

    def get_target_nodes_data(self):
        return self.__target_nodes_data

    def get_start_nodes_counter(self):
        return self.__start_node_counter

    def print_all_routes_data_info(self):
        self.__target_nodes_data.sort(key=lambda x: len(x.path_length_meters))
        self.__route_logger.print_routes_data_info(Message.target_nodes_data, self.__target_nodes_data)
