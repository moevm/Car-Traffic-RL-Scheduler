import time

import traci

from Facade.NodeData import NodeData
from Facade.Net import Net
import random

class RouteGeneration:
    def __init__(self, net):
        self.__coefficient = 10
        self.__route_counter = 0
        self.__last_n_routes = 0
        self.net = net
        self.__extreme_nodes = self.net.get_extreme_nodes()
        self.__start_node_counter = {node: 0 for node in self.__extreme_nodes}
        self.__target_nodes = self.net.get_clear_nodes()
        self.__target_nodes_data = self.__init_target_nodes_data()

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
    def generate(self):
        pass

    def __set_start_node(self, extreme_nodes, target_node_data):
        path_length_meters_counter = {}
        for path_length_in_meters in target_node_data.path_length_meters:
            try:
                path_length_meters_counter[path_length_in_meters] += 1
            except KeyError:
                path_length_meters_counter[path_length_in_meters] = 1
        if path_length_meters_counter:
            min_path_length_meters = min(path_length_meters_counter.items(), key=lambda item: item[1])[0]
        else:
            min_path_length_meters = -1
        for extreme_node in extreme_nodes:
            path = self.net.get_shortest_path(extreme_node, target_node_data.node_id)
            path_length_in_meters = self.net.get_path_length_in_meters(path)
            if path_length_in_meters not in path_length_meters_counter:
                return extreme_node
        for i in range(len(target_node_data.path_length_meters)):
            if target_node_data.path_length_meters[i] == min_path_length_meters:
                return target_node_data.start_nodes_ids[i]

    def uniform_distribution_for_target_nodes(self):
        if len(self.__target_nodes_data[0].start_nodes_ids) == self.__coefficient * len(self.__extreme_nodes):
            print("clear")
            self.__target_nodes_data = self.__init_target_nodes_data()
        self.__target_nodes_data.sort(key=lambda x: len(x.path_length_meters))
        n_routes = random.randint(1, len(self.__extreme_nodes))
        extreme_nodes = self.__extreme_nodes.copy()
        random.shuffle(extreme_nodes)
        print(f"len(extreme_nodes) = {len(extreme_nodes)}")
        for i in range(n_routes):
            start_time_1 = time.time()
            extreme_nodes_tmp = extreme_nodes.copy()
            if self.__target_nodes_data[i].node_id in extreme_nodes_tmp:
                extreme_nodes_tmp.remove(self.__target_nodes_data[i].node_id)
            print("1 ", time.time() - start_time_1)
            start_time_2 = time.time()
            start_node = self.__set_start_node(extreme_nodes_tmp, self.__target_nodes_data[i])
            print("2 ", time.time() - start_time_2)
            self.__start_node_counter[start_node] += 1
            if start_node in extreme_nodes:
                extreme_nodes.remove(start_node)
            start_time_3 = time.time()
            path = self.net.get_shortest_path(start_node, self.__target_nodes_data[i].node_id)
            print("3 ", time.time() - start_time_3)
            print(start_node, self.__target_nodes_data[i].node_id, path)
            path_length_in_meters = self.net.get_path_length_in_meters(path)
            path_length_in_edges = self.net.get_path_length_in_edges(path)
            self.__target_nodes_data[i].start_nodes_ids.append(start_node)
            self.__target_nodes_data[i].last_path = path
            self.__target_nodes_data[i].path_length_meters.append(path_length_in_meters)
            self.__target_nodes_data[i].path_length_edges.append(path_length_in_edges)
            self.__target_nodes_data[i].last_route_id = self.__route_counter
            self.__route_counter += 1
            print("4 ", time.time() - start_time_1)
        self.__last_n_routes = n_routes

    # def uniform_distribution_for_target_nodes(self):
    #     random.shuffle(self.__target_nodes_data)
    #     self.__target_nodes_data.sort(key=lambda x: len(x.start_nodes_ids))
    #     n_routes = random.randint(1, len(self.__extreme_nodes))
    #     extreme_nodes = self.__extreme_nodes.copy()
    #     for i in range(n_routes):
    #         extreme_nodes_tmp = extreme_nodes.copy()
    #         random.shuffle(extreme_nodes_tmp)
    #         if self.__target_nodes_data[i].node_id in extreme_nodes_tmp:
    #             extreme_nodes_tmp.remove(self.__target_nodes_data[i].node_id)
    #         start_node = random.choice(extreme_nodes_tmp)
    #         self.__start_node_counter[start_node] += 1
    #         extreme_nodes_tmp.remove(start_node)
    #         path = self.net.find_shortest_path(start_node, self.__target_nodes_data[i].node_id)
    #         path_length_in_meters = self.net.find_path_length_meters(path)
    #         path_length_in_edges = self.net.find_path_length_edges(path)
    #         self.__target_nodes_data[i].start_nodes_ids.append(start_node)
    #         self.__target_nodes_data[i].paths.append(path)
    #         self.__target_nodes_data[i].path_length_meters.append(path_length_in_meters)
    #         self.__target_nodes_data[i].path_length_edges.append(path_length_in_edges)
    #         self.__target_nodes_data[i].routes_ids.append(self.__route_counter)
    #         self.__route_counter += 1
    #     self.__last_n_routes = n_routes

    def get_last_target_nodes_data(self):
        last_target_nodes_data = []
        for target_node_data in self.__target_nodes_data[:self.__last_n_routes]:
            last_target_node_data = NodeData(
                node_id=target_node_data.node_id,
                last_path=target_node_data.last_path,
                start_nodes_ids=[target_node_data.start_nodes_ids[-1]],
                path_length_meters=[target_node_data.path_length_meters[-1]],
                path_length_edges=[target_node_data.path_length_meters[-1]],
                last_route_id=target_node_data.last_route_id
            )
            last_target_nodes_data.append(last_target_node_data)
        return last_target_nodes_data

    def get_target_nodes_data(self):
        return self.__target_nodes_data

    def get_start_nodes_counter(self):
        return self.__start_node_counter