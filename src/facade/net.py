import traci
import heapq
import random

from sumolib.net import readNet
from multiprocessing import Process, Manager, Lock, Pool, cpu_count
from facade.structures import NodeColor, NodePair
from facade.logger.network_logger import *


class Net:
    def __init__(self, net_config):
        self.__NET_CONFIG = net_config
        self.__sumolib_net = readNet(net_config)
        self.__network_logger = NetworkLogger()
        self.__INF = float("inf")
        self.__nodes = [node.getID() for node in self.__sumolib_net.getNodes()]
        self.__edges = [edge.getID() for edge in self.__sumolib_net.getEdges(withInternal=False)]
        self.__edges_length = {edge.getID(): edge.getLength() for edge in self.__sumolib_net.getEdges(
            withInternal=False)}
        self.__extreme_nodes = self.__find_extreme_nodes()
        self.__network_logger.print_graph_info(len(self.__edges), len(self.__nodes))
        self.__graph = self.__make_graph()
        (self.__poisson_generators_to_nodes, self.__poisson_generators_from_nodes, self.__poisson_generators_edges,
         self.__paths_for_way_back_routes) = [], [], [], {}
        self.__restore_path_matrix, self.__restore_path_matrix_for_way_back_routes = {}, {}
        self.__edges_dict = self.__make_edges_dict()
        self.__turned_on_traffic_lights_ids = []
        self.__paths = {}

    def init_poisson_generators(self, poisson_generators_edges: list[str]) -> None:
        self.__poisson_generators_to_nodes = [self.__sumolib_net.getEdge(edge).getToNode().getID() for edge in
                                              poisson_generators_edges]
        self.__poisson_generators_from_nodes = [self.__sumolib_net.getEdge(edge).getFromNode().getID() for edge in
                                                poisson_generators_edges]
        self.__poisson_generators_edges = poisson_generators_edges

    def __find_routes_worker(self, i, start_node, nodes, extreme_nodes, poisson_edges, paths, lock):
        local_paths = {}
        for j in range(len(nodes)):
            if start_node != nodes[j]:
                path = self.__get_shortest_path(start_node, nodes[j])
                if i >= len(extreme_nodes):
                    path.insert(0, poisson_edges[i - len(extreme_nodes)])
                if start_node not in local_paths:
                    local_paths[start_node] = {nodes[j]: path}
                else:
                    local_paths[start_node][nodes[j]] = path
        paths.update(local_paths)
        with lock:
            self.__network_logger.step_progress_bar()

    def parallel_find_routes(self) -> None:
        start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
        self.__network_logger.init_progress_bar(Message.find_all_routes, len(start_nodes))
        with Manager() as manager:
            paths = manager.dict()
            lock = Lock()
            processes = []
            for i, start_node in enumerate(start_nodes):
                p = Process(target=self.__find_routes_worker, args=(
                    i, start_node, self.__nodes, self.__extreme_nodes, self.__poisson_generators_edges, paths, lock
                ))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
                self.__network_logger.step_progress_bar()
            self.__network_logger.destroy_progress_bar()
            self.__paths = dict(paths)

    # def parallel_find_routes(self) -> None:
    #     start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
    #     paths = {}
    #     self.__network_logger.init_progress_bar(Message.find_all_routes, len(start_nodes))
    #     for i, start_node in enumerate(start_nodes):
    #         self.__network_logger.step_progress_bar()
    #         self.__find_routes(i, start_node, paths)
    #     self.__network_logger.destroy_progress_bar()
    #     # for start_node in paths.keys():
    #     #     print(f"{start_node}\n{paths[start_node]}\n\n")
    #     # self.__network_logger.init_progress_bar(Message.find_all_routes, len(start_nodes))
    #     # manager = Manager()
    #     # lock = Lock()
    #     # paths = manager.dict()
    #     # processes = []
    #     # for i, start_node in enumerate(start_nodes):
    #     #     process = Process(target=self.__find_routes, args=(i, start_node, paths))
    #     #     processes.append(process)
    #     #     process.start()
    #     # for process in processes:
    #     #     process.join()
    #     #     with lock:
    #     #         self.__network_logger.step_progress_bar()
    #     # self.__network_logger.destroy_progress_bar()
    #     self.__paths = paths
    #     print(self.__paths)
    #
    # def __find_routes(self, i, start_node: str, paths: dict[str, dict[str, list[str]]]) -> None:
    #     for j in range(len(self.__nodes)):
    #         if start_node != self.__nodes[j]:
    #             path = self.__get_shortest_path(start_node, self.__nodes[j])
    #             if i >= len(self.__extreme_nodes):
    #                 path.insert(0, self.__poisson_generators_edges[i - len(self.__extreme_nodes)])
    #             if start_node not in paths:
    #                 paths[start_node] = {self.__nodes[j]: path}
    #             else:
    #                 paths[start_node][self.__nodes[j]] = path

    def turn_off_traffic_lights(self, turned_off_traffic_lights) -> None:
        for off_traffic_light_id in turned_off_traffic_lights:
            traci.trafficlight.setProgram(off_traffic_light_id, "off")
        for traffic_light_id in traci.trafficlight.getIDList():
            is_on_traffic_light = True
            for off_traffic_light_id in turned_off_traffic_lights:
                if traffic_light_id == off_traffic_light_id:
                    is_on_traffic_light = False
                    break
            if is_on_traffic_light:
                self.__turned_on_traffic_lights_ids.append(traffic_light_id)

    def parallel_make_restore_path_matrix(self):
        start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix, len(start_nodes))
        manager = Manager()
        restore_path_matrix = manager.dict()
        processes = []
        for i, start_node in enumerate(start_nodes):
            if i < len(self.__extreme_nodes):
                process = Process(target=self.__make_restore_path_matrix,
                                  args=(start_node, restore_path_matrix, None, None))
            else:
                prev_node = self.__poisson_generators_from_nodes[i - len(self.__extreme_nodes)]
                process = Process(target=self.__make_restore_path_matrix,
                                  args=(start_node, restore_path_matrix, prev_node, None))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            self.__network_logger.step_progress_bar()
        self.__network_logger.destroy_progress_bar()
        self.__restore_path_matrix = restore_path_matrix

    # def parallel_make_restore_path_matrix(self):
    #     start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
    #     self.__network_logger.init_progress_bar(Message.init_restore_path_matrix, len(start_nodes))
    #     arg_list = []
    #     for i, start_node in enumerate(start_nodes):
    #         if i < len(self.__extreme_nodes):
    #             arg = (start_node, restore)
    #     with Pool(cpu_count() * 10) as p:

    def __make_restore_path_matrix(self, start_node: str, restore_path_matrix: dict[str, dict[str, str | None]],
                                   prev_node=None, blocked_node=None) -> None:
        if start_node in restore_path_matrix:
            print("ATTENTION: ", start_node)
            return
        distances = {node: self.__INF for node in self.__nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node, prev_node)]
        previous_nodes = {node: None for node in self.__nodes}
        while priority_queue:
            current_distance, current_node, prev_node = heapq.heappop(priority_queue)
            if current_distance > distances[current_node]:
                continue
            neighbors = list(self.__graph[current_node].items())
            random.shuffle(neighbors)
            for neighbor_node, edge in neighbors:
                if neighbor_node != prev_node and neighbor_node != blocked_node:
                    distance = current_distance + self.__edges_length[edge]
                    if distance < distances[neighbor_node]:
                        distances[neighbor_node] = distance
                        previous_nodes[neighbor_node] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor_node, current_node))
        restore_path_matrix[start_node] = previous_nodes

    def __find_intersection(self, to_node: str, from_node: str):
        current_node = to_node
        prev_node = from_node
        while len(self.__sumolib_net.getNode(current_node).getOutgoing()) < 3:
            neighbors = list(self.__graph[current_node].items())
            for node, edge in neighbors:
                if node != prev_node:
                    tmp = current_node
                    current_node = node
                    prev_node = tmp
                    break
        return current_node

    def __find_way_back(self, i: int, to_node: str, start_nodes, paths_for_way_back_routes) -> None:
        intersection_node = self.__find_intersection(to_node, self.__poisson_generators_from_nodes[i])
        path = self.__get_shortest_path(to_node, intersection_node)
        if intersection_node not in start_nodes:
            start_nodes.append(intersection_node)
        paths_for_way_back_routes[to_node] = path

    def __parallel_make_restore_path_matrix_for_way_back_routes(self, start_nodes: set[str]) -> None:
        manager = Manager()
        restore_path_matrix = manager.dict()
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix_for_cycles, len(start_nodes))
        processes = []
        for start_node in start_nodes:
            process = Process(target=self.__make_restore_path_matrix, args=(start_node,
                                                                            restore_path_matrix, None, None))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            self.__network_logger.step_progress_bar()
        self.__network_logger.destroy_progress_bar()
        self.__restore_path_matrix_for_way_back_routes = restore_path_matrix

    def parallel_find_way_back(self) -> None:
        self.__network_logger.init_progress_bar(Message.find_way_back_paths, len(self.__poisson_generators_to_nodes))
        manager = Manager()
        paths_for_way_back_routes = manager.dict()
        start_nodes = manager.list()
        processes = []
        for i, to_node in enumerate(self.__poisson_generators_to_nodes):
            process = Process(target=self.__find_way_back, args=(i, to_node, start_nodes, paths_for_way_back_routes))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            self.__network_logger.step_progress_bar()
        self.__paths_for_way_back_routes = paths_for_way_back_routes
        self.__network_logger.destroy_progress_bar()
        self.__parallel_make_restore_path_matrix_for_way_back_routes(set(start_nodes))

    def __make_edges_dict(self) -> dict[str, NodePair]:
        edges_dict = {}
        for edge in self.__edges:
            node_pair = NodePair(node_from=self.__sumolib_net.getEdge(edge).getFromNode().getID(),
                                 node_to=self.__sumolib_net.getEdge(edge).getToNode().getID())
            edges_dict[edge] = node_pair
        return edges_dict

    def __make_graph(self) -> dict[str, dict[str, str]]:
        graph = {}
        self.__network_logger.init_progress_bar(Message.init_incidence_matrix, len(self.__edges))
        for edge in self.__edges:
            self.__network_logger.step_progress_bar()
            node_1 = self.__sumolib_net.getEdge(edge).getFromNode().getID()
            node_2 = self.__sumolib_net.getEdge(edge).getToNode().getID()
            if node_1 in graph:
                graph[node_1][node_2] = edge
            else:
                value = {node_2: edge}
                graph[node_1] = value
        self.__network_logger.destroy_progress_bar()
        return graph

    def __find_clear_edges(self) -> list[str]:
        clear_edges = []
        for edge in self.__edges:
            if ":" not in edge:
                clear_edges.append(edge)
        return clear_edges

    def __find_clear_nodes(self) -> list[str]:
        clear_nodes = []
        for node in self.__nodes:
            if ":" not in node:
                clear_nodes.append(node)
        return clear_nodes

    def __find_extreme_nodes(self) -> list[str]:
        extreme_nodes = []
        for node in self.__sumolib_net.getNodes():
            if len(node.getOutgoing()) == 1:
                extreme_nodes.append(node.getID())
        return extreme_nodes

    def __get_shortest_path(self, start_node: str, end_node: str, restore_path_matrix=None) -> list[str]:
        if restore_path_matrix is not None:
            previous_nodes = restore_path_matrix[start_node]
        else:
            previous_nodes = self.__restore_path_matrix[start_node]
        path = []
        node = end_node
        while node != start_node:
            node_1 = node
            node_2 = previous_nodes[node_1]
            if (node_2 is None) and (restore_path_matrix is not None):
                return []
            elif (node_2 is None) and (restore_path_matrix is None):
                way_back_route = self.__paths_for_way_back_routes[start_node]
                if not way_back_route:
                    intersection_node = start_node
                else:
                    intersection_node = self.__sumolib_net.getEdge(way_back_route[-1]).getToNode().getID()
                path_from_intersection_node_to_end_node = self.__get_shortest_path(intersection_node, end_node,
                                                                                   self.__restore_path_matrix_for_way_back_routes)
                return way_back_route + path_from_intersection_node_to_end_node
            edge = self.__graph[node_2][node_1]
            node = node_2
            path.append(edge)
        return path[::-1]

    def get_restore_path_matrix(self) -> dict[str, dict[str, str | None]]:
        return self.__restore_path_matrix

    def get_graph(self) -> dict[str, dict[str, str]]:
        return self.__graph

    def get_extreme_nodes(self) -> list[str]:
        return self.__extreme_nodes

    def get_nodes(self) -> list[str]:
        return self.__nodes

    def get_edges(self) -> list[str]:
        return self.__edges

    def get_poisson_generators_edges(self) -> list[str]:
        return self.__poisson_generators_edges

    def get_poisson_generators_to_nodes(self) -> list[str]:
        return self.__poisson_generators_to_nodes

    def get_poisson_generators_from_nodes(self) -> list[str]:
        return self.__poisson_generators_from_nodes

    def get_path_length_in_meters(self, path: list[str]) -> int:
        length_meters = 0
        for edge in path:
            length_meters += self.__edges_length[edge]
        return length_meters

    @staticmethod
    def get_path_length_in_edges(path: list[str]) -> int:
        return len(path)

    def get_sumolib_net(self):
        return self.__sumolib_net

    def get_turned_on_traffic_lights_ids(self):
        return self.__turned_on_traffic_lights_ids

    def get_shortest_path(self, start_node: str, end_node: str) -> list[str]:
        return self.__paths[start_node][end_node]
