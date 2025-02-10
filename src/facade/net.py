import traci
import heapq
import random
from multiprocessing import Process, Manager, Lock

from facade.logger.network_logger import *


class Net:
    def __init__(self):
        self.__network_logger = NetworkLogger()
        self.__INF = float("inf")
        self.__nodes = traci.junction.getIDList()
        self.__edges = traci.edge.getIDList()
        self.__clear_nodes = self.__find_clear_nodes()
        self.__clear_edges = self.__find_clear_edges()
        self.__edges_length = {edge: traci.lane.getLength(f"{edge}_0") for edge in self.__clear_edges}
        self.__extreme_nodes = self.__find_extreme_nodes()
        self.__network_logger.print_graph_info(len(self.__clear_edges), len(self.__clear_nodes))
        self.__graph = self.__make_graph()
        (self.__poisson_generators_to_nodes, self.__poisson_generators_from_nodes, self.__poisson_generators_edges,
         self.__restore_path_matrix) = [], [], [], {}
        # self.__restore_path_matrix = self.__make_restore_path_matrix()

    def init_poisson_generators(self, poisson_generators_edges: list[str]):
        self.__poisson_generators_to_nodes = [traci.edge.getToJunction(edge) for edge in poisson_generators_edges]
        self.__poisson_generators_from_nodes = [traci.edge.getFromJunction(edge) for edge in poisson_generators_edges]
        self.__poisson_generators_edges = poisson_generators_edges

    def parallel_make_restore_path_matrix(self):
        start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix, len(start_nodes))
        manager = Manager()
        lock = Lock()
        restore_path_matrix = manager.dict()
        processes = []
        for i, start_node in enumerate(start_nodes):
            if i < len(self.__extreme_nodes):
                process = Process(target=self.__make_restore_path_matrix, args=(start_node, restore_path_matrix))
            else:
                prev_node = self.__poisson_generators_from_nodes[i - len(self.__extreme_nodes)]
                process = Process(target=self.__make_restore_path_matrix,
                                  args=(start_node, restore_path_matrix, prev_node))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            with lock:
                self.__network_logger.step_progress_bar()
        self.__network_logger.destroy_progress_bar()
        self.__restore_path_matrix = restore_path_matrix
        self.__find_way_back()

    '''
    Либо вызывать Дейкстру для каждой висячей вершины + точек спавна авто
    Либо вызывать A* в runtime
    '''

    # def __make_restore_path_matrix(self):
    #     restore_path_matrix = {}
    #     self.__network_logger.init_progress_bar(Message.init_restore_path_matrix, len(self.__extreme_nodes))
    #     for start_node in self.__extreme_nodes:
    #         self.__network_logger.step_progress_bar()
    #         distances = {node: self.__INF for node in self.__clear_nodes}
    #         distances[start_node] = 0
    #         priority_queue = [(0, start_node)]
    #         previous_nodes = {node: None for node in self.__clear_nodes}
    #         while priority_queue:
    #             current_distance, current_node = heapq.heappop(priority_queue)
    #             if current_distance > distances[current_node]:
    #                 continue
    #             neighbors = list(self.__graph[current_node].items())
    #             random.shuffle(neighbors)
    #             for neighbor_node, edge in neighbors:
    #                 distance = current_distance + traci.lane.getLength(f"{edge}_0")
    #                 if distance < distances[neighbor_node]:
    #                     distances[neighbor_node] = distance
    #                     previous_nodes[neighbor_node] = current_node
    #                     heapq.heappush(priority_queue, (distance, neighbor_node))
    #         restore_path_matrix[start_node] = previous_nodes
    #     self.__network_logger.destroy_progress_bar()
    #     return restore_path_matrix

    def __find_cycle(self, prev_node: str, current_node: str, colors: dict[str, int]) -> str:
        colors[current_node] = 1
        neighbors = self.__graph[current_node]
        for node, edge in neighbors.items():
            if node != prev_node:
                if colors[node] == 0:  # магические числа!!!
                    return self.__find_cycle(current_node, node, colors)
                if colors[node] == 1:
                    return node
        colors[current_node] = 2

    def __find_lightest_cycle(self, to_node, from_node):
        restore_path_matrix = {}
        colors = {node: 0 for node in self.__clear_nodes}
        start_node = self.__find_cycle(from_node, to_node, colors)
        neighbors = list(self.__graph[start_node].items())
        min_path_length_in_meters = self.__INF
        best_path = []
        for node_1, edge_1 in neighbors:
            for node_2, edge_2 in neighbors:
                if node_1 != node_2:
                    self.__make_restore_path_matrix(node_1, restore_path_matrix, start_node, start_node)
                    path = self.get_shortest_path(node_1, node_2, restore_path_matrix)
                    if len(path) > 0:
                        edge = self.__graph[node_2][start_node]
                        path_length_in_meters = self.get_path_length_in_edges(path) + self.__edges_length[edge]
                        if path_length_in_meters < min_path_length_in_meters:
                            path.append(edge)
                            best_path = path
                            min_path_length_in_meters = path_length_in_meters
        return best_path, start_node

    def __find_way_back(self):
        restore_path_matrix = {}
        for i, to_node in enumerate(self.__poisson_generators_to_nodes):
            best_path, start_node = self.__find_lightest_cycle(to_node, self.__poisson_generators_from_nodes[i])
            print('\n', best_path, start_node)

    def __make_restore_path_matrix(self, start_node, restore_path_matrix, prev_node=None, blocked_node=None):
        self.__network_logger.step_progress_bar()
        distances = {node: self.__INF for node in self.__clear_nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node, prev_node)]
        previous_nodes = {node: None for node in self.__clear_nodes}
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

    def __make_graph(self):
        graph = {}
        self.__network_logger.init_progress_bar(Message.init_incidence_matrix, len(self.__clear_edges))
        for edge in self.__clear_edges:
            self.__network_logger.step_progress_bar()
            node_1 = traci.edge.getFromJunction(edge)
            node_2 = traci.edge.getToJunction(edge)
            if node_1 in graph:
                graph[node_1][node_2] = edge
            else:
                value = {node_2: edge}
                graph[node_1] = value
        self.__network_logger.destroy_progress_bar()
        return graph

    def __find_clear_edges(self) -> list:
        clear_edges = []
        for edge in self.__edges:
            if ":" not in edge:
                clear_edges.append(edge)
        return clear_edges

    def __find_clear_nodes(self) -> list:
        clear_nodes = []
        for node in self.__nodes:
            if ":" not in node:
                clear_nodes.append(node)
        return clear_nodes

    def __find_extreme_nodes(self) -> list:
        nodes = self.get_clear_nodes()
        extreme_nodes = []
        for node in nodes:
            if len(traci.junction.getOutgoingEdges(node)) == 2:
                extreme_nodes.append(node)
        return extreme_nodes

    '''
    Сильно тормозит при edges кол-ве узлово около 10к. проблема?
    '''

    def get_shortest_path(self, start_node, end_node, restore_path_matrix=None):
        if restore_path_matrix is not None:
            previous_nodes = restore_path_matrix[start_node]
        else:
            previous_nodes = self.__restore_path_matrix[start_node]
        path = []
        node = end_node
        while node != start_node:
            node_1 = node
            node_2 = previous_nodes[node_1]
            if node_2 is None:
                return []
            edge = self.__graph[node_2][node_1]
            node = node_2
            path.append(edge)
        return path[::-1]

    def get_restore_path_matrix(self):
        return self.__restore_path_matrix

    def get_graph(self):
        return self.__graph

    def get_extreme_nodes(self):
        return self.__extreme_nodes

    def get_clear_nodes(self) -> list:
        return self.__clear_nodes

    def get_clear_edges(self) -> list:
        return self.__clear_edges

    def get_poisson_generators_edges(self):
        return self.__poisson_generators_edges

    def get_poisson_generators_to_nodes(self):
        return self.__poisson_generators_to_nodes

    def get_poisson_generators_from_nodes(self):
        return self.__poisson_generators_from_nodes

    def get_path_length_in_meters(self, path):
        length_meters = 0
        for edge in path:
            length_meters += traci.lane.getLength(f"{edge}_0")
        return length_meters

    def get_path_length_in_edges(self, path):
        return len(path)
