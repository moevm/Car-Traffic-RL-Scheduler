import traci
import heapq
import random
from multiprocessing import Process, Manager, Lock
from facade.structures import NodeColor, NodePair
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
         self.__paths_for_cyclic_routes) = [], [], [], {}
        self.__restore_path_matrix, self.__restore_path_matrix_for_cycles = {}, {}
        self.__edges_dict = self.__make_edges_dict()

    def init_poisson_generators(self, poisson_generators_edges: list[str]) -> None:
        self.__poisson_generators_to_nodes = [traci.edge.getToJunction(edge) for edge in poisson_generators_edges]
        self.__poisson_generators_from_nodes = [traci.edge.getFromJunction(edge) for edge in poisson_generators_edges]
        self.__poisson_generators_edges = poisson_generators_edges

    def parallel_make_restore_path_matrix(self) -> None:
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

    def __find_cycle(self, prev_node: str, current_node: str, colors: dict[str, NodeColor]) -> str:
        colors[current_node] = NodeColor.grey
        neighbors = self.__graph[current_node]
        for node, edge in neighbors.items():
            if node != prev_node:
                if colors[node] == NodeColor.white:
                    result = self.__find_cycle(current_node, node, colors)
                    if result is not None:
                        return result
                if colors[node] == NodeColor.grey:
                    return node
        colors[current_node] = NodeColor.black

    def __find_lightest_cycle(self, to_node: str, from_node: str) -> list[str]:
        colors = {node: NodeColor.white for node in self.__clear_nodes}
        start_node_of_cycle = self.__find_cycle(from_node, to_node, colors)
        neighbors = list(self.__graph[start_node_of_cycle].items())
        min_path_length_in_meters = self.__INF
        best_cyclic_path = []
        for node_1, edge_1 in neighbors:
            for node_2, edge_2 in neighbors:
                if node_1 != node_2:
                    restore_path_matrix = {}
                    self.__make_restore_path_matrix(node_1, restore_path_matrix, start_node_of_cycle,
                                                    start_node_of_cycle)
                    path = [self.__graph[start_node_of_cycle][node_1]] + self.get_shortest_path(node_1, node_2,
                                                                                                restore_path_matrix)
                    if len(path) > 1:
                        path.append(self.__graph[node_2][start_node_of_cycle])
                        path_length_in_meters = self.get_path_length_in_edges(path)
                        if path_length_in_meters < min_path_length_in_meters:
                            best_cyclic_path = path
                            min_path_length_in_meters = path_length_in_meters
        return best_cyclic_path

    def __find_way_back(self, i: int, to_node: str, start_nodes, paths_for_cyclic_routes) -> None:
        if all(node is not None for node in self.__restore_path_matrix[to_node].values()):
            return
        min_path_length = self.__INF
        index = 0
        cyclic_path = self.__find_lightest_cycle(to_node, self.__poisson_generators_from_nodes[i])
        for j, edge in enumerate(cyclic_path):
            node = self.__edges_dict[edge].node_from
            path = self.get_shortest_path(to_node, node)
            path_length_in_meters = self.get_path_length_in_edges(path)
            if path_length_in_meters < min_path_length:
                min_path_length = path_length_in_meters
                index = j
        if self.__edges_dict[cyclic_path[index]].node_from not in start_nodes:
            start_nodes.append(self.__edges_dict[cyclic_path[index]].node_from)
        paths_for_cyclic_routes[to_node] = cyclic_path[index:] + cyclic_path[:index]
        self.__network_logger.step_progress_bar()

    def __parallel_make_restore_path_matrix_for_cycles(self, start_nodes: set[str]) -> None:
        manager = Manager()
        lock = Lock()
        restore_path_matrix = manager.dict()
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix_for_cycles, len(start_nodes))
        processes = []
        for start_node in start_nodes:
            process = Process(target=self.__make_restore_path_matrix, args=(start_node, restore_path_matrix))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            with lock:
                self.__network_logger.step_progress_bar()
        self.__network_logger.destroy_progress_bar()
        self.__restore_path_matrix_for_cycles = restore_path_matrix

    def parallel_find_way_back(self) -> None:
        self.__network_logger.init_progress_bar(Message.find_way_back_paths, len(self.__poisson_generators_to_nodes))
        manager = Manager()
        lock = Lock()
        paths_for_cyclic_routes = manager.dict()
        start_nodes = manager.list()
        processes = []
        for i, to_node in enumerate(self.__poisson_generators_to_nodes):
            process = Process(target=self.__find_way_back, args=(i, to_node, start_nodes, paths_for_cyclic_routes))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
            with lock:
                self.__network_logger.step_progress_bar()
        self.__paths_for_cyclic_routes = paths_for_cyclic_routes
        self.__network_logger.destroy_progress_bar()
        self.__parallel_make_restore_path_matrix_for_cycles(set(start_nodes))

    def __make_edges_dict(self) -> dict[str, NodePair]:
        edges_dict = {}
        for edge in self.__clear_edges:
            node_pair = NodePair(node_from=traci.edge.getFromJunction(edge), node_to=traci.edge.getToJunction(edge))
            edges_dict[edge] = node_pair
        return edges_dict

    def __make_restore_path_matrix(self, start_node: str, restore_path_matrix: dict[str, dict[str, str | None]],
                                   prev_node=None, blocked_node=None) -> None:
        # self.__network_logger.step_progress_bar()
        if start_node in restore_path_matrix:
            return
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

    def __make_graph(self) -> dict[str, dict[str, str]]:
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
        nodes = self.get_clear_nodes()
        extreme_nodes = []
        for node in nodes:
            if len(traci.junction.getOutgoingEdges(node)) == 2:
                extreme_nodes.append(node)
        return extreme_nodes

    def get_shortest_path(self, start_node: str, end_node: str, restore_path_matrix=None) -> list[str]:
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
                cyclic_path = self.__paths_for_cyclic_routes[start_node]
                start_node_of_cycle = traci.edge.getFromJunction(cyclic_path[0])
                path_from_start_node_to_start_of_cycle = self.get_shortest_path(start_node, start_node_of_cycle)
                path_from_start_of_cycle_to_end_node = self.get_shortest_path(start_node_of_cycle, end_node,
                                                                              self.__restore_path_matrix_for_cycles)
                return path_from_start_node_to_start_of_cycle + cyclic_path + path_from_start_of_cycle_to_end_node
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

    def get_clear_nodes(self) -> list[str]:
        return self.__clear_nodes

    def get_clear_edges(self) -> list[str]:
        return self.__clear_edges

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

    def get_path_length_in_edges(self, path: list[str]) -> int:
        return len(path)
