import traci
import heapq
import random

from typing import Optional
from sumolib.net import readNet
from multiprocessing import Pool, cpu_count
from facade.structures import NodePair
from facade.logger.network_logger import *


class Net:
    def __init__(self, net_config: str, poisson_generators_edges=None, cpu_scale: int = 1):
        if poisson_generators_edges is None:
            poisson_generators_edges = []
        self.__CPU_SCALE = cpu_scale
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
        self.__poisson_generators_to_nodes, self.__poisson_generators_from_nodes, self.__poisson_generators_edges = (
            [], [], [])
        (self.__restore_path_matrix, self.__restore_path_matrix_for_way_back_routes,
         self.__paths_for_way_back_routes) = {}, {}, {}
        self.__edges_dict = self.__make_edges_dict()
        self.__turned_on_traffic_lights_ids = []
        self.__paths = {}
        for edge in poisson_generators_edges:
            self.__poisson_generators_to_nodes.append(self.__sumolib_net.getEdge(edge).getToNode().getID())
            self.__poisson_generators_from_nodes.append(self.__sumolib_net.getEdge(edge).getFromNode().getID())
        self.__poisson_generators_edges = poisson_generators_edges

    def __callback_find_routes(self, response):
        self.__paths.update(response)
        self.__network_logger.step_progress_bar()

    @staticmethod
    def _find_routes(start_node : str,
                     nodes: list[str],
                     graph: dict[str, dict[str, str]],
                     edges_dict: dict[str, NodePair],
                     restore_path_matrix: dict[str, dict[str, str]],
                     paths_for_way_back_routes: dict[str, list[str]],
                     restore_path_matrix_for_way_back_routes: dict[str, dict[str, str]],
                     connecting_edge=None
                     ) -> dict[str, dict[str, list[str]]]:
        local_paths = {}
        for j in range(len(nodes)):
            if start_node != nodes[j]:
                path = Net._get_shortest_path(start_node,
                                              nodes[j],
                                              graph,
                                              edges_dict,
                                              restore_path_matrix,
                                              paths_for_way_back_routes,
                                              restore_path_matrix_for_way_back_routes)
                if connecting_edge is not None:
                    path.insert(0, connecting_edge)
                if start_node not in local_paths:
                    local_paths[start_node] = {nodes[j]: path}
                else:
                    local_paths[start_node][nodes[j]] = path
        return local_paths

    def parallel_find_routes(self) -> None:
        start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
        self.__network_logger.init_progress_bar(Message.find_all_routes, len(start_nodes))
        args_list = []
        for i, start_node in enumerate(start_nodes):
            if i >= len(self.__extreme_nodes):
                args = (start_node,
                        self.__nodes,
                        self.__graph,
                        self.__edges_dict,
                        self.__restore_path_matrix,
                        self.__paths_for_way_back_routes,
                        self.__restore_path_matrix_for_way_back_routes,
                        self.__poisson_generators_edges[i - len(self.__extreme_nodes)])
            else:
                args = (start_node,
                        self.__nodes,
                        self.__graph,
                        self.__edges_dict,
                        self.__restore_path_matrix,
                        self.__paths_for_way_back_routes,
                        self.__restore_path_matrix_for_way_back_routes,
                        None)
            args_list.append(args)
        with Pool(cpu_count() * self.__CPU_SCALE) as p:
            for args in args_list:
                p.apply_async(Net._find_routes, args,
                              callback=self.__callback_find_routes)
            p.close()
            p.join()
        self.__network_logger.destroy_progress_bar()

    def turn_off_traffic_lights(self, turned_off_traffic_lights: list[str]) -> None:
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

    def __callback_make_restore_path_matrix(self, response: tuple[str, dict[str, str]]) -> None:
        start_node, previous_nodes = response
        self.__restore_path_matrix[start_node] = previous_nodes
        self.__network_logger.step_progress_bar()

    def parallel_make_restore_path_matrix(self) -> None:
        start_nodes = self.__extreme_nodes + self.__poisson_generators_to_nodes
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix, len(start_nodes))
        args_list = []
        for i, start_node in enumerate(start_nodes):
            if i < len(self.__extreme_nodes):
                args = (start_node, None, self.__INF, self.__nodes, self.__graph, self.__edges_length)
            else:
                prev_node = self.__poisson_generators_from_nodes[i - len(self.__extreme_nodes)]
                args = (start_node, prev_node, self.__INF, self.__nodes, self.__graph, self.__edges_length)
            args_list.append(args)
        with Pool(cpu_count() * self.__CPU_SCALE) as p:
            for args in args_list:
                p.apply_async(Net._make_restore_path_matrix, args,
                              callback=self.__callback_make_restore_path_matrix)
            p.close()
            p.join()
        self.__network_logger.destroy_progress_bar()

    @staticmethod
    def _make_restore_path_matrix(start_node: str,
                                  prev_node: Optional[str],
                                  inf: float,
                                  nodes: list[str],
                                  graph: dict[str, dict[str, str]],
                                  edges_length: dict[str, float]):
        distances = {node: inf for node in nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node, prev_node)]
        previous_nodes = {node: None for node in nodes}
        while priority_queue:
            current_distance, current_node, prev_node = heapq.heappop(priority_queue)
            if current_distance > distances[current_node]:
                continue
            neighbors = list(graph[current_node].items())
            random.shuffle(neighbors)
            for neighbor_node, edge in neighbors:
                if neighbor_node != prev_node:
                    distance = current_distance + edges_length[edge]
                    if distance < distances[neighbor_node]:
                        distances[neighbor_node] = distance
                        previous_nodes[neighbor_node] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor_node, current_node))
        return start_node, previous_nodes

    def __callback_make_restore_path_matrix_for_way_back_routes(self, response: tuple[str, dict[str, str]]):
        start_node, previous_nodes = response
        self.__restore_path_matrix_for_way_back_routes[start_node] = previous_nodes
        self.__network_logger.step_progress_bar()

    def __parallel_make_restore_path_matrix_for_way_back_routes(self, start_nodes: list[str]) -> None:
        self.__network_logger.init_progress_bar(Message.init_restore_path_matrix_for_cycles, len(start_nodes))
        args_list = []
        for start_node in start_nodes:
            args = (start_node, None, self.__INF, self.__nodes, self.__graph, self.__edges_length)
            args_list.append(args)
        with Pool(cpu_count() * self.__CPU_SCALE) as p:
            for args in args_list:
                p.apply_async(Net._make_restore_path_matrix, args,
                              callback=self.__callback_make_restore_path_matrix_for_way_back_routes)
            p.close()
            p.join()
        self.__network_logger.destroy_progress_bar()

    def __callback_for_find_way_back(self, response: tuple[str, list[str]]) -> None:
        to_node, path = response
        self.__paths_for_way_back_routes[to_node] = path
        self.__network_logger.step_progress_bar()

    @staticmethod
    def _error(e):
        print(e)

    def parallel_find_way_back(self):
        self.__network_logger.init_progress_bar(Message.find_way_back_paths, len(self.__poisson_generators_to_nodes))
        args_list = []
        for i, to_node in enumerate(self.__poisson_generators_to_nodes):
            args = (self.__poisson_generators_from_nodes[i], to_node, self.__graph,
                    self.__restore_path_matrix, self.__edges_dict)
            args_list.append(args)
        with Pool(cpu_count() * self.__CPU_SCALE) as p:
            for args in args_list:
                p.apply_async(Net._find_way_back, args=args, callback=self.__callback_for_find_way_back,
                              error_callback=Net._error)
            p.close()
            p.join()
        self.__network_logger.destroy_progress_bar()
        intersection_nodes = list(self.__paths_for_way_back_routes.keys())
        self.__parallel_make_restore_path_matrix_for_way_back_routes(intersection_nodes)

    @staticmethod
    def _find_way_back(from_node: str,
                       to_node: str,
                       graph: dict[str, dict[str, str]],
                       restore_path_matrix: dict[str, dict[str, str]],
                       edges_dict: dict[str, NodePair]):
        intersection_node = Net.find_intersection(from_node, to_node, graph)
        path = Net._get_shortest_path(to_node, intersection_node, graph, edges_dict, restore_path_matrix)
        return to_node, path

    @staticmethod
    def find_intersection(from_node: str,
                          to_node: str,
                          graph: dict[str, dict[str, str]]):
        current_node = to_node
        prev_node = from_node
        while len(graph[current_node]) < 3:
            neighbors = {node: edge for node, edge in graph[current_node].items() if node != prev_node}
            if len(neighbors) == 0:
                break
            else:
                tmp = current_node
                current_node = list(neighbors.keys())[0]
                prev_node = tmp
        return current_node

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

    @staticmethod
    def _get_shortest_path(start_node: str,
                           end_node: str,
                           graph: dict[str, dict[str, str]],
                           edges_dict: dict[str, NodePair],
                           restore_path_matrix=None,
                           paths_for_way_back_routes=None,
                           restore_path_matrix_for_way_back_routes=None) -> list[str]:
        previous_nodes = restore_path_matrix[start_node]
        path = []
        node = end_node
        while node != start_node:
            node_1 = node
            node_2 = previous_nodes[node_1]
            if (node_2 is None) and (restore_path_matrix is not None):
                return []
            elif (node_2 is None) and (restore_path_matrix is None):
                way_back_route = paths_for_way_back_routes[start_node]
                if not way_back_route:
                    intersection_node = start_node
                else:
                    intersection_node = edges_dict[way_back_route[-1]].node_to
                path_from_intersection_node_to_end_node = Net._get_shortest_path(intersection_node,
                                                                                 end_node,
                                                                                 graph,
                                                                                 edges_dict,
                                                                                 restore_path_matrix_for_way_back_routes,
                                                                                 None,
                                                                                 None)
                return way_back_route + path_from_intersection_node_to_end_node
            edge = graph[node_2][node_1]
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
