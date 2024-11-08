import traci, heapq

class Net:
    def __init__(self):
        self.__INF = 10 ** 10
        self.__nodes = traci.junction.getIDList()
        self.__edges = traci.edge.getIDList()
        self.__clear_nodes = self.__find_clear_nodes()
        self.__clear_edges = self.__find_clear_edges()
        self.__extreme_nodes = self.__find_extreme_nodes()
        self.__graph = self.__make_graph()

    def __make_graph(self):
        graph = {}
        for edge in self.__clear_edges:
            node_1 = traci.edge.getFromJunction(edge)
            node_2 = traci.edge.getToJunction(edge)
            try:
                graph[node_1][node_2] = traci.lane.getLength(f"{edge}_0")
            except KeyError:
                value = {node_2: traci.lane.getLength(f"{edge}_0")}
                graph[node_1] = value
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

    def get_graph(self):
        return self.__graph

    def get_extreme_nodes(self):
        return self.__extreme_nodes

    def get_clear_nodes(self) -> list:
        return self.__clear_nodes

    def __dijkstra_algorithm(self, start_node):
        distances = {node: self.__INF for node in self.__clear_nodes}
        distances[start_node] = 0
        priority_queue = [(0, start_node)]
        previous_nodes = {node: None for node in self.__clear_nodes}
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            if current_distance > distances[current_node]:
                continue
            for neighbor, edge_weight in self.__graph[current_node].items():
                distance = current_distance + edge_weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        return previous_nodes

    def find_shortest_path(self, start_node, end_node):
        path = []
        previous_nodes = self.__dijkstra_algorithm(start_node)
        node = end_node
        while node != start_node:
            edge = node
            node = previous_nodes[node]
            edge = node + edge
            path.append(edge)
        return path[::-1]

    def find_path_length_meters(self, path):
        length_meters = 0
        for edge in path:
            length_meters += traci.lane.getLength(f"{edge}_0")
        return length_meters

    def find_path_length_edges(self, path):
        return len(path)