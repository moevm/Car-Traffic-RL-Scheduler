import traci, heapq, random, tqdm


class Net:
    def __init__(self):
        self.__INF = float("inf")
        self.__nodes = traci.junction.getIDList()
        self.__edges = traci.edge.getIDList()
        self.__clear_nodes = self.__find_clear_nodes()
        self.__clear_edges = self.__find_clear_edges()
        self.__extreme_nodes = self.__find_extreme_nodes()
        print("Construction of incidence matrix...")
        self.__graph = self.__make_graph()
        print("Construction of restore path matrix...")
        self.__restore_path_matrix = self.__make_restore_path_matrix()

    def __make_edge_matrix(self):
        edge_matrix = {}
        for edge in tqdm.tqdm(self.__clear_edges):
            node_1 = traci.edge.getFromJunction(edge)
            node_2 = traci.edge.getToJunction(edge)
            try:
                edge_matrix[node_1][node_2] = edge
            except KeyError:
                value = {node_2: edge}
                edge_matrix[node_1] = value
        return edge_matrix

    def __init_distance_and_restore_path_matrices(self):
        dist_matrix = {}
        next_matrix = {}
        for node_1 in self.__clear_nodes:
            for node_2 in self.__clear_nodes:
                try:
                    dist_matrix[node_1][node_2] = self.__INF
                    next_matrix[node_1][node_2] = None
                except KeyError:
                    value = {node_2: self.__INF}
                    dist_matrix[node_1] = value
                    value = {node_2: None}
                    next_matrix[node_1] = value
        for node in self.__clear_nodes:
            dist_matrix[node][node] = 0
            next_matrix[node][node] = node
        for node_1 in self.__graph:
            for node_2 in self.__graph[node_1]:
                dist_matrix[node_1][node_2] = traci.lane.getLength(f"{self.__graph[node_1][node_2]}_0")
                next_matrix[node_1][node_2] = node_2
        return dist_matrix, next_matrix

    def __make_restore_path_matrix(self):
        distance_matrix, restore_path_matrix = self.__init_distance_and_restore_path_matrices()
        for k in tqdm.tqdm(distance_matrix):
            for i in distance_matrix:
                for j in distance_matrix:
                    if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                        restore_path_matrix[i][j] = restore_path_matrix[i][k]
        return restore_path_matrix

    def __make_graph(self):
        graph = {}
        for edge in tqdm.tqdm(self.__clear_edges):
            node_1 = traci.edge.getFromJunction(edge)
            node_2 = traci.edge.getToJunction(edge)
            try:
                graph[node_1][node_2] = edge
            except KeyError:
                value = {node_2: edge}
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

    def get_shortest_path(self, start_node, end_node):
        if self.__restore_path_matrix[start_node][end_node] is None:
            return []
        path_nodes = [start_node]
        path_edges = []
        node = start_node
        while node != end_node:
            node = self.__restore_path_matrix[node][end_node]
            path_nodes.append(node)
        if len(path_nodes) == 1:
            return []
        for i in range(1, len(path_nodes)):
            path_edges.append(self.__graph[path_nodes[i - 1]][path_nodes[i]])
        return path_edges

    def get_graph(self):
        return self.__graph

    def get_extreme_nodes(self):
        return self.__extreme_nodes

    def get_clear_nodes(self) -> list:
        return self.__clear_nodes

    def get_path_length_in_meters(self, path):
        length_meters = 0
        for edge in path:
            length_meters += traci.lane.getLength(f"{edge}_0")
        return length_meters

    def get_path_length_in_edges(self, path):
        return len(path)

    # def __dijkstra_algorithm(self, start_node):
    #     distances = {node: self.__INF for node in self.__clear_nodes}
    #     distances[start_node] = 0
    #     priority_queue = [(0, start_node)]
    #     previous_nodes = {node: None for node in self.__clear_nodes}
    #     while priority_queue:
    #         current_distance, current_node = heapq.heappop(priority_queue)
    #         if current_distance > distances[current_node]:
    #             continue
    #         neighbors_and_weights = list(self.__graph[current_node].items())
    #         random.shuffle(neighbors_and_weights)
    #         for neighbor, edge_weight in neighbors_and_weights:
    #             distance = current_distance + edge_weight
    #             if distance < distances[neighbor]:
    #                 distances[neighbor] = distance
    #                 previous_nodes[neighbor] = current_node
    #                 heapq.heappush(priority_queue, (distance, neighbor))
    #     return previous_nodes
    #
    # def __make_path(self, start_node, end_node, previous_nodes):
    #     path = []
    #     node = end_node
    #     while node != start_node:
    #         node_1 = node
    #         node_2 = previous_nodes[node]
    #         edge = node_2 + node_1
    #         path.append(edge)
    #     return path[::-1]

    # def find_shortest_path(self, start_node, end_node):
    #     previous_nodes = self.__dijkstra_algorithm(start_node)
    #     return self.__make_path(start_node, end_node, previous_nodes)