import random
import traci


class TransportGenerator:
    def __init__(self):
        self.__vehicles_data = {}

    def generate(self, last_target_nodes_data):
        for last_target_node_data in last_target_nodes_data:
            route_id = last_target_node_data.last_route_id
            path = last_target_node_data.last_path
            traci.route.add(route_id, path)
            traci.vehicle.add(route_id, route_id)
            traci.vehicle.setColor(route_id, self.__generate_color())
            self.__vehicles_data[str(route_id)] = last_target_node_data.path_length_meters[-1]

    def __generate_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b

    def get_vehicles_data(self):
        return self.__vehicles_data
