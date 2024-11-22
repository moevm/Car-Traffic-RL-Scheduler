import random

import traci

class TransportGeneration:
    def __init__(self):
        self.__transport_counter = 0

    def generate(self, last_target_nodes_data):
        for last_target_node_data in last_target_nodes_data:
            route_id = last_target_node_data.last_route_id
            path = last_target_node_data.last_path
            traci.route.add(route_id, path)
            traci.vehicle.add(self.__transport_counter, route_id)
            traci.vehicle.setColor(self.__transport_counter, self.__generate_color())
            self.__transport_counter += 1

    def __generate_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b