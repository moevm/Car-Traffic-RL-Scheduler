import traci

class TransportGeneration:
    def __init__(self):
        self.__transport_counter = 0

    def generate(self, last_target_nodes_data):
        for last_target_node_data in last_target_nodes_data:
            route_id = last_target_node_data.routes_ids[-1]
            path = last_target_node_data.paths[-1]
            print(route_id, path)
            traci.route.add(route_id, path)
            traci.vehicle.add(self.__transport_counter, route_id)
            traci.vehicle.setColor(self.__transport_counter, (255, 0, 0))
            self.__transport_counter += 1
