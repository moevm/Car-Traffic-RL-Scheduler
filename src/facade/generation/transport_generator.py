import random
import traci

from facade.structures import NodeData


class TransportGenerator:
    def __init__(self):
        self.__vehicles_data = {}

    def generate(self, last_target_nodes_data: list[NodeData]) -> None:
        for last_target_node_data in last_target_nodes_data:
            routes_ids = last_target_node_data.last_routes_ids
            paths = last_target_node_data.last_paths
            for i in range(len(paths)):
                traci.route.add(routes_ids[i], paths[i])
                traci.vehicle.add(routes_ids[i], routes_ids[i])
                traci.vehicle.setColor(routes_ids[i], self.__generate_color())
                self.__vehicles_data[str(routes_ids[i])] = last_target_node_data.path_length_meters[-1]

    def clean_vehicles_data(self):
        for arrived_vehicle_id in traci.simulation.getArrivedIDList():
            del self.__vehicles_data[arrived_vehicle_id]

    @staticmethod
    def __generate_color() -> (int, int, int):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b

    def get_vehicles_data(self) -> dict[str, int]:
        return self.__vehicles_data
