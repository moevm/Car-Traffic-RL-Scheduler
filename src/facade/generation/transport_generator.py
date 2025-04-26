import random
import traci

from facade.structures import NodeData


class TransportGenerator:
    def __init__(self, intensities: list[float],
                 generators_edges: list[str],
                 duration: int,
                 edges: list[str],
                 part_of_path: float):
        self.__vehicles_data = {}
        self.__intensities = intensities
        self.__generators_edges = generators_edges
        self.__duration = duration
        self.__edges = edges
        self.__part_of_path = part_of_path

    def generate_transport(self, last_target_nodes_data: list[NodeData]) -> None:
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
            self.__vehicles_data.pop(arrived_vehicle_id, None)

    @staticmethod
    def __generate_color() -> (int, int, int):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b

    def get_vehicles_data(self) -> dict[str, int]:
        return self.__vehicles_data

    def generate_schedule_for_poisson_flow(self, start_time: int) -> dict[int, list[str]]:
        data = {}
        for i, edge in enumerate(self.__generators_edges):
            prev_timestamp = start_time
            while prev_timestamp < self.__duration + start_time:
                if edge in data:
                    data[edge].append(prev_timestamp + round(random.expovariate(self.__intensities[i])))
                else:
                    data[edge] = [prev_timestamp + round(random.expovariate(self.__intensities[i]))]
                prev_timestamp = data[edge][-1]
        schedule = {}
        for edge, timestamps in data.items():
            for timestamp in timestamps:
                if timestamp in schedule:
                    schedule[timestamp].append(edge)
                else:
                    schedule[timestamp] = [edge]
        return schedule

    def have_vehicles_passed_part_of_path_in_total(self) -> bool:
        vehicles = traci.vehicle.getIDList()
        sum_full_distances = 0
        sum_current_distances = 0
        for vehicle_id in vehicles:
            sum_current_distances += traci.vehicle.getDistance(vehicle_id)
            sum_full_distances += self.__vehicles_data[vehicle_id]
        if sum_current_distances >= sum_full_distances * self.__part_of_path:
            return True
        else:
            return False
