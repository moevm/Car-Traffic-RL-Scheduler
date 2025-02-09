import random

import traci


class TrafficControl:
    def __init__(self, intensities, generators_edges, duration, clear_edges):
        self.__intensities = intensities
        self.__generators_edges = generators_edges
        self.__duration = duration
        self.__edges = clear_edges
        self.__vehicles_data = []

    def init_vehicles_data(self, vehicles_data):
        self.__vehicles_data = vehicles_data

    def generate_schedule_for_poisson_flow(self, start_time):
        data = {}
        for i, edge in enumerate(self.__generators_edges):
            prev_timestamp = start_time
            while prev_timestamp < self.__duration:
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

    def have_vehicles_passed_halfway_in_total(self):
        vehicles = traci.vehicle.getIDList()
        sum_full_distances = 0
        sum_current_distances = 0
        for vehicle_id in vehicles:
            sum_current_distances += traci.vehicle.getDistance(vehicle_id)
            sum_full_distances += self.__vehicles_data[vehicle_id]
        print(sum_current_distances / sum_full_distances)
        if sum_current_distances >= sum_full_distances / 2:
            return True
        else:
            return False
