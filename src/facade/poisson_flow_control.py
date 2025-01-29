import random


class PoissonFlowControl:
    def __init__(self, intensities, generators_edges, duration):
        self.__intensities = intensities
        self.__generators_edges = generators_edges
        self.__duration = duration

    def generate_schedule(self, start_time):
        prev_timestamp = start_time
        data = {}
        for i, edge in enumerate(self.__generators_edges):
            while prev_timestamp < self.__duration:
                try:
                    data[edge].append(round(random.expovariate(self.__intensities[i])))
                except KeyError:
                    data[edge] = [round(random.expovariate(self.__intensities[i]))]
                prev_timestamp += data[edge][-1]
        schedule = {}
        for edge, timestamps in data.items():
            for timestamp in timestamps:
                try:
                    schedule[timestamp].append(edge)
                except KeyError:
                    schedule[timestamp] = [edge]
        return schedule