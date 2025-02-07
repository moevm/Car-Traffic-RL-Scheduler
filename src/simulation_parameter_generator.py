import click
import random
import traci
import json
import queue
from facade.net import Net
from facade.logger.network_logger import NetworkLogger
from facade.logger.logger import Message

def generate_intensities(n_generators: int, duration: int) -> list:
    intensities = [random.uniform(1 / duration, 0.5) for i in range(n_generators)]
    return intensities

"""
выбираем только те рёбра-генраторы, из которых достижима любая точка в графе. недостижимость связана с тем, что
нельзя делать лупбэк по одному и тому же ребру
"""
def bfs(start_node: str, graph: dict, clear_nodes: list) -> bool:
    q = queue.Queue()
    q.put(start_node)
    visited = set()
    visited.add(start_node)
    while not q.empty():
        current_node = q.get()
        neighbors = graph[current_node]
        for node, edge in neighbors.items():
            if node not in visited:
                q.put(node)
                visited.add(node)
    if len(visited) == len(clear_nodes):
        return True
    else:
        return False


def generate_poisson_generators(sumo_config: str, n_generators: int) -> list:
    sumo_cmd = ["sumo", "-c", sumo_config]
    traci.start(sumo_cmd)
    traci.simulationStep()
    net = Net()
    network_logger = NetworkLogger()
    network_logger.init_progress_bar(Message.search_for_valid_edges, n_generators)
    possible_edges = []
    graph = net.get_graph()
    clear_edges = net.get_clear_edges()
    clear_nodes = net.get_clear_nodes()
    for i, edge in enumerate(clear_edges):
        network_logger.step_progress_bar()
        if bfs(traci.edge.getToJunction(edge), graph, clear_nodes):
            possible_edges.append(edge)
        if len(possible_edges) == n_generators:
            break
    network_logger.destroy_progress_bar()
    traci.close()
    if n_generators > len(possible_edges):
        """
        рассмотреть что будет при 4 более полоске в одном направлении
        """
        print(
            "Number of Poisson generators is greater than number of possible edges. The number of Poisson generators will be "
            "set equal to the number of possible edges in the graph.")
        n_generators = len(possible_edges)
    random.shuffle(possible_edges)
    return possible_edges[:n_generators]


@click.command()
@click.option('--duration', '-d', type=int, default=3600, help='simulation duration in steps.')
@click.option('--iterations', '-i', type=int, default=1, help='number '
                                                              'of iterations of initial traffic generation.')
@click.option('--generators', '-g', type=int, default=1, help='number of poisson generators.')
@click.option('--file', '-f', type=str, default='./configs/simulation-parameters/simulation_parameters.json',
              help='path to SUMO config file (.sumocfg).')
@click.option('--init-delay', '-n', type=int, default=10, help='delay between vehicle departures during map '
                                                               'initialization by traffic.')
@click.argument('sumo_config', nargs=1, type=str)
def main(duration: int, iterations: int, generators: int, file: str, init_delay, sumo_config: str) -> None:
    """
    This program generates a config with parameters for simulation.
    """
    intensities = generate_intensities(generators, duration)
    poisson_generators = generate_poisson_generators(sumo_config, generators)
    data = {"DURATION": duration, "INIT_DELAY": init_delay, "ITERATIONS": iterations,
            "intensities": intensities, "poisson_generators_edges": poisson_generators}
    with open(file, 'w') as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    main()
