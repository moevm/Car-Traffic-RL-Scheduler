import click
import random
import json
from facade.net import Net
from facade.logger.network_logger import NetworkLogger
from facade.logger.logger import Message
from facade.structures import NodeColor


def generate_intensities(n_generators: int, duration: int) -> list:
    intensities = [random.uniform(1 / duration, 0.1) for i in range(n_generators)]
    return intensities


"""
выбираем только те рёбра-генраторы, из которых достижима любая точка в графе. недостижимость связана с тем, что
нельзя делать лупбэк по одному и тому же ребру
"""


def is_there_cycle(prev_node: str, current_node: str, graph: dict[str, dict[str, str]],
                   colors: dict[str, NodeColor]) -> bool:
    colors[current_node] = NodeColor.grey
    neighbors = graph[current_node]
    for node, edge in neighbors.items():
        if node != prev_node:
            if colors[node] == NodeColor.white:
                result = is_there_cycle(current_node, node, graph, colors)
                if result:
                    return result
            if colors[node] == NodeColor.grey:
                return True
    colors[current_node] = NodeColor.black
    return False


def extract_net_config(sumo_config):
    slash_position = sumo_config.rfind('/')
    extension_position = sumo_config.rfind('.sumocfg')
    net_name = f"{sumo_config[slash_position + 1:extension_position]}.net.xml"
    return sumo_config[:slash_position + 1] + net_name


def generate_poisson_generators(part_generators: float, net: Net) -> list:
    sumolib_net = net.get_sumolib_net()
    possible_edges = []
    possible_nodes = set() # суть в том что нельзя доб
    graph = net.get_graph()
    edges = net.get_edges()
    nodes = net.get_nodes()
    network_logger = NetworkLogger()
    n_generators = int(part_generators * len(edges))
    network_logger.init_progress_bar(Message.search_for_valid_edges, n_generators)
    random.shuffle(edges)
    for i, edge in enumerate(edges):
        colors = {node: NodeColor.white for node in nodes}
        if (is_there_cycle(sumolib_net.getEdge(edge).getFromNode().getID(),
                           sumolib_net.getEdge(edge).getToNode().getID(), graph, colors) and
                sumolib_net.getEdge(edge).getToNode().getID() not in possible_nodes):
            possible_edges.append(edge)
            possible_nodes.add(sumolib_net.getEdge(edge).getToNode().getID())
            network_logger.step_progress_bar()
        if len(possible_edges) == n_generators:
            break
    network_logger.destroy_progress_bar()
    if n_generators > len(possible_edges):
        """
        рассмотреть что будет при 4 более полоске в одном направлении
        """
        print(
            "Number of Poisson generators is greater than number of possible edges. The number of Poisson generators will be "
            "set equal to the number of possible edges in the graph.")
        n_generators = len(possible_edges)
    return possible_edges[:n_generators]


def make_list_of_turned_off_traffic_lights(part_off_traffic_lights: float, net: Net) -> list[str]:
    sumolib_net = net.get_sumolib_net()
    tls_objects = sumolib_net.getTrafficLights()
    random.shuffle(tls_objects)
    n_off_traffic_lights = int(part_off_traffic_lights * len(tls_objects))
    if n_off_traffic_lights > len(tls_objects):
        print(
            "The number of switched off traffic lights is greater than the total number of traffic lights. All traffic "
            "lights will be turned off."
        )
        n_off_traffic_lights = len(tls_objects)
    return [tls_object.getID() for tls_object in tls_objects[:n_off_traffic_lights]]


@click.command()
@click.option('--duration', '-d', type=int, default=50000, help='simulation duration in steps.')
@click.option('--iterations', '-i', type=int, default=10, help='number '
                                                              'of iterations of initial traffic generation.')
@click.option('--part-generators', '-g', type=float, default=0.1, help='this part of the edges will act '
                                                                       'as flow generators.')
@click.option('--file', '-f', type=str, default='./configs/simulation-parameters/simulation_parameters.json',
              help='path to simulation parameters config file (.sumocfg).')
@click.option('--init-delay', '-n', type=int, default=10, help='delay between vehicle departures during map '
                                                               'initialization by traffic.')
@click.option('--part-of-the-path', '-p', type=float, default=0.5, help='the total part of the path that the '
                                                                        'initializing traffic must travel for generation '
                                                                        'to begin using Poisson flows.')
@click.option('--check-time', '-t', type=int, default=100, help='check time (in simulation ticks) whether '
                                                                'the traffic has traveled half of the total path in'
                                                                'meters.')
@click.option('--part-off-traffic-lights', '-l', default=0.2, type=float, help='this part of traffic lights '
                                                                               'will be turned off.')
@click.option('--threshold-edge-length', '-e', type=int, help='traffic lights will be installed every '
                                                              '[--threshold-edge-length] value meters on each edge, '
                                                              'simulating pedestrian crossings.')
@click.argument('sumo_config', nargs=1, type=str)
def main(duration: int, iterations: int, part_generators: float, file: str, init_delay: int, part_of_the_path: float,
         check_time: int, part_off_traffic_lights: float, threshold_edge_length: int, sumo_config: str) -> None:
    """
    This program generates a config with parameters for simulation.
    """
    net_config = extract_net_config(sumo_config)
    net = Net(net_config)
    poisson_generators = generate_poisson_generators(part_generators, net)
    intensities = generate_intensities(len(poisson_generators), duration)
    turned_off_traffic_lights = make_list_of_turned_off_traffic_lights(part_off_traffic_lights, net)
    data = {"DURATION": duration, "INIT_DELAY": init_delay, "ITERATIONS": iterations,
            "PART_OF_THE_PATH": part_of_the_path, "CHECK_TIME": check_time, "intensities": intensities,
            "poisson_generators_edges": poisson_generators, "turned_off_traffic_lights": turned_off_traffic_lights}
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
