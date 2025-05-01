import click
import random
import json

from facade.net import Net
from facade.logger.network_logger import NetworkLogger
from facade.logger.logger import Message


def generate_intensities(n_generators: int, lower_limit: float, upper_limit: float) -> list:
    intensities = [random.uniform(lower_limit, upper_limit) for _ in range(n_generators)]
    return intensities


def extract_net_config(sumo_config):
    slash_position = sumo_config.rfind('/')
    extension_position = sumo_config.rfind('.sumocfg')
    net_name = f"{sumo_config[slash_position + 1:extension_position]}.net.xml"
    return sumo_config[:slash_position + 1] + net_name


def generate_poisson_generators(part_generators: float, net: Net) -> list:
    sumolib_net = net.get_sumolib_net()
    possible_edges, required_edges = [], []
    edges = net.get_edges()
    network_logger = NetworkLogger()
    random.shuffle(edges)
    remain_edges = edges.copy()
    for i, edge in enumerate(edges):
        from_node = sumolib_net.getEdge(edge).getFromNode().getID()
        if len(sumolib_net.getNode(from_node).getOutgoing()) == 1:
            required_edges.append(edge)
            remain_edges.remove(edge)
    n_generators = int(part_generators * len(remain_edges))
    network_logger.init_progress_bar(Message.search_for_valid_edges, n_generators)
    for i, edge in enumerate(remain_edges):
        to_node = sumolib_net.getEdge(edge).getToNode().getID()
        if len(sumolib_net.getNode(to_node).getOutgoing()) > 1:
            possible_edges.append(edge)
            network_logger.step_progress_bar()
        if len(possible_edges) == n_generators:
            break
    network_logger.destroy_progress_bar()
    if n_generators > len(possible_edges):
        print(
            "Number of Poisson generators is greater than number of possible edges. The number of Poisson generators will be "
            "set equal to the number of possible edges in the graph.")
    generators = required_edges + possible_edges
    random.shuffle(generators)
    return generators


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
@click.option('--duration', '-d', type=int, default=1000000, help='Simulation duration in steps.\n')
@click.option('--iterations', '-i', type=int, default=5, help='Number '
                                                              'of iterations of initial traffic generation.\n')
@click.option('--part-generators', '-g', type=float, default=0.1, help='This part of non-extreme edges will act '
                                                                       'as flow generators. Extreme edges act as '
                                                                       'flow generators by default.\n')
@click.option('--file', '-f', type=str, default='./configs/learning-configs/learning_parameters.json',
              help='path to simulation parameters config file (.sumocfg).\n')
@click.option('--init-delay', '-n', type=int, default=10, help='Delay between vehicle departures during map '
                                                               'initialization by traffic.\n')
@click.option('--part-of-the-path', '-p', type=float, default=0.3, help='The total part of the path that the '
                                                                        'initializing traffic must travel for generation '
                                                                        'to begin using Poisson flows.\n')
@click.option('--check-time', '-t', type=int, default=100, help='Check time (in simulation ticks) whether '
                                                                'the traffic has traveled half of the total path in'
                                                                'meters.\n')
@click.option('--part-off-traffic-lights', '-t', default=0.0, type=float, help='This part of traffic lights '
                                                                               'will be turned off.\n')
@click.option('--cpu-scale', '-c', type=int, default=2, help="Process pool size during the operation of the Dijkstra "
                                                             "algorithm and the path reconstruction algorithm for each departure "
                                                             "node will be [number of cpu's] * [--cpu-scale]').\n")
@click.option('--lower-limit-of-intensity', '-l', type=float, default=0.0001,
              help='Lower bound of traffic intensity for each generator.\n')
@click.option('--upper-limit-of-intensity', '-u', type=float, default=0.05,
              help='Upper bound on traffic intensity for each generator.\n')
@click.argument('sumo_config', nargs=1, type=str)
def main(duration: int,
         iterations: int,
         part_generators: float,
         file: str,
         init_delay: int,
         part_of_the_path: float,
         check_time: int,
         part_off_traffic_lights: float,
         cpu_scale: int,
         upper_limit_of_intensity: float,
         lower_limit_of_intensity: float,
         sumo_config: str) -> None:
    """
    This program generates a config with parameters for simulation.
    """
    net_config = extract_net_config(sumo_config)
    net = Net(net_config, [])
    poisson_generators = generate_poisson_generators(part_generators, net)
    intensities = generate_intensities(len(poisson_generators), lower_limit_of_intensity, upper_limit_of_intensity)
    turned_off_traffic_lights = make_list_of_turned_off_traffic_lights(part_off_traffic_lights, net)
    data = {"DURATION": duration,
            "INIT_DELAY": init_delay,
            "ITERATIONS": iterations,
            "PART_OF_THE_PATH": part_of_the_path,
            "CHECK_TIME": check_time,
            "CPU_SCALE": cpu_scale,
            "intensities": intensities,
            "poisson_generators_edges": poisson_generators,
            "turned_off_traffic_lights": turned_off_traffic_lights}
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
