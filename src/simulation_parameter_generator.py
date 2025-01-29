from email.policy import default

import click
import random
import traci
import json

def generate_intensities(n_generators: int, duration: int) -> list:
    intensities = [random.uniform(1 / duration, 0.5) for i in range(n_generators)]
    return intensities


def find_clear_edges() -> list:
    edges = traci.edge.getIDList()
    clear_edges = []
    for edge in edges:
        if ":" not in edge:
            clear_edges.append(edge)
    return clear_edges


def generate_poisson_generators(sumo_config: str, n_generators: int) -> list:
    sumo_cmd = ["sumo", "-c", sumo_config]
    traci.start(sumo_cmd)
    traci.simulationStep()
    clear_edges = find_clear_edges()
    traci.close()
    if n_generators > len(clear_edges):
        """
        рассмотреть что будет при 4 более полоске в одном направлении
        """
        print("Number of Poisson generators is greater than number of edges. The number of Poisson generators will be "
              "set equal to the number of edges in the graph.")
        n_generators = len(clear_edges)
    random.shuffle(clear_edges)
    return clear_edges[:n_generators]


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
    data = {"duration": duration, "initialization_delay": init_delay, "iterations": iterations,
            "intensities": intensities, "poisson_generators_edges": poisson_generators}
    with open(file, 'w') as json_file:
        json.dump(data, json_file)
if __name__ == "__main__":
    main()
