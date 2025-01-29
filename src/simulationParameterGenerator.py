import click, random, traci, json


def generate_intensities(n_generators: int, duration: int) -> list:
    intensities = [random.uniform(1 / duration, 0.5) for i in range(n_generators)]
    return intensities


def find_clear_edges() -> list:
    nodes = traci.junction.getIDList()
    clear_nodes = []
    for node in nodes:
        if ":" not in node:
            clear_nodes.append(node)
    return clear_nodes


def generate_poisson_generators(sumo_config: str, n_generators: int) -> list:
    sumo_cmd = ["sumo", "-c", sumo_config]
    traci.start(sumo_cmd)
    traci.simulationStep()
    clear_edges = find_clear_edges()
    traci.close()
    random.shuffle(clear_edges)
    return clear_edges[:n_generators]


@click.command()
@click.option('--duration', '-d', type=int, default=3600, help='simulation duration in steps')
@click.option('--iterations', '-i', type=int, default=1, help='number '
                                                              'of iterations of initial traffic generation')
@click.option('--generators', '-g', type=int, default=1, help='number of poisson generators')
@click.option('--file', '-f', type=str, default='./configs/simulation-parameters/simulation_parameters.json',
              help='path to SUMO config file (.sumocfg)')
@click.argument('sumo_config', nargs=1, type=str)
def main(duration: int, iterations: int, generators: int, file: str, sumo_config: str) -> None:
    """
    This program generates a config with parameters for simulation.
    """
    intensities = generate_intensities(generators, duration)
    poisson_generators = generate_poisson_generators(sumo_config, generators)
    data = {"duration": duration, "iterations": iterations, "intensities": intensities,
            "poisson_generators": poisson_generators}
    with open(file, 'w') as json_file:
        json.dump(data, json_file)
if __name__ == "__main__":
    main()
