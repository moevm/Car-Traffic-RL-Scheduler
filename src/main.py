import click
from facade.facade import TrafficScheduler


@click.command()
@click.option('--sumo-config', '-s', type=str, help='path to SUMO config file (.sumocfg)')
@click.option('--simulation-parameters', '-p', type=str, help='path to config with parameters of '
                                                              'simulation (.json)')
def main(sumo_config: str, simulation_parameters: str) -> None:
    """
    This program maximizes the transport capacity of road network.
    """
    scheduler = TrafficScheduler(sumo_config, simulation_parameters)
    scheduler.learn()
    #scheduler.predict('trained_model')
    #scheduler.default_tls()

if __name__ == "__main__":
    main()
