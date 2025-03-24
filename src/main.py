import click
from facade.facade import Facade


@click.command()
@click.option('--sumo-config', '-s', type=str, help='path to SUMO config file (.sumocfg)')
@click.option('--simulation-parameters', '-p', type=str, help='path to config with parameters of '
                                                              'simulation (.json)')
def main(sumo_config: str, simulation_parameters: str) -> None:
    """
    This program maximizes the transport capacity of road network.
    """
    facade = Facade(sumo_config, simulation_parameters)
    facade.execute()


if __name__ == "__main__":
    main()
