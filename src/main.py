import click
from facade.facade import TrafficScheduler


@click.command()
@click.option('--sumo-config', '-s', type=str, help='Path to SUMO config file (.sumocfg).')
@click.option('--simulation-parameters', '-p', type=str, help='Path to config with parameters of '
                                                              'simulation (.json).')
@click.option('--mode', '-m', type=str, help='Mode of usage. Should be `train`, `train+`, '
                                             '`evaluation_trained_agent`, `evaluation_default_agent`.')
def main(sumo_config: str, simulation_parameters: str, mode: str) -> None:
    """
    This program maximizes the transport capacity of road network.
    """
    scheduler = TrafficScheduler(sumo_config, simulation_parameters)
    match mode:
        case 'train':
            scheduler.learn()
        case 'train+':
            scheduler.additional_learning()
        case 'evaluation_trained_agent':
            scheduler.trained_model_evaluation()
        case 'evaluation_default_agent':
            scheduler.default_agent_evaluation()
        case _:
            raise f'No such option {mode}'

if __name__ == "__main__":
    main()
