import click
from pygments.lexer import default

from facade.facade import TrafficScheduler


@click.command()
@click.option('--sumo-config', '-s', type=str, help='Path to SUMO config file (.sumocfg).')
@click.option('--simulation-parameters', '-p', type=str, help='Path to config with parameters of '
                                                              'simulation (.json).')
@click.option('--mode', '-m', type=str, help='Mode of usage. Should be `train`, `train+`, '
                                             '`evaluation_trained_agent`, `evaluation_default_agent`.')
@click.option('--vec-normalized', '-n', default='vec_normalized.pkl', type=str, help='Path to normalized '
                                                                                     'environment.')
@click.option('--model-weights', '-w', default='trained_model', type=str, help='Path to model weights.')
@click.option('--new-checkpoint', '-c', default=False, type=bool,
              help='Make new checkpoint or take from configs/checkpoints')
def main(sumo_config: str, simulation_parameters: str, mode: str, vec_normalized: str, model_weights,
         new_checkpoint: str) -> None:
    """
    This program maximizes the transport capacity of road network.
    """
    scheduler = TrafficScheduler(sumo_config, simulation_parameters, new_checkpoint)
    match mode:
        case 'train':
            scheduler.learn()
        case 'train+':
            scheduler.additional_learning()
        case 'evaluation_trained_agent':
            scheduler.trained_model_evaluation(vec_normalized, model_weights)
        case 'evaluation_default_agent':
            scheduler.default_agent_evaluation()
        case _:
            raise f'No such option {mode}'


if __name__ == "__main__":
    main()
