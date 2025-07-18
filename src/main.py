import click

from facade.facade import TrafficScheduler


@click.command()
@click.option('--sumo-config', '-s', type=str, help='Path to SUMO config file (.sumocfg).')
@click.option('--simulation-parameters', '-p', type=str, help='Path to config with parameters of '
                                                              'simulation (.json).')
@click.option('--mode', '-m', type=str, help='Mode of usage. Should be `train`,'
                                             '`evaluation_trained_agent`, `evaluation_default_agent`.')
@click.option('--vec-normalized', '-n', default='vec_normalized.pkl', type=str, help='Path to normalized '
                                                                                     'environment.')
@click.option('--model-weights', '-w', default='trained_model', type=str, help='Path to model weights.')
@click.option('--new-checkpoint', '-c', default=False, type=bool,
              help='Make new checkpoint or take from configs/checkpoints')
@click.option('--enable-gui', '-g', default=False, type=bool, help='Show gui.')
@click.option('--duration', '-d', default=6000, type=int, help='Duration of evaluation.')
@click.option('--cycle-time', '-t', default=90, type=int, help='Cycle time for traffic lights that did not '
                                                               'fit into any group.')
def main(sumo_config: str, simulation_parameters: str, mode: str, vec_normalized: str, model_weights,
         new_checkpoint: str, enable_gui: bool, duration: int, cycle_time: int) -> None:
    """
    This program is designed for training an agent using the Recurrent PPO algorithm and evaluating the trained agent.
    The program also allows evaluating the metrics of standard SUMO agents.
    """
    scheduler = TrafficScheduler(sumo_config, simulation_parameters, new_checkpoint, enable_gui, cycle_time)
    match mode:
        case 'train':
            scheduler.learn()
        case 'evaluation_trained_agent':
            scheduler.trained_model_evaluation(vec_normalized, model_weights, duration)
        case 'evaluation_default_agent':
            scheduler.default_agent_evaluation(duration)
        case _:
            raise f'No such option {mode}'


if __name__ == "__main__":
    main()
