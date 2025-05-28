import json

import numpy as np
import click


@click.command()
@click.option('--file', '-f', type=str, help='Path to file with runs info.')
def main(file: str):
    """
    This program computes the mean and standard deviation of runs for the same configuration of agent, network and
    simulation parameters.
    """
    with open(file, 'r') as json_file:
        data = json.load(json_file)
        for key, runs_list in data.items():
            print(key)
            print(f"\tmean = {np.mean(runs_list)}\n\tstd = {np.std(runs_list)}")


if __name__ == "__main__":
    main()
