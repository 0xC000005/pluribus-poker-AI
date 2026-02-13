import click

from poker_ai.ai.runner import train
from poker_ai.clustering.runner import cluster
from poker_ai.terminal.runner import run_terminal_app
from poker_ai.deep_cfr.trainer import train_deep_cfr
from poker_ai.deep_cfr.fast_trainer import train_fast_deep_cfr


@click.group()
def cli():
    """CLI for poker_ai: train, play, and cluster.

    Use 'train-deep-cfr' for full-deck Deep CFR training (recommended).
    Use 'train' for legacy short-deck tabular MCCFR training.
    Use 'play' to play against a trained agent.
    """
    pass


cli.add_command(train_fast_deep_cfr, name="train-fast-deep-cfr")
cli.add_command(train_deep_cfr, name="train-deep-cfr")
cli.add_command(train, name="train")
cli.add_command(cluster, name="cluster")
cli.add_command(run_terminal_app, name="play")
