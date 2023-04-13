import click
from rich.logging import RichHandler
import logging

from hopai import utils
from hopai.dataset import MyDataset
from hopai.model import MyModel


@click.group()
def cli():
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@cli.command()
@click.option("--epochs", default=10)
@click.option("--batch-size", default=64)
@click.option("--lr", default=1e-3)
@click.option("--log-interval", default=100)
def train(epochs, batch_size, lr, log_interval):
    train_dataset = MyDataset(train=True)
    test_dataset = MyDataset(train=False)
    model = MyModel()
    utils.train(model, train_dataset, test_dataset, epochs, batch_size, lr, log_interval)


if __name__ == "__main__":
    cli()
