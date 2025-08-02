from pathlib import Path

from datasets import load_dataset
from loguru import logger
import mlflow
from tap import Tap

from .utils import mlflow_end_run, mlflow_start_run


class Args(Tap):
    dataset_name: str
    output_filepath: Path


class Experiment:
    def __init__(self, args: Args):
        self.args = args

    def run(self) -> None:
        dataset = load_dataset(self.dataset_name)
        logger.info(dataset)


def main(args: Args) -> None:
    mlflow_start_run(expoeriment_name="download_dataset")
    mlflow.log_params(args.as_dict())
    logger.info(args.as_dict())

    experiment = Experiment(args)
    experiment.run()
    mlflow_end_run()


if __name__ == "__main__":
    cli_args = Args().parse_args()
    main(cli_args)
