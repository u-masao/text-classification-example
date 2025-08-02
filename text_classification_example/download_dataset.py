from pathlib import Path

from datasets import load_dataset
from loguru import logger
import mlflow
from tap import Tap

from .utils import mlflow_end_run, mlflow_start_run


class Args(Tap):
    dataset_name: str
    output_filepath: Path
    mlflow_run_name: str | None = None
    split_name: str = "train"


class Experiment:
    def __init__(self, args: Args):
        self.args = args

    def run(self) -> None:
        # download dataset
        dataset = load_dataset(self.args.dataset_name, split=self.args.split_name)

        # log summary
        logger.info(dataset)

        # make output dir
        self.args.output_filepath.parent.mkdir(parents=True, exist_ok=True)

        # output
        dataset.to_parquet(self.args.output_filepath)

        # logging
        mlflow.log_metrics(
            {
                "num_rows": dataset.num_rows,
            }
        )
        mlflow.log_text(str(dataset), "dataset_summary.txt")


def main(args: Args) -> None:
    mlflow_start_run(experiment_name="download_dataset")
    mlflow.log_params(args.as_dict())
    logger.info(args.as_dict())

    experiment = Experiment(args)
    experiment.run()
    mlflow_end_run()


if __name__ == "__main__":
    cli_args = Args().parse_args()
    main(cli_args)
