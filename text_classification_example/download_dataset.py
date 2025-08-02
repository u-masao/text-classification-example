from pathlib import Path

from datasets import load_dataset
from loguru import logger
import mlflow
from tap import Tap

from .utils import mlflow_start_run


class Args(Tap):
    """動作オプションの管理"""

    # dataset definition
    dataset_name: str
    split_name: str = "train"

    # output definition
    output_filepath: Path

    # experiment managemento definition
    mlflow_run_name: str | None = None


class Experiment:
    """実験の定義と処理"""

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
    """メイン処理: 実験記録の定義と実験の呼び出し"""

    # log cli args
    logger.info(args.as_dict())

    try:
        # init mlflow
        with mlflow_start_run(experiment_name="download_dataset", run_name=args.mlflow_run_name):
            # 引数を記録
            mlflow.log_params(args.as_dict())

            # 実験の初期化と実行
            experiment = Experiment(args)
            experiment.run()

            # 処理完了のログ
            logger.info("==== 処理完了 ====")

    except Exception as e:
        logger.error("==== 異常修了 ====")
        logger.error(f"エラー内容: {e}")
        raise e


if __name__ == "__main__":
    cli_args = Args().parse_args()
    main(cli_args)
