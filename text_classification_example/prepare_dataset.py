from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
from tap import Tap

from .utils import existing_filepath, mlflow_start_run


class Args(Tap):
    """動作オプションの管理"""

    # output definition
    output_train_filepath: Path
    output_valid_filepath: Path
    output_test_filepath: Path

    # experiment managemento definition
    mlflow_run_name: str | None = None

    def configure(self):
        # input definition
        self.add_argument("--input_filepath", type=existing_filepath)


class Experiment:
    """実験の定義と処理"""

    def __init__(self, args: Args):
        self.args = args

    def run(self) -> None:
        # load dataset
        input_df = pd.read_parquet(self.args.input_filepath)

        print(input_df)

        result_df = {
            "train": input_df.iloc[:100],
            "valid": input_df.iloc[100:200],
            "test": input_df.iloc[200:300],
        }

        # output dataset
        result_df["train"].to_parquet(self.args.output_train_filepath)
        result_df["valid"].to_parquet(self.args.output_valid_filepath)
        result_df["test"].to_parquet(self.args.output_test_filepath)


def main(args: Args) -> None:
    """メイン処理: 実験記録の定義と実験の呼び出し"""

    # log cli args
    logger.info(args.as_dict())

    try:
        # init mlflow
        with mlflow_start_run(experiment_name="prepare_dataset", run_name=args.mlflow_run_name):
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
