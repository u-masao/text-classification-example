import json
from pathlib import Path
import time

from loguru import logger
import mlflow
import numpy as np
from object_cache import object_cache
import ollama
import pandas as pd
from pydantic import BaseModel, Field
from tap import Tap
from tqdm import tqdm

from .utils import existing_filepath, mlflow_start_run


class CriteriaClasses(BaseModel):
    # 論文の分類基準をTrue/Falseで表すブール値の属性
    contains_experimentation: bool = Field(
        ..., description="論文内で実験（実測、制御された試行など）が行われているか？"
    )
    presents_novel_data: bool = Field(
        ..., description="論文内で新たに収集または生成されたデータが公開されているか？"
    )
    includes_numerical_analysis: bool = Field(
        ..., description="数値計算、統計分析、シミュレーションなどが主要な要素であるか？"
    )
    indicates_specific_industry_application: bool = Field(
        ..., description="研究成果が特定の産業分野への応用例や可能性を具体的に示しているか？"
    )
    aims_to_improve_existing_methods: bool = Field(
        ..., description="すでに存在する研究手法やアルゴリズムの性能向上や効率化を目指しているか？"
    )
    provides_open_source_code: bool = Field(
        ..., description="論文に関連するコードが公開されており、利用可能であるか？"
    )
    focuses_on_human_subjects: bool = Field(
        ...,
        description="アンケート調査、インタビュー、行動観察など、人間が研究対象に含まれているか？",
    )


class ModelResponseStructure(BaseModel):
    criterias: CriteriaClasses
    summary_text: str = Field(
        ...,
        description="日本語で要約",
    )


@object_cache
def generate_response_ollama(
    text: str,
    model_name: str,
    num_ctx: int = 128,
    max_retry: int = 5,
    temperature_delta: float = 0.01,
):
    return {}
    user_prompt = f"""
        評価規定に従い次のテキストを評価して。
        必ずすべての項目を評価して。

        ```
        {text}
        ```

        繰り返すが絶対に日本語で要約すること。
        テキスト本文が英語であっても日本語で要約すること。
        """
    try_count: int = 0
    temperature: float = 0.0
    while try_count < max_retry:
        try:
            response = ollama.generate(
                model=model_name,
                prompt=user_prompt,
                options={"temperature": temperature, "num_ctx": num_ctx},
                format=ModelResponseStructure.model_json_schema(),
            ).response
            logger.info(f"response: {response}")
            response_dict = json.loads(response)
            return response_dict
        except ollama.ResponseError as e:
            logger.info(f"Ollama API エラー: {e}")
        except json.JSONDecodeError:
            logger.info("JSON Decode エラー")
        except Exception as e:
            logger.info(f"予期しないエラー: {e}")
            raise e

        logger.info(f"リトライ中: {try_count + 1} / {max_retry}")
        temperature += temperature_delta
        try_count += 1
        time.sleep(1)
    logger.warning(f"{max_retry} 回リクエストしましたがすべて失敗でした")
    return {}


class Args(Tap):
    """動作オプションの管理"""

    # output definition
    output_train_filepath: Path
    output_valid_filepath: Path
    output_test_filepath: Path

    # split
    split_random_seed: int = 1234
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    test_ratio: float = 0.2

    # experiment managemento definition
    mlflow_run_name: str | None = None

    # ollama_model_name
    # ollama_model_name: str = "gemma3:4b"
    ollama_model_name: str = "gemma3:1b"
    ollama_num_ctx: int = 1024 * 31

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

        input_df = input_df.head(100)

        # log input
        print(input_df)

        # init result
        result_df = input_df.copy()

        # 分類
        tqdm.pandas(desc="処理中")
        generated_sr = input_df["text"].progress_apply(
            lambda x: generate_response_ollama(
                x,
                model_name=self.args.ollama_model_name,
                num_ctx=self.args.ollama_num_ctx,
            )
        )

        # デバッグ用にテンポラリファイルを出力
        result_df.to_parquet("data/interim/temp.parquet")

        # 分類結果を横に結合
        result_df = pd.concat([result_df, pd.json_normalize(generated_sr)], axis=1)

        # データ分割
        train_df, valid_df, test_df = self._split_dataframe(
            result_df,
            train_ratio=self.args.dataset.train_ratio,
            valid_ratio=self.args.dataset.valid_ratio,
            test_ratio=self.args.dataset.test_ratio,
            random_seed=self.args.dataset.split_random_seed,
        )

        # output dataset
        train_df.to_parquet(self.args.output_train_filepath)
        valid_df.to_parquet(self.args.output_valid_filepath)
        test_df.to_parquet(self.args.output_test_filepath)

        # ログ出力
        mlflow.log_metrics(
            {
                "length.train": len(train_df),
                "length.valid": len(valid_df),
                "length.test": len(test_df),
            }
        )

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 1234,
    ):
        total_ratio = train_ratio + valid_ratio + test_ratio
        train_ratio = train_ratio / total_ratio
        valid_ratio = valid_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
        split_points = [int(len(df) * train_ratio), int(len(df) * (train_ratio + valid_ratio))]
        return np.split(df.sample(frac=1, random_state=random_seed), split_points)


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
