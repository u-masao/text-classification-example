from pathlib import Path

from loguru import logger
import mlflow
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
    utilizes_long_term_data: bool = Field(
        ..., description="数ヶ月から数年以上にわたる時系列データや観測データを使用しているか？"
    )
    discusses_ethical_aspects: bool = Field(
        ..., description="研究の実施や結果における倫理的な問題点や配慮が議論されているか？"
    )
    deals_with_future_prediction: bool = Field(
        ..., description="将来のトレンド、イベント、状態などを予測することを目的としているか？"
    )
    establishes_proposes_theory: bool = Field(
        ..., description="新たな学術理論や概念モデルが構築または提唱されているか？"
    )
    primarily_a_literature_review: bool = Field(
        ..., description="特定のテーマに関する既存の論文を網羅的に調査・分析しているか？"
    )
    utilizes_machine_learning: bool = Field(
        ...,
        description="機械学習（深層学習、強化学習などを含む）のアルゴリズムが主要な手法として用いられているか？",
    )
    involves_software_system_development: bool = Field(
        ..., description="新しいソフトウェア、システム、ツールが開発または提案されているか？"
    )
    related_to_environmental_issues: bool = Field(
        ..., description="地球温暖化、汚染、生態系保護など、環境に関する問題を取り扱っているか？"
    )
    related_to_medical_health_field: bool = Field(
        ...,
        description="疾患の診断、治療、公衆衛生、医療技術など、医療や健康に関するテーマを扱っているか？",
    )
    includes_policy_recommendations: bool = Field(
        ..., description="研究結果に基づいて、特定の政策や制度に関する提言が行われているか？"
    )
    is_a_comparative_study: bool = Field(
        ..., description="複数の手法、モデル、システム、データセットなどを比較分析しているか？"
    )
    uses_interdisciplinary_approach: bool = Field(
        ..., description="複数の異なる学術分野の知識や手法を組み合わせて研究が行われているか？"
    )
    includes_qualitative_research: bool = Field(
        ...,
        description="インタビュー、フォーカスグループ、民族誌的手法など、非数値データを中心とした調査が行われているか？",
    )


class ModelResponseStructure(BaseModel):
    summary_text: str
    criterias: CriteriaClasses


@object_cache
def generate_response(text: str, model_name: str = "gemma3:4b", num_ctx: int = 1024 * 16) -> str:
    prompt = f"次の文章を要約して評価して。\n\n{text}"
    return ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": 0, "num_ctx": num_ctx},
        format=ModelResponseStructure.model_json_schema(),
    ).response


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

        # log input
        print(input_df)

        # 分類
        tqdm.pandas()
        input_df["generated"] = input_df["text"].progress_apply(generate_response)

        # split data
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
