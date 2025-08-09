# テキスト分類プロジェクト テンプレート

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

このプロジェクトは、Hugging Face Transformers、DVC、MLflowを使用したテキスト分類のためのテンプレートです。

**注意:** このプロジェクトは、すぐにモデルの学習や評価を実行できる完全なものではなく、あくまでパイプラインの骨格（スケルトン）を提供するものです。`train.py`や`evaluate.py`に必要な処理を実装することで、パイプラインを完成させる必要があります。

## 目的

*   再現可能な機械学習パイプラインの構築例を示す。
*   `dvc`によるデータと実験のバージョン管理。
*   `mlflow`による実験パラメータと結果の追跡。
*   `uv`によるクリーンなPython環境管理。

## 使い方

### 1. 環境構築

まず、Pythonの仮想環境を作成し、必要なライブラリをインストールします。

```bash
# 仮想環境を作成
make create_environment

# 仮想環境を有効化
# (Windowsの場合)
# .\.venv\Scripts\activate
# (Unix/macOSの場合)
source ./.venv/bin/activate

# 依存関係をインストール
make requirements
```

### 2. データパイプラインの実行

このプロジェクトでは、`dvc`を使ってデータ処理パイプラインを管理します。以下のコマンドで、データセットのダウンロードと前処理を実行できます。

```bash
# DVCパイプラインを実行（データのダウンロードと前処理）
dvc repro
```

`dvc.yaml`を見ると、このコマンドは以下のステージを実行します。
1.  `download_dataset`: `kunishou/J-ResearchCorpus` データセットをHugging Face Hubからダウンロードします。
2.  `prepare_dataset`: ダウンロードしたデータを訓練用、検証用、テスト用に分割します。

### 3. モデルの学習と評価 (実装が必要です)

`dvc.yaml`には`train`と`evaluate`ステージも定義されていますが、これらはダミーのコマンド(`echo`)が設定されているだけです。

パイプラインを完成させるには、以下のファイルを編集する必要があります。

*   `text_classification_example/train.py`: モデルの学習処理を実装します。
*   `text_classification_example/evaluate.py`: 学習済みモデルの評価処理を実装します。
*   `dvc.yaml`: `train`と`evaluate`ステージの`cmd`を、実装したスクリプトを実行する実際のコマンドに置き換えます。

### 4. 実験結果の確認

MLflow UIを起動することで、実験のパラメータや結果をブラウザで確認できます。

```bash
make mlflow_ui
```
ブラウザで `http://0.0.0.0:5000` を開いてください。

## プロジェクト構成

```
├── LICENSE
├── Makefile           <- `make requirements` や `make test` などの便利なコマンド
├── README.md          <- このREADMEファイル
├── data
│   ├── external       <- サードパーティのソースからのデータ
│   ├── interim        <- 変換された中間データ
│   └── raw            <- オリジナルの、変更不可能なデータダンプ
│
├── dvc.yaml           <- DVCパイプラインの定義ファイル
├── models             <- 学習済みモデルやモデルの予測結果
├── notebooks          <- Jupyterノートブック
├── params.yaml        <- DVCが追跡するパラメータ
├── pyproject.toml     <- プロジェクトのメタデータと依存関係
├── reports            <- 生成された分析レポート
│
└── text_classification_example   <- プロジェクトのソースコード
    │
    ├── __init__.py
    ├── download_dataset.py  <- データセットをダウンロードするスクリプト
    ├── prepare_dataset.py   <- データを前処理・分割するスクリプト
    ├── train.py             <- **(要実装)** モデル学習スクリプト
    ├── evaluate.py          <- **(要実装)** モデル評価スクリプト
    └── utils                <- 補助的なユーティリティコード
```
