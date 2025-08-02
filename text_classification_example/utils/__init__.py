import argparse
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow import ActiveRun


# ファイル存在チェックを行う関数
def existing_filepath(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"ファイルが見つかりません: '{path}'")
    return path


@contextmanager
def mlflow_start_run(
    experiment_name: str,
    run_name: str = "develop",
    tracking_uri: str = "sqlite:///mlruns.db",
    interval: int = 10,
    samples: int = 1,
) -> ActiveRun:
    """
    mlflow を初期化し終了時に mlflow.end_run() を呼び出す
    """

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    try:
        active_run = mlflow.start_run(run_name=run_name)
        mlflow.set_system_metrics_sampling_interval(interval)
        mlflow.set_system_metrics_samples_before_logging(samples)
        mlflow.enable_system_metrics_logging()
        yield active_run
    finally:
        mlflow.end_run()
