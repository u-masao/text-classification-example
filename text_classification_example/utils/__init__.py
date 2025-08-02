import mlflow
from mlflow import ActiveRun


def mlflow_start_run(
    experiment_name: str,
    run_name: str = "develop",
    tracking_uri: str = "sqlite:///mlruns.db",
    interval: int = 10,
    samples: int = 1,
) -> ActiveRun:
    mlflow.set_system_metrics_sampling_interval(interval)
    mlflow.set_system_metrics_samples_before_logging(samples)
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    active_run = mlflow.start_run(run_name=run_name)
    mlflow.enable_system_metrics_logging()
    return active_run


def mlflow_end_run():
    mlflow.end_run()
