"""
MLflow utility functions for logging experiments.
Provides a wrapper to conditionally log metrics, params, and artifacts based on ENABLE_MLFLOW flag.
"""
import os
import mlflow


def init_mlflow_run(tracking_uri, experiment_name, run_name=None):
    """
    Initialize MLflow run if enabled.
    
    Args:
        tracking_uri (str): MLflow tracking URI (local path or remote server)
        experiment_name (str): Name of the MLflow experiment
        run_name (str, optional): Name for this run
    
    Returns:
        bool: Whether MLflow is active
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize MLflow: {e}")
        return False


def log_params(params):
    """Log hyperparameters if MLflow"""
    if mlflow.active_run():
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"Warning: Failed to log params to MLflow: {e}")


def log_metrics(metrics, step=None):
    """Log metrics if MLflow"""
    if mlflow.active_run():
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}")


def log_artifact(local_path, artifact_path=None):
    """Log artifact (file) if MLflow"""
    if mlflow.active_run():
        try:
            if os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            elif os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path)
        except Exception as e:
            print(f"Warning: Failed to log artifact to MLflow: {e}")


def end_mlflow_run():
    """End MLflow run"""
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception as e:
        print(f"Warning: Failed to end MLflow run: {e}")
