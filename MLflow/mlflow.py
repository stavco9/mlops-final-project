import platform
import numpy as np
import pandas as pd
from IPython.display import display

if platform.system() != 'Windows':
    import MLflow.mlflow_mac as mlflow
    from MLflow.mlflow_mac.models import infer_signature
else:
    import MLflow.mlflow_windows as mlflow
    from MLflow.mlflow_windows.models import infer_signature

class MLflow:
    #
    # Please set the following environment variables before executing:
    # export MLFLOW_TRACKING_USERNAME=your_email@gmail.com
    # export MLFLOW_TRACKING_PASSWORD=********
    #
    def __init__(self, mlflow_url):
        mlflow.set_tracking_uri(mlflow_url)
        self.mlflow_client = mlflow.MlflowClient(
            tracking_uri=mlflow_url,
        )

    def run_experiment(self, experiment_name, model, model_name, train_x, valid_x, \
                        log_metrics_feats=None, log_metrics_vals=None, \
                        params=None):
        
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id=current_experiment['experiment_id']

        # Start an MLflow run
        run = self.mlflow_client.create_run(experiment_id)
        #print(run.info.artifact_uri)
        with mlflow.start_run(run_id=run.info.run_id):
            # Log the hyperparameters
            if params:
                mlflow.log_params(params)
            
            if log_metrics_feats is not None and log_metrics_vals is not None:
                for idx, feat in enumerate(log_metrics_feats.to_list()):
                    for metric_name, metric in log_metrics_vals.items():
                        metric_key = f"{feat}_{metric_name}"
                        metric_value = metric[idx]
                        mlflow.log_metric(metric_key, metric_value)
            
            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")
            
            signature = infer_signature(train_x, model.predict(valid_x))

            print('\n')
            # Log the model
            self.model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{run.info.artifact_uri}/skab_model",
                signature=signature,
                input_example=train_x,
                registered_model_name=model_name
            )

        print('\n')
        print("Experiment results are available in the following link: ")
        print(f"{mlflow.get_tracking_uri().rstrip('/')}/ml/experiments/{experiment_id}/runs/{run.info.run_id}")
            
    def show_experiment_results(self, test_x, test_y, test_x_df=None, start_index=0, end_index=-1):
        print('\n')
        loaded_model = mlflow.pyfunc.load_model(self.model_info.model_uri)

        predictions = np.where(loaded_model.predict(test_x) > 0.5, 1, 0)

        if (type(test_x) != pd.DataFrame and test_x_df is not None):
            # Convert X_test validation feature data to a Pandas DataFrame
            result = test_x_df
            predictions = np.array([1 if np.mean(items) > 0.5 else 0 for items in predictions])
        else:
            result = test_x

        if end_index == -1:
            end_index = len(result)

        # Add the actual classes to the DataFrame
        result["actual_class"] = test_y

        # Add the model predictions to the DataFrame
        result["predicted_class"] = predictions
        
        print('\n')
        print("Result comprasion between predicted anomaly of the download model from MLFlow vs the actual ones")
        display(result[start_index:end_index])

        return result
