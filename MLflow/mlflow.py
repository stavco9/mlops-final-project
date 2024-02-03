import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

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

    def run_experiment(self, experiment_name, test_acc, test_f1score, \
                       test_recallscore, test_cm, test_pred, \
                        params, model, model_name, train_x):
        
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id=current_experiment['experiment_id']
        print(f"Experiment id is {experiment_id}")

        # Start an MLflow run
        run = self.mlflow_client.create_run(experiment_id)
        print(run.info.artifact_uri)
        with mlflow.start_run(run_id=run.info.run_id):
            # Log the hyperparameters
            mlflow.log_params(params)
            
            # Log the loss metric
            mlflow.log_metric("accuracy", test_acc)
            
            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")
            
            # Infer the model signature
            signature = infer_signature(train_x, model.predict(train_x))

            # Log the model
            self.model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{run.info.artifact_uri}/skab_model",
                signature=signature,
                input_example=train_x,
                registered_model_name=model_name,
                await_registration_for=60
            )
            
    def show_experiment_results(self, test_x, test_y, features, start_index=0, end_index=-1):
        loaded_model = mlflow.pyfunc.load_model(self.model_info.model_uri)

        predictions = np.where(loaded_model.predict(test_x) > 0.5, 1, 0)

        # Convert X_test validation feature data to a Pandas DataFrame
        result = pd.DataFrame(test_x, columns=features)

        if end_index == -1:
            end_index = len(result)

        # Add the actual classes to the DataFrame
        result["actual_class"] = test_y

        # Add the model predictions to the DataFrame
        result["predicted_class"] = predictions

        print(result[start_index:end_index])
