from General_modules.dataset import Dataset
from LightGBM_modules.LightGBM import LightGBM
from MLflow.mlflow import MLflow

dataset = Dataset('./SKAB')
dataset.display_data()
#dataset.show_plt_data()
#dataset.show_heatmap_data()
dataset.split_data()
dataset.display_X()
#dataset.show_plt_free_anomaly()
dataset.standard_data()
#dataset.show_smooth_data()

lightgbm = LightGBM(dataset)
lightgbm.train()
lightgbm.hyperparams_optimization_results()
test_acc,test_f1score,test_recallscore,test_cm,test_pred,params,model = lightgbm.test()

# Please set the following environment variables before executing:
# export MLFLOW_TRACKING_USERNAME=your_email@gmail.com
# export MLFLOW_TRACKING_PASSWORD=********
mlflow_client = MLflow('https://dbc-c3108cf4-06da.cloud.databricks.com/')
mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABLightGBM", test_acc=test_acc,
                             test_f1score=test_f1score, test_recallscore=test_recallscore, test_cm=test_cm, test_pred=test_pred,
                             params=params, model=model, model_name="lightgbm-model" ,train_x=lightgbm.train_x)
mlflow_client.show_experiment_results(test_x=lightgbm.test_x, test_y=lightgbm.test_y, features=lightgbm.features,
                                      start_index=0, end_index=50)