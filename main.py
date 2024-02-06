from General_modules.dataset import Dataset
from LightGBM_modules.LightGBM import LightGBM
from Conv_AE_modules.Conv_AE_main import Conv_AE_Main
from MLflow.mlflow import MLflow
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['lightgbm', 'convae'],
                    help='Please select your desired training model',
                    required=True)

args = parser.parse_args(sys.argv[1:])

dataset = Dataset('./SKAB')
dataset.display_data()
dataset.split_data()
dataset.display_X()
dataset.standard_data()

# Please set the following environment variables before executing:
# export MLFLOW_TRACKING_USERNAME=your_email@gmail.com
# export MLFLOW_TRACKING_PASSWORD=********
mlflow_client = MLflow('https://dbc-c3108cf4-06da.cloud.databricks.com/')

if (args.model == 'lightgbm'):
    lightgbm = LightGBM(dataset)
    lightgbm.train()
    lightgbm.hyperparams_optimization_results()
    test_acc,test_f1score,test_recallscore,test_cm,test_pred,params,model = lightgbm.test()
     
    #######  

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABLightGBM", test_acc=test_acc,
                             test_f1score=test_f1score, test_recallscore=test_recallscore, test_cm=test_cm, test_pred=test_pred,
                             params=params, model=model, model_name="lightgbm-model" ,train_x=lightgbm.train_x, valid_x=lightgbm.valid_x)
    mlflow_client.show_experiment_results(test_x=lightgbm.test_x, test_y=lightgbm.test_y, features=lightgbm.features,
                                        start_index=0, end_index=50)
    
if (args.model == 'convae'):
    convae = Conv_AE_Main(dataset)
    #convae.train_and_test()

    convae.train()    
    test_acc,test_recallscore = convae.test()

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABConvAE", test_acc=test_acc,
                             test_recallscore=test_recallscore,
                             model=convae.model, model_name="convae-model", features=convae.features,
                             train_x=convae.X_train_df, valid_x=convae.X_valid_seq)
    mlflow_client.show_experiment_results(test_x=convae.X_test_seq, test_y=convae.Y_test, features=convae.features,
                                        start_index=0, end_index=50)