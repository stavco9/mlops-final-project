from General_modules.dataset import Dataset
from LightGBM_modules.LightGBM import LightGBM
from Conv_AE_modules.Conv_AE_main import Conv_AE_Main
from sklearn.metrics import accuracy_score
from MLflow.mlflow import MLflow
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
     
    #########
    # Complete here the decision tree implementation
    #########

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABLightGBM", test_acc=test_acc,
                             test_f1score=test_f1score, test_recallscore=test_recallscore, test_cm=test_cm, test_pred=test_pred,
                             params=params, model=model, model_name="lightgbm-model" ,train_x=lightgbm.train_x, valid_x=lightgbm.valid_x)
    mlflow_result = mlflow_client.show_experiment_results(test_x=lightgbm.test_x, test_y=lightgbm.test_y,
                                        start_index=0, end_index=50)
    
if (args.model == 'convae'):
    convae = Conv_AE_Main(dataset)
    convae.train()    
    test_acc,test_recallscore = convae.test()

    #########
    # Complete here the decision tree implementation
    #########

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABConvAE", test_acc=test_acc,
                             test_recallscore=test_recallscore,
                             model=convae.model, model_name="convae-model",
                             train_x=convae.X_train_df, valid_x=convae.X_valid_seq)

    mlflow_result = mlflow_client.show_experiment_results(test_x=convae.X_test_seq,
                                        test_x_df=convae.X_test_df.iloc[:-(convae.N_STEPS) + 1],
                                        test_y=convae.Y_test[:-(convae.N_STEPS) + 1],
                                        start_index=0, end_index=50)
    
mlflow_acc = accuracy_score(mlflow_result[["actual_class"]], mlflow_result["predicted_class"])
print(f"Mlflow accuracy: {str(mlflow_acc)}")