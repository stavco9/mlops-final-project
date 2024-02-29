from General_modules.dataset import Dataset
from LightGBM_modules.LightGBM import LightGBM
from Conv_AE_modules.Conv_AE_main import Conv_AE_Main
from sklearn.metrics import accuracy_score
from MLflow.mlflow import MLflow
import FraeAI.FreeAI as F
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    choices=['lightgbm', 'convae'],
                    help='Please select your desired training model',
                    required=True)

args = parser.parse_args(sys.argv[1:])

dataset = Dataset('./SKAB')
print("Displaying dataset after pre-processing")
print('\n')
dataset.display_data()
dataset.split_data()
dataset.display_X()
dataset.standard_data()

# Please set the following environment variables before executing:
# export MLFLOW_TRACKING_USERNAME=your_email@gmail.com
# export MLFLOW_TRACKING_PASSWORD=********
mlflow_client = MLflow('https://dbc-c3108cf4-06da.cloud.databricks.com/')

if (args.model == 'lightgbm'):
    print("Starting LightGBM model")
    print("\n")
    lightgbm = LightGBM(dataset)
    lightgbm.train()
    lightgbm.hyperparams_optimization_results()
    test_acc,test_f1score,test_recallscore,test_cm,test_pred,params,model = lightgbm.test()

    print("\n")
    print("Showing FreaAI metrics")
    df_fraeai_acc = F.main_func('LightGBM', lightgbm.test_x, test_pred, lightgbm.test_y, number_of_features=1, metric='accuracy')
    df_fraeai_acc_to_mlflow = df_fraeai_acc.loc[:, ['f2','f2 imp','size','min val1','max val1','min val2','max val2']].reset_index()
    df_features = df_fraeai_acc.loc[:, 'features']

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABLightGBM",
                             log_metrics_feats=df_features, log_metrics_vals=df_fraeai_acc_to_mlflow,
                             params=params, model=model, model_name="lightgbm-model",
                             train_x=lightgbm.train_x, valid_x=lightgbm.valid_x)
    mlflow_client.show_experiment_results(test_x=lightgbm.test_x, test_y=lightgbm.test_y,
                                        start_index=0, end_index=50)
    
if (args.model == 'convae'):
    print("Starting ConvAE model")
    print("\n")
    convae = Conv_AE_Main(dataset)
    convae.train()    
    test_acc,test_recallscore = convae.test()

    convae.X_test_df = convae.X_test_df.reset_index(drop=True).iloc[:-(convae.N_STEPS) + 1]
    convae.pred_test = convae.pred_test[:-(convae.N_STEPS) + 1]
    convae.Y_test = convae.Y_test[:-(convae.N_STEPS) + 1]

    print("\n")
    print("Showing FreaAI metrics")
    df_fraeai_acc = F.main_func('ConvAE', convae.X_test_df, convae.pred_test, convae.Y_test, number_of_features=1, metric='accuracy')
    df_fraeai_acc_to_mlflow = df_fraeai_acc.loc[:, ['f2','f2 imp','size','min val1','max val1','min val2','max val2']].reset_index()
    df_features = df_fraeai_acc.loc[:, 'features']

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABConvAE",
                             log_metrics_feats=df_features, log_metrics_vals=df_fraeai_acc_to_mlflow,
                             model=convae.model, model_name="convae-model",
                             train_x=convae.X_train_df, valid_x=convae.X_valid_seq)

    mlflow_client.show_experiment_results(test_x=convae.X_test_seq,
                                        test_x_df=convae.X_test_df,
                                        test_y=convae.Y_test,
                                        start_index=0, end_index=50)