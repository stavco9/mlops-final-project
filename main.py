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
     
    df2 = F.main_func_max(lightgbm.test_x, test_pred, lightgbm.test_y, number_of_features=1, metric='accuracy')
    #result = pd.concat([df1, df2], ignore_index=True).sort_values(by='acc', ascending=False)
    result = df2.sort_values(by=['feature','acc'], ascending=False)
    result1 = result.loc[result.groupby('feature')['acc'].idxmin()]

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABLightGBM", test_acc=test_acc,
                             test_f1score=test_f1score, test_recallscore=test_recallscore, test_cm=test_cm, test_pred=test_pred,
                             params=params, model=model, model_name="lightgbm-model" ,train_x=lightgbm.train_x, valid_x=lightgbm.valid_x)
    mlflow_result = mlflow_client.show_experiment_results(test_x=lightgbm.test_x, test_y=lightgbm.test_y,
                                        start_index=0, end_index=50)
    
if (args.model == 'convae'):
    convae = Conv_AE_Main(dataset)
    convae.train()    
    test_acc,test_recallscore = convae.test()

    print("Current sets")
    convae.X_test_df = convae.X_test_df.reset_index(drop=True).iloc[:-(convae.N_STEPS) + 1]
    convae.Y_train = convae.Y_train[:-(convae.N_STEPS) + 1]
    convae.Y_test = convae.Y_test[:-(convae.N_STEPS) + 1]

    df2 = F.main_func_max(convae.X_test_df, convae.Y_train, convae.Y_test, number_of_features=1, metric='accuracy')
    #result = pd.concat([df1, df2], ignore_index=True).sort_values(by='acc', ascending=False)
    result = df2.sort_values(by=['feature','acc'], ascending=False)
    result1 = result.loc[result.groupby('feature')['acc'].idxmin()]

    mlflow_client.run_experiment(experiment_name="/Users/stavco9@gmail.com/SKABConvAE", test_acc=test_acc,
                             test_recallscore=test_recallscore,
                             model=convae.model, model_name="convae-model",
                             train_x=convae.X_train_df, valid_x=convae.X_valid_seq)

    mlflow_result = mlflow_client.show_experiment_results(test_x=convae.X_test_seq,
                                        test_x_df=convae.X_test_df,
                                        test_y=convae.Y_test,
                                        start_index=0, end_index=50)
    
mlflow_acc = accuracy_score(mlflow_result[["actual_class"]], mlflow_result["predicted_class"])
print(f"Mlflow accuracy: {str(mlflow_acc)}")