from General_modules.dataset import Dataset
from Conv_AE_modules.Conv_AE import Conv_AE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Conv_AE_Main:
    def __init__(self, dataset: Dataset):
        self.model = Conv_AE()
        self.list_of_df = dataset.list_of_df
        self.features = list(dataset.list_of_df[0])
        self.features.remove('anomaly')
        self.features.remove('changepoint')
        print(self.features)

    @staticmethod
    # Generated training sequences for use in the model.
    def create_sequences(values, time_steps):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)
    
    @staticmethod
    def perf_measure(y_actual, y_hat, y_len: int):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return((TP / y_len), (FP / y_len), (TN / y_len), (FN / y_len))

    def train_and_test(self):
        # hyperparameters selection
        N_STEPS = 60
        Q = 0.999 # quantile for upper control limit (UCL) selection
        self.predicted_outlier, self.predicted_cp = [], []
        self.X_train_concat = []
        for df in self.list_of_df:
            X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
            
            # scaler init and fitting
            StSc = StandardScaler()
            StSc.fit(X_train)
            
            # convert into input/output
            X = self.create_sequences(StSc.transform(X_train), N_STEPS)
            
            # model fitting
            self.model.fit(X)
            
            # results predicting
            residuals = pd.Series(np.sum(np.mean(np.abs(X - self.model.predict(X)), axis=1), axis=1))
            UCL = residuals.quantile(Q) * 4/3
            
            # results predicting
            X = self.create_sequences(StSc.transform(df.drop(['anomaly','changepoint'], axis=1)), N_STEPS)
            cnn_residuals = pd.Series(np.sum(np.mean(np.abs(X - self.model.predict(X)), axis=1), axis=1))
            
            # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
            anomalous_data = cnn_residuals > UCL
            anomalous_data_indices = []
            for data_idx in range(N_STEPS - 1, len(X) - N_STEPS + 1):
                if np.all(anomalous_data[data_idx - N_STEPS + 1 : data_idx]):
                    anomalous_data_indices.append(data_idx)
            
            prediction = pd.Series(data=0, index=df.index)
            prediction.iloc[anomalous_data_indices] = 1
            
            # predicted outliers saving
            self.predicted_outlier.append(prediction)
            
            # predicted CPs saving
            prediction_cp = abs(prediction.diff())
            prediction_cp[0] = prediction[0]
            self.predicted_cp.append(prediction_cp)
            self.X_train_concat.append(X_train)

    def test_results(self):
        if not hasattr(self, 'predicted_cp') or not hasattr(self, 'predicted_outlier'):
            self.train_and_test()

        # true outlier indices selection
        true_outlier = [df.anomaly for df in self.list_of_df]
        outlier_len = len(true_outlier)
        X_train = np.concatenate([x.to_numpy() for x in self.X_train_concat])
        X_test = np.concatenate([df.drop(['anomaly', 'changepoint'], axis=1).to_numpy() for df in self.list_of_df])
        Y_train=np.concatenate([x.to_numpy() for x in self.predicted_outlier])
        Y_test=np.concatenate([x.to_numpy() for x in true_outlier])
        tp_rate, fp_rate, tn_rate, fn_rate = self.perf_measure(Y_test, Y_train, outlier_len)
        test_recallscore = tp_rate/(fn_rate + tp_rate)
        test_accuracy = (tp_rate+tn_rate)/(tp_rate+fp_rate+fn_rate+tn_rate)

        print('test_fn_count:' + str(fn_rate))
        print('test_fn_rate:' + str(fn_rate/(fn_rate + tp_rate)))
        print('test_recall_rate:' + str(test_recallscore))
        print('accuracy_rate:' + str(test_accuracy))

        return test_accuracy,test_recallscore,X_train,X_test,Y_train,Y_test