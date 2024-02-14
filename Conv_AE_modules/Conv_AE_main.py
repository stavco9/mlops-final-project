from General_modules.dataset import Dataset
from Conv_AE_modules.Conv_AE import Conv_AE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Conv_AE_Main:
    def __init__(self, dataset: Dataset):
        self.model = Conv_AE()
        self.list_of_df = [dataset.value1_data]
        self.features = list(dataset.list_of_df[0])
        self.features.remove('anomaly')
        self.features.remove('changepoint')
        self.N_STEPS = 60
        self.Q = 0.999 # quantile for upper control limit (UCL) selection
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

    def train(self):
        predicted_outlier, predicted_cp = [], []

        for df in self.list_of_df:
            train_pre_size = df.shape[0]
            self.train_size=int(train_pre_size*0.7)
            valid_pre_size=train_pre_size-self.train_size
            self.valid_size=int(valid_pre_size*0.66)

            #X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
            self.X_train_df = df[:self.train_size].drop(['anomaly', 'changepoint'], axis=1)
            self.X_valid_df = df[self.train_size:self.train_size+self.valid_size].drop(['anomaly', 'changepoint'], axis=1)
            self.X_test_df = df[self.train_size+self.valid_size:].drop(['anomaly', 'changepoint'], axis=1)

            # scaler init and fitting
            StSc = StandardScaler()
            StSc.fit(df.drop(['anomaly', 'changepoint'], axis=1))

            # convert into input/output
            self.X_train_seq = self.create_sequences(StSc.transform(self.X_train_df), self.N_STEPS)
            self.X_test_seq = self.create_sequences(StSc.transform(self.X_test_df), self.N_STEPS)
            self.X_valid_seq = self.create_sequences(StSc.transform(self.X_valid_df), self.N_STEPS)


            # model fitting
            self.model.fit(self.X_train_seq)

            # results predicting
            residuals = pd.Series(np.sum(np.mean(np.abs(self.X_valid_seq - self.model.predict(self.X_valid_seq)), axis=1), axis=1))
            UCL = residuals.quantile(self.Q) * 4/8   # 4/3

            # results predicting
            #X = create_sequences(StSc.transform(df.drop(['anomaly','changepoint'], axis=1)), N_STEPS)
            cnn_residuals = pd.Series(np.sum(np.mean(np.abs(self.X_valid_seq - self.model.predict(self.X_valid_seq)), axis=1), axis=1))

            # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
            anomalous_data = cnn_residuals > UCL
            anomalous_data_indices = []
            for data_idx in range(self.N_STEPS - 1, len(self.X_valid_seq) - self.N_STEPS + 1):
                if np.all(anomalous_data[data_idx - self.N_STEPS + 1 : data_idx]):
                    anomalous_data_indices.append(data_idx)

            prediction = pd.Series(data=0, index=self.X_valid_df.index)
            prediction.iloc[anomalous_data_indices] = 1

            # predicted outliers saving
            predicted_outlier.append(prediction)

            # predicted CPs saving
            prediction_cp = abs(prediction.diff())
            prediction_cp[0] = prediction[0]
            predicted_cp.append(prediction_cp)

    def test(self):
        N_STEPS = 60
        Q = 0.999 # quantile for upper control limit (UCL) selection

        # results predicting
        predicted_outlier = []
        residuals = pd.Series(np.sum(np.mean(np.abs(self.X_test_seq - self.model.predict(self.X_test_seq)), axis=1), axis=1))
        UCL = residuals.quantile(Q) * 4/8   # 4/3

        # results predicting
        #X = create_sequences(StSc.transform(df.drop(['anomaly','changepoint'], axis=1)), N_STEPS)
        cnn_residuals = pd.Series(np.sum(np.mean(np.abs(self.X_test_seq - self.model.predict(self.X_test_seq)), axis=1), axis=1))

        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data = cnn_residuals > UCL
        anomalous_data_indices = []
        for data_idx in range(self.N_STEPS - 1, len(self.X_test_seq) - self.N_STEPS + 1):
            if np.all(anomalous_data[data_idx - self.N_STEPS + 1 : data_idx]):
                anomalous_data_indices.append(data_idx)

        prediction = pd.Series(data=0, index=self.X_test_df.index)
        prediction.iloc[anomalous_data_indices] = 1

        # predicted outliers saving
        predicted_outlier.append(prediction)

        # true outlier indices selection
        true_outlier = [df[self.train_size+self.valid_size:].anomaly for df in self.list_of_df]

        predicted_outlier[0].plot(figsize=(12,3), label='predictions', marker='o', markersize=5)
        true_outlier[0].plot(marker='o', markersize=2)
        #plt.legend();

        outlier_len = len(true_outlier)
        true_outlier_np=np.concatenate([x.to_numpy() for x in true_outlier])
        predicted_outlier_np=np.concatenate([x.to_numpy() for x in predicted_outlier])
        tp_rate, fp_rate, tn_rate, fn_rate = self.perf_measure(true_outlier_np, predicted_outlier_np, outlier_len)

        test_recallscore = tp_rate/(fn_rate + tp_rate)
        test_accuracy = (tp_rate+tn_rate)/(tp_rate+fp_rate+fn_rate+tn_rate)

        self.Y_test = true_outlier_np
        self.pred_test = predicted_outlier_np

        print('test_fn_count:' + str(fn_rate))
        print('test_fn_rate:' + str(fn_rate/(fn_rate + tp_rate)))
        print('test_recall_rate:' + str(test_recallscore))
        print('accuracy_rate:' + str(test_accuracy))

        return test_accuracy,test_recallscore