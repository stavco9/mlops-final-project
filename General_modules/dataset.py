import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from IPython.display import display

class Dataset:

    def __init__(self, data_path):
        self.data_path = data_path
        self.value1_data = self.load_data()  

    def get_files(self):
        all_files = []
        for dirname, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                all_files.append(f'{dirname}/{filename}')
        all_files.sort()

        return all_files

    def load_data(self):
        all_files = self.get_files()

        warnings.filterwarnings('ignore')
        valve1_dat={file.split('/')[-1]:pd.read_csv(file,sep=';',index_col='datetime',parse_dates=True)
                for file in all_files if 'valve1' in file}
        valve1_data=pd.concat(list(valve1_dat.values()),axis=0).sort_index()

        return valve1_data
    
    def display_data(self):
        display(self.value1_data)

    def show_plt_data(self):
        #TS data plotting(Change point is not use)
        for column in self.value1_data.columns[:-1]:
            plt.figure(figsize=(13,8))
            plt.plot(self.value1_data[column].values)
            ax=plt.gca()
            
            plt.legend()
            plt.title(column)
            plt.show()

    def show_heatmap_data(self):
        #Correlation coefficient
        plt.figure(figsize=(8,8))
        data=self.value1_data.drop(columns='changepoint')
        sns.heatmap(data.corr(),annot=True,fmt='.2g')

    def show_smooth_data(self):
        if not hasattr(self, 'x_train_win') or not hasattr(self, 'x_test_win')  or not hasattr(self, 'x_valid_win'):
            self.window_data()

        x1=np.arange(0,12713)
        x2=np.arange(12713,16309)
        x3=np.arange(16309,18162)

        x_all_data=pd.concat([self.x_train,self.x_valid,self.x_test],ignore_index=False)

        features=['Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS']
        mat=np.concatenate([self.x_train_win,self.x_valid_win,self.x_test_win],axis=0)

        x_all_data_win=pd.DataFrame(mat,columns=features)

        for column in x_all_data.columns[:]:
            plt.figure(figsize=(13,8))
            plt.plot(x_all_data[column].values,color='b')
            plt.plot(x_all_data_win[column].values,color='r')
            ax=plt.gca()
            y_min,y_max=ax.get_ylim()
            
            y1=[y_min]*len(x1)
            y2=[y_max]*len(x1)
            plt.fill_between(x1,y1,y2,facecolor='g',alpha=0.3,label='train')
            
            y1=[y_min]*len(x2)
            y2=[y_max]*len(x2)
            plt.fill_between(x2,y1,y2,facecolor='b',alpha=0.3,label='valid')
            
            y1=[y_min]*len(x3)
            y2=[y_max]*len(x3)
            plt.fill_between(x3,y1,y2,facecolor='r',alpha=0.3,label='test')
            
            plt.legend()
            plt.title(column)
            plt.show()

    @staticmethod
    #*** split into free and anomaly ***
    def free_anomaly_split(X,Y):
        free=[]
        anomaly=[]
        
        for x,y in zip(X,Y):
            if y==0:
                free.append(x)
            elif y==1:
                anomaly.append(x)
        
        free=np.array(free)
        anomaly=np.array(anomaly)
            
        return free,anomaly
    
    @staticmethod
    def smooth_curve(x):
        #x=1 dimension array
        window_len = 11
        s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
        w = np.kaiser(window_len, 2)
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y[5:len(y)-5]

    def show_plt_free_anomaly(self):
        warnings.filterwarnings('ignore')

        if not hasattr(self, 'x_train') or not hasattr(self, 'x_test')  or not hasattr(self, 'x_valid'):
            self.split_data()
        
        x_train_free,x_train_anomaly=self.free_anomaly_split(self.x_train.values,self.y_train)
        x_valid_free,x_valid_anomaly=self.free_anomaly_split(self.x_valid.values,self.y_valid)
        x_test_free,x_test_anomaly=self.free_anomaly_split(self.x_test.values,self.y_test)

        #histgram of normal and anomaly
        features={1:'Accelerometer1RMS',2:'Accelerometer2RMS',3:'Current',4:'Pressure',5:'Temperature',6:'Thermocouple',7:'Voltage',8:'Volume Flow RateRMS'}
        bins=None

        for key,col in features.items():
            plt.figure(figsize=(16,6))
            sns.distplot(x_train_free[:,key-1],bins=bins,label='train_free')
            sns.distplot(x_train_anomaly[:,key-1],bins=bins,label='train_anomaly')
            sns.distplot(x_valid_free[:,key-1],bins=bins,label='valid_free')
            sns.distplot(x_valid_anomaly[:,key-1],bins=bins,label='valid_anomaly')
            sns.distplot(x_test_free[:,key-1],bins=bins,label='test_free')
            sns.distplot(x_test_anomaly[:,key-1],bins=bins,label='test_anomaly')   
            
            plt.title(col)
            plt.legend()
            plt.show()

    def split_data(self):
        #train_pre(valve1_data is dataframe)
        train_pre=self.value1_data

        #train_pre ⇒ train:valid_pre=7:3
        train_pre_size=len(train_pre)
        train_size=int(train_pre_size*0.7)
        train=train_pre[0:train_size]
        x_train_pre=train.drop('anomaly',axis=1)
        self.x_train=x_train_pre.drop('changepoint',axis=1)
        self.y_train=train['anomaly'].values

        #valid_pre ⇒ valid:test=2:1
        valid_pre_size=train_pre_size-train_size
        valid_size=int(valid_pre_size*0.66)
        valid=train_pre[train_size:train_size+valid_size]
        x_valid_pre=valid.drop('anomaly',axis=1)
        self.x_valid=x_valid_pre.drop('changepoint',axis=1)
        self.y_valid=valid['anomaly'].values

        test=train_pre[train_size+valid_size:]
        x_test_pre=test.drop('anomaly',axis=1)
        self.x_test=x_test_pre.drop('changepoint',axis=1)
        self.y_test=test['anomaly'].values

    def window_data(self):
        if not hasattr(self, 'x_train') or not hasattr(self, 'x_test')  or not hasattr(self, 'x_valid'):
            self.split_data()

        self.x_train_win=np.zeros_like(self.x_train.values)
        self.x_valid_win=np.zeros_like(self.x_valid.values)
        self.x_test_win=np.zeros_like(self.x_test.values)

        data_dim=8
        for i in range(0,data_dim):
            self.x_train_win[:,i]=self.smooth_curve(self.x_train.values[:,i].flatten())
            self.x_valid_win[:,i]=self.smooth_curve(self.x_valid.values[:,i].flatten())
            self.x_test_win[:,i]=self.smooth_curve(self.x_test.values[:,i].flatten())

    def standard_data(self):
        if not hasattr(self, 'x_train_win') or not hasattr(self, 'x_test_win')  or not hasattr(self, 'x_valid_win'):
            self.window_data()

        # Generate instance for standardization
        sc = StandardScaler()

        # Calculate the transform matrix and it is applied to valid and test data
        self.x_train_std = sc.fit_transform(self.x_train_win)
        self.x_valid_std = sc.transform(self.x_valid_win)
        self.x_test_std = sc.transform(self.x_test_win)

    def display_X(self):
        display(self.x_train)
        display(self.x_valid)
        display(self.x_test)