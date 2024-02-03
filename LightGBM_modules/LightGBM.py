import numpy as np
import pandas as pd
import tensorflow as tf
import random
import lightgbm as lgb
import matplotlib.pyplot as plt
from General_modules.dataset import Dataset
from IPython.display import display
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import os

class LightGBM:

    @staticmethod
    def create_dataset(dataset,look_back):
        
        data_X=np.zeros((len(dataset)-look_back+1,3))
        j=0
        
        for i in range(look_back-1,len(dataset)):
            
            data_pre=dataset[i-look_back+1:i+1,0]
        
            data_pre_mean=np.mean(data_pre,axis=0)
            data_pre_min=np.min(data_pre,axis=0)
            data_pre_max=np.max(data_pre,axis=0)
            
            data_X[j,:]=np.array([data_pre_mean,data_pre_min,data_pre_max])
            j+=1
        
        return np.array(data_X).reshape(-1,3)  

    def __init__(self, dataset: Dataset):
        self.features = ['A1_mean','A1_min','A1_max', \
          'A2_mean','A2_min','A2_max', \
          'Cur_mean','Cur_min','Cur_max', \
          'Pre_mean','Pre_min','Pre_max', \
          'Temp_mean','Temp_min','Temp_max', \
          'Ther_mean','Ther_min','Ther_max', \
          'Vol_mean','Vol_min','Vol_max', \
          'Flow_mean','Flow_min','Flow_max']

        self.init_dataset(dataset)

        # fix random seed
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)
        os.environ["PYTHONHASHSEED"] = "0"

    def init_dataset(self, dataset):
        #window parameter
        look_back=10

        #Dimension of input data
        data_dim=8

        for i in range(0,data_dim):
            
            if i==0:
                #train data
                x_train_win=self.create_dataset(dataset.x_train_std[:,i].reshape(-1,1),look_back)
                #valid data
                x_valid_win=self.create_dataset(dataset.x_valid_std[:,i].reshape(-1,1),look_back)
                #test data
                x_test_win=self.create_dataset(dataset.x_test_std[:,i].reshape(-1,1),look_back) 
            else:
                #train data
                x_train_win=np.concatenate([x_train_win,self.create_dataset( \
                                        dataset.x_train_std[:,i].reshape(-1,1),look_back)],axis=-1)
                #valid data
                x_valid_win=np.concatenate([x_valid_win,self.create_dataset( \
                                        dataset.x_valid_std[:,i].reshape(-1,1),look_back)],axis=-1)
                #test data
                x_test_win=np.concatenate([x_test_win,self.create_dataset( \
                                        dataset.x_test_std[:,i].reshape(-1,1),look_back)],axis=-1)
                
        #change the shape of data
        self.train_x_win=x_train_win.reshape(-1,3*data_dim)
        self.train_y=dataset.y_train[look_back-1:]

        self.valid_x_win=x_valid_win.reshape(-1,3*data_dim)
        self.valid_y=dataset.y_valid[look_back-1:]

        self.test_x_win=x_test_win.reshape(-1,3*data_dim)
        self.test_y=dataset.y_test[look_back-1:]

        self.train_x=pd.DataFrame(self.train_x_win,columns=self.features)
        self.valid_x=pd.DataFrame(self.valid_x_win,columns=self.features)
        self.test_x=pd.DataFrame(self.test_x_win,columns=self.features)

    def display_X(self):
        display(self.train_x)

    #LightGBM train predict function
    def lgb_train_predict(self, params, test_flag=False):
        
        lgb_train=lgb.Dataset(self.train_x,self.train_y)
        lgb_valid=lgb.Dataset(self.valid_x,self.valid_y)
        lgb_test=lgb.Dataset(self.test_x,self.test_y)
        
        model_lgb=lgb.train(params=params,train_set=lgb_train, \
                            valid_sets=[lgb_train,lgb_valid])
                            #verbose_eval=0, \
                            #early_stopping_rounds=20)
        
        if test_flag:
            test_pred=np.zeros((len(self.test_y),1))
            test_pred[:,0]=np.where(model_lgb.predict(self.test_x)>=0.5,1,0)
            test_acc=accuracy_score(self.test_y.reshape(-1,1),test_pred)
            test_f1score=f1_score(self.test_y.reshape(-1,1),test_pred)
            test_recallscore=recall_score(self.test_y.reshape(-1,1),test_pred)
            test_cm=confusion_matrix(self.test_y.reshape(-1,1),test_pred)
            
            return test_acc,test_f1score,test_recallscore,test_cm,test_pred,model_lgb
        
        else:
            train_pred=np.zeros((len(self.train_y),1))
            train_pred[:,0]=np.where(model_lgb.predict(self.train_x)>=0.5,1,0)
            train_acc=accuracy_score(self.train_y.reshape(-1,1),train_pred)
            
            valid_pred=np.zeros((len(self.valid_y),1))
            valid_pred[:,0]=np.where(model_lgb.predict(self.valid_x)>=0.5,1,0)
            valid_acc=accuracy_score(self.valid_y.reshape(-1,1),valid_pred) 
            
            return train_acc,valid_acc
        
    def train(self):
        # Ramdom search for hyper parameter
        self.optimization_trial = 100

        self.results_val_acc = {}
        self.results_train_acc= {}

        for _ in range(self.optimization_trial):
            # =====the searching area of hyper parameter =====
            lr = 10 ** np.random.uniform(-3, 0)
            min_data_in_leaf=np.random.choice(range(1,21),1)[0]
            max_depth=np.random.choice(range(3,31),1)[0]
            num_leaves=np.random.choice(range(20,41),1)[0]
            # ================================================
            
            #Hyper parameter
            lgb_params={'objective':'binary',
                        'metric':'binary_error',
                        'force_row_wise':True,
                        'seed':0,
                        'learning_rate':lr,
                        'min_data_in_leaf':min_data_in_leaf,
                        'max_depth':max_depth,
                        'num_leaves':num_leaves
                    }

            train_acc,valid_acc=self.lgb_train_predict(params=lgb_params,test_flag=False)
            print('optimization'+str(len(self.results_val_acc)+1))
            print("train acc:" + str(train_acc) + "valid acc:" + str(valid_acc) + " | lr:" + str(lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + \
                ",max_depth:" + str(max_depth) + ",num_leaves:" + str(num_leaves))
            key = " lr:" + str(lr) + ", min_data_in_leaf:" + str(min_data_in_leaf) + ", max_depth:" + str(max_depth) + ",num_leaves:" + str(num_leaves)
            self.results_val_acc[key] = valid_acc
            self.results_train_acc[key] = train_acc
    
    def hyperparams_optimization_results(self):
        if not hasattr(self, 'results_val_acc') or not hasattr(self, 'results_train_acc') or not hasattr(self, 'optimization_trial'):
            self.train()

        print("=========== Hyper-Parameter Optimization Result ===========")
        i = 0
        for key, val_acc in sorted(self.results_val_acc.items(), key=lambda x:x[1], reverse=True):
            
            print("Best-" + str(i+1) + "(val acc:" + str(val_acc) + ")| " + key)
            i += 1

            if i >= int(self.optimization_trial*0.05):
                break

    def test(self):
        #fine-tunned hyper paramter
        lgb_params={'objective':'binary',
                    'metric':'binary_error',
                    'force_row_wise':True,
                    'seed':0,
                    'learning_rate':0.0424127,
                    'min_data_in_leaf':15,
                    'max_depth':24,
                    'num_leaves':29
                }

        test_acc,test_f1score,test_recallscore,test_cm,test_pred,model_lgb=self.lgb_train_predict(params=lgb_params,test_flag=True)

        print('test_acc:' + str(test_acc))
        print('test_f1score:' + str(test_f1score))
        print('test_recallscore:' + str(test_recallscore))
        print('test_fn_rate:' + str(1 - test_recallscore))
        print('test_confusionMatrix')
        display(test_cm)

        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.test_y)),self.test_y,linestyle='none', marker='X', color='blue', markersize=5, label='Anomaly')
        plt.plot(range(len(test_pred)),test_pred,linestyle='none', marker='X', color='red', markersize=5, label='Predict',alpha=0.05)
        plt.title('Light GBM')
        plt.xlabel('index')
        plt.ylabel('label')
        plt.legend(loc='best')

        lgb.plot_importance(model_lgb,height=0.5,figsize=(4,8))

        return test_acc,test_f1score,test_recallscore,test_cm,test_pred,lgb_params,model_lgb