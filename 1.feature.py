import os 
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error

def eval_Features(df,feature_nan= {'nan': -1}):
    feature_tmp = []
    for index,feature in tqdm(enumerate(df['Features'].values)):
        feature_tmp.append(eval(feature,feature_nan))
    feature_tmp = pd.DataFrame(feature_tmp)
    feature_tmp.columns = ['feature_'+str(i) for i in range(feature_tmp.shape[1])]
    return feature_tmp

def get_dataset(path = './Molecule_prediction_20200312'):
    #读入数据
    df_train = pd.read_csv(f'{path}/train_0312.csv')
    df_test = pd.read_csv(f'{path}/test_noLabel_0312.csv')
    submit_examp = pd.read_csv(f'{path}/submit_examp_0312.csv')
    #处理训练集
    feature_tmp =  eval_Features(df_train,feature_nan= {'nan': -1})
    df_train = pd.concat([df_train,feature_tmp],axis=1)
    del df_train['Features']
    #处理测试集
    feature_tmp =  eval_Features(df_test,feature_nan= {'nan': -1})
    df_test = pd.concat([df_test,feature_tmp],axis=1)
    del df_test['Features']

    return df_train , df_test

df_train , df_test = get_dataset()

df_train.to_csv("df_train.csv" , index = None)
df_test.to_csv("df_test.csv" , index = None)

print('dataset creation finished!')

