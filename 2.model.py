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


def train_model_lgb(df_train , df_test , params , feature_col):
    
    X = df_train.copy()
    y = df_train['Label'].copy()
    test = df_test.copy()
    
    fi = []
    cv_score = []
    test_pred = np.zeros((test.shape[0],))
    train_pred = np.zeros((X.shape[0],))
    skf = KFold(n_splits=10, random_state=2019, shuffle=True)
    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index)
        train_x, test_x, train_y, test_y = X.iloc[train_index],X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_set = lgb.Dataset(train_x[feature_col], train_y)
        test_set = lgb.Dataset(test_x[feature_col],test_y)
        lgb_model = lgb.train(params,
                          train_set,
                          valid_sets=[train_set,test_set],
                          early_stopping_rounds=500, 
                          num_boost_round=10000 ,
                          verbose_eval=1000)

        y_val = lgb_model.predict(test_x[feature_col])
        train_pred[test_index] = y_val
        print( np.sqrt(mean_squared_error( test_y , y_val )) ) 

        cv_score.append( np.sqrt(mean_squared_error( test_y , y_val)) )
    
        print(cv_score[index])
    
        test_pred += lgb_model.predict(test[feature_col]) / 10
    
    return train_pred , test_pred

def submodel_lgb(train , test , subf , params , newtrain ,newtest, interval = 300 , extend_feature = set()):
    for i in tqdm(range(int(len(subf)/interval)+1)):
        feature = subf[i*interval:(i+1)*interval]
        print('----------------',i,'---------------------------')
        sup_trainp , sup_testp = train_model_lgb(train , test , params , list(set(feature) | extend_feature) )
        newtrain[str(i)] = sup_trainp
        newtest[str(i)] = sup_testp
    return newtrain , newtest

def train_model_xgb(df_train , df_test , params , feature_col):
    
    X = df_train.copy()
    y = df_train['Label'].copy()
    test = df_test.copy()
    
    fi = []
    cv_score = []
    test_pred = np.zeros((test.shape[0],))
    train_pred = np.zeros((X.shape[0],))
    skf = KFold(n_splits=10, random_state=2019, shuffle=True)
    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index)
        train_x, test_x, train_y, test_y = X.iloc[train_index],X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_set = xgb.DMatrix(train_x[feature_col] , train_y)
        test_set = xgb.DMatrix(test_x[feature_col] , test_y)
        
        xgb_model = xgb.train(params,train_set,num_boost_round=10000)

        y_val = xgb_model.predict(test_set)
        train_pred[test_index] = y_val
        print( np.sqrt(mean_squared_error( test_y , y_val )) ) 

        cv_score.append( np.sqrt(mean_squared_error( test_y , y_val)) )
    
        print(cv_score[index])
    
        test_pred += xgb_model.predict(xgb.DMatrix(test[feature_col])) / 10
    
    return train_pred , test_pred

def submodel_xgb(train , test , subf , params , newtrain ,newtest, interval = 300 , extend_feature = set()):
    for i in range(int(len(subf)/interval)+1):
        feature = subf[i*interval:(i+1)*interval]
        print('----------------',i,'---------------------------')
        sup_trainp , sup_testp = train_model_xgb(train , test , params , list(set(feature) | extend_feature))
        newtrain[str(i)] = sup_trainp
        newtest[str(i)] = sup_testp
    return newtrain , newtest

def feature_eng(df):
    df['Molecular_weight_log2'] = np.log2(df['Molecular weight'])
    df['AlogP_log2'] = np.log2(df['AlogP'])
    df['RO5_violations'].fillna(-1,inplace=True)
    return df

def get_submodel(xgb_params , lgb_params):


    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')

    subf = df_train.columns[6:]

    base_feature = ['ID', 'Molecule_max_phase', 'Molecular weight', 'RO5_violations','AlogP']
    print('getting the submodel of xgb...')
    for xgb_feature_num in [200,300,400]:
        print(f'creating the submodel of xgb{xgb_feature_num}')
        xgb_train , xgb_test = submodel_xgb(df_train , df_test ,  subf ,xgb_params,df_train[base_feature+['Label']].copy(),df_test[base_feature].copy() , interval=xgb_feature_num)
        xgb_train.to_csv(f'xgb_train_{xgb_feature_num}.csv' , index = None)
        xgb_test.to_csv(f'xgb_test_{xgb_feature_num}.csv' , index = None)

    df_train = feature_eng(df_train)
    df_test = feature_eng(df_test)

    base_feature = base_feature + ['Molecular_weight_log2' , 'AlogP_log2']
    print('getting the submodel of lgb...')

    for lgb_feature_num in [300,400,500]:
        print(f'creating the submodel of lgb{lgb_feature_num}')
        lgb_train , lgb_test = submodel_lgb(df_train , df_test ,  subf ,lgb_params,df_train[base_feature+['Label']].copy(),df_test[base_feature].copy(),interval = lgb_feature_num)
        lgb_train.to_csv(f'lgb_train_{lgb_feature_num}.csv' , index = None)
        lgb_test.to_csv(f'lgb_test_{lgb_feature_num}.csv' , index = None)
    extend_feature = { 'Molecule_max_phase', 'Molecular weight', 'RO5_violations','AlogP',}
    print('getting the submodel of fixed-feature-xgb...')
    
    for xgb_feature_num in [304,404]:
        print(f'creating the submodel of xgb{xgb_feature_num}')
        xgb_train , xgb_test = submodel_xgb(df_train , df_test ,  subf ,xgb_params,df_train[base_feature+['Label']].copy(),df_test[base_feature].copy() , interval=xgb_feature_num - 4 , extend_feature=extend_feature)
        xgb_train.to_csv(f'xgb_train_{xgb_feature_num}.csv' , index = None)
        xgb_test.to_csv(f'xgb_test_{xgb_feature_num}.csv' , index = None)
    print("submodel finished!!!")



xgb_params = {
            'booster':'gbtree',
            'objective':'reg:linear',
            'eta':0.02,
            'max_depth':6,
            'subsample':1.0,
            'min_child_weight':5,
            'colsample_bytree':0.2,
            'gamma':0.2,            
            'lambda':3,
            'nthread': -1,
            'early_stopping_rounds':1,
            'verbose_eval':1,
            'silient':1,
            'metric': 'rmse'
    }
lgb_params = {'objective': 'regression',
             'learning_rate': 0.02 ,
             'num_leaves': 40 , 
             'max_depth': 8 ,
             'feature_fraction': 0.8, 
             'bagging_fraction' : 0.8,
             'num_threads':-1,
             'metric': 'rmse',}

get_submodel(xgb_params , lgb_params)