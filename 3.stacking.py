import os 
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

from tqdm import tqdm
from scipy.stats import skew,kurtosis
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


def get_submodel_result_xgb(train_path , test_path , xgb_paras):

    train_xgb = pd.read_csv(train_path)
    test_xgb = pd.read_csv(test_path)

    tr_xgb = np.zeros((train_xgb.shape[0],))
    te_xgb = np.zeros((test_xgb.shape[0],))

    feature_col = [i for i in train_xgb.columns if i not in ['ID','Label']]
    X = train_xgb.copy()
    y = train_xgb['Label'].copy()
    test = test_xgb.copy()

    cv_score = []
    skf = KFold(n_splits=10, random_state=2019, shuffle=True)

    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index)
        train_x, test_x, train_y, test_y = X.iloc[train_index],X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_set = xgb.DMatrix(train_x[feature_col], train_y)
        test_set = xgb.DMatrix(test_x[feature_col], test_y)
        xgb_model = xgb.train(xgb_paras,
                          train_set,
                          evals=[(train_set,'train'),(test_set,'test')],
                          early_stopping_rounds=100, 
                          num_boost_round=10000,
                          verbose_eval=1000)


        y_val = xgb_model.predict(xgb.DMatrix(test_x[feature_col]))
    
        print( np.sqrt(mean_squared_error( test_y , y_val)) ) 

        cv_score.append( np.sqrt(mean_squared_error( test_y , y_val)) )
        tr_xgb[test_index,] = y_val
    
        print(cv_score[index])
    
        te_xgb += xgb_model.predict(xgb.DMatrix(test[feature_col])) / 10
    print(np.mean(cv_score))
    return tr_xgb , te_xgb

def get_submodel_result_lgb(train_path , test_path , lgb_paras):
    train_lgb = pd.read_csv(train_path)
    test_lgb = pd.read_csv(test_path)

    tr_lgb = np.zeros((train_lgb.shape[0],))
    te_lgb = np.zeros((test_lgb.shape[0],))

    feature_col = [i for i in train_lgb.columns if i not in ['ID','Label']]
    X = train_lgb.copy()
    y = train_lgb['Label'].copy()
    test = test_lgb.copy()

    cv_score = []
    skf = KFold(n_splits=10, random_state=2019, shuffle=True)

    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index)
        train_x, test_x, train_y, test_y = X.iloc[train_index],X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_set = lgb.Dataset(train_x[feature_col], train_y)
        test_set = lgb.Dataset(test_x[feature_col], test_y)
        lgb_model = lgb.train(lgb_paras,
                          train_set,
                          valid_sets=[train_set,test_set],
                          verbose_eval=100)

        y_val = lgb_model.predict(test_x[feature_col])
    
        print( np.sqrt(mean_squared_error( test_y , y_val)) ) 

        cv_score.append( np.sqrt(mean_squared_error( test_y , y_val)) )
    
    
        tr_lgb[test_index,] = y_val
    
        print(cv_score[index])
    
        te_lgb += lgb_model.predict(test[feature_col]) / 10
    print(np.mean(cv_score))
    return tr_lgb , te_lgb


def stacking(xgb_paras , lgb_paras , path = './Molecule_prediction_20200312'):
    res_train = {}
    res_test = {}
    submit_examp = pd.read_csv(f'{path}/submit_examp_0312.csv')
    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')
    print("getting the result of xgb-submodel...")
    for xgb_feature_num in [200,300,400]:
        res_train[f'xgb_{xgb_feature_num}'] , res_test[f'xgb_{xgb_feature_num}'] = get_submodel_result_xgb(f'xgb_train_{xgb_feature_num}.csv' , f'xgb_test_{xgb_feature_num}.csv' , xgb_paras)
    print("getting the result of lgb-submodel...")
    for lgb_feature_num in [300,400,500]:
        res_train[f'lgb_{lgb_feature_num}'] , res_test[f'lgb_{lgb_feature_num}']= get_submodel_result_lgb(f'lgb_train_{lgb_feature_num}.csv' , f'lgb_test_{lgb_feature_num}.csv' , lgb_paras)
    print("getting the result of fixed-feature-xgb-submodel...")
    for xgb_feature_num in [304,404]:
        res_train[f'xgb_{xgb_feature_num}'] , res_test[f'xgb_{xgb_feature_num}'] = get_submodel_result_xgb(f'xgb_train_{xgb_feature_num}.csv' , f'xgb_test_{xgb_feature_num}.csv' , xgb_paras)
    print("Stacking...")
    res_train = pd.DataFrame(res_train)
    res_test = pd.DataFrame(res_test)
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(res_train,df_train['Label'])
    test_pred = regr.predict(res_test)
    print("Producing submit-file...")
    submit_examp['Label'] = test_pred
    submit_examp.to_csv('stacking_submit.csv',index=False)
    print('finished!!!')
    return 

xgb_paras = {'objective': 'reg:squarederror',
 'tree_method': 'gpu_hist',
 'eval_metric': 'rmse',
 'learning_rate': 0.02,
 'alpha': 0.30328974897294075,
 'colsample_bytree': 0.5068082755866445,
 'lambda': 72.2173472522586,
 'max_depth': 9,
 'min_child_weight': 5,
 'subsample': 0.8170133539039669}
lgb_paras = {'objective': 'regression',
 'metric': 'rmse',
 'learning_rate': 0.02,
 'num_threads': -1,
 'early_stopping_rounds': 100,
 'num_boost_round': 10000,
 'bagging_fraction': 0.9978192061670864,
 'bagging_freq': 1,
 'feature_fraction': 0.5234178718477926,
 'max_depth': 7,
 'min_child_weight': 1,
 'num_leaves': 41,
 'reg_alpha': 0.1415592188002883,
 'reg_lambda': 2.2724007900790895}

stacking(xgb_paras , lgb_paras)