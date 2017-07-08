
# coding: utf-8

# # Stacking Experiment

# In[1]:

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import xgboost as xgb
from sklearn import model_selection
from datetime import datetime
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from argparse import ArgumentParser

X_PATH = '../../ML2017_data/bump/df_x.csv'
Y_PATH = '../../ML2017_data/bump/df_y.csv'
X_VAL_PATH = '../../ML2017_data/bump/df_x_val.csv'
Y_VAL_PATH = '../../ML2017_data/bump/df_y_val.csv'
X_TEST = '../../ML2017_data/bump/test_x.csv'

train_x_path = 'yx_x'
train_y_path = 'yx_y'
val_x_path = 'yx_x_val'
val_y_path = 'yx_y_val'
test_x_path = 'yx_x_test'


MODEL_NUM = 3
VALI = 1 
rf_voter_score = []
svc_voter_score = []
xgb_voter_score = []

for VOTER in range(1,20):
    VOTER = 0.015 * VOTER
    print('voter portion:', VOTER)
    x = np.array(pickle.load(open(train_x_path, 'rb')))
    y = np.array(pickle.load(open(train_y_path, 'rb')))

    randomize = np.arange(len(x))
    x_test = pickle.load(open(test_x_path, 'rb'))
    all_x = np.concatenate((x, x_test))

    scaler = StandardScaler()
    scaler.fit(all_x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    pickle.dump(scaler, open('scaler', 'wb'))
    print('Finish normizatilon')

    label_encorder = LabelEncoder()
    y = label_encorder.fit_transform(y)
    pickle.dump(label_encorder, open('label_encoder', 'wb'))
    print('Finish label')

#     clf = RandomForestClassifier(n_estimators = 1500,
#                 bootstrap = True,
#                 n_jobs = -1, 
#                 min_samples_split=7,
#                 oob_score = True)
#     clf.fit(x, y)
#     score = clf.oob_score_
#     test_ans = clf.predict_proba(x_test)
#     ans = clf.predict(x_test)
#     #     pickle.dump(clf, open('./model/{:.4f}'.format(score), 'wb'))
#     train_l = np.where(test_ans > 0.65)
#     #print(test_ans.shape)
#     #print(x_test.shape)
#     #print('ans shape', ans.shape)
#     x_test = np.array(x_test)
#     self_train_x = x_test[train_l[0]]
#     self_train_y = ans[train_l[0]]
#     x_test = np.delete(x_test, train_l[0], axis = 0)
#     x = np.concatenate((x, self_train_x))
#     y = np.concatenate((y, self_train_y))

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    VOTER = int(len(x)*VOTER)
    x_model = x[VOTER:]
    y_model = y[VOTER:]
    x_voter = x[:VOTER]
    y_voter = y[:VOTER]

    rf_clf = RandomForestClassifier(n_estimators = 1500,
                n_jobs = -1, 
                min_samples_split=10,
                oob_score = True)
    rf_clf.fit(x_model, y_model)
    print('Finish random forest.')


    for model_i in range(MODEL_NUM):
        np.random.seed(model_i)
        randomize = np.arange(len(x_model))
        np.random.shuffle(randomize)
        x_model = x_model[randomize]
        y_model = y_model[randomize]
        TRAIN = int(VALI * len(x_model))
        x_t = x_model[:TRAIN]
        y_t = y_model[:TRAIN]
        x_v = x_model[TRAIN:]
        y_v = y_model[TRAIN:]


        dtrain = xgb.DMatrix(x_t, y_t)
        dval = xgb.DMatrix(x_v, y_v)

        xgb_par = {
                'objective' : 'multi:softprob',
                'booster' : 'gbtree',
                'eta' : 0.03,
                'subsample' : 0.9,
                'num_class' : 3,
                'max_depth' : 15,
                'colsample_bytree' : .6,
                }

        clf = xgb.train(xgb_par, 
            dtrain = dtrain,
            num_boost_round = 300,
            )

        pickle.dump(clf, open('./stacking/' + str(model_i) + '_xgb', 'wb'))
    print('finish xgb bagging')

    for voter_name in ['avg']:
        rf_pred = rf_clf.predict_proba(x_voter)
        # rf_pred = rf_pred.reshape((rf_pred.shape[0], 1))

        # svc = svm.SVC(probability=True)
        # svc.fit(x, y)
        # svc_pred = svc.predict_proba(x_val)
        # all_pred = np.concatenate((rf_pred, svc_pred), axis = 1)
        # print('Finish svc')


        # dvoter = xgb.DMatrix(x_voter)
        # xgb_pred = np.zeros((VOTER,3))
        # for i in range(MODEL_NUM):
        #     model = pickle.load(open('stacking/' + str(i) + '_xgb', 'rb'))
        #     xgb_pred += model.predict(dvoter, ntree_limit = model.best_ntree_limit)
        # xgb_pred /= MODEL_NUM
        # print(rf_pred.shape)
        # print('xgb_pred shape',np.array(xgb_pred).shape)
        # all_pred = np.concatenate((all_pred,xgb_pred), axis = 1)
        # print(all_pred[0])

        dvoter = xgb.DMatrix(x_voter)
        xgb_pred = []
        for i in range(MODEL_NUM):
            model = pickle.load(open('stacking/' + str(i) + '_xgb', 'rb'))
            xgb_pred.append(model.predict(dvoter, ntree_limit = model.best_ntree_limit))
        xgb_pred = np.concatenate(xgb_pred, axis = 1)
        all_pred = np.concatenate((rf_pred,xgb_pred), axis = 1)

        # voter = svm.SVC()

        if voter_name == 'svc':
            voter = SVC()
            voter.fit(all_pred, y_voter)
        elif voter_name == 'rf':
            voter = RandomForestClassifier(n_estimators = 1000,
                        bootstrap = True,
                        n_jobs = -1, 
                        min_samples_split=200,
                        oob_score = True)
            voter.fit(all_pred, y_voter)
        else:
            # for average
            pass


        x_val = pickle.load(open(val_x_path, 'rb'))
        x_val = scaler.transform(x_val)
        y_val = pickle.load(open(val_y_path, 'rb'))
        y_val = label_encorder.transform(y_val)
        rf_pred = rf_clf.predict_proba(x_val)

        # xgb bagging
        # dval = xgb.DMatrix(x_val)
        # xgb_pred = np.zeros((11880,3))
        # # xgb_pred = []
        # for i in range(MODEL_NUM):
        #     model = pickle.load(open('stacking/' + str(i) + '_xgb', 'rb'))
        #     xgb_pred += model.predict(dval, ntree_limit = model.best_ntree_limit)
        # xgb_pred /= MODEL_NUM
        # val_pred = np.concatenate((rf_pred, xgb_pred), axis = 1)

        # xgb stacking
        dval = xgb.DMatrix(x_val)
        xgb_pred = []
        for i in range(MODEL_NUM):
            model = pickle.load(open('stacking/' + str(i) + '_xgb', 'rb'))
            xgb_pred.append(model.predict(dval, ntree_limit = model.best_ntree_limit))
        xgb_pred = np.concatenate(xgb_pred, axis = 1)
        val_pred = np.concatenate((rf_pred,xgb_pred), axis = 1)

        dtrain = xgb.DMatrix(all_pred, y_voter)
        dval = xgb.DMatrix(val_pred, y_val)

        # xgb_par = {
        #         'objective' : 'multi:softprob',
        #         'booster' : 'gbtree',
        #         'eta' : 0.03,
        #         'subsample' : 0.9,
        #         'num_class' : 3,
        #         'max_depth' : 15,
        #         'colsample_bytree' : .6,
        #         }

        # clf = xgb.train(xgb_par, 
        #         dtrain = dtrain,
        #         evals = [(dval, 'eval')],
        #         num_boost_round = 1000,
        #         early_stopping_rounds = 40,
        #         )

        if voter_name == 'svc':
            score = voter.score(val_pred, y_val)
            svc_voter_score.append(score)
        elif voter_name == 'rf':
            score = voter.score(val_pred, y_val)
            rf_voter_score.append(score)
        else:
            pickle.dump(val_pred, open('val_pred.pkl', 'wb'))
            val_pred.reshape(11880, 3, 4)
            
        print('score:', score)

    pickle.dump(svc_voter_score, open('svc_voter_score.pkl', 'wb'))
    pickle.dump(rf_voter_score, open('svm_voter_score.pkl', 'wb'))
    pickle.dump(xgb_voter_score, open('xgb_voter_score.pkl', 'wb'))
