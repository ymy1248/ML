
# coding: utf-8

# # Stacking Experiment

# In[1]:

import time
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
stack_rf_score = []
stack_avg_score = []

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

    x_val = pickle.load(open(val_x_path, 'rb'))
    x_val = scaler.transform(x_val)
    y_val = pickle.load(open(val_y_path, 'rb'))
    y_val = label_encorder.transform(y_val)

    rf_clf = RandomForestClassifier(n_estimators = 1000,
                n_jobs = -1, 
                min_samples_split=7,)
    rf_clf.fit(x_model, y_model)

    tree = DecisionTreeClassifier()
    tree.fit(x_model, y_model)
    print('Finish Model')

    rf_pred = rf_clf.predict_proba(x_voter)
    tree_pred = tree.predict_proba(x_voter)

    x_val = pickle.load(open(val_x_path, 'rb'))
    x_val = scaler.transform(x_val)
    y_val = pickle.load(open(val_y_path, 'rb'))
    y_val = label_encorder.transform(y_val)
    rf_val_pred = rf_clf.predict_proba(x_val)
    tree_val_pred = tree.predict_proba(x_val)

    for voter_name in ['avg', 'rf']:
        if voter_name == 'rf':
            all_pred = np.concatenate((rf_pred, tree_pred), axis = 1)
            voter = RandomForestClassifier(n_estimators = 1000,
                        bootstrap = True,
                        n_jobs = -1, 
                        min_samples_split=200,
                        oob_score = True)
            voter.fit(all_pred, y_voter)
            val_pred = np.concatenate((rf_val_pred, tree_val_pred), axis = 1)
            score = voter.score(val_pred, y_val)
            stack_rf_score.append(score)
            print('rf')

        else:
            all_pred = rf_val_pred + tree_val_pred
            all_pred /= 2
            all_pred = np.argmax(all_pred, axis = 1)
            print(all_pred.shape)
            val_count = 0
            for index in range(len(all_pred)):
                if all_pred[index] == y_val[index]:
                    val_count += 1
            score = val_count/len(y_val)
            stack_avg_score.append(score)
            print('average')
        print('score:', score)

    pickle.dump(stack_rf_score, open('stack_rf_score.pkl', 'wb'))
    pickle.dump(stack_avg_score, open('stack_avg_score.pkl', 'wb'))
