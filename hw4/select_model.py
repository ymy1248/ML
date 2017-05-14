import pickle
import numpy as np
import os
import random
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

TRAIN_DIR_PATH = '/home/ymy1248/Code/ML2017_data/hw4/self_data/'
DIR_PATH = '../../ML2017_data/hw4/data/'

x = np.load(TRAIN_DIR_PATH + '6_t_x.npy')
y = np.load(TRAIN_DIR_PATH + '6_t_y.npy')
print(x.shape)
scaler = StandardScaler()
x = scaler.fit_transform(x)
randomize = np.arange(len(x))

SCORE = 0.9948
best = 0
last_c = 0
pickle.dump( [], open(str(best) + '_' + str(0), 'wb'))
while True:
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    val_len = int(0.9 * len(x))
    x_T = x[:val_len]
    y_T = y[:val_len]
    x_test = x[val_len:]
    y_test = y[val_len:]
    c = random.uniform(30, 200)
    e = random.uniform(0, 1)
    svr = SVR(C = c, epsilon = e)
    svr.fit(x_T, y_T)
    score = svr.score(x_test, y_test)
    if score > best:
        os.remove(str(best) + '_' + str(last_c))
        pickle.dump( svr, open(str(score) + '_' + str(c), 'wb'))
        best = score
        last_c = c
    elif score > SCORE:
        pickle.dump( svr, open(str(score) + '_' + str(c), 'wb'))
