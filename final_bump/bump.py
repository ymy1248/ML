import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from argparse import ArgumentParser

VALI = 0.9

parser = ArgumentParser()
parser = ArgumentParser()
parser.add_argument('--csv')
parser.add_argument('--test', action = 'store_true') 
parser.add_argument('--xgb', action = 'store_true') 
parser.add_argument('--train', action = 'store_true')
parser.add_argument('--pre', action = 'store_true')
parser.add_argument('--pca', action = 'store_true')
args = parser.parse_args()

if args.pre:
    X_PATH = '/home/ymy1248/Code/ML2017_data/bump/x.csv'
    Y_PATH = '/home/ymy1248/Code/ML2017_data/bump/y.csv'
    X_TEST = '/home/ymy1248/Code/ML2017_data/bump/test_x.csv'
    
    x = np.array(list(csv.reader(open(X_PATH, 'r'))))
    y = np.array(list(csv.reader(open(Y_PATH, 'r'))))
    x_test = np.array(list(csv.reader(open(X_TEST, 'r'))))

    feature_name = x[0]
    x = x[1:, :]
    y = y[1:, 1:]
    x_test = x_test[1:, :]
    y = y.reshape((y.shape[0],))
    label_list = ['functional', 'functional needs repair', 'non functional']
    num_y = []
    for label in y:
        num_y.append(label_list.index(label))
    print(feature_name)
    delete_list = []
    delete_list.append(int(np.argwhere(feature_name == 'id')[0])) 
    #delete_list.append(int(np.argwhere(feature_name == 'date_recorded')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'wpt_name')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'num_private')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'recorded_by')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'extraction_type_group')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'extraction_type')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'payment_type')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'water_quality')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'district_code')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'region')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'region_code')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'subvillage')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'ward')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'installer')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'quantity_group')[0]))
    #delete_list.append(int(np.argwhere(feature_name == 'waterpoint_type_group')[0]))

    print(delete_list)
    x = np.delete(x, delete_list,  axis = 1)
    print(x[0])
    print(x[1])
    x_test = np.delete(x_test, delete_list, axis = 1)
    feature_list = []
    for i in range(len(x[0])):
        feature_list.append([])
    new_x = []
    for x_row in x:
        l = []
        for row_i in range(len(x_row)):
            if row_i == 1:
                dt = datetime.strptime(x_row[row_i], '%Y-%m-%d')
                l.append((dt-datetime(1976,1,1)).total_seconds())
                continue
            try:
                l.append(float(x_row[row_i]))
                continue
            except:
                pass
            if x_row[row_i].isdigit():
                l.append(int(x_row[row_i]))
            else:
                if x_row[row_i] not in feature_list[row_i]:
                    feature_list[row_i].append(x_row[row_i])
                l.append(feature_list[row_i].index(x_row[row_i]))
        new_x.append(l)
    new_x_test = []
    for x_row in x_test:
        l = []
        for row_i in range(len(x_row)):
            if row_i == 1:
                dt = datetime.strptime(x_row[row_i], '%Y-%m-%d')
                l.append((dt-datetime(2000,1,1)).total_seconds())
                continue
            try:
                l.append(float(x_row[row_i]))
                continue
            except:
                pass
            if x_row[row_i].isdigit():
                l.append(int(x_row[row_i]))
            else:
                if x_row[row_i] not in feature_list[row_i]:
                    feature_list[row_i].append(x_row[row_i])
                l.append(feature_list[row_i].index(x_row[row_i]))
        new_x_test.append(l)
    print('Length of new_x:', len(new_x))

    pickle.dump(new_x, open('x', 'wb'))
    print(new_x[0])
    pickle.dump(num_y, open('y', 'wb'))
    pickle.dump(new_x_test, open('x_test', 'wb'))
    print('Finish Preprocessing')

if args.train:
    x = np.array(pickle.load(open('x', 'rb')))
    y = np.array(pickle.load(open('y', 'rb')))
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    x_test = pickle.load(open('x_test', 'rb'))
    all_x = np.concatenate((x, x_test))

    TRAIN = int(VALI * len(x))
    x_t = x[:TRAIN]
    y_t = y[:TRAIN]
    x_v = x[TRAIN:]
    y_v = y[TRAIN:]

    scaler = StandardScaler()
    scaler.fit(all_x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    pickle.dump(scaler, open('scaler', 'wb'))
    print('Finish normizatilon')

    #pca = PCA(n_components = 36)
    #pca.fit(all_x)
    #x = pca.transform(x)
    #x_test = pca.transform(x_test)
    #print('Finish PCA')

    for i in range(2):
        clf = RandomForestClassifier(n_estimators = 1700,
                n_jobs = -1, 
                oob_score = True, 
                min_samples_leaf = 3,
                min_samples_split = 2)
        clf.fit(x, y)
        score = clf.oob_score_
        print(len(x_test))
        print(score)
        test_ans = clf.predict_proba(x_test)
        ans = clf.predict(x_test)
        pickle.dump(clf, open('./model/{:.4f}'.format(score), 'wb'))
        if np.max(test_ans) > 0.7:
            train_l = np.where(test_ans > 0.7)
            #print(test_ans.shape)
            #print(x_test.shape)
            #print('ans shape', ans.shape)
            x_test = np.array(x_test)
            self_train_x = x_test[train_l[0]]
            x_test = np.delete(x_test, train_l[0], axis = 0)
            self_train_y = ans[train_l[0]]
            x = np.concatenate((x, self_train_x))
            y = np.concatenate((y, self_train_y))
    print('Finish training')

if args.test:
    clf = pickle.load(open('./model/0.8182', 'rb'))
    scaler = pickle.load(open('scaler', 'rb'))
    label_list = ['functional', 'functional needs repair', 'non functional']
    x_test = pickle.load(open('x_test', 'rb'))
    x_test = scaler.transform(x_test)
    ans = clf.predict(x_test)
    X_TEST = '/home/ymy1248/Code/ML2017_data/bump/test_x.csv'
    x_test = np.array(list(csv.reader(open(X_TEST, 'r'))))
    id_list = x_test[1:, :1].reshape(x_test.shape[0]-1)
    print(id_list)
    with open('ans.csv', 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['id', 'status_group'])
        for i in range(len(ans)):
            writer.writerow([id_list[i], label_list[ans[i]]])

if args.xgb:
    x = np.array(pickle.load(open('x', 'rb')))
    y = np.array(pickle.load(open('y', 'rb')))
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    x_test = pickle.load(open('x_test', 'rb'))
    all_x = np.concatenate((x, x_test))

    TRAIN = int(VALI * len(x))
    x_t = x[:TRAIN]
    y_t = y[:TRAIN]
    x_v = x[TRAIN:]
    y_v = y[TRAIN:]

    all_xtrain = xgb.DMatrix(x, y)
    dtrain = xgb.DMatrix(x_t, y_t)
    dval = xgb.DMatrix(x_v, y_v)
    dtest = xgb.DMatrix(x_test)

    xgb_par = {
            'objective' : 'multi:softmax',
            'booster' : 'gbtree',
            'num_class' : 3,
            'eval_matric' : "merror",
            'max_depth' : 14,
            'colsample_bytree' : .4,
            }
    cv_out = xgb.cv(
            xgb_par,
            all_xtrain,
            num_boost_round = 1000,
            nfold = 5,
            early_stopping_rounds = 10,
            verbose_eval = 50,
            show_stdv = False)
    print(cv_out)
    print('Round:', len(cv_out))
    rounds = len(cv_out)

    clf = xgb.train(xgb_par, 
            dtrain = all_xtrain, 
            num_boost_round = rounds,
            )

    ans = clf.predict(dtest)
    X_TEST = '/home/ymy1248/Code/ML2017_data/bump/test_x.csv'
    x_test = np.array(list(csv.reader(open(X_TEST, 'r'))))
    id_list = x_test[1:, :1].reshape(x_test.shape[0]-1)
    print(id_list)
    label_list = ['functional', 'functional needs repair', 'non functional']
    with open('ans.csv', 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['id', 'status_group'])
        for i in range(len(ans)):
            writer.writerow([id_list[i], label_list[int(ans[i])]])


if args.pca:
    x = np.array(pickle.load(open('x', 'rb')))
    y = np.array(pickle.load(open('y', 'rb')))
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    x_test = pickle.load(open('x_test', 'rb'))
    all_x = np.concatenate((x, x_test))

    scaler = StandardScaler()
    scaler.fit(all_x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    score_list = []

    for dim in range(20, 36):
        print(dim)
        pca = PCA(n_components = dim)
        pca.fit(all_x)
        x_pca = pca.transform(x)
        x_test_pac = pca.transform(x_test)

        train_len = int(VALI * len(x))
        x_train = x_pca[:train_len]
        y_train = y[:train_len]
        x_val = x_pca[train_len:]
        y_val = y[train_len:]
        pickle.dump(scaler, open('scaler', 'wb'))
        clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, class_weight = 'balanced')
        clf.fit(x_train, y_train)
        score_list.append(clf.score(x_val, y_val))
    x_axis = np.arange(20,36)
    plt.plot(x_axis, score_list)
    pickle.dump(score_list, open('score_list', 'wb'))
    plt.show()
