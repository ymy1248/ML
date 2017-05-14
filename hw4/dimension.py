import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, scale, normalize, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR, SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

TRAIN_DIR_PATH = '/home/ymy1248/Code/ML2017_data/hw4/self_data/'
DIR_PATH = '../../ML2017_data/hw4/data/'
#fig = plt.figure()

#print(train[0].shape)
#final_train = [[],[]]
#for i in range(len(train[0])):
#    sub = list(train[0][i])
#    eig_samp = sub.pop()
#    sub.append(np.mean(eig_samp))
#    sub.append(np.std(eig_samp))
#    final_train[0].append(np.array(sub))
#final_train[0] = np.array(final_train[0])
#final_train[1] = np.array(train[1])
#np.save(TRAIN_DIR_PATH + '4_t_x', final_train[0])
#np.save(TRAIN_DIR_PATH + '4_t_y', final_train[1])

x = np.load(TRAIN_DIR_PATH + '6_t_x.npy')
y = np.load(TRAIN_DIR_PATH + '6_t_y.npy')
print(x.shape)
scaler = StandardScaler()
x = scaler.fit_transform(x)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]
val_len = int(0.9 * len(x))
x_T = x[:val_len]
y_T = y[:val_len]
x_test = x[val_len:]
y_test = y[val_len:]
#print(x.shape)
#svr = SVR(C = 50, epsilon = 0.1)
#svr.fit(np.array(x), np.array(y))
#print(svr.score(x_test, y_test))
svr = pickle.load(open('last_hope', 'rb'))
print(svr.score(x_test, y_test))

def get_eigenvalues(data):
    SAMPLE = 100 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)
    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals
ans = []
#for file_index in range(200):
#    print(file_index)
#    data = np.load(DIR_PATH + str(file_index) + '.npy')
#    pca = PCA()
#    pca.fit(data)
#    ratio_dim = 0
#    while pca.explained_variance_ratio_[ratio_dim] > 0.01:
#        ratio_dim += 1
#    eig = get_eigenvalues(data)
#    eig_dim = 0
#    for eig_dim in range(len(eig)):
#        if 0.01 * eig_dim > eig[eig_dim]:
#            break
#    ans.append([np.std(data), np.std(pca.explained_variance_ratio_), ratio_dim,
#        np.mean(eig), np.std(eig), eig_dim])
#ans = np.array(ans)
#ans = ans.reshape(len(ans), 6)
#ans = scaler.transform(ans)
ans = pickle.load(open('6_t_input', 'rb'))
ans = svr.predict(ans)
for i in range(len(ans)):
    if ans[i] < 1:
        ans[i] = 1
    if ans[i] > 60:
        ans[i] = 60
ans = np.round(ans)
with open('last_hope.csv', 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['SetId', 'LogDim'])
    for i in range(len(ans)):
        writer.writerow([i, np.log(ans[i])])
