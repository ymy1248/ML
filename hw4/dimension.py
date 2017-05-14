import csv
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, scale, normalize, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVR, SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

TRAIN_DIR_PATH = '/home/ymy1248/Code/ML2017_data/hw4/self_data/'

DATA_PATH = sys.argv[1]
ANS_PATH = sys.argv[2]
svr = pickle.load(open('last_hope', 'rb'))

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

data_npz = np.load(DATA_PATH)

for file_index in range(200):
   print(file_index)
   data = data_npz[str(file_index)]
   pca = PCA()
   pca.fit(data)
   ratio_dim = 0
   while pca.explained_variance_ratio_[ratio_dim] > 0.01:
       ratio_dim += 1
   eig = get_eigenvalues(data)
   eig_dim = 0
   for eig_dim in range(len(eig)):
       if 0.01 * eig_dim > eig[eig_dim]:
           break
   ans.append([np.std(data), np.std(pca.explained_variance_ratio_), ratio_dim,
       np.mean(eig), np.std(eig), eig_dim])
ans = np.array(ans)
ans = ans.reshape(len(ans), 6)
scaler = StandardScaler()
ans = scaler.fit_transform(ans)
ans = svr.predict(ans)

for i in range(len(ans)):
    if ans[i] < 1:
        ans[i] = 1
    if ans[i] > 60:
        ans[i] = 60
ans = np.round(ans)
with open(ANS_PATH, 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerow(['SetId', 'LogDim'])
    for i in range(len(ans)):
        writer.writerow([i, np.log(ans[i])])
