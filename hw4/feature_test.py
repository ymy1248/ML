import numpy as np
from sklearn.decomposition import PCA

DATA_PATH = '/home/ymy1248/Code/ML2017_data/hw4/self_data/'

for i in range(1,61):
    data = np.load(DATA_PATH + str(i) + '.npy')
    pca = PCA()
    pca.fit(data)
    ratio = pca.explained_variance_ratio_
    print('dim:' + str(i) + ' std:' + str(np.std(ratio)) + ' mean:' +
            str(np.mean(ratio)))
