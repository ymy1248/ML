import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA

HAND_PATH = '/home/ymy1248/Code/ML2017_data/hw4/hand/'

im = []
for i in range(1, 482):
    im.append(misc.imread(HAND_PATH + 'hand.seq' + str(i) + '.png'))

im = np.array(im)
im = im.reshape(481, 245760)
pca = PCA()

plt.plot(pca.explained_variance_ratio_)
plt.show()
