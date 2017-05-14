import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '/home/ymy1248/Code/ML2017_data/hw4/self_data/'
train = np.load(DATA_PATH + 'train.npy')

x = np.load(DATA_PATH + '5_t_x.npy') 
y = np.load(DATA_PATH + '5_t_y.npy')

x = x[:,2]
plt.plot(x, y, 'ro')
plt.show()
