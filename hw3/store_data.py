import pickle
import scipy.misc as m
import numpy as np

VALI = 26000
x = pickle.load(open("x", "rb"))
y = pickle.load(open("y", "rb"))
x = x.reshape(x.shape[0],48,48)
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]
xT = x[:VALI]
yT = y[:VALI]
xV = x[VALI:]
yV = y[VALI:]
for i in range(len(xT)):
	m.imsave("data/train/"+str(yT[i])+"/" + str(i)+".jpg", xT[i])
for i in range(len(xV)):
	m.imsave("data/validation/"+str(yV[i])+"/" + str(i)+".jpg", xV[i])