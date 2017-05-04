import sys
import csv
import numpy as np
import pickle
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.activations import relu

model = load_model("final.h5")
xStr = list(csv.reader(open(sys.argv[1])))
x = []
for i in range(1, len(xStr)):
	a = xStr[i][1].split(" ")
	x.append(list(map(float, a)))
x = np.array(x, dtype = np.float32)
x = x.reshape(x.shape[0],48,48,1)

ans = model.predict(x, batch_size = 200)

with open(sys.argv[2], "w") as f:
	writer = csv.writer(f,lineterminator='\n')
	writer.writerow(["id","label"])
	for i in range(len(ans)):
		# writer.writerow([i+1, model.predict(v)])
		writer.writerow([i, np.argmax(ans[i])])
		# writer.writerow([i, max(ans[i])])
