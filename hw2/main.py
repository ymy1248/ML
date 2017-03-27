import generative as gen
import logistic as log
import scaling
import numpy as np
import csv
import pickle

np.set_printoptions(threshold='nan')
X_TRAIN = "X_train"
Y_TRAIN = "Y_train"
X_TEST = "X_test"
xTrain = []
yTrain = []
xTest = []
with open(X_TRAIN, "r") as f:
	xTrainStr = list(csv.reader(f))

with open(Y_TRAIN, "r") as f:
	yTrainStr = list(csv.reader(f))

for i in range(1,len(xTrainStr)):
	vector = []
	for j in range(len(xTrainStr[i])):
		vector.append(float(xTrainStr[i][j]))
	xTrain.append(vector)

for i in range(len(yTrainStr)):
	yTrain.append(float(yTrainStr[i][0]))

config = {"func":0}
# model = log.LogisticRegression(xTrain[0:10000], yTrain[0:10000])
fs = scaling.FeatureScaling(xTrain)
newVector = []
for v in xTrain:
	newVector.append(fs.trans(v))
model = gen.BinaryGenerativeModel(newVector, yTrain,config)
pickle.dump(model,open("generativeModel.p", "wb"))

count = 0
for i in range(len(xTrain)):
	if model.predict(i) == yTrain[i]:
		count += 1
print(count/len(xTrain))

with open(X_TEST, "r") as f:
	xTestStr = list(csv.reader(f))

for i in range(1, len(xTestStr)):
	vector = []
	for j in range(len(xTestStr[i])):
		vector.append(float(xTestStr[i][j]))
	xTest.append(vector)

with open("ans.csv", "w") as f:
	writer = csv.writer(f)
	for i in range(xTest):
		writer.writerrow([i, model.predict(xTest[i])])
