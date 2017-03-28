import generative as gen
import logistic as log
import numpy as np
import scaling
import csv
import pickle

np.set_printoptions(threshold='nan')
X_TRAIN = "X_train"
Y_TRAIN = "Y_train"
X_TEST = "X_test"
xTrain = []
yTrain = []
xTest = []
# with open(X_TRAIN, "r") as f:
# 	xTrainStr = list(csv.reader(f))

# with open(Y_TRAIN, "r") as f:
# 	yTrainStr = list(csv.reader(f))

# for i in range(1,len(xTrainStr)):
# 	vector = []
# 	for j in range(len(xTrainStr[i])):
# 		vector.append(float(xTrainStr[i][j]))
# 	xTrain.append(vector)

# for i in range(len(yTrainStr)):
# 	yTrain.append(float(yTrainStr[i][0]))

# config = {"func":0}
# fs = scaling.FeatureScaling(xTrain)
# newVector = []
# for v in xTrain:
# 	newVector.append(fs.trans(v))
# model = log.LogisticRegression(newVector[1:30000], yTrain[1:30000], newVector[30000:], yTrain[30000:])
# pickle.dump(model, open("logi_init","wb"))
count = 0
lastName = "logi_init"
while True:
	count += 1000
	model = pickle.load(open(lastName,"rb"))
	model.train()
	valiScore = model.validation()
	lastName = "logi_" + str(count) + "_" + str(valiScore)
	pickle.dump(model,open(lastName,"wb"))
	print(lastName)
pickle.dump(model,open("logistic_iter6000.p", "wb"))
model = pickle.load(open("logistic_iter12000.p", "rb"))
model.train()
pickle.dump(model,open("logistic_iter18000.p", "wb"))
with open(X_TEST, "r") as f:
	xTestStr = list(csv.reader(f))

for i in range(1, len(xTestStr)):
	vector = []
	for j in range(len(xTestStr[i])):
		vector.append(float(xTestStr[i][j]))
	xTest.append(vector)

with open("ans.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerow(["id","label"])
	for i in range(len(xTest)):
		writer.writerow([i+1, model.predict(fs.trans(xTest[i]))])
# model = gen.BinaryGenerativeModel(xTrain, yTrain,config)
# pickle.dump(model,open("generativeModel.p", "wb"))
# count = 0
# for i in range(len(xTrain)):
# 	if model.predict(i) == yTrain[i]:
# 		count += 1
# print(count/len(xTrain))
# print(len(xTrain))
# print(len(yTrain))