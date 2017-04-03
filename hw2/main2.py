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
with open(X_TRAIN, "r") as f:
	xTrainStr = list(csv.reader(f))

with open(Y_TRAIN, "r") as f:
	yTrainStr = list(csv.reader(f))

for i in range(1,len(xTrainStr)):
	vector = []
	for j in range(len(xTrainStr[i])):
		vector.append(float(xTrainStr[i][j]))
	# vector.append(1.0)
	xTrain.append(vector)

for i in range(len(yTrainStr)):
	yTrain.append(float(yTrainStr[i][0]))

with open(X_TEST, "r") as f:
	xTestStr = list(csv.reader(f))

for i in range(1, len(xTestStr)):
	vector = []
	for j in range(len(xTestStr[i])):
		vector.append(float(xTestStr[i][j]))
	xTest.append(vector)

config = {"order":2,"lr":2, "iter": 100, "vali":30000, "lamda":10}
newVector = []
model = log.LogisticRegression(xTrain, yTrain, xTest, config)
# model = log.LogisticRegression(newVector[1:30000], yTrain[1:30000], newVector[30000:], yTrain[30000:], config)
lastName = "logi_init"
pickle.dump(model, open(lastName,"wb"))
count = 0
while True:
	count += config["iter"]
	model = pickle.load(open(lastName,"rb"))
	model.train()
	valiScore = model.validation()
	lastName = "logi_" + str(config["order"]) + "_" + str(count) + "_" + str(valiScore)
	pickle.dump(model,open(lastName,"wb"))
	print(lastName)
	print(model.loss())
# pickle.dump(model,open("logistic_iter6000.p", "wb"))
model = pickle.load(open("logi_115000_0.8457633736821554", "rb"))
# model.train()
# pickle.dump(model,open("logi_42000_0.8383443967200312", "wb"))


with open("ans.csv", "w") as f:
	writer = csv.writer(f,lineterminator='\n')
	writer.writerow(["id","label"])
	for i in range(len(xTest)):
		v = fs.trans(xTest[i])
		v = np.append(v,1.0)
		# writer.writerow([i+1, model.predict(v)])
		writer.writerow([i+1, model.predict(xTest[i])])
# model = gen.BinaryGenerativeModel(xTrain, yTrain,config)
# pickle.dump(model,open("generativeModel.p", "wb"))
# count = 0
# for i in range(len(xTrain)):
# 	if model.predict(i) == yTrain[i]:
# 		count += 1
# print(count/len(xTrain))
# print(len(xTrain))
# print(len(yTrain))