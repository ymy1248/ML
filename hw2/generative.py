import numpy as np
import pickle
import sys
import csv

class BinaryGenerativeModel:
	def __init__(self, x, y, xTest, config, w = None, b = None):
		self.c = config
		self.xTest = xTest
		self.xVal = np.array(x[self.c["vali"]:])
		self.yVal = np.array(y[self.c["vali"]:])
		x = np.array(x)
		self.x = [[],[]]
		i = 0
		while y[i] != 0:
			i = i+1
		self.x[0] = np.array(x[i])
		i = 0
		while y[i] != 1:
			i = i+1
		self.x[1] = np.array(x[i])
		self.y = np.array(y)
		self.count = [0,0]
		self.dim = len(x[0])
		self.total = self.c["vali"]
		if w is None or b is None:
			for i in range(self.c["vali"]):
				if y[i] == 0:
					self.x[0] = np.vstack((self.x[0], x[i]))
					self.count[0] = self.count[0] + 1
				else:
					self.x[1] = np.vstack((self.x[1], x[i]))
					self.count[1] = self.count[1] + 1
			self.mean = self.mean()
			self.cov = self.covariance()
			self.mean = [1,2]
			self.mean[1], self.mean[0], self.cov, self.count[1], self.count[0] = self.train(x[:self.c["vali"]], y[:self.c["vali"]])
			self.inv = np.linalg.inv(self.cov)
			self.w = np.dot((self.mean[1]-self.mean[0]), self.inv)
			self.b = (-0.5) * np.dot(np.dot(self.mean[1], self.inv), self.mean[1]) + (0.5) * np.dot(np.dot(self.mean[0], self.inv), self.mean[0]) + np.log(float(self.count[1])/self.count[0])
		else:
			self.w = w
			self.b = b

	def mean(self):
		mean = []
		mean.append(np.mean(self.x[0], axis = 0))
		mean.append(np.mean(self.x[1], axis = 0))
		return mean

	def covariance(self):
		cov0 = np.cov(self.x[0], rowvar = False)
		cov1 = np.cov(self.x[1], rowvar = False)
		res = (float(self.count[0])/self.total)* cov0 + (float(self.count[1])/self.total) * cov1
		return res

	def predict(self, x):
		x = x.T
		a = np.dot(w, x) + b
		y = self.sigmoid(a)
		return np.around(y)

	def sigmoid(self, z):
	    res = 1 / (1.0 + np.exp(-z))
	    return np.clip(res, 0.00000000000001, 0.99999999999999)

	def validation(self):
		total = len(self.xVal)
		count = 0
		res = self.predict(self.xVal)
		for i in range(total):
			if res[i] == self.yVal[i]:
				count += 1
		return count/total


if __name__ == "__main__":
	X_TRAIN = sys.argv[1]
	Y_TRAIN = sys.argv[2]
	X_TEST = sys.argv[3]
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

	with open(X_TEST, "r") as f:
		xTestStr = list(csv.reader(f))

	for i in range(1, len(xTestStr)):
		vector = []
		for j in range(len(xTestStr[i])):
			vector.append(float(xTestStr[i][j]))
		xTest.append(vector)

	config = {"func": 0, "vali": 30000}
	w = pickle.load(open("gen_w","rb"))
	b = pickle.load(open("gen_b","rb"))
	model = BinaryGenerativeModel(xTrain, yTrain, xTest,config, w, b)

	with open(sys.argv[4], "w") as f:
		ans = model.predict(np.array(xTest))
		writer = csv.writer(f)
		writer = csv.writer(f,lineterminator='\n')
		writer.writerow(["id","label"])
		for i in range(len(xTest)):
			writer.writerow([i+1, int(ans[i])])