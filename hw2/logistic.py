import numpy as np
import csv
import pickle
import sys

class LogisticRegression:
	def __init__ (self, x, y, xTest, config, w = None, b = None):
		self.c = config
		xNorm, self.xTest = self.feature_normalize(np.array(x), np.array(xTest))
		self.x = np.ndarray((self.c["vali"],self.c["order"],106))
		for i in range(self.c["vali"]):
			order = np.array(xNorm[i], dtype = np.float64)
			for j in range(self.c["order"]):
				self.x[i][j] = order
				order *= order
		self.y = np.array(y)
		self.dim = len(x[0])
		if w is None:
			self.w = np.random.rand(config["order"],self.dim)
		else:
			self.w = w
		if b is None:
			self.b = np.random.rand()
		else:
			self.b = b
		self.lr = config["lr"]
		self.wRate = np.zeros((config["order"],self.dim))
		self.bRate = 0
		self.xVal = np.array(xNorm[config["vali"]+1:])
		self.yVal = y[config["vali"]+1:]
		self.lamda = self.c["lamda"]

	def loss(self):
		loss = 0.0
		for i in range(len(self.x)):
			Y = self.y[i]
			y = self.sigTrain(i)
			loss += -(np.dot(Y, np.log(y)) + np.dot((1-Y), np.log(1-y)))
		loss += self.lamda*np.sum(self.w**2)
		return loss

	def gradient(self):
		num = len(self.x)
		g = np.zeros((self.c["order"],self.dim))
		b = 0
		for i in range(len(self.x)):
			base = -(self.y[i] - self.sigTrain(i))
			b += base
			for j in range(self.c["order"]):
				g[j] += base*self.x[i][j]
		return (g + self.lamda * self.w)/num, b/num

	def train(self):
		gGra, bGra = self.gradient()
		for i in range(self.c["iter"]):
			self.wRate += gGra**2
			self.bRate += bGra**2
			self.w -= self.lr / np.sqrt(self.wRate)*gGra
			self.b -= self.lr / np.sqrt(self.bRate)*bGra
			gGra, bGra = self.gradient()

	def shuffle(self, x, y):
		randomize = np.arange(len(x))
		np.random.shuffle(randomize)
		return x[randomize], y[randomize]


	def sigTrain(self, index):
		base = np.ones(self.dim)
		z = 0
		for i in range(self.c["order"]):
			z += np.dot(self.w[i],self.x[index][i])
		z += self.b
		# print("b:", self.b)
		# print("z:",z)
		res = 1/(1+np.exp(-z))
		# print("res:",res)
		return np.clip(res, 0.00000000000001, 0.99999999999999)

	def sigFunc(self, vector):
		base = np.ones(self.dim)
		z = 0
		for i in range(self.c["order"]):
			z += np.dot(self.w[i], vector)
			vector *= vector
		res = 1/(1+np.exp(-(z + self.b)))
		return np.clip(res, 0.00000000000001, 0.99999999999999)

	def predict(self, vector):
		out = self.sigFunc(vector)
		if out >= 0.5:
			return 1
		else:
			return 0

	def validation(self):
		total = len(self.xVal)
		count = 0
		for i in range(total):
			if self.predict(self.xVal[i]) == self.yVal[i]:
				count += 1
		return count/total

	def test(self):
		self.train()

	def feature_normalize(self, X_train, X_test):
		# feature normalization with all X
		X_all = np.concatenate((X_train, X_test))
		mu = np.mean(X_all, axis=0)
		sigma = np.std(X_all, axis=0)
		
		# only apply normalization on continuos attribute
		index = [0, 1, 3, 4, 5]
		mean_vec = np.zeros(X_all.shape[1])
		std_vec = np.ones(X_all.shape[1])
		mean_vec[index] = mu[index]
		std_vec[index] = sigma[index]

		X_all_normed = (X_all - mean_vec) / std_vec

		# split train, test again
		X_train_normed = X_all_normed[0:X_train.shape[0]-1]
		X_test_normed = X_all_normed[X_train.shape[0]:]

		return X_train_normed, X_test_normed

	def write(self, fileName):
		with open(fileName, "w") as f:
			writer = csv.writer(f,lineterminator='\n')
			writer.writerow(["id","label"])
			for i in range(len(self.xTest)):
				# writer.writerow([i+1, model.predict(v)])
				writer.writerow([i+1, self.predict(self.xTest[i])])

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

	config = {"order":2,"lr":1, "iter": 500, "vali":30000, "lamda":20}
	newVector = []

	# model = log.LogisticRegression(newVector[1:30000], yTrain[1:30000], newVector[30000:], yTrain[30000:], config)
	w = pickle.load(open("logistic_w", "rb"))
	b = pickle.load(open("logistic_b", "rb"))
	model = LogisticRegression(xTrain, yTrain, xTest, config,w,b)
	model.write(sys.argv[4])