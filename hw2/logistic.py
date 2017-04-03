import numpy as np
import csv
# TODO bias

class LogisticRegression:
	def __init__ (self, x, y, xTest, config):
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
		self.w = np.random.rand(config["order"],self.dim)
		self.b = np.random.rand()
		self.lr = config["lr"]
		self.wRate = np.zeros((config["order"],self.dim))
		self.bRate = 0
		self.xVal = np.array(x[config["vali"]+1:])
		self.yVal = y[config["vali"]+1:]

	def loss(self):
		loss = 0.0
		for i in range(len(self.x)):
			Y = self.y[i]
			y = self.sigTrain(i)
			loss += -(np.dot(Y, np.log(y)) + np.dot((1-Y), np.log(1-y)))
		return loss

	def gradient(self):
		num = len(self.x)
		g = np.zeros((self.c["order"],self.dim))
		b = 0
		for i in range(len(self.x)):
			base = -(self.y[i] - self.sigTrain(i))
			# print("base:",base)
			b += base
			for j in range(self.c["order"]):
				g[j] += base*self.x[i][j]
		return g/num, b/num

	def train(self):
		gGra, bGra = self.gradient()
		# print(g)
		for i in range(self.c["iter"]):
			# self.x, self.y = self.shuffle(self.x, self.y)
			# print(gGra[0][0])
			self.wRate += gGra**2
			self.bRate += bGra**2
			self.w -= self.lr / np.sqrt(self.wRate)*gGra
			self.b -= self.lr / np.sqrt(self.bRate)*bGra
			# print(str(self.wRate[0][0]))
			# print("b: ", bGra)
			# self.w -= 0.02*gGra
			# self.b -= 0.02*bGra
			# print("flag")
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

	def wirte(self):
		with open("ans.csv", "w") as f:
			writer = csv.writer(f,lineterminator='\n')
			writer.writerow(["id","label"])
			for i in range(len(self.xTest)):
				v = np.append(v,1.0)
				# writer.writerow([i+1, model.predict(v)])
				writer.writerow([i+1, self.predict(self.xTest[i])])

if __name__ == "__main__":
	np.set_printoptions(threshold='nan')