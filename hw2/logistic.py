import numpy as np
# TODO bias

class LogisticRegression:
	def __init__ (self, vectors, labels):
		self.vectors = [[],[]]
		self.count = [0,0]
		self.dim = len(vectors[0])
		self.w = np.array([2.0]*self.dim)
		self.lr = 50.0
		self.wRate = np.array([0.0]*self.dim)

		i = 0
		while labels[i] != 0:
			i += 1
		self.vectors[0] = np.array(vectors[i])
		i = 0
		while labels[i] != 1:
			i += 1
		self.vectors[1] = np.array(vectors[i])
		for i in range(len(labels)):
			if labels[i] == 0:
				self.vectors[0] = np.vstack((self.vectors[0], vectors[i]))
				self.count[0] = self.count[0] + 1
			else:
				self.vectors[1] = np.vstack((self.vectors[1], vectors[i]))
				self.count[1] = self.count[1] + 1

	def loss(self):
		loss = self.sigFunc(self.vectors[0][0])
		for v in self.vectors[0][1:]:
			loss *= self.sigFunc(v)
		for v in self.vectors[1]:
			loss *= (1-self.sigFunc(v))
		return loss

	def gradient(self):
		g = np.array([0.0]*self.dim)
		for v in self.vectors[0]:
			g += -(0-self.sigFunc(v))*v
		for v in self.vectors[1]:
			g += -(1-self.sigFunc(v))*v
		return g

	def train(self):
		g = self.gradient()
		# print(g)
		for i in range(6000):
			if i % 200 == 0:
				print("i =",i)
				print(self.loss())
				# print(g)
			self.wRate += g**2
			self.w -= self.lr / np.sqrt(self.wRate) * g
			g = self.gradient()


	def sigFunc(self, vector):
		z = np.dot(self.w,vector)
		return 1/(1+np.exp(-z))

	def predict(self, vector):
		out = self.sigFunc(vector)
		if out >= 0:
			return 1
		else:
			return 0

	def test(self):
		self.train()

if __name__ == "__main__":
	np.set_printoptions(threshold='nan')