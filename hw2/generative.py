import numpy as np

class BinaryGenerativeModel:
	def __init__(self, vectors, labels):
		self.vectors = vectors
		self.labels = labels
		self.dim = len(vectors[0])
		self.mean = self.mean()
		self.train()

	def mean(self):
		mean = []
		mean0 = np.array([0] * self.dim)
		mean1 = np.array([0] * self.dim)
		count0 = 0
		count1 = 0

		for i in len(labels):
			if label[i] == 0:
				count0 = count0 + 1
				mean0 = np.add(mean0, vectors[i])
			else:
				count = count1 + 1
				mean1 = np.add(mean1, vectors[i])
		mean.append(np.divide(mean0, count0))
		mean.append(np.divide(mean1, count1))
		return mean

	def covariance(self):
		pass

	def train(self):
		pass

	def predict():
		pass