import numpy as np

class BinaryGenerativeModel:
	def __init__(self, vectors, labels, valiVec, valiLabels, config):
		self.vectors = [[],[]]
		i = 0
		while labels[i] != 0:
			i = i+1
		self.vectors[0] = np.array(vectors[i])
		i = 0
		while labels[i] != 1:
			i = i+1
		self.vectors[1] = np.array(vectors[i])
		self.labels = np.array(labels)
		self.count = [0,0]
		self.dim = len(vectors[0])
		self.total = len(labels)
		self.valiVec = valiVec
		self.valiLabel = valiLabels
		for i in range(len(labels)):
			if labels[i] == 0:
				self.vectors[0] = np.vstack((self.vectors[0], vectors[i]))
				self.count[0] = self.count[0] + 1
			else:
				self.vectors[1] = np.vstack((self.vectors[1], vectors[i]))
				self.count[1] = self.count[1] + 1
		self.mean = self.mean()
		self.cov = self.covariance()
		self.inv = np.linalg.inv(self.cov)
		self.down = 1/np.sqrt(np.linalg.det(self.cov))
		self.down *= 1/(np.sqrt(2*np.pi)**self.dim)

	def mean(self):
		mean = []
		mean.append(np.mean(self.vectors[0], axis = 0))
		mean.append(np.mean(self.vectors[1], axis = 0))
		return mean

	def covariance(self):
		cov0 = np.cov(self.vectors[0], rowvar = False)
		cov1 = np.cov(self.vectors[1], rowvar = False)
		res = np.multiply(cov0,self.count[0]/self.total)
		res += np.multiply(cov1,self.count[1]/self.total)
		return res

	def prob(self, vector):
		pC0 = self.count[0]/self.total
		pC1 = self.count[1]/self.total
		pxC0 = self.gaussian(self.mean[0], vector)
		pxC1 = self.gaussian(self.mean[1], vector)
		# print("pC0:", pC0, ", pC1", pC1, ", pxC0:", pxC0, ", pxC1:", pxC1)
		# print(pC0*pxC0/(pC1*pxC1 + pC0*pxC0))
		if pxC1 == float("inf") or pxC0 == float("inf"):
			return 0.0
		if pxC1 == 0.0 and pxC0 == 0.0:
			return 1.0
		return pC0*pxC0/(pC1*pxC1 + pC0*pxC0)

	def gaussian(self, mean, vector):
		diff = np.array(vector)-mean
		# print("diff:",diff)
		last = np.dot(self.inv, diff)
		# print("dot:", np.dot(diff,last))
		up = np.exp(-0.5*np.dot(diff,last))
		# print("up:", up)
		# print("down:", self.down)
		return up/self.down

	def validation(self):
		total = len(self.valiVec)
		count = 0
		for i in range(total):
			if self.predict(self.valiVec[i]) == self.valiLabel[i]:
				count += 1
		return count/total


	def predict(self, vector):
		predict = self.prob(vector)
		if predict >= 0.5 and predict <= 1:
			return 1
		elif predict >=0 and predict < 0.5:
			return 0
		else:
			print("something wrong, predict = ", predict)
			return 1

if __name__ == "__main__":
	pass