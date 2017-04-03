import numpy as np

class FeatureScaling:
	def __init__(self, vectors):
		self.mean = np.mean(vectors, axis = 0)
		self.var = np.var(vectors, axis = 0)

	def trans(self, vector):
		new = (vector - self.mean)/ self.var
		for j in range(len(new)):
			if j == 3 or j == 4 or (j >= 15 and j <= 21):
				new[j] *= 5
		return new