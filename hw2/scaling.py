import numpy as np

class FeatureScaling:
	def __init__(self, vectors):
		self.mean = np.mean(vectors, axis = 0)
		self.var = np.var(vectors, axis = 0)

	def trans(self, vector):
		return (vector - self.mean)/ self.var
