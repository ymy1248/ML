# PM2.5 is an interger value

import csv
import math
import time
import config
# import matplotlib.pyplot as plt
class LinearModel:
	def __init__(self, config, data):
		self.lRate = config["lRate"]				# learning rate factor
		self.e 		= config["e"]					# stop criteria
		self.order  = config["model"]["order"]		# model order
		self.w 		= config["model"]["weight"]		# weight
		self.b      = config["model"]["bias"]		# bias
		self.data 	= data							# input vector with last is ans
		self.feaNum = len(data[0]) - 1				# input feature number
		self.bRate  = 0.0							# b learning rate
		self.wRate  = [0.0]*(self.order*self.feaNum)

	# inPut: input all the feature as a 1d list
	def modelResult(self, inPut):
		outPut = self.b
		for i in range(len(inPut)):
			power = 1.0
			for j in range(self.order):
				power *= inPut[i]
				outPut += self.w[i*self.order+j] * power
		return outPut

	# self.data: input all the feature with the last one is answer 2d list
	def gradientFunc(self):
		gradient = [0.0]*(self.order*self.feaNum+1)
		for d in self.data:
			diff = d[self.feaNum] - self.modelResult(d[:len(d)-1])		# last one is ans
			for i in range(self.feaNum):
				power = 1.0
				for j in range(self.order):
					power *= d[i]
					gradient[i*self.order+j] += -2*diff*power
			gradient[self.order*self.feaNum] += -2 * diff	# bias gradient
		return gradient

	# loss function with all the data
	def lossFunction(self):
		loss = 0.0
		for d in self.data:
			loss +=  (d[self.feaNum] - self.modelResult(d[:len(d)-1]))**2
		return loss

	def decent(self):
		global b
		gra = self.gradientFunc() 
		# gra = first_gradient(data)
		flag = True
		loss = []
		k = 0
		while flag == True:
		# for k in range(3000):
			if k%1000 == 0:
				loss_num = self.lossFunction()
				loss.append(loss_num)
				print("k = ", k)
				print("loss: ",loss_num)
				print("gradient: ", gra)
				print("weight: ", self.w)
				print("b: ", self.b)
				print("----------------------------------------")
			for i in range(len(self.w)):
				self.wRate[i] += gra[i]**2
				self.w[i] -= self.lRate / math.sqrt(self.wRate[i]) * gra[i]
			self.bRate += gra[len(gra)-1]**2
			self.b -= self.lRate/ math.sqrt(self.bRate)*gra[len(gra)-1]
			gra = self.gradientFunc()
			# gra = first_gradient(data)
			# print(gra)
			flag = False
			k = k + 1
			for i in gra:
				if abs(i) >= self.e:
					flag = True
					break
		# plt.plot(loss)
		print("weight:", self.w)
		print("b:", b)

if __name__ == "__main__":
	start = time.time()
	# w = []
	# b = 0.0
	# e = 0.001				# tolerance
	# rate = 0.00007
	# self.bRate = 0.0
	# w_rate = []
	# order = 2

	train_file_name = "train.csv"
	test_file_name = "test_X.csv"

	with open(train_file_name, "r", encoding='utf-8', errors='ignore') as f:
		reader = csv.reader(f)
		raw_data = list(reader)

	pm_data = []
	train_data = []

	for i in range(10,len(raw_data),18):
		for j in range(3,len(raw_data[i])):
			pm_data.append(float(raw_data[i][j]))
	for i in range(10,len(pm_data)):
		per_data = []
		for j in range(i-9,i+1):
			per_data.append(pm_data[j])
		train_data.append(per_data)

	with open(test_file_name, "r", encoding='utf-8', errors='ignore') as f:
		reader = csv.reader(f)
		test_raw_data = list(reader)

	test_data = []

	for i in range(9, len(test_raw_data), 18):
		per_data = []
		for j in range(2,len(test_raw_data[i])):
			per_data.append(float(test_raw_data[i][j]))
		test_data.append(per_data)

	# init_w(order)
	# init_w_rate(order)
	# decent(order, train_data, gradient_func, e)
	model = LinearModel(config.modelConfig, train_data)
	model.decent()

	ans = []

	for i in range(len(test_data)):
		ans.append(model.modelResult(test_data[i]))
		test_data[i].append(ans[i])


	with open("ans.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(["id", "value"])
		for i in range(len(ans)):
			row = ["id_"+str(i), ans[i]]
			writer.writerow(row)
	# plt.show()
	print(model.lossFunction())
	# end = time.time()
	# print("time: ", end - start)
