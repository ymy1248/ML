import csv
import math
import time
import config

# this is the training model for linear regression
class LinearModel:
	def __init__(self, config, data):
		self.lRate 	= config["lRate"]				# learning rate factor
		self.e 		= config["e"]					# stop criteria
		self.order  = config["model"]["order"]		# model order
		self.w 		= config["model"]["weight"].copy()		# weight
		self.b      = config["model"]["bias"]		# bias
		self.data 	= data							# input vector with last is ans
		self.feaNum = len(data[0]) - 1				# input feature number
		self.bRate  = 0.0							# b learning rate
		self.wRate  = [0.0]*(self.order*self.feaNum)
		self.lamda  = config["lamda"]
		# self.part	+ config["model"]["part"]

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
					gradient[i*self.order+j] += -2*diff*power + 2*self.lamda*self.w[i*self.order+j]
			gradient[self.order*self.feaNum] += -2 * diff	# bias gradient
		return gradient

	# loss function with all the data
	def lossFunction(self):
		loss = 0.0
		for d in self.data:
			loss +=  (d[self.feaNum] - self.modelResult(d[:len(d)-1]))**2
			wSum = 0.0
			for w in self.w:
				wSum += w**2;
			loss += self.lamda *  wSum
		return loss

	def vali(self, data):
		loss = 0.0
		for d in data:
			loss +=  (d[self.feaNum] - self.modelResult(d[:len(d)-1]))**2
			wSum = 0.0
			for w in self.w:
				wSum += w**2;
			loss += self.lamda *  wSum
		return loss

	def train(self):
		gra = self.gradientFunc() 
		# gra = first_gradient(data)
		flag = True
		loss = []
		k = 0
		# while flag == True:
		for k in range(4000):
			if k%2000 == 0:
				loss_num = self.lossFunction()
				loss.append(loss_num)
				print("k = ", k)
				print("loss: ",loss_num)
				print("gradient: ", gra)
				print("\"order\": ", self.order)
				print("\"weight\": ", self.w)
				print("\"bias\": ", self.b)
				print("\"lamda\": ", self.lamda)
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
		print("bias:", self.b)

	# def testTrain(self):
	# 	gra = self.gradientFunc() 
	# 	# gra = first_gradient(data)
	# 	flag = True
	# 	k = 0
	# 	while flag == True:
	# 	# for k in range(3000):
	# 		if k%1000 == 0:
	# 			loss_num = self.lossFunction()
	# 			test = self.testScore()
	# 			print("k = ", k)
	# 			print("loss: ",loss_num)
	# 			print("test: ", test)
	# 			print("gradient: ", gra)
	# 			print("weight: ", self.w)
	# 			print("b: ", self.b)
	# 			print("----------------------------------------")
	# 		for i in range(len(self.w)):
	# 			self.wRate[i] += gra[i]**2
	# 			self.w[i] -= self.lRate / math.sqrt(self.wRate[i]) * gra[i]
	# 		self.bRate += gra[len(gra)-1]**2
	# 		self.b -= self.lRate/ math.sqrt(self.bRate)*gra[len(gra)-1]
	# 		gra = self.gradientFunc()
	# 		# gra = first_gradient(data)
	# 		# print(gra)
	# 		flag = False
	# 		k = k + 1
	# 		for i in gra:
	# 			if abs(i) >= self.e:
	# 				flag = True
	# 				break
	# 	# plt.plot(loss)
	# 	print("weight:", self.w)
	# 	print("bias:", self.b)

	# def testScore(self):
	# 	loss = 0.0
	# 	for i in range(int(len(self.data)*self.part),len(self.data)):
	# 		loss +=  (self.data[i][self.feaNum] - self.modelResult(d[:len(d)-1]))**2
	# 	return loss

if __name__ == "__main__":
	start = time.time()
	# w = []
	# b = 0.0
	# e = 0.001				# tolerance
	# rate = 0.00007
	# self.bRate = 0.0
	# w_rate = []
	# order = 2

	trainFileName = "train.csv"
	testFileName = "test_X.csv"

	with open(trainFileName, "r", encoding='utf-8', errors='ignore') as f:
		reader = csv.reader(f)
		rawData = list(reader)

	pmData = []
	trainData = []
	day = 0
	for i in range(10,len(rawData),18):
		for j in range(3,len(rawData[i])):
			pmData.append(float(rawData[i][j]))

	for i in range(10,len(pmData)):
		perData = []
		for j in range(i-9,i+1):
			perData.append(pmData[j])
		trainData.append(perData)

	with open(testFileName, "r", encoding='utf-8', errors='ignore') as f:
		reader = csv.reader(f)
		testRawData = list(reader)

	testData = []

	for i in range(9, len(testRawData), 18):
		perData = []
		for j in range(2,len(testRawData[i])):
			perData.append(float(testRawData[i][j]))
		testData.append(perData)

	# init_w(order)
	# init_w_rate(order)
	# decent(order, trainData, gradient_func, e)
	model = LinearModel(config.modelConfig, trainData)
	model.train()

	ans = []

	for i in range(len(testData)):
		ans.append(model.modelResult(testData[i]))
		testData[i].append(ans[i])


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
