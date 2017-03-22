import csv
import LinearModel as lm
import config
import matplotlib.pyplot as plt
import model as m

TRAIN_FILE = "train.csv"
TEST_FILE = "test_X.csv"

with open(TRAIN_FILE, "r", encoding='utf-8', errors='ignore') as f:
	reader = csv.reader(f)
	rawData = list(reader)

pmData = []
monthData = []
trainData = []
day = 1

for i in range(10,len(rawData),18):
	for j in range(3,len(rawData[i])):
		monthData.append(float(rawData[i][j]))
	if day%20 == 0:
		pmData.append(monthData)
		# print(monthData)
		# print("--------------------------------------------")
		# print(pmData)
		# print("--------------------------------------------")
		monthData = []
	day = day +1

for data in pmData:
	for i in range(10, len(data)):
		perData = []
		for j in range(i-8,i):
			perData.append(data[j] - data[j-1])
		for j in range(i-9,i+1):
			perData.append(data[j])
		trainData.append(perData)

print(len(trainData))

with open(TEST_FILE, "r", encoding='utf-8', errors='ignore') as f:
	reader = csv.reader(f)
	testRawData = list(reader)

testData = []

for i in range(9, len(testRawData), 18):
	perData = []
	for j in range(3,len(testRawData[i])):
		perData.append(float(testRawData[i][j]) - float(testRawData[i][j-1]))
	for j in range(2,len(testRawData[i])):
		perData.append(float(testRawData[i][j]))
	testData.append(perData)

# valiScore = []
# for i in range(200, len(trainData)-1000, 200):
# 	print("i = " , i)
# 	model = lm.LinearModel(config.modelConfig, trainData[0:i])
# 	model.train()
# 	valiScore.append(model.vali(trainData[-1000:]))

lamdaScore = []
lamda = []

model = lm.LinearModel(config.modelConfig,trainData[0:-500])

print(model.vali(trainData[-500:]))

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
print("-------------------------------------------------------")
print(lamdaScore)
print(model.lossFunction())
