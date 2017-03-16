import csv
import LinearModel as lm
import config

TRAIN_FILE = "train.csv"
TEST_FILE = "test_X.csv"

with open(TRAIN_FILE, "r", encoding='utf-8', errors='ignore') as f:
	reader = csv.reader(f)
	rawData = list(reader)

pmData = []
trainData = []

for i in range(10,len(rawData),18):
	for j in range(3,len(rawData[i])):
		pmData.append(float(rawData[i][j]))

for i in range(10,len(pmData)):
	perData = []
	for j in range(i-8,i):
		perData.append(pmData[j] - pmData[j-1])
	for j in range(i-9,i+1):
		perData.append(pmData[j])
	trainData.append(perData)

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

model = lm.LinearModel(config.modelConfig, trainData)
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
