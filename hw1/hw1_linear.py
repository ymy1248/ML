# PM2.5 is an interger value

import csv
import math
import time

start = time.time()
w = []
b = 0.0
e = 100					# tolerance
rate = 100
b_rate = 0.0
w_rate = []
order = 2


def init_w (order):
	global w
	w = [0.0] * (order*9)

def init_w_rate(order):
	global w_rate
	w_rate = [0.0] * (order*9)

# TODO other model_func
def first_order (in_put):
	global w
	global b
	out_put = b
	for i in range(9):
		out_put += w[i]*in_put[i]
	return out_put

def first_gradient(in_put_all):
	gradient = [0.0]*10
	for d in in_put_all:
		# diff = d[9] - first_order(d)
		diff = d[9] - model_function(order, d)
		for i in range(len(d) - 1):
			gradient[i] += 2*diff*(-d[i])
		gradient[9] += -2 * diff
	return gradient

def model_function(order, in_put):
	global w
	global b
	out_put = b
	for i in range(9):
		power = 1.0
		for j in range(order):
			power *= in_put[i]
			out_put += w[i*order+j] * power
	return out_put

def gradient_func(order, in_put_all):
	gradient = [0.0]*(order*9+1)
	for d in in_put_all:
		diff = d[9] - model_function(order, d)
		for i in range(9):
			power = 1.0
			for j in range(order):
				power *= d[i]
				gradient[i*order+j] += -2*diff*power
		gradient[order*9] += -2 * diff
	return gradient

def loss_function(data, model_func = first_order):
	loss = 0.0
	for d in data:
		loss +=  (d[9] - model_func(d[:9]))**2
	return loss

def decent(order, data, gradient = gradient_func, tolerance = 0.00001):
	# TODO 
	global w
	global b
	global b_rate
	gra = gradient(order, data) 
	# gra = first_gradient(data)
	flag = True
	j = 0
	while flag == True:
	# for k in range(100):
		if j == 2000:
			print(gra)
			print("weight:", w)
			print("b:", b)
			print("-------------------------------------------------")
			j = 0
		for i in range(len(w)):
			w_rate[i] += gra[i]**2
			w[i] -= rate / math.sqrt(w_rate[i]) * gra[i]
		b_rate += gra[len(gra)-1]**2
		b -= rate/ math.sqrt(b_rate)*gra[len(gra)-1]
		gra = gradient(order, data)
		j = j + 1
		# gra = first_gradient(data)
		# print(gra)
		flag = False
		for i in gra:
			if abs(i) >= tolerance:
				flag = True
				break
	print("weight:", w)
	print("b:", b)

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

init_w_rate(order)
init_w(order)
decent(order, train_data, gradient_func, e)

ans = []

for i in range(len(test_data)):
	ans.append(first_order(test_data[i]))
	test_data[i].append(ans[i])


with open("ans.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerow(["id", "value"])
	for i in range(len(ans)):
		row = ["id_"+str(i), ans[i]]
		writer.writerow(row)
print(loss_function(train_data))
end = time.time()
print("time: ", end - start)
