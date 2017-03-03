# PM2.5 is an interger value

import csv
import math

w = []
b = 0.0
e = 1					# tolerance
rate = 10
b_rate = 0.0
w_rate = []
order = 1


def init_w (order):
	global w
	global w_rate
	w = [0.0] * (order*9)
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
		diff = d[9] - first_order(d)
		for i in range(len(d) - 1):
			gradient[i] += 2*diff*(-d[i])
		gradient[9] += -2 * diff
	return gradient

def loss_function(data, model_func = first_order):
	loss = 0.0
	for d in data:
		loss +=  (data[9] - model_func(data[:9]))**2
	return loss

def decent(data, gradient = first_gradient, tolerance = 0.00001):
	# TODO 
	global w
	global b
	global b_rate
	gra = gradient(data) 
	flag = True
	while flag == True:
		for i in range(len(w)):
			w_rate[i] += gra[i]**2
			w[i] -= rate / math.sqrt(w_rate[i]) * gra[i]
		b_rate += gra[len(gra)-1]**2
		b -= rate/ math.sqrt(b_rate)*gra[len(gra)-1]
		gra = gradient(data)
		flag = False
		for i in gra:
			if abs(i) >= tolerance:
				flag = True
				break
	print("gradient:",gra)
	print("b:", b)
	print("weight:", w)

file_name = "train.csv"
with open(file_name, "r") as f:
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
init_w(order)
decent(train_data, first_gradient, e)