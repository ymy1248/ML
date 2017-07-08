import sys
import csv
import numpy as np
import pickle
import shutil
import os
import scipy.misc as m
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.activations import relu
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

class CNN:
	def __init__(self, cnn, dnn, dropout, regu, rota, zoom, flip, border, opt, title):
		self.score = 0
		self.rota = rota
		self.zoom = zoom
		self.title = title
		self.flip = flip

		reg = regularizers.l2(regu)
		self.model = Sequential()

		self.model.add(Conv2D(cnn[1],(cnn[2],cnn[2]),padding=border, input_shape = (48,48,1)))
		self.model.add(Activation('relu'))
		for i in range(len(cnn)//4):
			for j in range(cnn[i*4]):
					self.model.add(Conv2D(cnn[i*4 + 1],(cnn[i*4+2],cnn[i*4+2]),padding=border))
					self.model.add(Activation('relu'))
			if cnn[i*4] != 0 or i == 0:
				self.model.add(MaxPooling2D(pool_size=(cnn[i*4+3], cnn[i*4+3])))
				self.model.add(Dropout(dropout))

		self.model.add(Flatten())
		
		for i in range(dnn[0]):
			self.model.add(Dense(units = dnn[1], kernel_regularizer = reg))
			self.model.add(Activation('relu'))

		self.model.add(Dense(units = 7))
		self.model.add(Activation('softmax'))
		self.model.summary()
		self.model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ["accuracy"])

		self.reg_model_name = str(self.score) + "_" + title
		self.model.save_weights(self.reg_model_name + ".h5")



	def fit(self, x, y, nb_train_samples):
		class_weight = {0:1.8, 1:16.54, 2:1.76, 3:1, 4:1.49, 5:2.28, 6:1.45}
		if self.zoom == 0 and self.rota == 0 and self.flip == False:
			his = self.model.fit(x, y, batch_size = 100, epochs = 1, validation_split = 0.05,class_weight = class_weight)

		else:
			batch_size = 512
			epochs = 1
			nb_validation_samples = 2709
			datagen = ImageDataGenerator(
				width_shift_range = 0.05,
				height_shift_range = 0.05,
				rotation_range = self.rota,
				zoom_range = self.zoom,
				horizontal_flip = self.flip,
				fill_mode = "nearest"
				)
			test_datagen = ImageDataGenerator(
				rotation_range = 0,
				zoom_range = 0.0,
				horizontal_flip = self.flip,
				fill_mode = "nearest")
			train_gen = datagen.flow_from_directory(
				"data/train",
				color_mode = "grayscale",
				target_size = (48,48),
				batch_size = batch_size)
			validation_generator = test_datagen.flow_from_directory(
			    "data/validation",
			    color_mode = "grayscale",
			    target_size = (48,48),
			    batch_size=batch_size
			    )
			his = self.model.fit_generator(train_gen,
				steps_per_epoch = nb_train_samples // batch_size,
				validation_data=validation_generator,
				validation_steps=nb_validation_samples // batch_size,
				# class_weight = class_weight,
				epochs=epochs)

		if his.history["val_acc"][0] > 0.67:
			self.model.save(str(his.history["val_acc"][0]) + '_' + str(his.history["val_loss"][0]) + "_" + self.title+ ".h5")

		if his.history["val_acc"][0] > self.score:
			os.remove(self.reg_model_name + ".h5")
			self.reg_model_name = str(his.history["val_acc"][0]) + "_" + self.title
			self.model.save_weights(self.reg_model_name + ".h5")
			self.score = his.history["val_acc"][0]

		if his.history["val_loss"][0] < 0.94:
			self.model.save(str(his.history["val_acc"][0]) + '_' + str(his.history["val_loss"][0]) + "_" + self.title+ ".h5")

		return his.history["val_acc"][0], his.history["val_loss"][0], his.history["acc"][0]

	def store(self, vali_loss):
		os.remove(self.reg_model_name + ".h5")
		self.model.save(str(self.score) + '_' + str(vali_loss) + "_" + self.title+ ".h5")

	def predict(self, x):
		return self.model.predict(x, batch_size = 100)

	def roll_back(self):
		self.model.load_weights(self.reg_model_name + ".h5")



if __name__ == "__main__":
	TRAIN = 26000
	SELF_TRAIN_THRE = 0.85
	TRAIN_STOP_THRE = 25
	# CONFIG = {}
	TRAIN_FILE = sys.argv[1]
	with open(TRAIN_FILE, "r") as f:
		xyStr = list(csv.reader(f))
	y = []
	x = []
	# for i in range(1, len(xyStr)):
	# 	y.append(float(xyStr[i][0]))
	# 	xStr = xyStr[i][1].split(" ")
	# 	x.append(list(map(float, xStr)))
	# x = np.array(x, dtype = np.float32)
	# x = x.reshape(x.shape[0],48,48,1)
	# y = np.array(y, dtype = np.int)
	# pickle.dump(x, open("x", "wb"))
	# pickle.dump(y, open("y", "wb"))
	xTest = pickle.load(open("xTest", "rb"))
	x = pickle.load(open("x", "rb"))
	y = pickle.load(open("y", "rb"))

	all_x = np.concatenate((x,xTest), axis = 0)
	x_mean = np.mean(all_x, axis = 0)
	x_var = np.std(all_x, axis = 0)
	# x -= x_mean
	# x /= x_var
	count = [0]*7
	for i in y:
		count[i] += 1
	print(count)

	# print("x_mean:", x_mean)
	# print("x_var:", x_var)
	model_score = pickle.load(open("model_score", "rb"))
	randomize = np.arange(len(x))
	np.random.shuffle(randomize)
	x = x[randomize]
	y = y[randomize]
	xT = x[:TRAIN]
	yT = y[:TRAIN]
	xV = x[TRAIN:]
	yV = y[TRAIN:]

	# cnn, dnn, dropout, regu, rota, zoom, flip, border, opt,
	config_list = [
	[[1, 64, 3, 2, 2, 128, 3, 2, 2, 256, 3, 2, 3, 512, 3, 2], [2, 64], 0.4, 0.01, 20, 0.1, True, 'same', 'adam'],
	]
	best_vali = []
	message = 'Model Finished!!!\n'
	# for i in range(10):
	for CONFIG in config_list:
		TRAIN = 26000
		randomize = np.arange(len(x))
		np.random.shuffle(randomize)
		xTest = pickle.load(open("xTest", "rb"))
		# xTest -= x_mean
		# xTest /= x_var
		# print("x_mean:", x_mean)
		# print("x_var:", x_var)
		x = x[randomize]
		y = y[randomize]
		xT = x[:TRAIN]
		yT = y[:TRAIN]
		xV = x[TRAIN:]
		yV = y[TRAIN:]
		val_data = []
		val_data.append(xV)
		val_data.append(yV)
		pickle.dump(val_data, open(str(CONFIG) + "_val", "wb"))

		xT = xT.reshape(xT.shape[0],48,48)
		xV = xV.reshape(xV.shape[0],48,48)

		if os.path.isdir('data'):
			shutil.rmtree('data')

		for i in range(7):
			os.makedirs('data/train/' + str(i))

		for i in range(7):
			os.makedirs('data/validation/' + str(i))

		for i in range(len(xT)):
			m.imsave("data/train/"+str(yT[i])+"/" + str(i)+".jpg", xT[i])

		for i in range(len(xV)):
			m.imsave("data/validation/"+str(yV[i])+"/" + str(i)+".jpg", xV[i])

		x = x.reshape(x.shape[0],48,48, 1)
		message += str(CONFIG) + "\n"
		model_input = CONFIG.copy()
		model_input.append(str(CONFIG))
		model = CNN(*model_input)
		model_score[str(CONFIG)] = {"vali":[], "vali_loss":[],"train":[]}
		best = 0.0
		min_loss = float("inf")
		count = 0
		last_train_acc = 0.0
		best_train = 0.0
		ans = []

		while True:
			count = 0
			delete_list = []
			for i in range(len(ans)):
				if np.max(ans[i]) >= SELF_TRAIN_THRE:
					index = np.argmax(ans[i])
					xSave = xTest[i].reshape(48,48)
					m.imsave("data/train/"+str(index)+"/" + str(TRAIN)+".jpg", xSave)
					TRAIN += 1
					delete_list.append(i)
			xTest = np.delete(xTest, delete_list, 0)
			while count < TRAIN_STOP_THRE:
				print("count:", count)
				y_ng = np_utils.to_categorical(y, 7)
				vali_acc, vali_loss, train_acc = model.fit(x, y_ng, TRAIN)
				model_score[str(CONFIG)]["vali"].append(vali_acc)
				model_score[str(CONFIG)]["vali_loss"].append(vali_loss)
				model_score[str(CONFIG)]["train"].append(train_acc)
				best_train = max(best_train, train_acc)
				if train_acc >= last_train_acc - 0.0001 and train_acc <= last_train_acc + 0.0001 and train_acc < 0.4:
					count += 4
				elif vali_acc > best:
					best = vali_acc
					min_loss = vali_loss
					count = 0
				else:
					count += 1
				last_train_acc = train_acc
			model.roll_back()
			ans = model.predict(xTest)
			if np.max(ans) < SELF_TRAIN_THRE:
				break

		best_vali.append(best)
		model.store(min_loss)
		message += "vali:" + str(best) + "\n"
		message += "train:" + str(train_acc) + "\n"

	pickle.dump(best_vali, open("best_vali", "wb"))
	pickle.dump(model_score, open("model_score", "wb"))

	def send_email(user, pwd, recipient, subject, body):
		import smtplib

		gmail_user = user
		gmail_pwd = pwd
		FROM = user
		TO = recipient if type(recipient) is list else [recipient]
		SUBJECT = subject
		TEXT = body

		# Prepare actual message
		message = """From: %s\nTo: %s\nSubject: %s\n\n%s
		""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
		try:
			server = smtplib.SMTP("smtp.gmail.com", 587)
			server.ehlo()
			server.starttls()
			server.ehlo()
			server.login(gmail_user, gmail_pwd)
			server.sendmail(FROM, TO, message)
			server.close()
			print("successfully sent the mail")
		except Exception as e:
			print(e)
			print("failed to send mail")

	send_email("yemengyuan0405", "********", "carlosyex@gmail.com", "r04921094@ntu.edu.tw", message)
