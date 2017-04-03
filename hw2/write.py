import csv
import pickle
import logistic as log

model = pickle.load(open("logi_2_2200_0.74921875", "rb"))
model.write()