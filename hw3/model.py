import pickle

score = pickle.load(open("model_score", "rb"))

for model, value in  score.items():
	