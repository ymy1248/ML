from keras.models import load_model
from keras.utils import plot_model

model_path = 'model/final.h5'
model = load_model(model_path)
plot_model(model, to_file = 'model.png')
