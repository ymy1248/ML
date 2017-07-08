import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
	rotation_range = 40,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = "nearest")
x = pickle.load(open("x","rb"))
# x = x/255
print(x.shape)
a = x[1].reshape((1,) + x[1].shape)
b = x.reshape(x.shape[0],48,48)
i = 0
for batch in datagen.flow(a, batch_size = 1, save_to_dir='preview', save_prefix='face', save_format='jpeg'):
	i += 1
	if i > 20:
		break
plt.imshow(b[1])
plt.show()