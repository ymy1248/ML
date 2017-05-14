import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

FACE_PATH = '../../ML2017_data/hw4/face/'

images = []
for i in range(10):
    for j in range(10):
        images.append(misc.imread(FACE_PATH + chr(ord('A') + i) + '0' + str(j) + '.bmp'))

mean = np.mean(images, axis = 0)
plt.imsave('mean.png', mean, cmap = 'gray')
mean = mean.reshape(4096)
images = np.array(images, dtype = np.float64)
images = images.reshape(100, 4096)
print('mean shape:', mean.shape)
images_con = images - mean
images_con = images_con.reshape(100, 4096)
U, s, V = np.linalg.svd(images_con)

#fig = plt.figure(figsize = (64,64))
#for i in range(9):
#    subplot = fig.add_subplot(3,3,i+1)
#    #subplot.imshow(eigen_face[:,i].reshape(64,64), cmap = 'gray')
#    subplot.imshow(V[i].reshape(64, 64), cmap = 'gray')
#fig.suptitle('Top 9 eigenface')
#fig.savefig('eigen_face_t.png')
#
#fig = plt.figure(figsize = (64,64))
#for i in range(100):
#   subplot = fig.add_subplot(10,10,i+1)
#   subplot.imshow(images[i].reshape(64,64), cmap = 'gray')
#fig.savefig('origional.png')

def re_func (eig_num):
    image_reduced = []
    for i in range(100):
        reduced_list = []
        for j in range(eig_num):
            reduced = np.dot((images[i] - mean).T, V[j])
            reduced_list.append(reduced)
        image_reduced.append(np.array(reduced_list))
    image_reduced = np.array(image_reduced)

    reconstruct = []
    for i in range(100):
        s = np.zeros(4096)
        for j in range(eig_num):
            s += np.dot(image_reduced[i][j], V[j])
        reconstruct.append(s + mean)
    reconstruct = np.array(reconstruct)
    return reconstruct

reconstruct = re_func(5)
fig = plt.figure(figsize = (64,64))
for i in range(100):
   subplot = fig.add_subplot(10,10,i+1)
   subplot.imshow(reconstruct[i].reshape(64,64), cmap = 'gray')
fig.savefig('reconstruct by 5 eigen faces.png')

eigen_num = 1 
while True:
    reimages = re_func(eigen_num)
    rmse = (np.mean(((reimages - images) / 256)**2))**0.5
    if rmse < 0.01:
        break
    else:
        eigen_num += 1
print(eigen_num)
