import os
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from keras import backend as K
from utils import *
from scipy.misc import imsave
# from marcos import *
import numpy as np

model_path = 'model/final.h5'
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    for i in range(1300):
        loss_value, grads_value = iter_func([input_image_data])
        input_image_data += grads_value * 1 
    return input_image_data 

def plot_filter():
    fig = plt.figure(figsize = (14, 8))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(cnn_filter(i), cmap = 'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel(str(i))
        plt.tight_layout()
    fig.suptitle('conv2d_2')
    fig.savefig('p5/1.png')
        
def cnn_filter(index):
    model = load_model(model_path)
    input_img = model.input
    layer_name = 'conv2d_2'
    layer_dict = dict([layer.name, layer] for layer in model.layers[1:])
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, index])
    grads = normalize(K.gradients(loss, input_img)[0])
    iterate = K.function([input_img], [loss,grads])
    input_image_data = np.random.random((1, 48, 48, 1)) * 20 + 128.
    result_pic = grad_ascent(10, input_image_data, iterate) 

    img = result_pic[0]
    img = deprocess_image(img)
    img = img.reshape(48, 48)
    imsave('p5/%d.png' % index, img)
    return img
#def cnn_filter():
#    NUM_STEPS = 10
#    RECORD_FREQ = 2
#    nb_filter = 64 
#    model_path = 'model/final.h5'
#    emotion_classifier = load_model(model_path)
#    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
#    input_img = emotion_classifier.input
#
#    name_ls = ["conv2d_2"]
#    collect_layers = [ layer_dict[name].output for name in name_ls ]
#
#    for cnt, c in enumerate(collect_layers):
#        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
#        for filter_idx in range(nb_filter):
#            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
#            target = K.mean(c[:, :, :, filter_idx])
#            grads = normalize(K.gradients(target, input_img)[0])
#            iterate = K.function([input_img], [target, grads])
#
#            filter_imgs = grad_ascent(, input_img_data, iterate)
#
#        for it in range(NUM_STEPS//RECORD_FREQ):
#            fig = plt.figure(figsize=(14, 8))
#            for i in range(nb_filter):
#                ax = fig.add_subplot(nb_filter/16, 16, i+1)
#                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
#                plt.xticks(np.array([]))
#                plt.yticks(np.array([]))
#                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
#                plt.tight_layout()
#            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
#            img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
#            if not os.path.isdir(img_path):
#                os.mkdir(img_path)
#            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ))) 

def filter_out():
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_2"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    private_pixels = pickle.load(open('x', 'rb'))
    choose_id = 20 
    photo = private_pixels[choose_id]
    photo = photo.reshape(1, 48, 48, 1)
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.savefig(os.path.join('p5','layer{}'.format(cnt)))
if __name__ == "__main__":
    filter_out()
    plot_filter()
