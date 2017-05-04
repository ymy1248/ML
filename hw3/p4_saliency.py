import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
from termcolor import colored,cprint
from utils import *
from sklearn.preprocessing import normalize

#base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#img_dir = os.path.join(base_dir, 'image')
#if not os.path.exists(img_dir):
#    os.makedirs(img_dir)
#cmap_dir = os.path.join(img_dir, 'cmap')
#if not os.path.exists(cmap_dir):
#    os.makedirs(cmap_dir)
#partial_see_dir = os.path.join(img_dir,'partial_see')
#if notos.path.exists(partial_see_dir):
#    os.makedirs(partial_see_dir)
#model_dir = os.path.join(base_dir, 'model')

def main():
    #parser = argparse.ArgumentParser(prog='plot_saliency.py',
    #        description='ML-Assignment3 visualize attention heat map.')
    #parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    #args = parser.parse_args()
    #model_name = "model-%s.h5" %str(args.epoch)
    #model_path = os.path.join(model_dir, model_name)
    #model = load_model(model_path)
    #print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels = pickle.load(open('x','rb')) 
    private_pixels = [private_pixels[i].reshape((1, 48, 48, 1)) 
            for i in range(len(private_pixels)) ]
    model_path = "model/final.h5"
    model = load_model(model_path)   
    input_img = model.input
    img_ids = list(range(0,40))
    for idx in img_ids:
        #x = pickle.load(open("x", "rb"))
        #x.reshape(len(x), 48, 48, 1)
        val_proba = model.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        # print('pred', pred)
        target = K.mean(model.output[:, pred])
        # print('model output: ', model.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        heatmap = np.array(fn([np.reshape(private_pixels[idx], (1, 48, 48, 1)), 1]))
        heatmap = heatmap[0, 0, :, :, 0]
        heatmap = np.abs(heatmap)
        heatmap /= np.std(heatmap)
        heatmap -= np.mean(heatmap)
        #heatmap = normalize(heatmap)
        heatmap /= 2
        heatmap += 0.5

        plt.figure()
        plt.imshow(private_pixels[idx].reshape(48,48), cmap = 'gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('p4', '{}.png'.format(idx)), dpi=100)

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)

        
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('p4', '{}_saliency.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('p4', '{}_mask.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()
