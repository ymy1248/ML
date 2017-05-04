import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,title='confusion matrix',cmap=plt.cm.jet):
    """
    this function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('true label')
            plt.xlabel('predicted label')

if __name__ == "__main__":
    model_path = "model/final.h5"
    val_data_path = "final_val"
    model = load_model(model_path)
    np.set_printoptions(precision=2)
    val_data = pickle.load(open(val_data_path, "rb"))
    x = val_data[0]
    y = val_data[1]
    predictions = model.predict_classes(x)
    conf_mat = confusion_matrix(y,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()
