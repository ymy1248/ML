import word2vec
import nltk
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
from argparse import ArgumentParser
from sklearn.manifold import TSNE

PLOT_NUM = 1000 
word2vec.word2phrase('/home/ymy1248/Code/ML2017_data/hw4/all.txt', '/home/ymy1248/Code/ML2017_data/hw4/all_phrases.txt', verbose = True)
word2vec.word2vec('/home/ymy1248/Code/ML2017_data/hw4/all_phrases.txt', '/home/ymy1248/Code/ML2017_data/hw4/hp.bin', size = 100, verbose = True)

word_model = word2vec.load('/home/ymy1248/Code/ML2017_data/hw4/hp.bin')
words = []
vecs = []
for vocab in word_model.vocab:
    words.append(vocab)
    vecs.append(word_model[vocab])
words = np.array(words)[:PLOT_NUM]
vecs = np.array(vecs)[:PLOT_NUM]

tsne = TSNE(n_components = 2)
reduced_vecs = tsne.fit_transform(vecs)
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':',';', ',', '?', '!', u"’"]

plt.figure()
texts = []
for i, label in enumerate(words):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags and all(c not in label for c in puncts)):
        x, y = reduced_vecs[i, :]
       	texts.append(plt.text(x, y, label))
       	plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.show()
