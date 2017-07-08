import sys
import csv
import pickle
import numpy as np
import keras.backend as K
from collections import OrderedDict
from argparse import ArgumentParser
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Conv1D, Flatten,MaxPooling1D, Dropout, GRU, Merge
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import MLutil

VALI = 0.9
MAX_NB_WORDS = 20000

parser = ArgumentParser()
parser.add_argument('--test') 
parser.add_argument('--csv')
parser.add_argument('--crazytest')
parser.add_argument('--crazyout')
parser.add_argument('--train', action = 'store_true')
parser.add_argument('--pre', action = 'store_true')
parser.add_argument('--crazy', action = 'store_true')
parser.add_argument('--bag', action = 'store_true')
args = parser.parse_args()

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

if args.test:
    test_file = open(args.test, 'r', encoding = 'utf-8')
    content = test_file.readlines()
    test_in = [x.strip() for x in content]  
    print('test line num:', len(test_in))
    tokenizer = pickle.load(open('tokenizer', 'rb'))
    test_in = tokenizer.texts_to_sequences(test_in)
    test_in = sequence.pad_sequences(test_in, maxlen = 500)
    model = load_model("./model/0.5200_['RNN', [128, 128, 128], [0.4, 0.4, 0.4], [256, 128, 64], [0.1, 0.1, 0.1], True, 'GRU', 200, 'rmsprop', 64].h5",
            custom_objects = {'f1_score': f1_score})
    test_out = model.predict(test_in[1:])
    test_out = np.round(test_out)
    num_list = pickle.load(open('num_list', 'rb'))

    print(num_list)
    with open('ans.csv', 'w') as f:
        writer = csv.writer(f, lineterminator = '\n', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['id', 'tags'])
        for index in range(len(test_out)):
            tags = '' 
            first_tag = True
            for j in range(len(test_out[index])):
                if test_out[index][j] == 1:
                    tag = num_list[int(j)]
                    if tag == "CHILDREN'S-LITERATURE":
                        tag = """CHILDREN''S-LITERATURE"""
                    if first_tag:
                        tags += num_list[int(j)]
                        first_tag = False
                    else:
                        tags += ' ' + num_list[int(j)]
            #tags = '\"' + tags + '\"'
            writer.writerow([str(index), tags])

if args.pre:
    TRAIN_PATH = '/home/ymy1248/Code/ML2017_data/hw5/train_data.csv'
    TEST_PATH = '/home/ymy1248/Code/ML2017_data/hw5/test_data.csv'
    train_file = open(TRAIN_PATH, 'r', encoding = 'utf-8')
    content = train_file.readlines()
    train = [x.strip() for x in content]
    train = [str(x).split(',') for x in train]
    test_file = open(TEST_PATH, 'r', encoding = 'utf-8')
    content = test_file.readlines()
    test = [x.strip() for x in content]
    test = [str(x).split(',') for x in train]
    y = np.zeros((len(train) - 1, 38)) 
    y_count = np.zeros(38) 
    x = []
    label_dic = {}
    num_list = [] 
    label_count = 0
    max_str = 0
    
    for index in range(1,len(train)):
        label = train[index][1][1:-1].split(' ')
        for l in label:
            if l not in label_dic:
                label_dic[l] = label_count
                num_list.append(l)
                label_count += 1
            y[index-1][label_dic[l]] = 1 
            y_count[label_dic[l]] += 1
        x_str = train[index][2]
        for x_str_sub in train[index][3:]:
            x_str += x_str_sub
        x.append(x_str)
    print(y_count)
    print(label_dic)
    y_count = np.full(38, 1672)/y_count
    y_weight = {}
    for i in range(len(y_count)):
        y_weight[i] = y_count[i]
    print(y_weight)
    pickle.dump(num_list, open('num_list', 'wb'))
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    #pickle.dump(x, open('x', 'wb'))
    #pickle.dump(y, open('y', 'wb'))
    #pickle.dump(num_list, open('num_list', 'wb'))
    #pickle.dump(tokenizer, open('tokenizer', 'wb'))
    #pickle.dump(y_weight, open('y_weight', 'wb'))
    #
if args.train:
    x = pickle.load(open('x', 'rb'))
    y = pickle.load(open('y', 'rb'))
    y_weight = pickle.load(open('y_weight', 'rb'))
    
    storer = MLutil.Storer('rnn_model')
    # [RNN/CNN, GRU/LSTM, rnn units, rnn dropout, dnn units, dnn drop, Bidirection, optimizer, batch]
    configure = [
            ['RNN', [128, 128, 128], [0.4, 0.4, 0.4], [256, 128, 64], [0.1, 0.1, 0.1], False, 'GRU', 200, 'rmsprop', 200],
            ]
    for conf in configure:
        ROC = conf[0]

        rnn_units = conf[1]
        rnn_drop = conf[2]
        rnn_dnn_units = conf[3]
        rnn_dnn_drop = conf[4]
        bi = conf[5]
        GOL = conf[6]
        word_dim = conf[7]
        opt = conf[8]
        batch = conf[9]

        cnn_filters = conf[1]
        cnn_kernel = conf[2]
        cnn_drop = conf[3]
        cnn_pooling = conf[4]
        cnn_dnn_units = conf[5]
        cnn_dnn_drop = conf[6]

        tokenizer = pickle.load(open('tokenizer', 'rb'))
        word_index = tokenizer.word_index
        embeddings_index = {}
        f = open('/home/ymy1248/Code/ML2017_data/hw5/glove.6B.' + str(word_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = coefs
        f.close()

        num_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((num_words, word_dim))
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        TRAIN = int(VALI * len(x))

        x = sequence.pad_sequences(x, maxlen = 500)
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        x_train = x[:TRAIN]
        y_train = y[:TRAIN]
        x_val = x[TRAIN:]
        y_val = y[TRAIN:]

        model = Sequential()
        model.add(Embedding(num_words, word_dim, 
            weights = [embedding_matrix],
            input_length = 500, 
            trainable = False, 
            input_shape = (500,)))

        try:
            if ROC == 'RNN':
                if GOL == 'LSTM':
                    if bi == True:
                        for rnn_i in range(len(rnn_units[:-1])):
                            model.add(LSTM(units = rnn_units[rnn_i], 
                                dropout = rnn_drop[rnn_i], 
                                return_sequences = True))
                        model.add(Bidirectional(LSTM(units = rnn_units[-1], dropout = rnn_drop[-1])))
                    else:
                        for rnn_i in range(len(rnn_units[:-1])):
                            model.add(LSTM(units = rnn_units[rnn_i], 
                                dropout = rnn_drop[rnn_i], 
                                return_sequences = True))
                        model.add(LSTM(units = rnn_units[-1], dropout = rnn_drop[-1])) 

                if GOL == 'GRU':
                    if bi == True:
                        for rnn_i in range(len(rnn_units[:-1])):
                            model.add(GRU(units = rnn_units[rnn_i], 
                                dropout = rnn_drop[rnn_i], 
                                return_sequences = True))
                        model.add(Bidirectional(GRU(units = rnn_units[-1], dropout = rnn_drop[-1])))
                    else:
                        for rnn_i in range(len(rnn_units[:-1])):
                            model.add(GRU(units = rnn_units[rnn_i], 
                                dropout = rnn_drop[rnn_i], 
                                return_sequences = True))
                        model.add(GRU(units = rnn_units[-1], dropout = rnn_drop[-1])) 
                
                for dnn_i in range(len(rnn_dnn_units)):
                    model.add(Dense(units = rnn_dnn_units[dnn_i]))
                    model.add(Dropout(rnn_dnn_drop[dnn_i]))

            if ROC == 'CNN':
                for cnn_i in range(len(cnn_filters)):
                    model.add(Conv1D(cnn_filters[cnn_i], cnn_kernel[cnn_i]))
                    model.add(Dropout(cnn_drop[cnn_i]))
                    model.add(MaxPooling1D(cnn_pooling[cnn_i]))

                model.add(Flatten())
                for dnn_i in range(len(cnn_dnn_units)):
                    model.add(Dense(cnn_dnn_units[dnn_i]))
                    model.add(Dropout(cnn_dnn_drop[dnn_i]))

            model.add(Dense(units = 38, activation = 'softmax'))
            model.summary()

            model.compile(optimizer = opt, 
                    loss = 'categorical_crossentropy', 
                    metrics = ['accuracy', f1_score])
            earlystopping = EarlyStopping(monitor = 'val_f1_score', 
                    patience = 21, 
                    verbose = 1, 
                    mode = 'max')
            checkpoitn = ModelCheckpoint('./model/{val_f1_score:.4f}_{epoch:02d}_' + str(conf) + '.h5', 
                    monitor='val_f1_score', 
                    save_best_only=True, 
                    verbose=0, 
                    mode = 'max')
            reduce_lr = ReduceLROnPlateau(factor = 0.5, patience=7)
            his = model.fit(x_train, y_train, 
                    #class_weight = y_weight,
                    batch_size = batch, 
                    epochs = 1000, 
                    validation_data = (x_val, y_val), 
                    callbacks = [earlystopping, checkpoitn, reduce_lr])
        except:
            model_dict = OrderedDict()
            model_dict['loss'] = float('nan')
            model_dict['acc'] = float('nan')
            model_dict['f1_score'] = float('nan')
            model_dict['val_loss'] = float('nan')
            model_dict['val_acc'] = float('nan')
            model_dict['val_f1_scores'] = float('nan')
            storer.store(str(conf), model_dict) 
            continue

        model_dict = OrderedDict()
        model_dict['loss'] = his.history['loss']
        model_dict['acc'] = his.history['acc']
        model_dict['f1_score'] = his.history['f1_score']
        model_dict['val_loss'] = his.history['val_loss']
        model_dict['val_acc'] = his.history['val_acc']
        model_dict['val_f1_scores'] = his.history['val_f1_score']
        storer.store(str(conf) + '_softmax', model_dict) 
    storer.close()

if args.crazy:
    ensamble = 7
    sample = 0.7
    x = pickle.load(open('x', 'rb'))
    y = pickle.load(open('y', 'rb'))
    y_weight = pickle.load(open('y_weight', 'rb'))
    
    storer = MLutil.Storer('rnn_model')
    model_list = []
    # [RNN/CNN, GRU/LSTM, rnn units, rnn dropout, dnn units, dnn drop, Bidirection, optimizer, batch]
    conf = ['RNN', [129, 128, 128], [0.4, 0.4, 0.4], [256, 128, 64], [0.1, 0.1, 0.1], True, 'GRU', 200, 'rmsprop', 150]
    for model_i in range(14, 100):
        ROC = conf[0]

        rnn_units = conf[1]
        rnn_drop = conf[2]
        rnn_dnn_units = conf[3]
        rnn_dnn_drop = conf[4]
        bi = conf[5]
        GOL = conf[6]
        word_dim = conf[7]
        opt = conf[8]
        batch = conf[9]

        cnn_filters = conf[1]
        cnn_kernel = conf[2]
        cnn_drop = conf[3]
        cnn_pooling = conf[4]
        cnn_dnn_units = conf[5]
        cnn_dnn_drop = conf[6]

        tokenizer = pickle.load(open('tokenizer', 'rb'))
        word_index = tokenizer.word_index
        embeddings_index = {}
        f = open('/home/ymy1248/Code/ML2017_data/hw5/glove.6B.' + str(word_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = coefs
        f.close()

        num_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((num_words, word_dim))
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        TRAIN = int(sample * len(x))

        x = sequence.pad_sequences(x, maxlen = 500)
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]
        x_train = x[:TRAIN]
        y_train = y[:TRAIN]
        x_val = x[TRAIN:]
        y_val = y[TRAIN:]

        model = Sequential()
        model.add(Embedding(num_words, word_dim, 
            weights = [embedding_matrix],
            input_length = 500, 
            trainable = False, 
            input_shape = (500,)))

        if ROC == 'RNN':
            if GOL == 'LSTM':
                if bi == True:
                    for rnn_i in range(len(rnn_units[:-1])):
                        model.add(LSTM(units = rnn_units[rnn_i], 
                            dropout = rnn_drop[rnn_i], 
                            return_sequences = True))
                    model.add(Bidirectional(LSTM(units = rnn_units[-1], dropout = rnn_drop[-1])))
                else:
                    for rnn_i in range(len(rnn_units[:-1])):
                        model.add(LSTM(units = rnn_units[rnn_i], 
                            dropout = rnn_drop[rnn_i], 
                            return_sequences = True))
                    model.add(LSTM(units = rnn_units[-1], dropout = rnn_drop[-1])) 

            if GOL == 'GRU':
                if bi == True:
                    for rnn_i in range(len(rnn_units[:-1])):
                        model.add(GRU(units = rnn_units[rnn_i], 
                            dropout = rnn_drop[rnn_i], 
                            return_sequences = True))
                    model.add(Bidirectional(GRU(units = rnn_units[-1], dropout = rnn_drop[-1])))
                else:
                    for rnn_i in range(len(rnn_units[:-1])):
                        model.add(GRU(units = rnn_units[rnn_i], 
                            dropout = rnn_drop[rnn_i], 
                            return_sequences = True))
                    model.add(GRU(units = rnn_units[-1], dropout = rnn_drop[-1])) 
            
            for dnn_i in range(len(rnn_dnn_units)):
                model.add(Dense(units = rnn_dnn_units[dnn_i]))
                model.add(Dropout(rnn_dnn_drop[dnn_i]))

        if ROC == 'CNN':
            for cnn_i in range(len(cnn_filters)):
                model.add(Conv1D(cnn_filters[cnn_i], cnn_kernel[cnn_i]))
                model.add(Dropout(cnn_drop[cnn_i]))
                model.add(MaxPooling1D(cnn_pooling[cnn_i]))

            model.add(Flatten())
            for dnn_i in range(len(cnn_dnn_units)):
                model.add(Dense(cnn_dnn_units[dnn_i]))
                model.add(Dropout(cnn_dnn_drop[dnn_i]))

        model.add(Dense(units = 38, activation = 'sigmoid'))
        model.summary()

        model.compile(optimizer = opt, 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy', f1_score])
        try:
            best_f1 = 0.0
            best_model_name = ''
            count = 0
            while True:
                his = model.fit(x_train, y_train,
                        batch_size = batch,
                        epochs = 1,
                        validation_data = (x_val, y_val))
                val_f1 = his.history['val_f1_score'][0]
                print('count:', count)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    count = 0
                    model.save('./model/{:d}_{:.4f}'.format(model_i, val_f1) + '.h5')
                    best_model_name = './model/{:d}_{:.4f}'.format(model_i, val_f1) + '.h5'
                else:
                    count += 1

                if count == 15:
                    model = load_model(best_model_name, custom_objects = {'f1_score': f1_score})
                    model_list.append(model)
                    break
        except:
            continue

    pickle.dump(model_list, open('crazy_model', 'wb'))

if args.crazytest:
    test_file = open(args.crazytest, 'r', encoding = 'utf-8')
    content = test_file.readlines()
    test_in = [x.strip() for x in content]  
    tokenizer = pickle.load(open('tokenizer', 'rb'))
    test_in = tokenizer.texts_to_sequences(test_in)
    test_in = sequence.pad_sequences(test_in, maxlen = 500)
    test_out = np.zeros((1234, 38))
    model_list = [
    load_model("./model/0.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/1.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/2.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/3.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/4.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/5.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/6.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/7.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/8.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/9.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/10.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/11.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/12.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/13.h5", custom_objects = {'f1_score': f1_score}),
    load_model("./model/14.h5", custom_objects = {'f1_score': f1_score}),
    ]

    for model in model_list:
        test_out += model.predict(test_in[1:], batch_size = 256)

    test_out /= len(model_list)
    test_out = np.round(test_out)
    num_list = pickle.load(open('num_list', 'rb'))

    with open(args.crazyout, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['id', 'tags'])
        for index in range(len(test_out)):
            tags = '' 
            first_tag = True
            for j in range(len(test_out[index])):
                if test_out[index][j] == 1:
                    tag = num_list[int(j)]
                    if tag == "CHILDREN'S-LITERATURE":
                        tag = """CHILDREN''S-LITERATURE"""
                    if first_tag:
                        tags += num_list[int(j)]
                        first_tag = False
                    else:
                        tags += ' ' + num_list[int(j)]
            #tags = '\"' + tags + '\"'
            writer.writerow([str(index), tags])

if args.bag:
    TRAIN_PATH = '/home/ymy1248/Code/ML2017_data/hw5/train_data.csv'
    TEST_PATH = '/home/ymy1248/Code/ML2017_data/hw5/test_data.csv'
    train_file = open(TRAIN_PATH, 'r', encoding = 'utf-8')
    content = train_file.readlines()
    train = [x.strip() for x in content]
    train = [str(x).split(',') for x in train]
    test_file = open(TEST_PATH, 'r', encoding = 'utf-8')
    content = test_file.readlines()
    test = [x.strip() for x in content]
    test = [str(x).split(',') for x in train]
    y = np.zeros((len(train) - 1, 38)) 
    y_count = np.zeros(38) 
    x = []
    label_dic = {}
    num_list = [] 
    label_count = 0
    max_str = 0
    
    for index in range(1,len(train)):
        label = train[index][1][1:-1].split(' ')
        for l in label:
            if l not in label_dic:
                label_dic[l] = label_count
                num_list.append(l)
                label_count += 1
            y[index-1][label_dic[l]] = 1 
            y_count[label_dic[l]] += 1
        x_str = train[index][2]
        for x_str_sub in train[index][3:]:
            x_str += x_str_sub
        x.append(x_str)
    print(y_count)
    print(label_dic)
    y_count = np.full(38, 1672)/y_count
    y_weight = {}
    for i in range(len(y_count)):
        y_weight[i] = y_count[i]
    print(y_weight)
    pickle.dump(num_list, open('num_list', 'wb'))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_matrix(x)
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    TRAIN = int(VALI * len(x))
    x = x[randomize]
    y = y[randomize]
    x_train = x[:TRAIN]
    y_train = y[:TRAIN]
    x_val = x[TRAIN:]
    y_val = y[TRAIN:]
    print('Shape of x:', x.shape)

    model = Sequential()
    model.add(Dense(64, input_shape = (x.shape[1],)))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(38, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy', f1_score])
    earlystopping = EarlyStopping(monitor = 'val_f1_score', 
            patience = 21, 
            verbose = 1, 
            mode = 'max')
    checkpoitn = ModelCheckpoint('./model/{val_f1_score:.4f}_{epoch:02d}_' + 'bag' + '.h5', 
            monitor='val_f1_score', 
            save_best_only=True, 
            verbose=0, 
            mode = 'max')
    reduce_lr = ReduceLROnPlateau(factor = 0.5, patience=7)
    his = model.fit(x_train, y_train, 
            #class_weight = y_weight,
            batch_size = 512, 
            epochs = 1000, 
            validation_data = (x_val, y_val), 
            callbacks = [earlystopping, checkpoitn, reduce_lr])
    storer = MLutil.Storer('rnn_model')
    model_dict = OrderedDict()
    model_dict['loss'] = his.history['loss']
    model_dict['acc'] = his.history['acc']
    model_dict['f1_score'] = his.history['f1_score']
    model_dict['val_loss'] = his.history['val_loss']
    model_dict['val_acc'] = his.history['val_acc']
    model_dict['val_f1_scores'] = his.history['val_f1_score']
    storer.store('bag of words', model_dict) 
    storer.close()
