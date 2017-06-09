import csv
import MLutil
import pickle
import numpy as np
import keras.backend as K
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Dropout, BatchNormalization
from keras.layers import Concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import OrderedDict

DIM = 250
USER_NUM = 6041
MOVIE_NUM = 3953
VALI = 0.9
RATING_MEAN = 3.58171208604

parser = ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--testout')
parser.add_argument('--entest')
parser.add_argument('--pre', action = 'store_true')
parser.add_argument('--en', action = 'store_true')
parser.add_argument('--train', action = 'store_true')
args = parser.parse_args()

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred-y_true) ** 2))

if args.test:
    #scaler = pickle.load(open('scaler', 'rb'))
    reader = csv.reader(open(args.test))
    test_data = list(reader)
    test_data = np.array(test_data[1:], dtype = np.dtype('float64'))
    model = load_model('./model/0.8540_006_240.h5', custom_objects = {'rmse':rmse})
    ans = model.predict(np.hsplit(test_data[:,1:], 2), batch_size = 512)
    ans += RATING_MEAN
    #ans = scaler.inverse_transform(ans)
    count = 0
    for i in ans:
        if np.isnan(i):
            count += 1
    ans[np.isnan(ans)] = RATING_MEAN
    with open(args.testout, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['TestDataID','Rating'])
        for i in range(len(ans)):
            writer.writerow([i+1, ans[i][0]])

if args.entest:
    reader = csv.reader(open(args.entest))
    test_data = list(reader)
    test_data = np.array(test_data[1:], dtype = np.dtype('float64'))
    model_list = []
    ans = np.zeros((100336,1))
    for en_i in range(50):
        model_list.append(load_model(
            './ensemble/' + '{:02d}'.format(en_i) + '.h5',
            custom_objects = {'rmse': rmse}))
    for model in model_list:
        ans += model.predict(np.hsplit(test_data[:,1:], 2), batch_size = 512)
    ans /= len(model_list)
    ans += RATING_MEAN
    ans += RATING_MEAN
    count = 0
    for i in ans:
        if np.isnan(i):
            count += 1
    ans[np.isnan(ans)] = RATING_MEAN
    with open(args.testout, 'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(['TestDataID','Rating'])
        for i in range(len(ans)):
            writer.writerow([i+1, ans[i][0]])

if args.pre:
    TRAIN_CSV = '../../ML2017_data/hw6/train.csv'
    USER_CSV = '../../ML2017_data/hw6/users.csv'
    MOVIE_CSV = '../../ML2017_data/hw6/movies.csv'
    reader = csv.reader(open(TRAIN_CSV))
    train_data = list(reader)
    train_data = np.array(train_data[1:], dtype = np.dtype('float64'))
    x = train_data[:, 1:3]
    y = train_data[:, 3]
    y -= RATING_MEAN

    user_sum = np.zeros((USER_NUM,))
    movie_sum = np.zeros((MOVIE_NUM,))
    user_len = np.zeros((USER_NUM,))
    movie_len = np.zeros((MOVIE_NUM,))
    
    for row in train_data:
        user_sum[int(row[1])] += row[3]
        user_len[int(row[1])] += 1
        movie_sum[int(row[2])] += row[3]
        movie_len[int(row[2])] += 1
    user_b = user_sum / user_len - RATING_MEAN
    movie_b = movie_sum / movie_len - RATING_MEAN

    reader = csv.reader(open(USER_CSV))
    user_data = list(reader)
    gender = {'F':0, 'M':1}
    user_age = np.zeros((USER_NUM,))
    user_occ = np.zeros((USER_NUM,))
    user_gender = np.zeros((USER_NUM,))

    for row in user_data[1:]:
        user_id = int(row[0])
        user_gen = row[1]
        age = int(row[2])
        occ = int(row[3])
        if age <= 12:
            age_group = 0
        elif age <= 18:
            age_group = 1
        elif age <= 24:
            age_group = 2
        elif age <= 30:
            age_group = 3
        elif age <= 45:
            age_group = 4
        elif age <= 60:
            age_group = 5
        else:
            age_group = 6
        user_gender[user_id] = gender[user_gen]
        user_age[user_id] = age_group
        user_occ[user_id] = occ

    content = open(MOVIE_CSV, 'r', encoding = 'latin-1').readlines()
    movie_data = [x.strip() for x in content]
    movie_data = [str(x).split('::') for x in movie_data]
    genres_list = ['dumy']
    movie_gen = np.zeros((MOVIE_NUM,6))
    for row in movie_data[1:]:
        movie_id = int(row[0])
        genres = row[2].split('|')
        for gen_i in range(len(genres)):
            if genres[gen_i] not in genres_list:
                genres_list.append(genres[gen_i])
            movie_gen[movie_id][gen_i] = genres_list.index(genres[gen_i])


    pickle.dump(x, open('x', 'wb'))
    pickle.dump(y, open('y', 'wb'))
    pickle.dump(user_b, open('user_b', 'wb'))
    pickle.dump(movie_b, open('movie_b', 'wb'))
    pickle.dump(user_gender, open('user_gender', 'wb'))
    pickle.dump(user_age, open('user_age', 'wb'))
    pickle.dump(user_occ, open('user_occ', 'wb'))
    pickle.dump(movie_gen, open('movie_gen', 'wb'))

    #print(train_data.shape)
    #print(train_data.shape)
    #print('User ID, min:', np.min(train_data[:, 0]), 'max:', np.max(train_data[:, 0]))
    #print('Movie ID, min:', np.min(train_data[:, 1]), 'max:', np.max(train_data[:, 1]))

if args.train:
    DIM = 240
    storer = MLutil.Storer('mf')
    model_name = 'implicit'
    x = pickle.load(open('x', 'rb'))
    y = pickle.load(open('y', 'rb'))
    user_b = pickle.load(open('user_b', 'rb'))
    movie_b = pickle.load(open('movie_b', 'rb'))
    user_occ = pickle.load(open('user_occ', 'rb'))
    user_age = pickle.load(open('user_age', 'rb'))
    user_gender = pickle.load(open('user_gender', 'rb'))
    movie_gen = pickle.load(open('movie_gen', 'rb'))
    user_b = user_b.reshape(user_b.shape[0], 1)
    movie_b = movie_b.reshape(movie_b.shape[0], 1)
    user_occ = user_occ.reshape(user_occ.shape[0], 1)
    user_age = user_age.reshape(user_age.shape[0], 1)
    user_gender = user_gender.reshape(user_gender.shape[0], 1)
    #y = StandardScaler().fit_transform(y)

    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    
    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))
    user_v = Embedding(input_dim = USER_NUM, 
            output_dim = DIM, 
            embeddings_regularizer = l2(0.000005),
            input_length = 1)(user_input)
    movie_v = Embedding(input_dim = MOVIE_NUM, 
            output_dim = DIM, 
            embeddings_regularizer = l2(0.000005),
            input_length = 1)(movie_input)

    #movie_gen_embed = Embedding(input_dim = MOVIE_NUM, 
    #        output_dim = 6, 
    #        weights = [movie_gen], 
    #        trainable = False)(movie_input)
    movie_bias = Embedding(input_dim = MOVIE_NUM,
            output_dim = 1,
            weights = [movie_b],
            trainable = False)(movie_input)
    user_bias = Embedding(input_dim = USER_NUM, 
            output_dim = 1, 
            weights = [user_b], 
            trainable = True)(user_input)
    user_gender_embed = Embedding(input_dim = USER_NUM, 
            output_dim = 1, 
            weights = [user_gender], 
            trainable = False)(user_input)
    user_occ_embed = Embedding(input_dim = USER_NUM, 
            output_dim = 1, 
            weights = [user_occ], 
            trainable = False)(user_input)
    user_age_embed = Embedding(input_dim = USER_NUM, 
            output_dim = 1, 
            weights = [user_age], 
            trainable = False)(user_input)
    user_gender_embed = Embedding(input_dim = 2, 
            output_dim = DIM, 
            embeddings_regularizer = l2(0.003),
            input_length = 1)(user_gender_embed)
    user_age_embed = Embedding(input_dim = 7, 
            output_dim = DIM, 
            embeddings_regularizer = l2(0.003),
            input_length = 1)(user_age_embed)
    user_occ_embed = Embedding(input_dim = int(np.max(user_occ))+1,
            output_dim = DIM, 
            embeddings_regularizer = l2(0.008),
            input_length = 1)(user_occ_embed)
    user_bias = Flatten()(user_bias)
    movie_bias = Flatten()(movie_bias)
    user_gender_embed = Flatten()(user_gender_embed)
    user_age_embed = Flatten()(user_age_embed)
    user_occ_embed = Flatten()(user_occ_embed)
    user_im = Add()([user_v, user_gender_embed, user_age_embed, user_occ_embed])
    user_im = Flatten()(user_im)
    movie_v = Flatten()(movie_v)
    user_v = Flatten()(user_v)
    #user_im = BatchNormalization()(user_im)
    #movie_v = BatchNormalization()(movie_v)
    dot = Dot(axes = 1)([user_v, movie_v])
    add = Add()([dot, user_bias, movie_bias])
    model = Model([user_input, movie_input], dot)
    model.summary()

    opt = Adam(lr = 0.0005)
    model.compile(optimizer = opt, loss = 'mse', metrics = [rmse])
    earylystopping = EarlyStopping(monitor = 'val_rmse',
            patience = 8,
            verbose = 1)
    checkpoint = ModelCheckpoint('./model/{val_rmse:.4f}_{epoch:03d}_' + str(DIM) + '.h5',
            monitor = 'val_rmse',
            save_best_only = True,
            verbose = 0)
    reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 3)
    his = model.fit(np.hsplit(x, 2), y,
            batch_size = 1024, 
            epochs = 1000, 
            validation_split = 0.1,
            callbacks = [earylystopping, checkpoint, reduce_lr])
    model_dict = OrderedDict()
    model_dict['rmse'] = his.history['rmse']
    model_dict['val_rmse'] = his.history['val_rmse']
    storer.store(model_name + str(DIM), model_dict)
    storer.close()
    

if args.en:
    x = pickle.load(open('x', 'rb'))
    y = pickle.load(open('y', 'rb'))
    y -= RATING_MEAN
    user_b = pickle.load(open('user_b', 'rb'))
    movie_b = pickle.load(open('movie_b', 'rb'))
    user_occ = pickle.load(open('user_occ', 'rb'))
    user_age = pickle.load(open('user_age', 'rb'))
    user_gender = pickle.load(open('user_gender', 'rb'))
    movie_gen = pickle.load(open('movie_gen', 'rb'))
    user_b = user_b.reshape(user_b.shape[0], 1)
    movie_b = movie_b.reshape(movie_b.shape[0], 1)
    user_occ = user_occ.reshape(user_occ.shape[0], 1)
    user_age = user_age.reshape(user_age.shape[0], 1)
    user_gender = user_gender.reshape(user_gender.shape[0], 1)

    for en_i in range(1,50):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x = x[randomize]
        y = y[randomize]

        user_input = Input(shape = (1,))
        movie_input = Input(shape = (1,))
        user_v = Embedding(input_dim = USER_NUM, 
                output_dim = 200, 
                embeddings_regularizer = l2(0.000005),
                input_length = 1)(user_input)
        movie_v = Embedding(input_dim = MOVIE_NUM, 
                output_dim = 200, 
                embeddings_regularizer = l2(0.000005),
                input_length = 1)(movie_input)
        user_v = Flatten()(user_v)
        movie_v = Flatten()(movie_v)

        #movie_gen_embed = Embedding(input_dim = MOVIE_NUM, 
        #        output_dim = 6, 
        #        weights = [movie_gen], 
        #        trainable = False)(movie_input)
        movie_bias = Embedding(input_dim = MOVIE_NUM,
                output_dim = 1,
                weights = [movie_b],
                trainable = False)(movie_input)
        user_bias = Embedding(input_dim = USER_NUM, 
                output_dim = 1, 
                weights = [user_b], 
                trainable = False)(user_input)
        user_gender_embed = Embedding(input_dim = USER_NUM, 
                output_dim = 1, 
                weights = [user_gender], 
                trainable = False)(user_input)
        user_occ_embed = Embedding(input_dim = USER_NUM, 
                output_dim = 1, 
                weights = [user_occ], 
                trainable = False)(user_input)
        user_age_embed = Embedding(input_dim = USER_NUM, 
                output_dim = 1, 
                weights = [user_age], 
                trainable = False)(user_input)
        user_gender_embed = Embedding(input_dim = 2, 
                output_dim = 100, 
                #embeddings_regularizer = l2(0.01),
                input_length = 1)(user_gender_embed)
        user_age_embed = Embedding(input_dim = 7, 
                output_dim = 100, 
                #embeddings_regularizer = l2(0.03),
                input_length = 1)(user_age_embed)
        user_occ_embed = Embedding(input_dim = int(np.max(user_occ))+1,
                output_dim = 100, 
                #embeddings_regularizer = l2(0.03),
                input_length = 1)(user_occ_embed)
        user_bias = Flatten()(user_bias)
        movie_bias = Flatten()(movie_bias)
        user_gender_embed = Flatten()(user_gender_embed)
        user_age_embed = Flatten()(user_age_embed)
        user_occ_embed = Flatten()(user_occ_embed)
        user_im = Concatenate()([user_v, user_gender_embed, user_age_embed, user_occ_embed])
        #user_im = BatchNormalization()(user_im)
        #movie_v = BatchNormalization()(movie_v)
        dot = Dot(axes = 1)([user_v, movie_v])
        add = Add()([dot, user_bias, movie_bias])
        model = Model([user_input, movie_input], add)
        model.summary()

        opt = Adam(lr = 0.0005)
        model.compile(optimizer = opt, loss = 'mse', metrics = [rmse])
        earylystopping = EarlyStopping(monitor = 'val_rmse',
                patience = 8,
                verbose = 1)
        checkpoint = ModelCheckpoint('./ensemble/' + '{:02d}'.format(en_i) + '.h5',
                monitor = 'val_rmse',
                save_best_only = True,
                verbose = 0)
        reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 3)
        his = model.fit(np.hsplit(x, 2), y,
                batch_size = 1024, 
                epochs = 1000, 
                validation_split = 0.33,
                callbacks = [earylystopping, checkpoint, reduce_lr])
