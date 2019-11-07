# Load Libraries
import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import KFold
from pathlib import Path
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python.platform import gfile

import json

import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

# General save model to disk function
def save_model(fileModelJSON, fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)

# General load model from disk function
def load_model(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(fileWeights)
    return model

with tf.device("/GPU:1"):
    # Load data
    DATA_HOME ='/home/jhnamgung/kcyber/data/'
    df = pd.read_csv(DATA_HOME + 'dga_1st_round_train.csv',encoding='ISO-8859-1', sep=',')



    x_trains, x_tests = model_selection.train_test_split(df, test_size=0.2, random_state=33)
    x_trains.to_csv("./trainsXLSTM.csv", mode='w', index=False)
    x_tests.to_csv("./testsXLSTM.csv",mode='w', index=False)
    # Sampling

    # Initial Data Preparation URL

    # Step 1: Convert raw URL string in list of lists where characters that are contained in "printable" are sotred encoded as integer
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.domain]

    # Step 2: Cut URL string at max_len or pad with zeros if shorter
    max_len = 74
    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)

    # Step 3: Extract labels form df to nupy array
    y = np.array(df.nclass)

    X_train, X_test, y_train0, y_test0 = model_selection.train_test_split(X, y, test_size=0.2, random_state=33)
    y_train = np_utils.to_categorical(y_train0, 20)
    y_test = np_utils.to_categorical(y_test0, 20)
    # Embedding
    # 1D Convolution and Fully Connected Layers
with tf.device("/GPU:1"):

    def simple_lstm(max_len=74, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input) 

        # LSTM layer
        lstm = LSTM(lstm_output_size)(emb)
        lstm = Dropout(0.5)(lstm)
    
        # Output layer (last fully connected layer)
        output = Dense(20, activation='sigmoid', name='output')(lstm)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

        # Output layer (last fully connected layer)
        output = Dense(20, activation='sigmoid', name='main_output')(hidden2)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
with tf.device("/GPU:1"):
    epochs = 6
    batch_size = 32

    model = simple_lstm()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('\nFinal Cross-Validation Accuracy', accuracy, '\n')


    model_name = "Predictdeeplearning_LSTM"
    save_model("./" + model_name + ".json", DATA_HOME + model_name + ".h5")


    model = load_model("./" + model_name + ".json", DATA_HOME + model_name + ".h5")

    df_Test = pd.read_csv("./testsXLSTM.csv",encoding='ISO-8859-1', sep=',')

    url_test_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df_Test.domain]

    max_len = 74

    X_test = sequence.pad_sequences(url_test_int_tokens, maxlen=max_len)

    target_prob = model.predict(X_test, batch_size=1)
    target_list = []

    for k in target_prob:
        target_label = k.tolist().index(max(k.tolist()))
        #print(target_label)
        target_list.append(target_label)

    #print(target_list)

    df_Test["predict"] = target_list
    df_Test.to_csv("./result/LSTM_result.csv",mode='w')
    #print("###############################WITH PREDICT")
    #print(df_Test)
