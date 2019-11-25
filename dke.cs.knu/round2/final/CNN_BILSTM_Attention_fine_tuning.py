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
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras_self_attention import SeqSelfAttention
from numpy import argmax
from keras.callbacks import ModelCheckpoint

import json

import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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


with tf.device("/GPU:0"):

    # Load data
    DATA_HOME ='../../data/'
    #DATA_HOME ='/home/jhnamgung/kcyber/data/'
    #df = pd.read_csv(DATA_HOME + 'sample.csv',encoding='ISO-8859-1', sep=',')
    df = pd.read_csv(DATA_HOME + 'dga_2nd_round_train.csv',encoding='ISO-8859-1', sep=',')
    m = "CNN_BILSTM_ATT"
    with open('./models/' + m + '.json', 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json, custom_objects={'SeqSelfAttention':SeqSelfAttention})
    model.load_weights('./models/' + m + '.h5')


    # Convert domain string to integer
    # URL 알파벳을 숫자로 변경
    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.domain]

    # Padding domain integer max_len=74
    # 최대길이 74로 지정
    max_len = 74

    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    y = np.array(df['class'])

    # Using all training data to train final model
    X_train = X
    y_train = np_utils.to_categorical(y, 20) # dga class: 0~19: 20개

with tf.device("/GPU:0"):
    epochs = 4
    batch_size = 512
  
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    cb_checkpoint = ModelCheckpoint(filepath='./models/', monitor='val_loss',
                                   verbose=1, save_best_only=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[cb_checkpoint])

    # Save final training model
    model_name = "CNN_BILSTM_ATT_FINE"
    save_model("./models/" + model_name + ".json", "./models/" + model_name + ".h5")