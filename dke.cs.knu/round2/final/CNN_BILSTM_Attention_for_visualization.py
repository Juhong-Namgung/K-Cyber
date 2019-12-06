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

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
        conv = ELU()(conv)
        # conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)

        return conv

    def bidirection_lstm_with_attention(max_len=74, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
        # Input
        main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

        # Embedding layer
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

        conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=256)

        merged = concatenate([ conv2, conv3])

        # Bi-directional LSTM layer
        lstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)
        
        att = SeqSelfAttention(attention_activation='softmax')(lstm)

        cnnlstm_merged = concatenate([merged, att])

        s_lstm = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(cnnlstm_merged)
        s_lstm = Dropout(0.2)(s_lstm)

        s_att = SeqSelfAttention(attention_activation='softmax')(s_lstm)
        s_att = Flatten()(s_att)

        hidden1 = Dense(4096)(s_att)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(1024)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization(mode=0)(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        hidden3 = Dense(1024)(hidden2)
        hidden3 = ELU()(hidden3)
        hidden3 = BatchNormalization(mode=0)(hidden3)
        hidden3 = Dropout(0.5)(hidden3)

        # Output layer (last fully connected layer)
        output = Dense(20, activation='softmax', name='output')(hidden3)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

with tf.device("/GPU:0"):

    model = bidirection_lstm_with_attention()

    # model structure visualization
    SVG(model_to_dot(model, show_shapes=True).create(prog='dot',format='svg'))
    plot_model(model,to_file='CNN_BILSTM_Attention.png',show_shapes=True,show_layer_names=True)

    # model.summary()
