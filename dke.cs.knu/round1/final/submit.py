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
from keras.models import model_from_json
from numpy import argmax

import json

import warnings
warnings.filterwarnings("ignore")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
K.tensorflow_backend.set_session(tf.Session(config=config))

# General load model from disk function
def load_model(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(fileWeights)
    return model

with tf.device("/GPU:0"):

    # Load pre-trained model.   
    model_name = "1DConv_f"
    model = load_model("./models/" + model_name + ".json","./models/"  + model_name + ".h5")

    # Load test data
    DATA_HOME ='/home/jhnamgung/kcyber/data/'
    df_Test = pd.read_csv(DATA_HOME + 'dga_1st_round_test.csv',encoding='ISO-8859-1', sep=',')

    # Preprocesing input data
    url_test_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df_Test.domain]

    max_len = 74
     
    print("Preprocessing input data")
    X_test = sequence.pad_sequences(url_test_int_tokens, maxlen=max_len)
   
    # Predict test data 
    target_prob = model.predict(X_test, batch_size=128) # batch_size가 작으면 속도가 오래 걸림
    print("Predict complete!!")
    
    # Select class label
    yhat = argmax(target_prob, axis=1)

    # Define DGA(yes or no) using class label  
    dgalist = []
    for x in yhat.tolist():
        if x != 0:
            dgalist.append("yes")
        else:
            dgalist.append("no")
    print("Make dgalist")

    # Save test result
    x_input = df_Test['domain'].tolist()
    archive = pd.DataFrame(columns=['domain',"dga","class"])
    archive["domain"] = x_input
    archive["dga"] = dgalist
    archive["class"] = yhat.tolist()

    # Write to file
    output_file = "finalDKE.csv"
    print("Writing file...")
    archive.to_csv("./outputfile" + output_file, mode='w', index=False)

