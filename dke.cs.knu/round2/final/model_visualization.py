
# Load Libraries
import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

# import gensim
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
from keras_self_attention import SeqSelfAttention

import json

import warnings
warnings.filterwarnings("ignore")

m = "STACKING"
with open('./models/' + m + '.json', 'r') as f:
    model_json = json.load(f)
model = model_from_json(model_json, custom_objects={'SeqSelfAttention':SeqSelfAttention})
model.load_weights('./models/' + m + '.h5')

SVG(model_to_dot(model, show_shapes=True).create(prog='dot',format='svg'))
plot_model(model,to_file='STACKING.png',show_shapes=False,show_layer_names=True)