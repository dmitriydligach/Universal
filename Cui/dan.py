#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as bke
s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

# the rest of imports
import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
import configparser

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from keras import Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.callbacks import Callback

import dataset

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_model(vocabulary_size, max_seq_len):
  """Model definition"""

  # input tensor not including batch size
  input_tensor = Input(shape=(max_seq_len,))

  x = Embedding(
    input_dim=vocabulary_size,
    output_dim=300,
    input_length=max_seq_len)(input_tensor)

  x = GlobalAveragePooling1D(name='AL')(x)

  x = Dense(512, activation='relu', name='HL')(x)

  x = Dropout(0.25)(x)

  output_tensor = Dense(1, activation='sigmoid')(x)

  model = Model(input_tensor, output_tensor)

  model.summary()

  return model

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))

  dp = dataset.DatasetProvider(train_dir)
  x, y = dp.load()
  print('x shape:', x.shape)
  print('y shape:', y.shape)

  train_x, val_x, train_y, val_y = train_test_split(
    x, y,
    test_size=cfg.getfloat('args', 'test_size'))

  model = get_model(len(dp.tokenizer.word_index)+1, x.shape[1])
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit(train_x,
            train_y,
            validation_data=(val_x, val_y),
            epochs=cfg.getint('dan', 'epochs'),
            batch_size=cfg.getint('dan', 'batch'),
            validation_split=0.0)

  # probability for each class; (test size, num of classes)
  distribution = model.predict(val_x)

  # turn into an indicator matrix
  distribution[distribution < 0.5] = 0
  distribution[distribution >= 0.5] = 1

  f1 = f1_score(val_y, distribution, average='macro')
  p = precision_score(val_y, distribution, average='macro')
  r = recall_score(val_y, distribution, average='macro')
  print("\nmacro: p: %.3f - r: %.3f - f1: %.3f" % (p, r, f1))
  f1 = f1_score(val_y, distribution, average='micro')
  p = precision_score(val_y, distribution, average='micro')
  r = recall_score(val_y, distribution, average='micro')
  print("micro: p: %.3f - r: %.3f - f1: %.3f" % (p, r, f1))
