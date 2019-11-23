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

import keras.optimizers as optimizers
from keras import Input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import concatenate, dot
from keras.models import load_model
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import dataset, word2vec

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

RESULTS_FILE = 'Model/results.txt'
MODEL_FILE = 'Model/model.h5'

def get_model(vocabulary_size, n_targets):
  """Average pooling model"""

  project = Dense(
    cfg.getint('bow', 'hidden'),
    activation=cfg.get('bow', 'activation'),
    name='HL')
  drop = Dropout(cfg.getfloat('bow', 'dropout'))

  input_tensor = Input(shape=(vocabulary_size,))
  x = project(input_tensor)
  x = drop(x)
  output_tensor = Dense(n_targets, activation='sigmoid')(x)

  model = Model(input_tensor, output_tensor)
  plot_model(model, show_shapes=True, to_file='Model/model.png')
  model.summary()

  return model

def main():
  """Driver function"""

  base = os.environ['DATA_ROOT']

  dp = dataset.DatasetProvider(
    os.path.join(base, cfg.get('data', 'cuis')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.get('args', 'max_cuis'),
    cfg.get('args', 'max_codes'))
  x, y = dp.load()

  print('x shape:', x.shape)
  print('y shape:', y.shape)

  # are we training the best model?
  if cfg.getfloat('args', 'test_size') != 0:
    train_x, val_x, train_y, val_y = train_test_split(
      x, y, test_size=cfg.getfloat('args', 'test_size'))
    validation_data = (val_x, val_y)
  else:
    train_x, train_y = x, y
    validation_data = None

  # need to add one to account for the index 0 which is not used
  model = get_model(cfg.getint('args', 'max_cuis'), y.shape[1])
  optim = getattr(optimizers, cfg.get('bow', 'optimizer'))
  model.compile(loss='binary_crossentropy',
                optimizer=optim(lr=10**cfg.getint('bow', 'log10lr')),
                metrics=['accuracy'])

  # save the model after every epoch
  callback = ModelCheckpoint(
    cfg.get('data', 'model_dir') + 'model.h5',
    verbose=1,
    save_best_only=True)

  model.fit(train_x,
            train_y,
            validation_data=validation_data,
            epochs=cfg.getint('bow', 'epochs'),
            batch_size=cfg.getint('bow', 'batch'),
            validation_split=0.0,
            callbacks=[callback])

  # are we training the best model?
  if cfg.getfloat('args', 'test_size') == 0:
    model.save(cfg.get('data', 'model_dir') + 'model.h5')
    exit()

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

if __name__ == "__main__":
  """Run something"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  main()
