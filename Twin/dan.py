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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras import Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import concatenate, dot
from keras.models import load_model
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

import dataset_split

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_model_dot(vocabulary_size, max_seq_len):
  """Model definition"""

  embed = Embedding(
    input_dim=vocabulary_size,
    output_dim=cfg.getint('dan', 'emb_dim'),
    input_length=max_seq_len,
    name='EL')
  average = GlobalAveragePooling1D(name='AL')
  project = Dense(
    cfg.getint('dan', 'hidden'),
    activation='relu',
    name='DL')

  input_tensor1 = Input(shape=(max_seq_len,))
  x1 = embed(input_tensor1)
  x1 = average(x1)
  x1 = project(x1)

  input_tensor2 = Input(shape=(max_seq_len,))
  x2 = embed(input_tensor2)
  x2 = average(x2)
  x2 = project(x2)

  x = dot([x1, x2], axes=-1)
  output_tensor = Dense(1, activation='sigmoid')(x)

  model = Model([input_tensor1, input_tensor2], output_tensor)

  plot_model(model, show_shapes=True, to_file='Model/model.png')
  model.summary()

  return model

def get_model_concat(vocabulary_size, max_seq_len):
  """Model definition"""

  embed = Embedding(
    input_dim=vocabulary_size,
    output_dim=cfg.getint('dan', 'emb_dim'),
    input_length=max_seq_len,
    name='EL')
  average = GlobalAveragePooling1D(name='AL')
  project = Dense(
    cfg.getint('dan', 'hidden'),
    activation='relu',
    name='DL')

  input_tensor1 = Input(shape=(max_seq_len,))
  x1 = embed(input_tensor1)
  x1 = average(x1)
  x1 = project(x1)

  input_tensor2 = Input(shape=(max_seq_len,))
  x2 = embed(input_tensor2)
  x2 = average(x2)
  x2 = project(x2)

  x = concatenate([x1, x2], axis=-1)
  x = Dense(cfg.getint('dan', 'hidden'), activation='relu')(x)

  output_tensor = Dense(1, activation='sigmoid')(x)

  model = Model([input_tensor1, input_tensor2], output_tensor)

  plot_model(model, show_shapes=True, to_file='Model/model.png')
  model.summary()

  return model

def main():
  """Driver function"""

  base = os.environ['DATA_ROOT']

  dp = dataset_split.DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.getint('args', 'max_seq_len'),
    cfg.get('args', 'n_files'),
    cfg.getfloat('args', 'split'))
  x1, x2, y = dp.load()

  print('x1 shape:', x1.shape)
  print('x2 shape:', x2.shape)
  print('y shape:', y.shape)

  train_x1, val_x1, train_x2, val_x2, train_y, val_y = train_test_split(
    x1, x2, y, test_size=cfg.getfloat('args', 'test_size'))

  model = get_model_concat(len(dp.tokenizer.word_index) + 1, x1.shape[1])

  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

  # save the model after every epoch
  callback = ModelCheckpoint(
    cfg.get('data', 'model_dir') + 'model.h5',
    verbose=1,
    save_best_only=True)

  model.fit([train_x1, train_x2],
            train_y,
            validation_data=([val_x1, val_x2], val_y),
            epochs=cfg.getint('dan', 'epochs'),
            batch_size=cfg.getint('dan', 'batch'),
            validation_split=0.0,
            callbacks=[callback])

  # for now best model saved after each epoch
  # model.save(cfg.get('data', 'model_dir') + 'model.h5')

  # do we need to evaluate?
  if cfg.getfloat('args', 'test_size') == 0:
    exit()

  probs = model.predict([val_x1, val_x2])
  predictions = (probs > 0.5).astype(int)
  accuracy = accuracy_score(val_y, predictions)
  print('accuracy: ', accuracy)

if __name__ == "__main__":

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  main()
