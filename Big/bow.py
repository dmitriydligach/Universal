#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
import os, math
tf.logging.set_verbosity(tf.logging.ERROR)
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

def configure_model_dir():
  """Configure dir to store model, tokenizer, etc."""

  # create model dir if needed
  if not os.path.isdir('Model'):
    print('making a new model dir...')
    os.mkdir('Model')

  # delete old model prior to training
  if os.path.exists('Model/model.h5'):
    if os.path.exists('Model/model.h5'):
      print('removing old model...')
      os.remove('Model/model.h5')

  # remove old alphabet if making a new one
  if cfg.getboolean('args', 'make_alphabet'):
    if os.path.exists('Model/tokenizer.p'):
      print('removing old alphabet...')
      os.remove('Model/tokenizer.p')

def main():
  """Do out-of-core training here"""

  configure_model_dir()
  base = os.environ['DATA_ROOT']

  dp = dataset.DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('args', 'max_files'),
    cfg.getint('args', 'max_cuis'),
    cfg.getint('args', 'samples_per_doc'),
    cfg.getint('bow', 'batch'),
    cfg.getboolean('args', 'make_alphabet'))

  max_cuis = int(cfg.get('args', 'max_cuis'))
  model = get_model(max_cuis, max_cuis - 1)
  optim = getattr(optimizers, cfg.get('bow', 'optimizer'))

  model.compile(
    loss='binary_crossentropy',
    optimizer=optim(lr=10**cfg.getint('bow', 'log10lr')),
    metrics=['accuracy'])

  callback = ModelCheckpoint(
    'Model/model.h5',
    verbose=1,
    save_best_only=True)

  # load validation data
  val_x, val_y = dp.load(os.path.join(base, cfg.get('data', 'dev')))
  print('dev x, y shapes:', val_x.shape, val_y.shape)

  steps = math.ceil(dp.train_size / cfg.getint('bow', 'batch'))
  print('steps per epoch:', steps)

  model.fit_generator(
    dp.stream(),
	  validation_data=(val_x, val_y),
    epochs=cfg.getint('bow', 'epochs'),
    steps_per_epoch=steps,
    callbacks=[callback])

  # save final model
  model.save('Model/final.h5')

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

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  main()
