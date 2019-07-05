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

from scipy.stats import uniform
from scipy.stats import randint

from keras import Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
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

import dataset, word2vec, rndsearch

def make_model(args):
  """Average pooling model"""

  embed = Embedding(
    input_dim=args['vocabulary_size'],
    output_dim=args['emb_dim'],
    input_length=args['max_seq_len'],
    weights=args['init_vectors'],
    name='EL')
  average = GlobalAveragePooling1D(name='AL')
  project = Dense(
    units=args['hidden'],
    activation=args['activation'],
    name='HL')
  drop = Dropout(args['dropout'])

  input_tensor = Input(shape=(args['max_seq_len'],))
  x = embed(input_tensor)
  x = average(x)
  x = project(x)
  x = drop(x)
  output_tensor = Dense(args['n_targets'], activation='sigmoid')(x)

  model = Model(input_tensor, output_tensor)

  return model

def main():
  """Driver function"""

  base = os.environ['DATA_ROOT']

  dp = dataset.DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.get('args', 'n_x_cuis'),
    cfg.get('args', 'n_y_cuis'),
    cfg.getfloat('args', 'min_examples_per_targ'))
  x, y = dp.load()

  print('x shape:', x.shape)
  print('y shape:', y.shape)

  fixed_args = {
    'vocabulary_size': len(dp.tokenizer.word_index) + 1,
    'max_seq_len': x.shape[1],
    'n_targets': y.shape[1],
    'init_vectors': None,
    'loss': 'binary_crossentropy',
    'epochs': cfg.getint('search', 'max_epochs')}

  param_space = {
    'emb_dim': (512, 1024, 2048),
    'hidden': (1000, 3000, 5000, 10000),
    'activation': ('linear', 'tanh', 'relu'),
    'dropout': uniform(0, 0.75),
    'optimizer': ('RMSprop', 'Adam'),
    'log10lr': (-5, -4, -3, -2, -1),
    'batch': (4, 8, 16, 32, 64)}

  config2score = rndsearch.run(
    make_model,
    fixed_args,
    param_space,
    x,
    y,
    n=cfg.getint('search', 'n'),
    verbose=1)

  # display configs sorted by f1
  print('\nconfigurations sorted by score:')
  sorted_by_value = sorted(config2score, key=config2score.get)
  for config in sorted_by_value:
    print('%s: %.3f' % (config, config2score[config]))

  best_config = dict(sorted_by_value[-1])
  print('best config:', best_config)
  print('best score:', config2score[sorted_by_value[-1]])

if __name__ == "__main__":

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  main()
