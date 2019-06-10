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
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalAveragePooling1D
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

def get_model(vocabulary_size, max_seq_len, n_targets, init_vectors):
  """Define model"""

  model = Sequential()
  model.add(Embedding(input_dim=vocabulary_size,
                      output_dim=cfg.getint('dan', 'emb_dim'),
                      input_length=max_seq_len,
                      weights=init_vectors,
                      name='EL'))
  model.add(GlobalAveragePooling1D(name='AL'))

  model.add(Dense(cfg.getint('dan', 'hidden'), name='HL'))
  model.add(Activation(cfg.get('dan', 'activation')))

  model.add(Dropout(cfg.getfloat('dan', 'dropout')))

  model.add(Dense(n_targets))
  model.add(Activation('sigmoid'))

  plot_model(model, show_shapes=True, to_file='Model/model.png')
  model.summary()

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

  train_x, val_x, train_y, val_y = train_test_split(
    x, y, test_size=cfg.getfloat('args', 'test_size'))

  # TODO: figure out what to do about negated cuis
  init_vectors = None
  if cfg.has_option('data', 'embed'):
    embed_file = os.path.join(base, cfg.get('data', 'embed'))
    w2v = word2vec.Model(embed_file, verbose=True)
    init_vectors = [w2v.select_vectors(dp.tokenizer.word_index)]

  model = get_model(
    len(dp.tokenizer.word_index)+1,
    x.shape[1],
    y.shape[1],
    init_vectors)

  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

  # save the model after every epoch
  callback = ModelCheckpoint(
    cfg.get('data', 'model_dir') + 'model.h5',
    verbose=1,
    save_best_only=True)

  model.fit(train_x,
            train_y,
            validation_data=(val_x, val_y),
            epochs=cfg.getint('dan', 'epochs'),
            batch_size=cfg.getint('dan', 'batch'),
            validation_split=0.0,
            callbacks=[callback])

  # are we training the best model?
  if cfg.getfloat('args', 'test_size') == 0:
    model.save(cfg.get('data', 'model_dir') + 'model.h5')
    exit()

  probs = model.predict(val_x)
  predictions = (probs > 0.5).astype(int)
  accuracy = accuracy_score(val_y, predictions)
  print('accuracy: ', accuracy)

if __name__ == "__main__":

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  main()
