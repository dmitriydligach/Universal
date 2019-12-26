#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle, shutil

import keras.optimizers as optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Dense

from phenot_dataset import DatasetProvider
import i2b2, metrics

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_model(num_labels=2):
  """Load pre-trained model and get ready for fine-tunining"""

  # load pre-trained model
  rep_layer = cfg.get('data', 'rep_layer')
  pretrained = load_model(cfg.get('data', 'model_file'))
  base = Model(
    inputs=pretrained.input,
    outputs=pretrained.get_layer(rep_layer).output)

  # add logistic regression layer
  model = Sequential()
  model.add(base)
  model.add(Dropout(cfg.getfloat('bow', 'dropout')))
  model.add(Dense(num_labels, activation='softmax'))

  model.summary()

  return model

def eval():
  """Train and evaluate"""

  data_root = os.environ['DATA_ROOT']

  if os.path.isdir('./Model/'):
    shutil.rmtree('./Model/')
  os.mkdir('./Model/')

  train_data_provider = DatasetProvider(
    os.path.join(data_root, cfg.get('data', 'train')),
    cfg.get('data', 'tokenizer_pickle'),
    None)
  x_train, y_train = train_data_provider.load_as_one_hot()

  test_data_provider = DatasetProvider(
    os.path.join(data_root, cfg.get('data', 'test')),
    cfg.get('data', 'tokenizer_pickle'),
    None)
  x_test, y_test = test_data_provider.load_as_one_hot()

  callback = ModelCheckpoint(
    './Model/model.h5',
    verbose=1,
    save_best_only=True)

  model = get_model()
  optim = getattr(optimizers, cfg.get('bow', 'optimizer'))
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optim(lr=10**cfg.getint('bow', 'log10lr')),
                metrics=['accuracy'])

  model.fit(x_train,
            y_train,
            epochs=cfg.getint('bow', 'epochs'),
            batch_size=cfg.getint('bow', 'batch'),
            validation_split=0.2,
            callbacks=[callback])

  # (test size, num of classes)
  distribution = model.predict(x_test)
  predictions = np.argmax(distribution, axis=1)

  metrics.report_f1(y_test, predictions, 'macro')
  metrics.report_f1(y_test, predictions, 'micro')
  metrics.report_accuracy(y_test, predictions)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  eval()
