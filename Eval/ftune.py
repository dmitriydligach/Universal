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

from sklearn.model_selection import train_test_split

import keras.optimizers as optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Dense

from dataphenot import DatasetProvider
import i2b2, metrics

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_model(num_labels):
  """Load pre-trained model and get ready for fine-tunining"""

  # https://stackoverflow.com/questions/41668813/
  # how-to-add-and-remove-new-layers-in-keras-after-loading-weights
  pretrained = load_model(cfg.get('data', 'model_file'))

  # remove code prediction and dropout layers
  pretrained.layers.pop()
  pretrained.layers.pop()

  # freeze pretrained weights
  for layer in pretrained.layers:
    layer.trainable = False

  # pre-trained model's hidden layer output
  output = pretrained.layers[-1].output

  output = Dropout(cfg.getfloat('linear', 'dropout'))(output)
  output = Dense(num_labels, activation='softmax')(output)

  model = Model(inputs=pretrained.input, outputs=output)

  model.summary()

  print()
  for layer in model.layers:
    print('%s: %s' % (layer.name, layer.trainable))
  print()

  return model

def main():
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
  print('loaded x_train:', x_train.shape)

  # are we evaluating on test or dev?
  if cfg.getfloat('data', 'val_size') != 0:
    x_train, x_val, y_train, y_val = train_test_split(
      x_train,
      y_train,
      test_size=cfg.getfloat('data', 'val_size'))
    callbacks = [ModelCheckpoint(
      './Model/model.h5',
      verbose=1, save_best_only=True)]
    validation_data = (x_val, y_val)
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)

  else:
    test_data_provider = DatasetProvider(
      os.path.join(data_root, cfg.get('data', 'test')),
      cfg.get('data', 'tokenizer_pickle'),
      None)
    x_test, y_test = test_data_provider.load_as_one_hot()
    print('loaded x_test:', x_test.shape)
    validation_data = None
    callbacks = None

  # train the linear classification layer
  model = get_model(len(train_data_provider.label2int))
  optim = getattr(optimizers, cfg.get('linear', 'optimizer'))
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optim(lr=cfg.getfloat('linear', 'lr')),
                metrics=['accuracy'])
  model.fit(x_train,
            y_train,
            validation_data=validation_data,
            epochs=cfg.getint('linear', 'epochs'),
            batch_size=cfg.getint('linear', 'batch'),
            validation_split=0.0,
            callbacks=callbacks)

  # fine-tune the pre-trained layers
  # https://stackoverflow.com/questions/47995324/
  # does-model-compile-initialize-all-the-weights-and-biases-in-keras-tensorflow/47996024

  if cfg.getboolean('base', 'finetune'):

    print()
    for layer in model.layers:
      layer.trainable = True
      print('%s: %s' % (layer.name, layer.trainable))

    optim = getattr(optimizers, cfg.get('base', 'optimizer'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optim(lr=cfg.getfloat('base', 'lr')),
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              validation_data=validation_data,
              epochs=cfg.getint('base', 'epochs'),
              batch_size=cfg.getint('base', 'batch'),
              validation_split=0.0,
              callbacks=callbacks)

  if cfg.getfloat('data', 'val_size') != 0:
    # during validation, load last best model
    model = load_model('./Model/model.h5')
    x_test, y_test = x_val, y_val

  # distribution.shape: (test size, num of classes)
  distribution = model.predict(x_test)
  predictions = np.argmax(distribution, axis=1)

  pos_label =train_data_provider.label2int['yes']
  metrics.report_roc_auc(y_test, distribution[:, pos_label])
  metrics.report_pr_auc(y_test, distribution[:, pos_label])

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  main()
