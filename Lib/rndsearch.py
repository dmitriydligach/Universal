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
import sys, random, gc, keras
sys.path.append('../Lib/')
sys.dont_write_bytecode = True
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def sample(params):
  """Sample a configuration from param space"""

  config = {}

  for param, value in params.items():
    if hasattr(value, 'rvs'):
      # this is a scipy.stats distribution
      config[param] = value.rvs()
    else:
      # this is a tuple
      config[param] = random.choice(value)

  return config

def run(
  make_model,  # function that returns a keras model
  fixed_args,  # make_model and other fixed arguments
  param_space, # possible hyperparameter values
  x_train,     # training examples
  y_train,     # training labels
  x_val=None,  # validation examples
  y_val=None,  # validation labels
  n=100,       # number of iterations
  verbose=0):  # suppress output
  """Random search"""

  # need a validation set?
  if x_val == None:
    x_train, x_val, y_train, y_val = \
      train_test_split(x_train, y_train, test_size=0.2)

  # configurations and their scores
  config2score = {}

  for i in range(n):

    # prevent OOM errors
    gc.collect()
    bke.clear_session()

    config = sample(param_space)
    args = config.copy()
    args.update(fixed_args)

    model = make_model(args)

    erstop = EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=2,
      restore_best_weights=True)

    optim = getattr(keras.optimizers, args['optimizer'])
    model.compile(
      loss=args['loss'],
      optimizer=optim(lr=10**args['log10lr']),
      metrics=['accuracy'])

    model.fit(
      x_train,
      y_train,
      validation_data=(x_val, y_val),
      epochs=args['epochs'],
      batch_size=args['batch'],
      verbose=0,
      callbacks=[erstop])

    # add effective number of epochs to config
    if erstop.stopped_epoch > 0:
      config['epochs'] = erstop.stopped_epoch - 1
    else:
      config['epochs'] = 0

    predictions = model.predict_classes(x_val)
    f1 = f1_score(y_val, predictions, average='macro')
    config2score[tuple(config.items())] = f1

    if verbose == 1:
      print('[%d] %s' % (i + 1, config))
      print('[%d] score: %.3f' % (i + 1, f1))

  return config2score

if __name__ == "__main__":

  pass
