#!/usr/bin/env python3

import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import configparser, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def grid_search(x, y):
  """Find best model"""

  param_grid = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LinearSVC(class_weight='balanced')
  grid_search = GridSearchCV(
    lr,
    param_grid,
    scoring='f1_macro',
    cv=10,
    n_jobs=-1) # -1 fails on mac os
  grid_search.fit(x, y)

  return grid_search.best_estimator_

def run_evaluation_dense(disease, judgement):
  """Use pre-trained patient representations"""

  print('disease:', disease)
  x_train, y_train, x_test, y_test = data_dense(disease, judgement)

  if cfg.get('data', 'classif_param') == 'search':
    classifier = grid_search(x_train, y_train)
  else:
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_test)
  p = precision_score(y_test, predictions, average='macro')
  r = recall_score(y_test, predictions, average='macro')
  f1 = f1_score(y_test, predictions, average='macro')
  print("precision: %.3f - recall: %.3f - f1: %.3f\n" % (p, r, f1))

  return p, r, f1

def data_dense(disease, judgement):
  """Data to feed into code prediction model"""

  base = os.environ['DATA_ROOT']
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer(cfg.get('data', 'rep_layer')).output)
  maxlen = model.get_layer(name='EL').get_config()['input_length']

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    cfg.get('data', 'tokenizer_pickle'),
    maxlen)
  x1_train, x2_train, y_train = train_data_provider.load()

  print('unique labels in train:', set(y_train))

  # make training vectors for target task
  print('original x_train shape:', x1_train.shape)
  x_train = interm_layer_model.predict([x1_train, x2_train])
  print('new x_train shape:', x_train.shape)

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    cfg.get('data', 'tokenizer_pickle'),
    maxlen)
  x1_test, x2_test, y_test = test_data_provider.load()

  # make test vectors for target task
  print('original x_test shape:', x1_test.shape)
  x_test = interm_layer_model.predict([x1_test, x2_test])
  print('new x_test shape:', x_test.shape)

  return x_train, y_train, x_test, y_test

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  base = os.environ['DATA_ROOT']
  judgement = cfg.get('data', 'judgement')
  evaluation = cfg.get('data', 'evaluation')
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  ps = []; rs = []; f1s = []
  for disease in i2b2.get_disease_names(test_annot, set()):
    p, r, f1 = run_evaluation_dense(disease, judgement)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

  print('average p = %.3f' % np.mean(ps))
  print('average r = %.3f' % np.mean(rs))
  print('average f1 = %.3f' % np.mean(f1s))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  run_evaluation_all_diseases()
