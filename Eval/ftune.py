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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.decomposition import TruncatedSVD

import keras.optimizers as optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Dense

from phenot_dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def report_f1(y_test, predictions, average):
  """Report p, r, and f1"""

  p = precision_score(y_test, predictions, average=average)
  r = recall_score(y_test, predictions, average=average)
  f1 = f1_score(y_test, predictions, average=average)
  print("[%s] p: %.3f - r: %.3f - f1: %.3f" % (average, p, r, f1))

def report_accuracy(y_test, predictions):
  """Accuracy score"""

  accuracy = accuracy_score(y_test, predictions)
  print('accuracy: %.3f' % accuracy)

def report_roc_auc(y_test, probs):
  """ROC and PR AUC scores"""

  roc_auc = roc_auc_score(y_test, probs[:, 1])
  print('roc auc: %.3f' % roc_auc)

def report_pr_auc(y_true, probs):
    """PR AUC; x-axis should be recall, y-axis precision"""

    precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
    pr_auc = auc(recall, precision)
    print('pr auc: %.3f' % pr_auc)

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
    cfg.get('data', 'model_dir') + 'model.h5',
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

  report_f1(y_test, predictions, 'macro')
  report_f1(y_test, predictions, 'micro')
  report_accuracy(y_test, predictions)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  eval()
