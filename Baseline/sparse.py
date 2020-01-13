#!/usr/bin/env python3

# the rest of the imports
import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import configparser, os
import dataset, metrics

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def grid_search(x, y, scoring):
  """Find best model and fit it"""

  param_grid = {
    'penalty': ['l1', 'l2'],
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced')
  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=10)
  gs.fit(x, y)

  print('best model:')
  print(gs.best_estimator_)

  return gs.best_estimator_

def run_eval(x_train, y_train, x_test, y_test, search=True):
  """Evaluation on test set"""

  if search:
    classifier = grid_search(x_train, y_train, 'roc_auc')
  else:
    classifier = LogisticRegression(class_weight='balanced')
    model = classifier.fit(x_train, y_train)

  probs = classifier.predict_proba(x_test)
  metrics.report_roc_auc(y_test, probs[:, 1])
  metrics.report_pr_auc(y_test, probs[:, 1])

def data_sparse():
  """Bag-of-cuis data for sparse evaluation"""

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))
  test_dir = os.path.join(base, cfg.get('data', 'test'))

  # load training data
  dataset_provider = dataset.DatasetProvider(train_dir)
  x_train, y_train = dataset_provider.load_sklearn()

  # load test data
  dataset_provider = dataset.DatasetProvider(test_dir)
  x_test, y_test = dataset_provider.load_sklearn()

  # turn xs into tfidf vectors
  vectorizer = TfidfVectorizer()
  x_train = vectorizer.fit_transform(x_train)
  x_test = vectorizer.transform(x_test)

  return x_train, y_train, x_test, y_test

def main():
  """Driver function"""

  x_train, y_train, x_test, y_test = data_sparse()
  run_eval(x_train, y_train, x_test, y_test)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  main()
