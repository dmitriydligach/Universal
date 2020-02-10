#!/usr/bin/env python3

import scipy
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def report_f1(y_test, predictions, average):
  """F1 score"""

  f1 = f1_score(y_test, predictions, average=average)
  print("f1-%s: %.3f" % (average, f1))

def report_accuracy(y_test, predictions):
  """Accuracy score"""

  accuracy = accuracy_score(y_test, predictions)
  print('accuracy: %.3f' % accuracy)

def report_roc_auc(y_test, probs):
  """ROC and PR AUC scores"""

  roc_auc = roc_auc_score(y_test, probs)
  print('roc auc: %.3f' % roc_auc)

def report_pr_auc(y_true, probs):
    """PR AUC; x-axis should be recall, y-axis precision"""

    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    print('pr auc: %.3f' % pr_auc)

def report_roc_auc_ci(y_test, probs, n_samples=100000):
  """Confidence 95% confidence intervals on ROC AUC"""

  # source: https://stackoverflow.com/questions/19124239/
  # scikit-learn-roc-curve-with-confidence-intervals

  rs = np.random.RandomState(2020)
  y_test = np.array(y_test)
  probs = np.array(probs)

  scores = []
  for _ in range(n_samples):
    indices = rs.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        continue

    score = roc_auc_score(y_test[indices], probs[indices])
    scores.append(score)

  sorted = np.array(scores)
  sorted.sort()

  lower = sorted[int(0.025 * len(sorted))]
  upper = sorted[int(0.975 * len(sorted))]
  mean = np.mean(scores)

  print('boostrapping CIs: %.3f < %.3f < %.3f' % (lower, mean, upper))

  # sanity check
  std_err = np.std(scores) / np.sqrt(len(y_test))
  ci = 1.96 * std_err
  lower = mean - ci
  upper = mean + ci
  print('approximate CIs: %.3f < %.3f < %.3f' % (lower, mean, upper))

if __name__ == "__main__":

  print()
