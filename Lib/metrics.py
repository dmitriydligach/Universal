#!/usr/bin/env python3

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

if __name__ == "__main__":

  print()
