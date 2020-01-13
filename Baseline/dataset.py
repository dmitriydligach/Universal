#!/usr/bin/env python3

import numpy, pickle
import configparser, os, nltk, pandas, sys
sys.dont_write_bytecode = True
import glob, string, collections, operator

# negation prefix e.g. nC0032326
CUI_NEG_PREF = 'neg'

class DatasetProvider:
  """Data for BOW eval using sklearn"""

  def __init__(self, corpus_path):
    """Constructor"""

    self.corpus_path = corpus_path
    self.label2int = {'no':0, 'yes':1}

  def load_sklearn(self):
    """Assume each subdir is a separate class"""

    labels = []    # int labels
    examples = []  # examples as strings

    for dir in os.listdir(self.corpus_path):
      dir_path = os.path.join(self.corpus_path, dir)

      for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        file_text = open(file_path).read().rstrip()
        examples.append(file_text)
        int_label = self.label2int[dir.lower()]
        labels.append(int_label)

    return examples, labels

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train'))

  dataset = DatasetProvider(data_dir)
  x, y = dataset.load_sklearn()
  print(x[:3])
  print(y)
