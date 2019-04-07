#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MODEL_DIR = 'Model/'
TRAIN_SIZE = 0.50

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(self, corpus_path):
    """Constructor"""

    self.corpus_path = corpus_path
    self.tokenizer = Tokenizer(oov_token='oov_token')

    # prepare model directory
    if os.path.isdir(MODEL_DIR):
      shutil.rmtree(MODEL_DIR)
    os.mkdir(MODEL_DIR)

  def load(self):
    """Make x and y"""

    x = []
    y = []

    for file in os.listdir(self.corpus_path)[:500]:
      path = os.path.join(self.corpus_path, file)
      text = open(path).read().split()
      unique = list(set(text))

      random.shuffle(unique)

      train_examples = round(len(unique) * TRAIN_SIZE)
      test_examples = len(unique) - train_examples
      x.append(' '.join(unique[:train_examples]))
      y.append(' '.join(unique[train_examples:]))

    self.tokenizer.fit_on_texts(x)
    x = self.tokenizer.texts_to_sequences(x)
    y = self.tokenizer.texts_to_sequences(y)

    x = pad_sequences(x, maxlen=1000)
    y = pad_sequences(y, maxlen=1000)

    # y = numpy.random.randint(0, 2, size=x.shape[0])

    return x, y

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))

  dat_prov = DatasetProvider(train_dir)
  x, y = dat_prov.load()
