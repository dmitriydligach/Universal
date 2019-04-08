#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MODEL_DIR = 'Model/'
TRAIN_SIZE = 0.50
MAXLEN = 1000

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

    x1 = []
    x2 = []

    for file in os.listdir(self.corpus_path)[:500]:
      path = os.path.join(self.corpus_path, file)
      tokens = open(path).read().split()
      unique = list(set(tokens))

      random.shuffle(unique)
      x1_count = round(len(unique) * TRAIN_SIZE)
      x1.append(' '.join(unique[:x1_count]))
      x2.append(' '.join(unique[x1_count:]))

    self.tokenizer.fit_on_texts(x1 + x2)
    x1 = self.tokenizer.texts_to_sequences(x1)
    x2 = self.tokenizer.texts_to_sequences(x2)
    
    x1 = pad_sequences(x1, maxlen=MAXLEN)
    x2 = pad_sequences(x2, maxlen=MAXLEN)

    return x1, x2

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))

  dat_prov = DatasetProvider(train_dir)
  x, y = dat_prov.load()
