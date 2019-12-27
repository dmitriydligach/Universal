#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import configparser, os, pickle, random
from keras.preprocessing.sequence import pad_sequences
import i2b2

class DatasetProvider:
  """Read data from files and make keras inputs/outputs"""

  def __init__(self,
               corpus_path,
               tokenizer_pickle,
               max_seq_len):
    """Index words by frequency in a file"""

    self.corpus_path = corpus_path
    self.max_seq_len = max_seq_len
    self.label2int = {'no':0, 'yes':1}

    pkl = open(tokenizer_pickle, 'rb')
    self.tokenizer = pickle.load(pkl)

  def load_as_int_seqs(self):
    """Convert examples into lists of indices"""

    x = [] # (n_docs, max_seq_len)
    y = [] # int labels

    for d in os.listdir(self.corpus_path):
      label_dir = os.path.join(self.corpus_path, d)

      for f in os.listdir(label_dir):
        y.append(self.label2int[d.lower()])

        file_path = os.path.join(label_dir, f)
        tokens = open(file_path).read().split()
        x.append(' '.join(set(tokens)))

    x = self.tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=self.max_seq_len)

    return x, y

  def load_as_one_hot(self):
    """Convert examples into lists of indices"""

    x = [] # eventually ndarray (n_docs, num_features)
    y = [] # int labels

    for d in os.listdir(self.corpus_path):
      label_dir = os.path.join(self.corpus_path, d)

      for f in os.listdir(label_dir):
        y.append(self.label2int[d.lower()])

        file_path = os.path.join(label_dir, f)
        tokens = open(file_path).read().split()
        x.append(' '.join(set(tokens)))

    x = self.tokenizer.texts_to_matrix(x, mode='binary')

    return x, y

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train'))
  tokenizer_pickle = cfg.get('data', 'tokenizer_pickle')

  dp = DatasetProvider(data_dir, tokenizer_pickle, None)
  x, y = dp.load_as_one_hot()

  print(x.shape)
  print(len(y))
