#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import configparser, os, pickle, random
from keras.preprocessing.sequence import pad_sequences
import i2b2

class DatasetProvider:
  """Comorboditiy data loader"""

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

  def load(self):
    """Convert examples into lists of indices"""

    x1 = [] # first halfs (n_docs, max_seq_len)
    x2 = [] # second halfs (n_docs, max_seq_len)
    y = []  # int labels

    for d in os.listdir(self.corpus_path):
      label_dir = os.path.join(self.corpus_path, d)

      for f in os.listdir(label_dir):
        y.append(self.label2int[d.lower()])

        file_path = os.path.join(label_dir, f)
        tokens = open(file_path).read().split()
        unique = list(set(tokens))
        random.shuffle(unique)

        x1_count = round(len(unique) * 0.5)
        x1.append(' '.join(unique[:x1_count]))
        x2.append(' '.join(unique[x1_count:]))

    x1 = self.tokenizer.texts_to_sequences(x1)
    x2 = self.tokenizer.texts_to_sequences(x2)

    x1 = pad_sequences(x1, maxlen=self.max_seq_len)
    x2 = pad_sequences(x2, maxlen=self.max_seq_len)

    return x1, x2, y

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train'))
  tokenizer_pickle = cfg.get('data', 'tokenizer_pickle')

  dp = DatasetProvider(data_dir, tokenizer_pickle, 1000)
  x1, x2, y = dp.load()

  print(x1.shape)
  print(x2.shape)

  print(len(y))

  print('x1:', x1)
  print('x2:', x2)
  print('y:', y)
