#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    model_dir,
    max_seq_len,
    n_files,
    split):
    """Constructor"""

    self.tokenizer = Tokenizer(oov_token='oov_token')

    self.train_dir = train_dir
    self.max_seq_len = max_seq_len
    self.n_files = None if n_files == 'all' else int(n_files)
    self.split = split

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

  def load(self):
    """Make x and y"""

    x1 = [] # to turn into a np array (n_docs, max_seq_len)
    x2 = [] # to turn into a np array (n_docs, max_seq_len)

    for file in os.listdir(self.train_dir)[:self.n_files]:
      path = os.path.join(self.train_dir, file)
      tokens = open(path).read().split()
      unique = list(set(tokens))
      random.shuffle(unique)

      x1_count = round(len(unique) * self.split)
      x1.append(' '.join(unique[:x1_count]))
      x2.append(' '.join(unique[x1_count:]))

    self.tokenizer.fit_on_texts(x1 + x2)

    x1 = self.tokenizer.texts_to_sequences(x1)
    x2 = self.tokenizer.texts_to_sequences(x2)

    x1 = pad_sequences(x1, maxlen=self.max_seq_len)
    x2 = pad_sequences(x2, maxlen=self.max_seq_len)

    # twice the size of x1 with  half as 1s and the rest 0s
    y = numpy.concatenate((
      numpy.ones(x1.shape[0], dtype='int'),
      numpy.zeros(x1.shape[0], dtype='int')))

    # make negative examples by pairing x1 with permuted x2
    x1 = numpy.concatenate((x1, x1))
    x2 = numpy.concatenate((x2, numpy.random.permutation(x2)))

    return x1, x2, y

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))

  dat_prov = DatasetProvider(train_dir)
  x1, x2, y = dat_prov.load()

  print('x1.shape:', x1.shape)
  print('x2.shape:', x2.shape)
  print('y.shape:', y.shape)
  print('y:', y)
