#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy, pickle, glob, operator, collections
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_tokens(file_path, dropout=None):
  """Read n tokens from specified file into a list"""

  tokens = open(file_path).read().split()

  if dropout is not None:
    tokens_to_keep = round(len(tokens) * (1 - dropout))
    tokens = random.sample(tokens, tokens_to_keep)

  return tokens

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    model_dir,
    n_examples):
    """Constructor"""

    self.train_dir = train_dir
    self.n_examples = n_examples

    self.tokenizer = Tokenizer(oov_token='oovtok', lower=False)

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

  def load(self):
    """Process notes to make x and y"""

    x = [] # input documents (n_docs, max_seq_len)
    labels = [] # targets we are predicting for each input

    for file_path in glob.glob(self.train_dir + '*.txt'):
      tokens = read_tokens(file_path)
      unique = list(set(tokens))
      x_count = round(len(unique) * 0.85)

      for _ in range(self.n_examples):
        random.shuffle(unique)
        x.append(' '.join(unique[:x_count]))
        labels.append(unique[x_count:])

    # make x
    self.tokenizer.fit_on_texts(x)
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    x = self.tokenizer.texts_to_matrix(x, mode='binary')
    print('input vocabulary size:', len(self.tokenizer.word_index))

    # determine unique targets
    uniq_targs = set()
    for targ_list in labels:
      uniq_targs.update(set(targ_list))

    # map targets to indices
    index = 0
    targ2int = {}
    for targ in uniq_targs:
      targ2int[targ] = index
      index = index + 1

    # convert labels to one-hot numpy arrays
    y = []
    for targ_list in labels:
      targ_vec = numpy.zeros(len(uniq_targs))
      for targ in targ_list:
        targ_vec[targ2int[targ]] = 1
      y.append(targ_vec)

    return x, numpy.array(y)

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.getint('args', 'n_examples'))

  x, y = dp.load()
  print('x:', x.shape)
  print('y:', y.shape)
