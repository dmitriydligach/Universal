#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, pickle, glob, operator, collections
import numpy, pandas
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_tokens(file_path, dropout=None):
  """Read n tokens from specified file"""

  tokens = open(file_path).read().split(' ')

  if dropout is not None:
    tokens_to_keep = round(len(tokens) * (1 - dropout))
    tokens = random.sample(tokens, tokens_to_keep)

  return tokens

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    targ_file,
    model_dir,
    min_examples_per_targ):
    """Constructor"""

    # for converting input tokens to int sequences
    self.tokenizer = Tokenizer(
      num_words=None,
      oov_token='oovtok',
      lower=False)

    # encounter id -> set of targets
    self.enc2targs = {}

    # target -> int index
    self.targ2int = {}

    self.train_dir = train_dir
    self.targ_file = targ_file
    self.model_dir = model_dir
    self.min_examples_per_targ = min_examples_per_targ

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    self.index_targets()

  def index_targets(self):
    """Process medication file"""

    df = pandas.read_csv(self.targ_file, dtype='str')

    for enc, drug in zip(df['HADM_ID'], df['DRUG']):
      if enc not in self.enc2targs:
        self.enc2targs[enc] = set()
      self.enc2targs[enc].add(drug.lower())

    targ_counter = collections.Counter()
    for targs in self.enc2targs.values():
      targ_counter.update(targs)

    outfile = open(os.path.join(self.model_dir, 'targs.txt'), 'w')
    for med, count in targ_counter.most_common():
      outfile.write('%s|%s\n' % (med, count))

    # make alphabet for *frequent* targets
    index = 0
    for targ, count in targ_counter.items():
      if count > self.min_examples_per_targ:
        self.targ2int[targ] = index
        index = index + 1

  def load(self):
    """Process notes to make x and y"""

    x = [] # input documents (n_docs, max_seq_len)
    y = [] # targets we are predicting for each input

    for train_file in os.listdir(self.train_dir):

      targ_vec = numpy.zeros(len(self.targ2int))
      enc = train_file.split('.')[0]

      if enc not in self.enc2targs:
        continue

      no_labels_for_this_file = True
      for targ in self.enc2targs[enc]:
        if targ in self.targ2int:
          targ_vec[self.targ2int[targ]] = 1
          no_labels_for_this_file = False

      if no_labels_for_this_file:
        continue # all rare codes

      y.append(targ_vec)

      tokens = read_tokens(os.path.join(self.train_dir, train_file))
      x.append(' '.join(set(tokens)))

    self.tokenizer.fit_on_texts(x)
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    print('input vocabulary size:', len(self.tokenizer.word_index))

    x = self.tokenizer.texts_to_sequences(x)
    max_seq_len = max(len(seq) for seq in x)
    x = pad_sequences(x, maxlen=max_seq_len)

    return x, numpy.array(y)

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    os.path.join(base, cfg.get('data', 'targs')),
    cfg.get('data', 'model_dir'),
    cfg.getfloat('args', 'min_examples_per_targ'))

  x, y = dp.load()
  print('x:', x.shape)
  print('y:', y.shape)
