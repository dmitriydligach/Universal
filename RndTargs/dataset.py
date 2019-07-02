#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy, pickle, glob, operator, collections
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    model_dir,
    min_examples_per_targ):
    """Constructor"""

    # input text tokenizer
    self.tokenizer = Tokenizer(oov_token='oovtok', lower=False)
    # encounter id -> set of targets
    self.enc2targs = {}
    # target -> index
    self.targ2int = {}

    self.train_dir = train_dir
    self.min_examples_per_targ = min_examples_per_targ

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

  def load(self):
    """Process notes to make x and y"""

    x = [] # input documents (n_docs, max_seq_len)
    y = [] # targets we are predicting for each input

    targ_counter = collections.Counter()

    # make x, count targets, and map enc_id to targets
    for file_path in glob.glob(self.train_dir + '*.txt'):

      tokens = open(file_path).read().split()
      unique = list(set(tokens))
      random.shuffle(unique)

      x_count = round(len(unique) * 0.85)
      x.append(' '.join(unique[:x_count]))

      targs = unique[x_count:]
      enc_id = file_path.split('.')[0]
      self.enc2targs[enc_id] = targs
      targ_counter.update(targs)

    # make x
    self.tokenizer.fit_on_texts(x)
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    print('input vocabulary size:', len(self.tokenizer.word_index))

    x = self.tokenizer.texts_to_sequences(x)
    max_seq_len = max(len(seq) for seq in x)
    x = pad_sequences(x, maxlen=max_seq_len)

    # figure out what targets to predict
    index = 0
    for targ, count in targ_counter.items():
      if count > self.min_examples_per_targ:
        self.targ2int[targ] = index
        index = index + 1

    # make y
    for enc_id, targs in self.enc2targs.items():
      targ_vec = numpy.zeros(len(self.targ2int))
      for targ in targs:
        if targ in self.targ2int:
          targ_vec[self.targ2int[targ]] = 1

      y.append(targ_vec)

    return x, numpy.array(y)

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.getfloat('args', 'min_examples_per_targ'))

  x, y = dp.load()
  print('x:', x.shape)
  print('y:', y.shape)
