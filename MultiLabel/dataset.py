#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy, pickle, glob, operator, collections
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def read_tokens(file_path, n_tokens, dropout):
  """Read n tokens from specified file"""

  tokens = []
  for line in open(file_path).readlines()[:n_tokens]:
    token, score = line.split(' ')
    tokens.append(token)

  if dropout is not None:
    tokens_to_keep = round(len(tokens) * (1 - dropout))
    tokens = random.sample(tokens, tokens_to_keep)

  return ' '.join(set(tokens))

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    model_dir,
    max_seq_len,
    n_x_cuis,
    n_y_cuis,
    min_examples_per_targ):
    """Constructor"""

    self.tokenizer = Tokenizer(oov_token='oovtok', lower=False)

    self.enc2targs = {} # encounter id -> set of targets
    self.targ2int = {}  # target -> index

    self.train_dir = train_dir
    self.max_seq_len = max_seq_len
    self.min_examples_per_targ = min_examples_per_targ

    self.n_x_cuis = None if n_x_cuis == 'all' else int(n_x_cuis)
    self.n_y_cuis = None if n_y_cuis == 'all' else int(n_y_cuis)

    self.index() # make all sort of indicies

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

  def index(self):
    """Map encounters to targets and make target alphabet"""

    for disch_file in glob.glob(self.train_dir + '*_discharge.txt'):

      # map encounters to targets
      tokens = []
      for line in open(disch_file).readlines()[:self.n_x_cuis]:
        token, score = line.split(' ')
        tokens.append(token)

      enc_id = disch_file.split('/')[-1].split('_')[0]
      self.enc2targs[enc_id] = set(tokens)

    # count target frequencies
    targ_counter = collections.Counter()
    for targs in self.enc2targs.values():
      targ_counter.update(targs)

    # make target alplhabet
    index = 0
    for targ, count in targ_counter.items():
      if count > self.min_examples_per_targ:
        self.targ2int[targ] = index
        index = index + 1

  def load(self):
    """Make x and y"""

    x = [] # (n_docs, max_seq_len)
    y = [] # targets

    for disch_path in glob.glob(self.train_dir + '*_discharge.txt'):

      rest_path = disch_path.split('_')[0] + '_rest.txt'
      if not os.path.exists(rest_path):
        continue
      x.append(read_tokens(rest_path, self.n_x_cuis, None))

      targ_vec = numpy.zeros(len(self.targ2int))
      enc_id = disch_path.split('/')[-1].split('_')[0]

      for targ in self.enc2targs[enc_id]:
        targ_vec[self.targ2int[targ]] = 1
      y.append(targ_vec)

    self.tokenizer.fit_on_texts(x)
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)

    x = self.tokenizer.texts_to_sequences(x)
    x = pad_sequences(x2, maxlen=self.max_seq_len)

    return x, numpy.array(y)

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.getint('args', 'max_seq_len'),
    cfg.get('args', 'n_x_cuis'),
    cfg.get('args', 'n_y_cuis'),
    cfg.getfloat('args', 'min_examples_per_targ'))

  dp.load()
