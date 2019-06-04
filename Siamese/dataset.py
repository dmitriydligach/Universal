#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil, random, numpy, pickle, glob, operator
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
    n_files,
    n_x1_cuis,
    n_x2_cuis):
    """Constructor"""

    self.tokenizer = Tokenizer(oov_token='oovtok', lower=False)

    self.train_dir = train_dir
    self.max_seq_len = max_seq_len

    self.n_files = None if n_files == 'all' else int(n_files)
    self.n_x1_cuis = None if n_x1_cuis == 'all' else int(n_x1_cuis)
    self.n_x2_cuis = None if n_x2_cuis == 'all' else int(n_x2_cuis)

    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

  def load(self):
    """Make x and y"""

    x1 = [] # to turn into a np array (n_docs, max_seq_len)
    x2 = [] # to turn into a np array (n_docs, max_seq_len)

    disch_file_pattern = self.train_dir + '*_discharge.txt'

    for disch_file in glob.glob(disch_file_pattern)[:self.n_files]:
      rest_file = disch_file.split('_')[0] + '_rest.txt'
      if not os.path.exists(rest_file):
        continue

      x1.append(read_tokens(rest_file, self.n_x1_cuis, None))
      x2.append(read_tokens(disch_file, self.n_x2_cuis, None))

    self.tokenizer.fit_on_texts(x1 + x2)

    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)

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

  dp = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    cfg.get('data', 'model_dir'),
    cfg.getint('args', 'max_seq_len'),
    cfg.get('args', 'n_files'))

  dp.targets()

  # x1, x2, y = dp.load()
  # print('x1.shape:', x1.shape)
  # print('x2.shape:', x2.shape)
  # print('y.shape:', y.shape)
  # print('y:', y)
