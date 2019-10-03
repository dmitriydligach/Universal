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
    max_files,
    max_cuis,
    samples_per_doc,
    fetch_batches,
    batch_size):
    """Constructor"""

    self.samples_per_doc = samples_per_doc
    self.fetch_samples = batch_size * fetch_batches
    self.max_files = None if max_files == 'all' else int(max_files)

    # remove old model dir just in case
    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # file paths for tokenzier and training
    self.file_paths = glob.glob(train_dir + '*.txt')
    random.shuffle(self.file_paths)
    self.file_paths = self.file_paths[:self.max_files]
    print('total files:', len(self.file_paths))

    self.tokenizer = Tokenizer(
      num_words=max_cuis,
      oov_token='oovtok',
      lower=False)
    self.tokenize()

  def tokenize(self):
    """Read data and map words to ints"""

    x = [] # input documents
    for file_path in self.file_paths:
      file_as_string = open(file_path).read()
      x.append(file_as_string)

    self.tokenizer.fit_on_texts(x)
    print('vocabulary size:', len(self.tokenizer.word_index))
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.tokenizer, pickle_file)
    print('tokenizer saved: Model/tokenizer.p')

  def load(self):
    """Generate n examples at a time"""

    x = [] # one chunk of samples
    y = [] # labels for these samples

    pkl = open('Model/tokenizer.p', 'rb')
    self.tokenizer = pickle.load(pkl)

    count = 0 # track num of examples generated so far
    total_examples = len(self.file_paths) * self.samples_per_doc
    print('total examples:', total_examples)

    # loop over all files multiple times
    for pass_num in range(self.samples_per_doc):
      print('pass %d over files...' % pass_num)

      # loop over all files
      for file_path in self.file_paths:
        tokens = read_tokens(file_path)
        unique = list(set(tokens))
        x_count = round(len(unique) * 0.85)

        random.shuffle(unique)
        x.append(' '.join(unique[:x_count]))
        y.append(' '.join(unique[x_count:]))
        count = count + 1

        if len(x) == self.fetch_samples:
          print('fetching %d samples...' % self.fetch_samples)
          print('%d/%d generated so far...' % (count, total_examples))
          x = self.tokenizer.texts_to_matrix(x, mode='binary')
          y = self.tokenizer.texts_to_matrix(y, mode='binary')
          yield x, y[:, 1:]
          x, y = [], []

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
