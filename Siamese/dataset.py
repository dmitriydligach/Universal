#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, shutil, random, numpy, pickle, glob, operator
from configparser import ConfigParser

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(
    self,
    train_dir,
    model_dir,
    max_seq_len,
    n_files,
    n_cuis,
    tokenizer_pickle='tokenizer.p',
    vectorizer_pickle='vectorizer.p',
    target_pickle='target.p'):
    """Constructor"""

    # if os.path.isdir(model_dir):
    #   shutil.rmtree(model_dir)
    # os.mkdir(model_dir)

    self.tokenizer_pickle = model_dir + tokenizer_pickle
    self.vectorizer_pickle = model_dir + vectorizer_pickle
    self.target_pickle = model_dir + target_pickle

    self.train_dir = train_dir
    self.max_seq_len = max_seq_len
    self.n_files = None if n_files == 'all' else int(n_files)
    self.n_cuis = None if n_cuis == 'all' else int(n_cuis)

    self.tokenizer = Tokenizer(oov_token='oovtok', lower=False)

    self.get_vectorizer()
    self.get_targets()

  def get_vectorizer(self):
    """Train or load a tfidf vectorizer"""

    if not os.path.isfile(self.vectorizer_pickle):
      train_files = glob.glob(self.train_dir + '*.txt')

      self.vectorizer = TfidfVectorizer(
        input='filename',
        ngram_range=(1,1))
      self.vectorizer.fit(train_files)

      print('saving vectorizer:', self.vectorizer_pickle)
      pickle_file = open(self.vectorizer_pickle, 'wb')
      pickle.dump(vectorizer, pickle_file)

    else:
      print('loading vectorizer:', self.vectorizer_pickle)
      pl = open(self.vectorizer_pickle, 'rb')
      self.vectorizer = pickle.load(pl)

  def get_targets(self):
    """Rank tokens in all files by tfidf"""

    doc2target = {} # document -> list of target tokens

    disch_files = glob.glob(self.train_dir + '*_discharge.txt')
    doc_term_matrix = self.vectorizer.transform(disch_files).toarray()
    features = self.vectorizer.get_feature_names()

    for row_n in range(doc_term_matrix.shape[0]):

      token2score = {}
      for dim, score in enumerate(doc_term_matrix[row_n, :]):
        if score > 0:
          token2score[features[dim]] = score

      ranked = sorted(
        token2score.items(),
        key=operator.itemgetter(1),
        reverse=True)

      doc2target[disch_files[row_n]] = []
      for token, score in ranked[:self.n_cuis]:
        doc2target[disch_files[row_n]].append(token)

    print('saving targets:', self.target_pickle)
    pickle_file = open(self.target_pickle, 'wb')
    pickle.dump(doc2target, pickle_file)

  def targets(self):
    """Look at discharge summaries and figure out what to predict"""

    # TODO: one path to allow filtering of CUIs based on frequencies
    # for both rest and discharge summaries is to tokenize the entire
    # corpus here and make one set for targets and one for the rest

    # tokenizer for discharge summaries
    tokenizer = Tokenizer(lower=False)

    texts = []
    discharge_files = self.train_dir + '*_discharge.txt'
    for disch_file in glob.glob(discharge_files)[:self.n_files]:
      text = open(disch_file).read().replace('n', '')
      texts.append(text)

    tokenizer.fit_on_texts(texts)
    counts = sorted(
      tokenizer.word_counts.items(),
      # tokenizer.word_docs.items(),
      key=operator.itemgetter(1),
      reverse=True)

    target_set = set()
    for cui, count in counts[:self.n_cuis]:
      target_set.add(cui)

    return target_set

  def load(self):
    """Make x and y"""

    x1 = [] # to turn into a np array (n_docs, max_seq_len)
    x2 = [] # to turn into a np array (n_docs, max_seq_len)

    target_set = self.targets()
    disch_file_pattern = self.train_dir + '*_discharge.txt'
    for disch_file in glob.glob(disch_file_pattern)[:self.n_files]:

      rest_file = disch_file.split('_')[0] + '_rest.txt'
      if not os.path.exists(rest_file):
        continue

      x1_text = open(rest_file).read().replace('n', '')
      x2_text = open(disch_file).read().replace('n', '')

      x1_tokens = set(x1_text.split())
      x2_tokens = set(x2_text.split())

      x2_tokens = x2_tokens.intersection(target_set)
      if len(x2_tokens) == 0:
        print('wow such empty')
        continue

      x1.append(' '.join(x1_tokens))
      x2.append(' '.join(x2_tokens))

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
    cfg.get('args', 'n_files'),
    cfg.get('args', 'n_cuis'))

  # x1, x2, y = dp.load()
  # print('x1.shape:', x1.shape)
  # print('x2.shape:', x2.shape)
  # print('y.shape:', y.shape)
  # print('y:', y)
