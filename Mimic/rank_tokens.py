#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, shutil, numpy, pickle, glob, operator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix

vectorizer_pickle = './Model/vectorizer.p'

source_dir = '/Users/Dima/Loyola/Data/MimicIII/Discharge/Cuis/'
target_dir = '/Users/Dima/Loyola/Data/MimicIII/Discharge/Ranked/'

def get_vectorizer():
  """Train or load a tfidf vectorizer"""

  if not os.path.isfile(vectorizer_pickle):
    file_paths = glob.glob(source_dir + '*.txt')

    vectorizer = TfidfVectorizer(
      lowercase=False,
      input='filename',
      ngram_range=(1, 1),
      max_df=0.95,
      min_df=5)
    vectorizer.fit(file_paths)

    print('saving vectorizer:', vectorizer_pickle)
    pickle_file = open(vectorizer_pickle, 'wb')
    pickle.dump(vectorizer, pickle_file)
  else:
    print('loading vectorizer:', vectorizer_pickle)
    pl = open(vectorizer_pickle, 'rb')
    vectorizer = pickle.load(pl)

  return vectorizer

def write_ranked_tokens(vectorizer):
  """Rank tokens in all files by tfidf"""

  file_paths = glob.glob(source_dir + '*.txt')
  features = vectorizer.get_feature_names()
  doc_term_mat = vectorizer.transform(file_paths)
  print('done generating doc term matrix...')

  # write ranked tokens for each file
  for file_index, file_path in enumerate(file_paths):
    token2score = {}

    # save tokens and their scores in a dictionary
    row_mat = coo_matrix(doc_term_mat.getrow(file_index))
    for col, score in zip(row_mat.col, row_mat.data):
      token2score[features[col]] = score

    # sort tokens by score
    ranked = sorted(
      token2score.items(),
      key=operator.itemgetter(1),
      reverse=True)

    # save resulting list of token-score tuples
    out = open(target_dir + file_path.split('/')[-1], 'w')
    for token, score in ranked:
      out.write('%s %s\n' % (token, score))

if __name__ == "__main__":

  vectorizer = get_vectorizer()
  write_ranked_tokens(vectorizer)
