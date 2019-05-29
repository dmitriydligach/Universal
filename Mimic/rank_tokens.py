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
    train_files = glob.glob(source_dir + '*.txt')
    vectorizer = TfidfVectorizer(
      input='filename',
      ngram_range=(1,1))
    vectorizer.fit(train_files)
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

  disch_files = glob.glob(source_dir + '*_discharge.txt')
  features = vectorizer.get_feature_names()
  doc_term_mat = vectorizer.transform(disch_files)
  print('done generating doc term matrix...')

  # write ranked tokens for each discharge summary
  for disch_file_index, disch_file in enumerate(disch_files):
    token2score = {}

    # save tokens and their scores in a dictionary
    row_mat = coo_matrix(doc_term_mat.getrow(disch_file_index))
    for col, score in zip(row_mat.col, row_mat.data):
      token2score[features[col]] = score

    # sort tokens by score
    ranked = sorted(
      token2score.items(),
      key=operator.itemgetter(1),
      reverse=True)

    # save resulting list of tuples
    out = open(target_dir + disch_file.split('/')[-1], 'w')
    for token, score in ranked:
      out.write('%s %s\n' % (token, score))

if __name__ == "__main__":

  vectorizer = get_vectorizer()
  write_ranked_tokens(vectorizer)
