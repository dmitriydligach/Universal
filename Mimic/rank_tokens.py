#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, shutil, glob, operator, configparser
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix

def main():
  """Rank tokens in all files by tfidf"""

  from_dir = os.path.join(base, cfg.get('args', 'from'))
  to_dir = os.path.join(base, cfg.get('args', 'to'))
  file_paths = glob.glob(from_dir + '*.txt')

  if cfg.get('args', 'max_features') == 'all':
    max_features = None
  else:
    max_features = int(cfg.get('args', 'max_features'))

  vectorizer = TfidfVectorizer(
    lowercase=False,
    input='filename',
    ngram_range=(1, 1),
    max_df=cfg.getfloat('args', 'max_df'),
    min_df=cfg.getfloat('args', 'min_df'),
    max_features=max_features)
  doc_term_mat = vectorizer.fit_transform(file_paths)

  features = vectorizer.get_feature_names()
  print('done generating doc term matrix...')
  print('vocabulary size:', len(vectorizer.vocabulary_))

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
    out = open(to_dir + file_path.split('/')[-1], 'w')
    for token, score in ranked:
      out.write('%s %s\n' % (token, score))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  main()
