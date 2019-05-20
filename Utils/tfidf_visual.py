#!/usr/bin/env python3

import sys, numpy, operator, os, glob, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = '/Users/Dima/Loyola/Data/MimicIII/Admissions/Cuis/'
vectorizer_pickle = 'Model/vectorizer.p'

max_files = None
top_tokens = 50

def show_top_tokens(train_files, target_file):
  """Vectorize and print"""

  if not os.path.isfile(vectorizer_pickle):
    vectorizer = TfidfVectorizer(
      input='filename',
      ngram_range=(1,1),
      stop_words='english')
    vectorizer.fit(train_files)
    pickle_file = open(vectorizer_pickle, 'wb')
    pickle.dump(vectorizer, pickle_file)
  else:
    pkl = open(vectorizer_pickle, 'rb')
    vectorizer = pickle.load(pkl)

  matrix = vectorizer.transform(target_file).toarray()

  token2weight = {}
  features = vectorizer.get_feature_names()

  for dim, weight in enumerate(matrix[0, :]):
    if weight > 0:
      token2weight[features[dim]] = weight

  ranked = sorted(
    token2weight.items(),
    key=operator.itemgetter(1),
    reverse=True)

  for token, weight in ranked[:top_tokens]:
    print(token, weight)

if __name__ == "__main__":

  train = glob.glob(corpus + '*.txt')
  target = [corpus + sys.argv[1]]

  show_top_tokens(train[:max_files], target)
