#!/usr/bin/env python3

import sys, numpy, operator, os, glob
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = '/Users/Dima/Loyola/Data/MimicIII/Admissions/Text/'

max_files = 10000
top_tokens = 50

def show_top_tokens(train_files, target_file):
  """Vectorize and print"""

  vectorizer = TfidfVectorizer(
    input='filename',
    ngram_range=(1,1),
    stop_words='english')
  vectorizer.fit(train_files)

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
