import numpy as np
import os, os.path

def read_tokens(file_path):
  """Return file as a list of ngrams"""

  text = open(file_path).read().lower()

  tokens = [] 
  for token in text.split():
    if token.isalpha():
      tokens.append(token)

  return tokens

def read_cuis(file_path, ignore_negation=False):
  """Return a file as a list of CUIs"""

  text = open(file_path).read()

  if ignore_negation:
    tokens = []
    for token in text.split():
      if token.startswith('n'):
        tokens.append(token[1:])
      else:
        tokens.append(token)
    return tokens

  else:
    return text.split()

if __name__ == "__main__":

  print()
