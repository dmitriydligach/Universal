#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import os, shutil
from configparser import ConfigParser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MODEL_DIR = 'Model/'

class DatasetProvider:
  """Make x and y from raw data"""

  def __init__(self, corpus_path):
    """Constructor"""

    texts = []
    for file in os.listdir(corpus_path):
      path = os.path.join(corpus_path, file)
      text = open(path).read()
      texts.append(text)
    print('finished reading files')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    print('finished building vocabulary')

    if os.path.isdir(MODEL_DIR):
      print('removing old model directory...')
      shutil.rmtree(MODEL_DIR)
    os.mkdir(MODEL_DIR)

    # self.make_and_write_token_alphabet()

  def load(self,
           maxlen=float('inf'),
           tokens_as_set=True):
    """Convert examples into lists of indices"""

    codes = []    # each example has multiple codes
    examples = [] # int sequence represents each example

    for file in os.listdir(self.corpus_path):
      file_ngram_list = None
      if self.use_cuis == True:
        file_ngram_list = self.read_cuis(file)
      else:
        file_ngram_list = self.read_tokens(file)
      if file_ngram_list == None:
        continue # file too long

      # make code vector for this example
      subj_id = file.split('.')[0]
      if subj_id not in self.subj2codes:
        continue # subject was present once with no code
      if len(self.subj2codes[subj_id]) == 0:
        continue # shouldn't happen

      code_vec = [0] * len(self.code2int)
      for icd9_category in self.subj2codes[subj_id]:
        if icd9_category in self.code2int:
          # this icd9 has enough examples
          code_vec[self.code2int[icd9_category]] = 1

      if sum(code_vec) == 0:
        continue # all rare codes for this file

      codes.append(code_vec)

      # represent this example as a list of ints
      example = []

      if tokens_as_set:
        file_ngram_list = set(file_ngram_list)

      for token in file_ngram_list:
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      examples.append(example)

    return examples, codes

if __name__ == "__main__":

  cfg = ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, cfg.get('data', 'train'))

  dataset = DatasetProvider(train_dir)
  # x, y = dataset.load()
