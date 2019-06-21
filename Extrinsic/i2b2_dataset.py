#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import configparser, os, pickle, random
from keras.preprocessing.sequence import pad_sequences
import i2b2

# can be used to turn this into a binary task
LABEL2INT = {'Y':0, 'N':1, 'Q':2, 'U':3}

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               corpus_path,
               annot_xml,
               disease,
               judgement,
               tokenizer_pickle,
               max_seq_len):
    """Constructor"""

    self.corpus_path = corpus_path
    self.annot_xml = annot_xml
    self.disease = disease
    self.judgement = judgement
    self.max_seq_len = max_seq_len

    pkl = open(tokenizer_pickle, 'rb')
    self.tokenizer = pickle.load(pkl)

  def load(self):
    """Convert examples into lists of indices for keras"""

    x = [] # to turn into a np array (n_docs, max_seq_len)
    y = [] # int labels

    # document id -> label mapping
    doc2label = i2b2.parse_standoff(
      self.annot_xml,
      self.disease,
      self.judgement)

    # load examples and labels
    for f in os.listdir(self.corpus_path):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.corpus_path, f)

      # no labels for some documents for some reason
      if doc_id in doc2label:
        string_label = doc2label[doc_id]
        int_label = LABEL2INT[string_label]
        y.append(int_label)
      else:
        continue

      # tokens = open(file_path).read().replace('n', '').split()
      tokens = open(file_path).read().split()
      x.append(' '.join(set(tokens)))

    x = self.tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=self.max_seq_len)

    return x, y

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, cfg.get('data', 'train_data'))
  annot_xml = os.path.join(base, cfg.get('data', 'train_annot'))
  tokenizer_pickle = cfg.get('data', 'tokenizer_pickle')

  dp = DatasetProvider(
    data_dir,
    annot_xml,
    'CHF',
    'intuitive',
    tokenizer_pickle,
    1000)
  x1, x2, y = dp.load()

  print(x1.shape)
  print(x2.shape)
  print(len(y))
  print('x1:', x1)
  print('x2:', x2)
  print('y:', y)
