#!/usr/bin/env python3

import configparser, os, pandas, sys, glob
sys.dont_write_bytecode = True
import collections, pickle, shutil
from keras.preprocessing.text import Tokenizer

model_dir = 'Model/'
alphabet_file = 'Model/alphabet.txt'
alphabet_pickle = 'Model/alphabet.p'
diag_icd_file = 'DIAGNOSES_ICD.csv'
proc_icd_file = 'PROCEDURES_ICD.csv'
cpt_code_file = 'CPTEVENTS.csv'

def read_tokens(file_path, dropout=None):
  """Read n tokens from specified file into a list"""

  tokens = open(file_path).read().split()

  if dropout is not None:
    tokens_to_keep = round(len(tokens) * (1 - dropout))
    tokens = random.sample(tokens, tokens_to_keep)

  return tokens

class DatasetProvider:
  """Notes and ICD code data"""

  def __init__(self,
               input_dir,
               output_dir,
               max_cuis,
               max_codes):
    """Construct it"""

    self.input_dir = input_dir
    self.output_dir = output_dir

    # remove old model directory and make a fresh one
    if os.path.isdir(model_dir):
      shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # map encounters to icd codes
    self.enc2codes = {}

    diag_code_path = os.path.join(self.output_dir, diag_icd_file)
    proc_code_path = os.path.join(self.output_dir, proc_icd_file)
    cpt_code_path = os.path.join(self.output_dir, cpt_code_file)

    self.index_codes(diag_code_path, 'ICD9_CODE', 'diag', 3)
    self.index_codes(proc_code_path, 'ICD9_CODE', 'proc', 2)
    self.index_codes(cpt_code_path, 'CPT_NUMBER', 'cpt', 5)

    # index input text
    self.input_tokenizer = Tokenizer(
      num_words=None if max_cuis == 'all' else int(max_cuis),
      oov_token='oovtok',
      lower=False)
    self.tokenize_input()

    # index outputs (codes)
    self.output_tokenizer = Tokenizer(
      num_words=None if max_codes == 'all' else int(max_codes),
      oov_token='oovtok',
      lower=False)
    self.tokenize_output()

  def index_codes(self, code_file, code_col, prefix, num_digits):
    """Map encounters to codes"""

    frame = pandas.read_csv(code_file, dtype='str')

    for id, code in zip(frame['HADM_ID'], frame[code_col]):
      if pandas.isnull(id):
        continue # some subjects skipped (e.g. 13567)
      if pandas.isnull(code):
        continue

      if id not in self.enc2codes:
        self.enc2codes[id] = set()

      short_code = '%s_%s' % (prefix, code[0:num_digits])
      self.enc2codes[id].add(short_code)

  def tokenize_input(self):
    """Read text and map tokens to ints"""

    x = [] # input documents
    for file_path in glob.glob(self.input_dir + '*.txt'):
      file_as_string = open(file_path).read()
      x.append(file_as_string)

    # index inputs and save to use in evaluation
    self.input_tokenizer.fit_on_texts(x)
    pickle_file = open('Model/tokenizer.p', 'wb')
    pickle.dump(self.input_tokenizer, pickle_file)
    print('input vocab:', len(self.input_tokenizer.word_index))

  def tokenize_output(self):
    """Map codes to ints"""

    y = [] # codes for input documents
    for _, codes in self.enc2codes.items():
      codes_as_string = ' '.join(codes)
      y.append(codes_as_string)

    self.output_tokenizer.fit_on_texts(y)
    print('output vocab:', len(self.output_tokenizer.word_index))

  def load(self):
    """Make x and y"""

    x = []
    y = []

    # make a list of inputs and outputs to vectorize
    for file_path in glob.glob(self.input_dir + '*.txt'):
      id = file_path.split('/')[-1].split('.')[0]
      if id not in self.enc2codes:
        continue

      file_as_string = open(file_path).read()
      x.append(file_as_string)

      codes_as_string = ' '.join(self.enc2codes[id])
      y.append(codes_as_string)

    # make x and y matrices
    x = self.input_tokenizer.texts_to_matrix(x, mode='binary')
    y = self.output_tokenizer.texts_to_matrix(y, mode='binary')

    # column zero is empty
    return x, y[:,1:]

if __name__ == "__main__":
  """Test dataset class"""

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dataset = DatasetProvider(
    os.path.join(base, cfg.get('data', 'train')),
    os.path.join(base, cfg.get('data', 'codes')),
    cfg.getint('args', 'max_cuis'))
  x, y = dataset.load()

  print('x shape:', x.shape)
  print('y shape:', y.shape)
