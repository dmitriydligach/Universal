#!/usr/bin/env python3

import pandas, os

# where to read and store the data
ROOT = '/Users/Dima/Loyola/Data/Injury/'
CSV = 'Csv/001_Chest-Trauma-CT-All.csv'

# convert csv values into path
test2path = {'0': 'Train/', '1': 'Test/'}
injury2path = {'TRUE': 'Yes/', 'FALSE': 'No/'}

import sys
sys.dont_write_bytecode = True

def make_dir_struct():
  """Train/Test with Yes/No subdirectories"""

  os.mkdir(os.path.join(ROOT, 'Train/'))
  os.mkdir(os.path.join(ROOT, 'Train/Yes/'))
  os.mkdir(os.path.join(ROOT, 'Train/No/'))
  os.mkdir(os.path.join(ROOT, 'Test/'))
  os.mkdir(os.path.join(ROOT, 'Test/Yes/'))
  os.mkdir(os.path.join(ROOT, 'Test/No/'))

def parse_csv():
  """Write individual encounter files"""

  csv_path = os.path.join(ROOT, CSV)
  df = pandas.read_csv(csv_path, dtype='str')

  for har, test, injury, cuis in zip(df.har, df.test, df.thoracic_injury, df.cuis):

    out_path = os.path.join(ROOT, test2path[test], injury2path[injury])
    out_file = open('%s%s.txt' % (out_path, har), 'w')
    out_string = ' '.join(cuis.split(',')) + '\n'
    out_file.write(out_string)

if __name__ == "__main__":

  make_dir_struct()
  parse_csv()
