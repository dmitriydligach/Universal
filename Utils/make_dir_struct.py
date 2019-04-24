#!/usr/bin/env python3

import sys, pandas, shutil, os
sys.dont_write_bytecode = True

base = '/Users/Dima/Loyola/Data/Opioids1k/'
split = '/Users/Dima/Loyola/Data/Opioids1k/split.csv'

partition2dir = {1:'Train', 2:'Dev', 3:'Test'}
label2dir = {0:'No', 1:'Yes'}

def make_dirs():
  """Create directory structure"""

  for partition in partition2dir.values():
    os.mkdir(os.path.join(base, partition))
    os.mkdir(os.path.join(base, partition, 'Yes'))
    os.mkdir(os.path.join(base, partition, 'No'))

def main():
  """Read split info"""

  df = pandas.read_csv(split, sep='|')
  for id, partition, label in zip(df.hsp_account_id, df.splitType, df.gold_label):
    src = os.path.join(base, 'All/', str(id) + '.txt')
    dst = os.path.join(base, partition2dir[partition], label2dir[label], str(id) + '.txt')
    shutil.copyfile(src, dst)

if __name__ == "__main__":

  make_dirs()
  main()
