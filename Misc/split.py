#!/usr/bin/env python3

import sys, random, glob, shutil, os

source_dir = 'MimicIII/Encounters/Cuis/Train/'
dest_dir = 'MimicIII/Encounters/Cuis/Dev/'

base = os.environ['DATA_ROOT']
source_dir = os.path.join(base, source_dir)
dest_dir = os.path.join(base, dest_dir)

# same split every time
random.seed(0)

def select_and_move(split=0.20):
  """Pick random files and move to another dir"""

  all = glob.glob(source_dir + '*.txt')
  sample = random.sample(all, int(len(all) * split))
  print('source dir:', source_dir)
  print('dest dir:', dest_dir)
  print('moving {} files...'.format(len(sample)))

  for item in sample:
    shutil.move(item, dest_dir)

if __name__ == "__main__":

  select_and_move()
