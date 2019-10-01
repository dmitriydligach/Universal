#!/usr/bin/env python3

import sys, random, glob, shutil

source_dir = '1/'
dest_dir = '2/'

def select_and_move(split=0.2):
  """Pick random files and move to another dir"""

  all = glob.glob(source_dir + '*.txt')
  sample = random.sample(all, int(len(all) * split))
  print('moving %d files...' % len(sample))

  for item in sample:
    shutil.move(item, dest_dir)

if __name__ == "__main__":

  select_and_move()
