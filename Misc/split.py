#!/usr/bin/env python3

import sys, random, glob, shutil

source_dir = '/home/dima/Temp/Train/'
dest_dir = '/home/dima/Temp/Dev/'

def select_and_move(split=0.15):
  """Pick random files and move to another dir"""

  all = glob.glob(source_dir + '*.txt')
  sample = random.sample(all, int(len(all) * split))
  print('moving %d files...' % len(sample))

  for item in sample:
    shutil.move(item, dest_dir)

if __name__ == "__main__":

  select_and_move()
