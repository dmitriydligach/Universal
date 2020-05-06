#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
import glob, numpy

def main():
  """Count tokens in files"""

  cui_files = '/Users/Dima/Work/Data/MimicIII/Encounters/Cuis/All/*.txt'

  counts = []
  for file_path in glob.glob(cui_files):

    file_as_string = open(file_path).read()
    n_tokens = len(file_as_string.split())
    counts.append(n_tokens)

  print('mean:', numpy.mean(counts))
  print('median:', numpy.median(counts))
  print('std:', numpy.std(counts))
  print('min:', numpy.min(counts))
  print('max:', numpy.max(counts))



if __name__ == "__main__":

  main()