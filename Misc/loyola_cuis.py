#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

codes_file = '/Users/Dima/Temp/rxnorm.codes'
output_dir = '/Users/Dima/Temp/Text/'

def parse():
  """Parse cui file"""

  for line in open(codes_file):
    elements = line.strip().split('|')
    report_id = elements[0]
    cui = elements[5]
    count = elements[9]

    out = open('%s%s.txt' % (output_dir, report_id), 'a')

    for _ in range(int(count)):
      out.write(cui + ' ')

if __name__ == "__main__":

  parse()
