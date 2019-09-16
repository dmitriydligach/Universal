#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import configparser

# CUI file header:
# RptID|RptDesc|MRN|Hsp_Account_ID|CodeDomain|Cui|Tui|PreferredText|Polarity|Count

def parse():
  """Parse cui file"""

  cui_file = cfg.get('args', 'cui_file')
  output_dir = cfg.get('args', 'output_dir')
  note_type = cfg.get('args', 'note_type')

  for line in open(cui_file):
    elements = line.strip().split('|')
    report_id = elements[0]
    report_type = elements[1]
    cui = elements[5]
    count = elements[9]

    if report_type != note_type:
      continue

    out = open('%s%s.txt' % (output_dir, report_id), 'a')
    for _ in range(int(count)):
      out.write(cui + ' ')

if __name__ == "__main__":

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  parse()
