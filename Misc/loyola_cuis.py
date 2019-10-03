#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import configparser

# CUI file header:
# RptID|RptDesc|MRN|Hsp_Account_ID|CodeDomain|Cui|Tui|PreferredText|Polarity|Count

def write_notes(cui_file, out_dir):
  """Parse cui file"""

  print('parsing file %s...' % cui_file)

  for line in open(cui_file):
    elements = line.strip().split('|')
    report_id = elements[0]
    cui = elements[5]
    polarity = elements[8]
    count = elements[9]

    out = open('%s%s.txt' % (out_dir, report_id), 'a')
    cui = cui if polarity == '1' else 'n%s' % cui

    for _ in range(int(count)):
      out.write(cui + ' ')

def write_notes_with_prefixes(cui_file, note_prefix, out_dir):
  """Parse cui file"""

  print('searching %s prefix in %s...' % (note_prefix, cui_file))

  for line in open(cui_file):
    elements = line.strip().split('|')
    report_id = elements[0]
    report_type = elements[1]
    cui = elements[5]
    count = elements[9]

    if report_type.startswith(note_prefix):
      out = open('%s%s.txt' % (out_dir, report_id), 'a')
      cui = cui if polarity == '1' else 'n%s' % cui

      for _ in range(int(count)):
        out.write(cui + ' ')

def write_encounters(cui_file, out_dir):
  """Parse notes into encounters"""

  print('parsing file %s...' % cui_file)

  for line in open(cui_file):
    elements = line.strip().split('|')
    encounter_id = elements[3]
    cui = elements[5]
    polarity = elements[8]
    count = elements[9]

    out = open('%s%s.txt' % (out_dir, encounter_id), 'a')
    cui = cui if polarity == '1' else 'n%s' % cui

    for _ in range(int(count)):
      out.write(cui + ' ')

def notes_to_files():
  """Main driver"""

  root_dir = cfg.get('args', 'root_dir')
  out_dir = cfg.get('args', 'out_dir')
  cui_files = cfg.get('args', 'cui_files')

  for cui_file in cui_files.split('|'):
    write_notes(root_dir + cui_file, root_dir + out_dir)
    
def notes_with_prefixes_to_files():
  """Main driver"""

  root_dir = cfg.get('args', 'root_dir')
  out_dir = cfg.get('args', 'out_dir')
  note_prefixes = cfg.get('args', 'note_prefixes')
  cui_files = cfg.get('args', 'cui_files')

  for cui_file in cui_files.split('|'):
    for note_prefix in note_prefixes.split('|'):
      write_notes_with_prefixes(root_dir + cui_file, note_prefix, root_dir + out_dir)

def encounters_to_files():
  """Main driver"""

  root_dir = cfg.get('args', 'root_dir')
  out_dir = cfg.get('args', 'out_dir')
  cui_files = cfg.get('args', 'cui_files')

  for cui_file in cui_files.split('|'):
    write_encounters(root_dir + cui_file, root_dir + out_dir)

if __name__ == "__main__":

  cfg = configparser.ConfigParser(allow_no_value=True)
  cfg.read(sys.argv[1])

  encounters_to_files()
  # notes_to_files()
