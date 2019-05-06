#! /usr/bin/env python3
import pandas, string, os, math

NOTES_CSV = '/Users/Dima/Loyola/Data/MimicIII/Source/NOTEEVENTS.csv'
OUT_DIR = '/Users/Dima/Loyola/Data/MimicIII/Discharge/Text/'

def write_admissions_to_files():
  """Each admission is written to a separate file"""

  frame = pandas.read_csv(NOTES_CSV, dtype='str')

  for rowid, hadmid, text in zip(frame.ROW_ID, frame.HADM_ID, frame.TEXT):
    if pandas.isnull(hadmid):
      print('empty hadmid for rowid', rowid)
    else:
      printable = ''.join(c for c in text if c in string.printable)
      outfile = open('%s%s.txt' % (OUT_DIR, hadmid), 'a')
      outfile.write(printable + '\n')
      outfile.write('\n**************************\n\n')

def write_patients_to_files():
  """Write files to one directory. Group by patient."""

  frame = pandas.read_csv(NOTES_CSV, dtype='str')

  for row_id, subj_id, text in zip(frame.ROW_ID, frame.SUBJECT_ID, frame.TEXT):
    printable = ''.join(c for c in text if c in string.printable)
    outfile = open('%s%s.txt' % (OUT_DIR, subj_id), 'a')
    outfile.write(printable + '\n')

def separate_discharge_summaries():
  """Each admission is written to a separate file"""

  df = pandas.read_csv(NOTES_CSV, dtype='str')

  for rowid, hadmid, cat, text in zip(df.ROW_ID, df.HADM_ID, df.CATEGORY, df.TEXT):
    if pandas.isnull(hadmid):
      print('empty hadmid for rowid', rowid)
      continue

    printable = ''.join(c for c in text if c in string.printable)

    if cat == 'Discharge summary':
      outfile = open('%s%s_discharge.txt' % (OUT_DIR, hadmid), 'a')
    else:
      outfile = open('%s%s_other.txt' % (OUT_DIR, hadmid), 'a')

    outfile.write(printable + '\n')
    outfile.write('\n**************************\n\n')

def note_type_viewer():
  """Generate stats"""

  df = pandas.read_csv(NOTES_CSV, dtype='str')

  for hid, cat, desc in zip(df.HADM_ID, df.CATEGORY, df.DESCRIPTION):
    print(id, cat.replace(' ', '_'), desc)

if __name__ == "__main__":

  separate_discharge_summaries()
