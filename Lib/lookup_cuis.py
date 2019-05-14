#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from umls import UMLSClient

#
# E.g. lookup_cuis.py /Users/Dima/Loyola/Data/MimicIII/Patients/Cuis/99999.txt
#

def process_file():
  """Lookup CUIs from a file"""
  
  cui_file = sys.argv[1]

  APIKey = 'c24cc91f-0c2f-4380-aa36-3f375cc78030'
  UMLSCUIs = UMLSClient(APIKey)
  getTd = UMLSCUIs.getst()

  for cui in open(cui_file).read().split():
    text = UMLSCUIs.query_umls(cui)
    print('%s: %s' % (cui, text['name']))

def process_single_cui():
  """Lookup a single cui"""

  cui = sys.argv[1].upper()

  APIKey = 'c24cc91f-0c2f-4380-aa36-3f375cc78030'
  UMLSCUIs = UMLSClient(APIKey)
  getTd = UMLSCUIs.getst()

  text = UMLSCUIs.query_umls(cui)
  print('%s: %s' % (cui, text['name']))
    
if __name__ == "__main__":

  process_single_cui()
  # process_file()

