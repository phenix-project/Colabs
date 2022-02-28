from __future__ import division
from __future__ import print_function

import os, sys, shutil

"""
 Utility to install updates in alphafold_with_density colab
 Need for this utility:  This allows testing of changes in the Colab notebook 
  without affecting current use of the existing notebook. It also allows
  passing updates to users without affecting other users

 Use:
 1. You package up any python files you want to update into the file updates.tgz
  and you put code install_updates() below that says where those files go. You
  check these to files in to github here.

 2. End user (or you) checks the box "install_custom_updates" in the
AlphaFoldWithDensityMap Colab notebook.

 3. On running cell 1 of that notebook, the install_updates.py and install.tgz
 files from this github are downloaded and the install_updates() method
 of install_updates.py is run.

  
"""

def same_file(f1,f2):
  if not f1 or not f2 or not os.path.isfile(f1) or not os.path.isfile(f2):
    return False
  return os.path.samefile(os.path.abspath(f1),os.path.abspath(f2))

def check_and_copy(a,b):
  if (not a) or (not os.path.isfile(a)):
    return # Nothing to do
  if not same_file(a, b):
     shutil.copyfile(a,b)

def install_updates():
  print("Installing updates")
  if not os.path.isfile("updates.tgz"):
    print("No updates.tgz file ... skipping")
    return

  os.system("tar xzvf updates.tgz")

  # Copy files where they go

  file_dict = {
    'alphafold_with_density_map.py':'.',
   } 
  for key in list(file_dict.keys()):
    if not os.path.isfile(key):
      print("Missing the file %s" %(key))
    else:
      check_and_copy(key,os.path.join(file_dict[key],key))
      print("Copied %s to %s" %(key,os.path.join(file_dict[key],key)))
  print("Done with updates")
    

