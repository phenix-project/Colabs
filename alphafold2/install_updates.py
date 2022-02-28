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
    try:
      shutil.copyfile(a,b)
    except Exception as e:
      print("Unable to copy %s to %s" %(a,b))

def install_updates():
  print("Installing updates")
  if not os.path.isfile("updates.tgz"):
    print("No updates.tgz file ... skipping")
    return

  if not os.path.isdir("updates"):
    os.mkdir("updates")
  here = os.getcwd()
  os.chdir("updates")
  print("Unpacking in %s" %(os.getcwd())
  os.system("tar xzvf ../updates.tgz")
  file_list = os.listdir(".")
  print("Files unpacked: %s" %(" ".join(file_list)))
  os.chdir(here)

  # Copy files where they go

  default_directory = "/usr/local/lib/python3.7/site-packages/Colab/alphafold2"
  file_directory_dict = {
    'alphafold_with_density_map.py':
       '/usr/local/lib/python3.7/site-packages/phenix/programs',
   } 
  for file_name in file_list:
    full_file = os.path.join("updates",file_name)
    if not os.path.isfile(file_name):
      print("Missing the file %s" %(key))
      continue
    
    for dd in [".", file_directory_dict.get(file_name,default_directory)]:
      check_and_copy(file_name, os.path.join(dd, file_name))
      print("Copied %s to %s" %(file_name, os.path.join(dd, file_name)))
  print("Done with updates")
    

