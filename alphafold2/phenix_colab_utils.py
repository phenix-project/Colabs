from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

# Utilities for setting up and running Phenix in Colab

def shell(text):
  """ Utility to run a string as a shell script
  """
  result = os.system(text)
  return result

def install_condacolab():
  if os.path.isfile("CONDA_READY"):
    print ("CondaColab is already installed")
    return
  print("Installing condacolab...", end = "")
  shell("pip install -q condacolab")
  import condacolab
  condacolab.install()
  shell("touch CONDA_READY")
  print("Done...please ignore the crash messege...")

def install_bioconda():
  pass

def test():
  print("testing")
def import_tensorflow():
  import importlib
  tf = importlib.import_module("tensorflow")
  if not tf:
    print("Unable to import tensorflow")
  else: # usual
    print("Tensorflow available")
    return tf

def install_pdb_to_cif():
  if not os.path.isfile('PDB_TO_CIF_READY'):
    print("Downloading pdb_to_cif...")
    shell('wget https://phenix-online.org/phenix_data/terwilliger/colab_data/maxit-v11.100-prod-src.tgz')
    shell('tar xzf maxit-v11.100-prod-src_essentials.tgz')
    shell('rm -f maxit-v11.100-prod-src_essentials.tgz')
    shell('touch PDB_TO_CIF_READY')
    print("Ready with pdb_to_cif")

def run_pdb_to_cif(f):
    if not os.path.isfile("/content/maxit-v11.100-prod-src/bin/process_entry"):
      print("Sorry, pdb_to_cif is not available..."+
        "install with phenix_colab_utils.install_pdb_to_cif")
      return
    if hasattr(f,'as_posix'):
      f = f.as_posix()  # make it a string
    output_file = f.replace(".pdb",".cif")
    import shutil
    shutil.copyfile(f,'/content/pdb.pdb')
    print("RUNNING")
    print("chmod +x /content/maxit-v11.100-prod-src/bin/process_entry; RCSBROOT=/content/maxit-v11.100-prod-src; export RCSBROOT; echo $RCSBROOT;  /content/maxit-v11.100-prod-src/bin/process_entry -input /content/pdb.pdb -input_format pdb -output /content/pdb.cif -output_format cif")

    shell("chmod +x /content/maxit-v11.100-prod-src/bin/process_entry; RCSBROOT=/content/maxit-v11.100-prod-src; export RCSBROOT; echo $RCSBROOT;  /content/maxit-v11.100-prod-src/bin/process_entry -input /content/pdb.pdb -input_format pdb -output /content/pdb.cif -output_format cif")
    shutil.copyfile('/content/pdb.cif',output_file)
    return Path(output_file)
