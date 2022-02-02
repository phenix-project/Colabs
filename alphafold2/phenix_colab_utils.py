from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

# Utilities for setting up and running Phenix in Colab

def shell(text):
  """ Utility to run a string as a shell script
  """
  import subprocess
  print("RUNNING:",text)
  result = subprocess.call(text.split())
  return result

def clear_python_caches(modules = None):
  import sys
  print("Clearing python caches ...")
  if modules is None:
    modules = ['phenix_colab_utils','phenix_alphafold_utils','install_alphafold','cu','install_phenix','install_software',
     'alphafold','protein', 'Alphafold', 'Protein', 'colabfold', ]
  for x in list(sys.modules.keys(  )) + list(globals()):
    for key in modules:
       if x.find(key)>-1:
        if x in list(sys.modules.keys()):
          del(sys.modules[x])
        if x in list(globals().keys()):
          del globals()[x]
          assert not x in list(globals().keys())
          break

def install_software(
  bioconda = True,
  phenix = True,
    phenix_version = None,
    phenix_password = None,
  alphafold = True,
    alphafold_version = '0bab1bf84d9d887aba5cfb6d09af1e8c3ecbc408',
  pdb_to_cif = True,
  fix_paths = True,
  content_dir = None,
    ):

  if content_dir is None:
    content_dir = os.getcwd()

  if bioconda:
    install_bioconda()

  if phenix:
    install_phenix(password = phenix_password, version = phenix_version)

  if alphafold:
    install_alphafold(version = alphafold_version, content_dir = content_dir)

  if pdb_to_cif:
    install_pdb_to_cif()

  if fix_paths:
    run_fix_paths()



def install_alphafold(version = None, content_dir = None):
  assert content_dir is not None
  os.chdir(content_dir)

  if not version:
    raise AssertionError("Need alphafold version for installation")

  if os.path.isfile("AF2_READY"):
    print("AF2 is already installed")
    return

  # install dependencies
  print( "Installing biopython and colabfold...")
  shell("pip -q install biopython dm-haiku ml-collections py3Dmol")
  shell("wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/96fe2446f454eba38ea34ca45d97dc3f393e24ed/beta/colabfold.py")
  # download model
  if not os.path.isdir("alphafold"):
    print("Installing AlphaFold...")
    shell("git clone https://github.com/deepmind/alphafold.git --quiet")
    shell("(cd alphafold; git checkout %s --quiet)" %(version))
    shell("mv alphafold alphafold_")
    shell("mv alphafold_/alphafold .")
    # remove "END" from PDBs, otherwise biopython complains
    dd = os.path.join(content_dir, "alphafold","common","protein.py")
    shell("""sed -i "s/pdb_lines.append('END')//" %s""" %(dd))
    shell("""sed -i "s/pdb_lines.append('ENDMDL')//" %s""" %(dd))

  # download model params (~1 min)
  if not os.path.isdir("params"):
    print("Installing AlphaFold parameters...")
    shell("mkdir params")
    shell("curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar| tar x -C params")

  shell("touch AF2_READY")


  # download libraries for interfacing with MMseqs2 API
  if not os.path.isfile("MMSEQ2_READY"):
    print( "Installing mmseq2 ...")
    shell("apt-get -qq -y update 2>&1 1>/dev/null")
    shell("apt-get -qq -y install jq curl zlib1g gawk 2>&1 1>/dev/null")
    shell("touch MMSEQ2_READY")


def run_fix_paths():
  #  Hacks to fix some paths

  # Get path correct in pulchra.sh
  print("Fixing path in pulchra.sh...")
  if os.path.isfile("/usr/local/lib/python3.7/site-packages/phenix/command_line/pulchra.sh"):
    f = open(
     "/usr/local/lib/python3.7/site-packages/phenix/command_line/pulchra.sh",
     'w')
    print("""exec /usr/local/share/cctbx/pulchra/exe/pulchra "$@" """,
      file = f)
    f.close()


  print("Updating python paths...")

  shell ( """ cd /usr/local/lib/python3.7/site-packages/; tar czvf - phenix/refinement/*/*.params > phenix/tmp_phenix.tgz; cd phenix; tar xzvf tmp_phenix.tgz""")

  import sys
  for d in ['/usr/local/lib', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages']:
    if d not in sys.path:
       sys.path.append(d)
  print("Done with patches")


def install_phenix(password = None, version = None):
  if os.path.isfile("PHENIX_READY"):
    print("Phenix is already installed")
    return

  if ((not version) or (not password)):
    raise AssertionError("Need version and password for Phenix")

  if (os.path.isfile("PHENIX_DOWNLOADED") or os.path.isfile("PHENIX_READY")):
    print("Phenix is already downloaded")
  else:
    print("Downloading Phenix...")
    shell("wget -q --user user --password %s -r -l1 https://phenix-online.org/download/installers/%s/linux-64/ -A phenix*.tar.bz2" %(password, version))
    if not os.path.isdir("phenix-online.org"):
      # try with user as trusted
      shell("wget -q --user trusted --password %s -r -l1 https://phenix-online.org/download/installers/%s/linux-64/ -A phenix*.tar.bz2" %(password, version))
    if not os.path.isdir("phenix-online.org"):
      raise AssertionError("Unable to download...please check your Phenix version and password?")

    # Move files to working directory
    print("ZZ download:",os.listdir("phenix-online.org/download/installers/%s/linux-64/" %(version))
    print("ZZ here:",os.listdir("."))
    shell("mv phenix-online.org/download/installers/%s/linux-64/* ." %(version))
    #shell("rm -fr phenix-online.org")
    shell("touch PHENIX_DOWNLOADED")
    print("Phenix has been downloaded.")

  if os.path.isfile("PHENIX_READY"):
    print("Phenix is already installed")
  else:
    print("Installing Phenix")

    # Check that there is only one version downloaded (can be more than one
    #   by accident
    bz2_file = get_last_bz2_file()
    print("Zip file is %s" %(bz2_file))

    shell("mamba install -q -y %s" %(bz2_file))
    shell("mamba install -q -c conda-forge -y boost=1.74 boost-cpp mrcfile numpy=1.20 scipy")  # >& /dev/null")
    shell("cp -a /usr/local/share/cctbx /usr/share")
    shell("pip install psutil")
    shell("touch PHENIX_READY")
    print("Phenix has been installed.")

def get_last_bz2_file():
  file_list = os.listdir(".")
  bz2_files = []
  for f in file_list:
    if f.startswith("phenix") and f.endswith(".bz2"):
      bz2_files.append(f)
  if not bz2_files:
    raise AssertionError(
       "No Phenix downloaded...please check version and password?")
  bz2_files = sorted(bz2_files, key = lambda b: b, reverse = True)
  return bz2_files[0]

def install_condacolab():
  if os.path.isfile("CONDA_READY"):
    print ("CondaColab is already installed")
    return
  print("Installing condacolab...")
  shell("pip install -q condacolab")
  import condacolab
  condacolab.install()
  shell("touch CONDA_READY")
  print("Done...please ignore the crash messege...")

def install_bioconda():
  # set up bioconda and mount drive to patch in pdb_to_cif
  if os.path.isfile("HH_READY"):
    print("Bioconda already installed")
  else:
    print("Installing bioconda")
    shell("conda install -y -q -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 python=3.7 2>&1 1>/dev/null")
    shell("touch HH_READY")

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
  if os.path.isfile('PDB_TO_CIF_READY'):
    print("pdb_to_cif is already downloaded")
  else:
    print("Downloading pdb_to_cif...")
    shell('wget https://phenix-online.org/phenix_data/terwilliger/colab_data/maxit-v11.100-prod-src.tgz')
    shell('tar xzf maxit-v11.100-prod-src.tgz')
    shell('rm -f maxit-v11.100-prod-src.tgz')
    shell('touch PDB_TO_CIF_READY')
    print("Ready with pdb_to_cif")

def run_pdb_to_cif(f, content_dir = None):
    assert content_dir is not None

    if not os.path.isfile(os.path.join(content_dir,
        "maxit-v11.100-prod-src/bin/process_entry")):
      print("Sorry, pdb_to_cif is not available..."+
        "install with phenix_colab_utils.install_pdb_to_cif")
      return
    if hasattr(f,'as_posix'):
      f = f.as_posix()  # make it a string
    output_file = f.replace(".pdb",".cif")

    p = os.path.join(content_dir,"maxit-v11.100-prod-src")
    b = os.path.join(p, "bin","process_entry")
    shell("chmod +x %s; RCSBROOT=%s; export RCSBROOT; echo $RCSBROOT;  %s -input %s -input_format pdb -output %s -output_format cif" %(b,p,b,f,output_file))
    return Path(output_file)
