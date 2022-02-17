from __future__ import division
from __future__ import print_function

import os, sys
from pathlib import Path

# Utilities for setting up and running Phenix in Colab

def get_helper_files():
  # Get the helper python files
  for file_name in [
      'alphafold_with_density_map.py',
      'alphafold_utils.py',
      'run_alphafold_with_density_map.py',
      'phenix_alphafold_utils.py',
      'colabfold.py']:
    if os.path.isfile(file_name):
      os.remove(file_name)
    os.environ['file_name'] = file_name
    print("Getting %s" %(file_name))
    result = os.system(
      "wget -qnc https://raw.githubusercontent.com/phenix-project/Colabs/main/alphafold2/$file_name")


def run_command(command, log = sys.stdout):
    "Run and get output to terminal"
    import shlex
    import subprocess
    process = subprocess.run(shlex.split(command), stdout=subprocess.PIPE)
    print (process.stdout.decode("utf-8"), file = log)

def runsh(text):
  """ Utility to run a string as a shell script and toss output
  """
  import subprocess
  import shlex
  print("RUNNING:",text)
  result = os.system(text) # ZZZsubprocess.run(shlex.split(text))
  return result

def clear_python_caches(modules = None, keep_list = None):
  import sys
  print("Clearing python caches ...")
  if modules is None:
    modules = ['phenix_colab_utils','phenix_alphafold_utils',
     'install_alphafold','cu','install_phenix','install_software',
     'alphafold','protein', 'Alphafold', 'Protein', 'colabfold', ]
  if keep_list is None:
    keep_list = "ls cd rm mv cat ".split()
  for x in list(sys.modules.keys(  )) + list(globals()):
    for key in modules:
       if x in keep_list: continue
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
  biopython = True,
  mmseq2 = True,
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

  if alphafold or biopython or mmseq2:
    install_alphafold(version = alphafold_version, content_dir = content_dir,
      biopython = biopython,
      mmseq2 = mmseq2,
      alphafold = alphafold)

  if pdb_to_cif:
    install_pdb_to_cif()

  if fix_paths:
    run_fix_paths()



def install_alphafold(version = None, content_dir = None,
    biopython = True,
    mmseq2 = True,
    alphafold = True,
    ):
  assert content_dir is not None
  os.chdir(content_dir)

  if not version and alphafold:
    raise AssertionError("Need alphafold version for installation")

  if os.path.isfile("AF2_READY"):
    print("AF2 is already installed")
    return

  # install dependencies
  print( "Installing biopython ...")
  if biopython:
    runsh("pip -q install biopython dm-haiku==0.0.5 ml-collections py3Dmol")
  # download model
  if alphafold and (not os.path.isdir("alphafold")):
    print("Installing AlphaFold...")
    runsh("git clone https://github.com/deepmind/alphafold.git --quiet")
    here = os.getcwd()
    os.chdir(os.path.join(here,"alphafold"))
    runsh("git checkout %s --quiet" %(version))
    os.chdir(here)
    runsh("mv alphafold alphafold_")
    runsh("mv alphafold_/alphafold .")
    # remove "END" from PDBs, otherwise biopython complains
    dd = os.path.join(content_dir, "alphafold","common","protein.py")
    runsh("""sed -i "s/pdb_lines.append('END')//" %s""" %(dd))
    runsh("""sed -i "s/pdb_lines.append('ENDMDL')//" %s""" %(dd))

  # download model params (~1 min)
  if alphafold and (not os.path.isdir("params")):
    print("Installing AlphaFold parameters...")
    runsh("mkdir params")
    import subprocess
    subprocess.check_output("/usr/bin/curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params", shell = True)

  runsh("touch AF2_READY")


  # download libraries for interfacing with MMseqs2 API
  if mmseq2 and (not os.path.isfile("MMSEQ2_READY")):
    print( "Installing mmseq2 ...")
    runsh("apt-get -qq -y update ")
    runsh("apt-get -qq -y install jq curl zlib1g gawk ")
    runsh("touch MMSEQ2_READY")


def run_fix_paths():
  #  Hacks to fix some paths
  if not os.path.isdir("/usr/local/lib/python3.7/site-packages/phenix"):
    return # nothing to do


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

  here = os.getcwd()
  os.chdir("/usr/local/lib/python3.7/site-packages/")
  runsh (" tar czvf - phenix/refinement/*/*.params > phenix/tmp_phenix.tgz")
  os.chdir("/usr/local/lib/python3.7/site-packages/phenix")
  runsh("tar xzvf tmp_phenix.tgz")
  os.chdir(here)

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
    runsh("wget -q --user download --password %s -r -l1 https://phenix-online.org/download/installers/%s/linux-64/ -A phenix*.tar.bz2" %(password, version))
    if not os.path.isdir("phenix-online.org"):
      # try with user as trusted
      runsh("wget -q --user trusted --password %s -r -l1 https://phenix-online.org/download/installers/%s/linux-64/ -A phenix*.tar.bz2" %(password, version))
    if not os.path.isdir("phenix-online.org"):
      raise AssertionError("Unable to download...please check your Phenix version and password?")

    # Move files to working directory
    file_list = os.listdir(
       "phenix-online.org/download/installers/%s/linux-64/" %(version))
    if len(file_list)< 1:
      raise AssertionError("Unable to download...please check your Phenix version and password?")
    file_name = file_list[0]
    print("Downloaded bz2 file: %s" %(file_name))
    if not file_name.endswith(".bz2"):
      raise AssertionError("Downloaded file does not end with .bz2")

    runsh("mv phenix-online.org/download/installers/%s/linux-64/%s ." %(
     version, file_name))
    runsh("rm -fr phenix-online.org")
    runsh("touch PHENIX_DOWNLOADED")
    print("Phenix has been downloaded.")

  if os.path.isfile("PHENIX_READY"):
    print("Phenix is already installed")
  else:
    print("Installing Phenix")

    # Check that there is only one version downloaded (can be more than one
    #   by accident
    bz2_file = get_last_bz2_file()
    print("Zip file is %s" %(bz2_file))

    runsh("mamba install -q -y %s" %(bz2_file))
    runsh("mamba install -q -c conda-forge -y boost=1.74 boost-cpp mrcfile numpy=1.20 scipy")
    runsh("cp -a /usr/local/share/cctbx /usr/share")
    runsh("pip install psutil")
    runsh("touch PHENIX_READY")
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
  runsh("pip install -q condacolab")
  import condacolab
  condacolab.install()
  runsh("touch CONDA_READY")
  print("Done...please ignore the crash messege...")

def install_bioconda():
  # set up bioconda and mount drive to patch in pdb_to_cif
  if os.path.isfile("HH_READY"):
    print("Bioconda already installed")
  else:
    print("Installing bioconda")
    runsh("conda install -y -q -c conda-forge -c bioconda kalign3=3.2.2 hhsuite=3.3.0 python=3.7 ")
    runsh("touch HH_READY")

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

def install_pdb_to_cif(content_dir = None):
  if not content_dir:
    content_dir = "/content"
  if os.path.isfile('PDB_TO_CIF_READY'):
    print("pdb_to_cif is already downloaded")
  else:
    print("Downloading pdb_to_cif...")
    runsh('wget https://phenix-online.org/phenix_data/terwilliger/colab_data/maxit-v11.100-prod-src.tgz')
    runsh('tar xzf maxit-v11.100-prod-src.tgz')
    runsh('rm -f maxit-v11.100-prod-src.tgz')
    p = os.path.join(content_dir,"maxit-v11.100-prod-src")
    b = os.path.join(p, "bin","process_entry")
    runsh("chmod +x %s" %(b))
    runsh('touch PDB_TO_CIF_READY')
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
    here = os.getcwd()
    os.environ['RCSBROOT'] = p
    runsh("%s -input %s -input_format pdb -output %s -output_format cif" %(b,f,output_file))
    return Path(output_file)
