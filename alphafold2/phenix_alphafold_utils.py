from __future__ import division
from __future__ import print_function

# IMPORTS, STANDARD PARAMETERS AND METHODS

import os, sys
import re
import hashlib

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import shutil

# Local methods

from pathlib import Path

# Utilities for running alphafold in Colab


class save_locals:
  """ Class to save working variables in locals() in a .pkl file and restore them """
  def __init__(self, file_name = "LOCALS.pkl", special_locals_to_ignore = None,
       names_to_ignore = None):

    if names_to_ignore is None: # does not seem to work
      names_to_ignore = ['function', 'module', 'Feature']
    if special_locals_to_ignore is None:
      special_locals_to_ignore = ['In','Out','sys','os',
        'exit','quit','get_ipython','Path','re','StringIO','redirect','hashlib',
        'shutil','ascii_uppercase','files','redirect_stderr','redirect_stdout']
    self.special_locals_to_ignore = special_locals_to_ignore
    self.names_to_ignore = names_to_ignore
    self.file_name = file_name

  def save(self, local_variables, verbose = False):
    import pickle
    new_locals = {}
    for k in list(local_variables.keys()):
      if k.startswith("_"):
        continue
      if k in self.special_locals_to_ignore:
        continue
      ok = True
      for x in self.names_to_ignore:
        if str(local_variables[k]).find(x) > -1:
          ok = False
          break
      if ok:
        if verbose:
          print("Saved variable %s with value %s" %(k, local_variables[k]))
        new_locals[k] = local_variables[k]

    pickle.dump(new_locals, open(self.file_name, "wb" ) )
    print("Saved working local variables in %s" %(self.file_name))

  def restore(self, local_variables, verbose = False):
    if not os.path.isfile(self.file_name):
       print("No saved parameters to restore...")
       return

    import pickle
    new_locals = pickle.load(open( self.file_name, "rb" ) )
    for k in new_locals.keys():
      local_variables[k] = new_locals[k]
      if verbose:
        print("Set variable %s as %s" %(k, new_locals[k]))

def get_input_output_dirs(params):
  """
     Identify input directory as either a directory named with the value
     of input_directory in default directory, or as a directory with this
     name in user's Google drive.

     If save_outputs_in_google_drive is set, create an output_dir as
     well in Google drive
  """
  input_directory = params.get('input_directory',None)
  create_output_dir = params.get('save_outputs_in_google_drive',None)
  output_dir = params.get('output_dir',"ColabOutputs")

  have_input_directory = False
  need_google_drive = False

  if os.path.isdir(input_directory):
    input_directory = Path(input_directory)
    print("Input files will be taken from %s" %(
        input_directory))
    have_input_directory = True
  else:
    need_google_drive = True

  if create_output_dir:
    need_google_drive = True

  if need_google_drive:
    gdrive_dir = '/content/gdrive/MyDrive'
    if not os.path.isdir('/content/gdrive'):
      from google.colab import drive
      drive.mount('/content/gdrive')
      if not os.path.isdir(gdrive_dir):
        raise Exception("Sorry, cannot find the Google drive directory %s" %(
           gdrive_dir))

  if not have_input_directory:  # get it
    print("Input files will be taken from Google drive folder %s" %(
        input_directory))
    input_directory = os.path.join(gdrive_dir, input_directory)
    if not os.path.isdir(input_directory):
      raise Exception("Sorry, cannot find the Google drive directory %s" %(
           input_directory))
    input_directory = Path(input_directory)

  if create_output_dir:
    full_output_dir = os.path.join(gdrive_dir, output_dir)
    if not os.path.isdir(full_output_dir):
      os.mkdir(full_output_dir)
    print("Output files will be copied to %s in Google drive" %(output_dir))
    full_output_dir = Path(full_output_dir)
  else:
    full_output_dir = None

  params['input_directory'] = input_directory
  params['output_directory'] = full_output_dir
  return params

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

def clear_directories(all_dirs):

  for d in all_dirs:
    if d.exists():
      shutil.rmtree(d)
    d.mkdir(parents=True)


def clean_query(query_sequence):
  query_sequence = "".join(query_sequence.split())
  query_sequence = re.sub(r'[^a-zA-Z]','', query_sequence).upper()
  return query_sequence

def clean_jobname(jobname, query_sequence):
  jobname = "".join(jobname.split())
  jobname = re.sub(r'\W+', '', jobname)
  if len(jobname.split("_")) == 1:
    jobname = add_hash(jobname, query_sequence)
  return jobname

def save_sequence(jobname, query_sequence):
  # save sequence as text file
  filename = f"{jobname}.fasta"
  with open(filename, "w") as text_file:
    text_file.write(">1\n%s" % query_sequence)
  print("Saved sequence in %s: %s" %(filename, query_sequence))

def upload_templates(params):
  manual_templates_uploaded = []
  maps_uploaded = []
  from google.colab import files
  with redirect_stdout(StringIO()) as out:
    uploaded = files.upload()
    for filename,contents in uploaded.items():
      sys.stdout.flush()

      if params.get('upload_maps',None) and \
           (str(filename).lower().endswith(".ccp4") or
           str(filename).lower().endswith(".map") or
           str(filename).lower().endswith(".mrc")):

        filepath = Path(cif_dir,filename)
        ff = open(filepath, 'wb')
        ff.write(contents)
        maps_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".cif"):

        filepath = Path(cif_dir,filename)
        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".pdb"):

        pdb_filepath = Path(cif_dir,filename)
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = pdb_to_cif(pdb_filepath)
        manual_templates_uploaded.append(cif_filepath)

  if params.get('upload_maps'):
    print("Maps uploaded: %s" %(maps_uploaded))

  print("Templates uploaded: %s" %(manual_templates_uploaded))
  if (not upload_maps) and (not manual_templates_uploaded):
    print("\n*** WARNING: no templates uploaded...Please use only .cif files ***\n")
  return manual_templates_uploaded, maps_uploaded

def get_templates_from_drive(params):
  manual_templates_uploaded = []
  maps_uploaded = []

  input_directory = params.get('input_directory',None)
  print("Input directory:",input_directory)
  if input_directory is None:
    raise AssertionError(
      "No Google drive folder available. Please specify input_directory")

  jobname = params.get('jobname',None)
  cif_dir = params.get('cif_dir','.')

  filename_list = os.listdir(input_directory)

  for filename in filename_list:
      sys.stdout.flush()
      full_file_name = os.path.join(input_directory, filename)
      if not os.path.isfile(full_file_name):
        continue
      contents = open(os.path.join(input_directory, filename),'rb').read()
      if params.get('upload_maps',None) and \
           (str(filename).lower().endswith(".ccp4") or
           str(filename).lower().endswith(".map") or
           str(filename).lower().endswith(".mrc")):

        filepath = Path(cif_dir,filename)

        ff = open(filepath, 'wb')
        ff.write(contents)
        maps_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".cif"):

        filepath = Path(cif_dir,filename)

        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".pdb"):

        pdb_filepath = Path(cif_dir,filename)
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = pdb_to_cif(pdb_filepath)
        manual_templates_uploaded.append(cif_filepath)

  if params.get('upload_maps',None):
    print("Maps uploaded: %s" %(maps_uploaded))

  print("Templates uploaded: %s" %(manual_templates_uploaded))
  if (not params.get('upload_maps',None)) and (not manual_templates_uploaded):
    print("\n*** WARNING: no templates uploaded...***\n")
  if manual_templates_uploaded:
    manual_templates_uploaded = select_matching_template_files(
      manual_templates_uploaded, jobname)
  if maps_uploaded:
    maps_uploaded = select_matching_template_files(
      maps_uploaded, jobname)

  return manual_templates_uploaded, maps_uploaded

def select_matching_template_files(uploaded_template_files,
    jobname):
  if not jobname:
    return uploaded_template_files

  matching_files = []
  for file_name in uploaded_template_files:
    if file_name.parts[-1].startswith(jobname.split("_")[0]):
      matching_files.append(file_name)
  return matching_files

def get_jobnames_sequences_from_file(params):

  from io import StringIO
  from google.colab import files
  print("Upload file with one jobname, a space, resolution, space,"+
     " and one sequence on each line")
  uploaded_job_file = files.upload()
  if params.get('upload_manual_templates',None) or params.get(
    'upload_maps', None):

    if params.get('input_directory',None):
      uploaded_template_files, uploaded_maps = \
        get_templates_from_drive(params)
    else:
      print("\nUpload your templates now, all at once. " +\
           "NOTE: template file names must start with your job names")
      uploaded_template_files, uploaded_maps = upload_templates(params)
    print("Total of %s template files uploaded: %s" %(
          len(uploaded_template_files), uploaded_template_files))


  else:
    uploaded_template_files = []
    uploaded_maps = []

  s = StringIO()
  query_sequences = []
  jobnames = []
  resolutions = []
  cif_filename_dict = {}
  map_filename_dict = {}
  for filename,contents in uploaded_job_file.items():
    print(contents.decode("UTF-8"), file = s)
    text = s.getvalue()
    for line in text.splitlines():
      spl = line.split()
      if len(spl) < 2:
        pass # empty line
      else: # usual
        jobname = spl[0]
        if params.get('upload_maps',None):
          resolution = float(spl[1])
          query_sequence = "".join(spl[2:])
        else:
          resolution = 3
          query_sequence = "".join(spl[1:])
        jobname = clean_jobname(jobname, query_sequence)
        query_sequence = clean_query(query_sequence)

        if jobname in jobnames:
          pass # already there
        else:
          query_sequences.append(query_sequence)
          jobnames.append(jobname)
          resolutions.append(resolution)
          if uploaded_template_files:
              cif_filename_dict[jobname] = \
                select_matching_template_files(
                    uploaded_template_files,
                     jobname)
          if uploaded_maps:
              map_filename_dict[jobname] = \
                select_matching_template_files(
                    uploaded_maps,
                     jobname)

  params['jobnames'] = jobnames
  params['resolutions'] = resolutions
  params['map_filename_dict'] = map_filename_dict
  params['cif_filename_dict'] = cif_filename_dict
  params['query_sequences'] = query_sequences
  return params


def get_helper_files():

  # Upload utilities python files to get started
  import os
  result = os.system("wget -qnc https://raw.githubusercontent.com/phenix-project/Colabs/main/alphafold2/phenix_alphafold_utils.py")
  result = os.system("wget -qnc https://raw.githubusercontent.com/phenix-project/Colabs/main/alphafold2/phenix_colab_utils.py")
  for file_name in ['phenix_alphafold_utils.py', 'phenix_colab_utils.py']:
    if not os.path.isfile(file_name):
      print('Unable to set up the helper file %s' %(file_name))
      raise AssertionError('Unable to set up the helper file %s' %(file_name))

def set_up_input_files(params):

  # Default working directory
  if not params.get('content_dir',None):
    params['content_dir'] = "/content/"

  from pathlib import Path
  import os, sys

  from google.colab import files

  content_dir = params.get('content_dir',".")

  # Set working directory
  os.chdir(content_dir)

  # Clear out directories
  parent_dir = Path(os.path.join(content_dir,"manual_templates"))

  if not params.get('cif_dir',None):
    params['cif_dir'] = Path(parent_dir,"mmcif")

  # get input and output directories
  params = get_input_output_dirs(params)

  # Initialize
  query_sequences = []
  jobnames = []
  resolutions = []
  cif_filename_dict = {}
  map_filename_dict = {}
  clear_directories([parent_dir,params.get('cif_dir','')])

  if params.get(
     'upload_file_with_jobname_resolution_sequence_lines',None):
    params = get_jobnames_sequences_from_file(params)
    jobnames = params['jobnames']
    resolutions = params['resolutions']
    map_filename_dict = params['map_filename_dict']
    cif_filename_dict = params['cif_filename_dict']
    query_sequences = params['query_sequences']
  else: # usual
    jobname = params.get('jobname',None)
    query_sequence = params.get('query_sequence',None)
    resolution = params.get('resolution',None)
    upload_manual_templates = params.get('upload_manual_templates',None)
    upload_maps = params.get('upload_maps',None)

    jobname = clean_jobname(jobname, query_sequence)
    query_sequence = clean_query(query_sequence)
    if query_sequence and not jobname:
      print("Please enter a job name and rerun")
      raise AssertionError("Please enter a job name and rerun")

    if jobname and not query_sequence:
      print("Please enter a query_sequence and rerun")
      raise AssertionError("Please enter a query_sequence rerun")

    # Add sequence and jobname if new
    if (jobname and query_sequence) and (
         not query_sequence in query_sequences) and (
         not jobname in jobnames):
        query_sequences.append(query_sequence)
        jobnames.append(jobname)
        resolutions.append(resolution)
        if upload_manual_templates or upload_maps:
          if params.get('input_directory', None):
            cif_filename_dict[jobname], map_filename_dict[jobname] = \
              get_templates_from_drive(params)
          else:
            print("\nPlease upload %s for %s" %(
              "template and map" if upload_manual_templates and upload_maps
              else "template" if upload_manual_templates
              else "map",
              jobname))
            sys.stdout.flush()
            cif_filename_dict[jobname], map_filename_dict[jobname] = \
              upload_templates(params)


  # Save sequence
  for i in range(len(query_sequences)):
    # save the sequence as a file with name jobname.fasta
    save_sequence(jobnames[i], query_sequences[i])

  print("\nCurrent jobs, resolutions, sequences, templates, and maps:")

  for qs,jn,res in zip(query_sequences, jobnames, resolutions):
    template_list = []
    for t in cif_filename_dict.get(jn,[]):
      template_list.append(os.path.split(str(t))[-1])
    map_list = []
    for t in map_filename_dict.get(jn,[]):
      map_list.append(os.path.split(str(t))[-1])
    print(jn, res, qs, template_list, map_list)

    if len(qs) < 20:
      print("\n\nMinimum sequence length is 20 residues...\n\n",
            "Please enter a longer sequence and \n",
            "run again\n\n")
      sys.stdout.flush()
      raise AssertionError("Sequence must be 20 residues or more")
    if not map_list:
      print("\n\nNeed a map for each sequence...\n\n",
            "Please be sure input_directory is set and run again\n\n")
      sys.stdout.flush()
      raise AssertionError("Map file needed for job %s" %(jn))

  if not query_sequences:
    print("Please supply a query sequence and run again")
    raise AssertionError("Need a query sequence")

  params['jobnames'] = jobnames
  params['resolutions'] = resolutions
  params['map_filename_dict'] = map_filename_dict
  params['cif_filename_dict'] = cif_filename_dict
  params['query_sequences'] = query_sequences
  return params
