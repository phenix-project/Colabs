from __future__ import division
from __future__ import print_function

from pathlib import Path

# Utilities for running and setting up Phenix in Colab


class save_locals:
  """ Class to save working variables in locals() in a .pkl file and restore them """
  def __init__(self, file_name = "LOCALS.pkl", special_locals_to_ignore = None,
       names_to_ignore = None):

    if names_to_ignore is None:
      names_to_ignore = ['<function', '<module', '_Feature']
    if special_locals_to_ignore is None:
      special_locals_to_ignore = ['In','Out','sys','os',
        'exit','quit','get_ipython','Path','re','StringIO','redirect','hashlib',
        'shutil','ascii_uppercase']
    self.special_locals_to_ignore = special_locals_to_ignore
    self.names_to_ignore = names_to_ignore
    self.file_name = file_name

  def save(self, local_variables):
    import pickle
    new_locals = {}
    for k in list(local_variables.keys()):
      if k.startswith("_"):
        continue
      if k in self.special_locals_to_ignore:
        continue
      ok = True
      for x in self.names_to_ignore:
        if str(type(local_variables[k])).find(x) > -1:
          ok = False
          break
      if ok:
        print("Saved variable %s with value %s" %(k, local_variables[k]))
        new_locals[k] = local_variables[k]

    pickle.dump(new_locals, open(self.file_name, "wb" ) )
    print("Saved working local variables in %s" %(self.file_name))

  def restore(self, local_variables):
    if not os.path.isfile(self.file_name):
       print("No saved parameters to restore...")
       return

    import pickle
    new_locals = pickle.load(open( self.file_name, "rb" ) )
    for k in new_locals.keys():
      local_variables[k] = new_locals[k]
      print("Set variable %s as %s" %(k, new_locals[k]))

def get_input_directory(input_directory):
  if not input_directory:
    return None

  elif os.path.isdir(input_directory):
    input_directory = Path(input_directory)
    print("Input files will be taken from %s" %(
        input_directory))
  else:  # get it
    print("Input files will be taken from Google drive folder %s" %(
        input_directory))
    gdrive_dir = '/content/gdrive/MyDrive'
    if not os.path.isdir('/content/gdrive'):
      from google.colab import drive
      drive.mount('/content/gdrive')
      if not os.path.isdir(gdrive_dir):
        raise Exception("Sorry, cannot find the Google drive directory %s" %(
           gdrive_dir))

    input_directory = os.path.join(gdrive_dir, input_directory)
    if not os.path.isdir(input_directory):
      raise Exception("Sorry, cannot find the Google drive directory %s" %(
           input_directory))
  return Path(input_directory)



# IMPORTS, STANDARD PARAMETERS AND METHODS

import os, sys
import os.path
import re
import hashlib

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from google.colab import files
import shutil
from string import ascii_uppercase

# Local methods

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

def upload_templates(cif_dir, upload_maps = False,
     upload_manual_templates = None):
  manual_templates_uploaded = []
  maps_uploaded = []

  with redirect_stdout(StringIO()) as out:
    uploaded = files.upload()
    for filename,contents in uploaded.items():
      sys.stdout.flush()

      if upload_maps and \
           (str(filename).lower().endswith(".ccp4") or
           str(filename).lower().endswith(".map") or
           str(filename).lower().endswith(".mrc")):

        filepath = Path(cif_dir,filename)
        ff = open(filepath, 'wb')
        ff.write(contents)
        maps_uploaded.append(filepath)

      elif upload_manual_templates and \
           str(filename).endswith(".cif"):

        filepath = Path(cif_dir,filename)
        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif upload_manual_templates and \
           str(filename).endswith(".pdb"):

        pdb_filepath = Path(cif_dir,filename)
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = pdb_to_cif(pdb_filepath)
        manual_templates_uploaded.append(cif_filepath)

  if upload_maps:
    print("Maps uploaded: %s" %(maps_uploaded))

  print("Templates uploaded: %s" %(manual_templates_uploaded))
  if (not upload_maps) and (not manual_templates_uploaded):
    print("\n*** WARNING: no templates uploaded...Please use only .cif files ***\n")
  return manual_templates_uploaded, maps_uploaded

def get_templates_from_drive(cif_dir, upload_maps = False,
     upload_manual_templates = None,
     input_directory = None,
     jobname = None):
  manual_templates_uploaded = []
  maps_uploaded = []

  filename_list = os.listdir(input_directory)
  for filename in filename_list:
      sys.stdout.flush()
      contents = open(os.path.join(input_directory, filename),'rb').read()
      if upload_maps and \
           (str(filename).lower().endswith(".ccp4") or
           str(filename).lower().endswith(".map") or
           str(filename).lower().endswith(".mrc")):

        filepath = Path(cif_dir,filename)

        ff = open(filepath, 'wb')
        ff.write(contents)
        maps_uploaded.append(filepath)

      elif upload_manual_templates and \
           str(filename).endswith(".cif"):

        filepath = Path(cif_dir,filename)

        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif upload_manual_templates and \
           str(filename).endswith(".pdb"):

        pdb_filepath = Path(cif_dir,filename)
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = pdb_to_cif(pdb_filepath)
        manual_templates_uploaded.append(cif_filepath)

  if upload_maps:
    print("Maps uploaded: %s" %(maps_uploaded))

  print("Templates uploaded: %s" %(manual_templates_uploaded))
  if (not upload_maps) and (not manual_templates_uploaded):
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

def get_jobnames_sequences_from_file(
    upload_manual_templates = None,
    upload_maps = False,
    cif_dir = None,
    input_directory = None):
  from io import StringIO
  from google.colab import files
  print("Upload file with one jobname, a space, resolution, space, and one sequence on each line")
  uploaded_job_file = files.upload()
  if upload_manual_templates or upload_maps:

    if input_directory:
      uploaded_template_files, uploaded_maps = \
        get_templates_from_drive(cif_dir, upload_maps = upload_maps,
            upload_manual_templates = upload_manual_templates,
            input_directory = input_directory )
    else:
      print("\nUpload your templates now, all at once. " +\
           "NOTE: template file names must start with your job names")
      uploaded_template_files, uploaded_maps = upload_templates(
        cif_dir, upload_maps = upload_maps,
        upload_manual_templates= upload_manual_templates)
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
        if upload_maps:
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


  return jobnames, resolutions, \
     query_sequences, cif_filename_dict, map_filename_dict
