from __future__ import division
from __future__ import print_function

# IMPORTS, STANDARD PARAMETERS AND METHODS

import os, sys
import re
import hashlib

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import shutil
from phenix_colab_utils import exit, run_pdb_to_cif, make_four_char_name

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

def params_as_dict(params):
  # Convert params object to a dict (only works if 1 level)
  p = {}
  for key in dir(params):
    if key.startswith("_"): continue
    p[key] = getattr(params, key)
  return p

def get_input_output_dirs(params, log = sys.stdout):
  """
     Identify input directory as either a directory named with the value
     of input_directory in default directory, or as a directory with this
     name in user's Google drive.

     If output_dir is set or save_outputs_in_google_drive is set,
     create an output_dir as
     well. Save in Google drive if input_directory is in Google drive
  """
  input_directory = params.get('input_directory',None)
  save_outputs_in_google_drive = params.get('save_outputs_in_google_drive',None)
  output_dir = params.get('output_directory', None)

  create_output_dir = save_outputs_in_google_drive or output_dir

  if save_outputs_in_google_drive and not output_dir:
    output_dir = "ColabOutputs"

  content_dir = params.get('content_dir')
  assert content_dir is not None

  have_input_directory = False
  need_google_drive = False
  gdrive_dir = None

  if input_directory and os.path.isdir(input_directory):
    input_directory = Path(input_directory)
    print("Input files will be taken from %s" %(
        input_directory), file = log)
    have_input_directory = True
  elif input_directory:
    need_google_drive = True

  if save_outputs_in_google_drive:
    need_google_drive = True

  if need_google_drive:
    gdrive_path = os.path.join(content_dir,'gdrive')
    gdrive_dir = os.path.join(content_dir,'gdrive','MyDrive')
    if not os.path.isdir(gdrive_path):
      try:
        from google.colab import drive
      except Exception as e:
        if input_directory:
          raise Exception("Sorry, cannot find the directory "+
          "%s and cannot mount Google drive directory %s" %(input_directory,
            gdrive_dir))
        else:
          raise Exception("Sorry, cannot find the Google drive directory %s" %(
            gdrive_dir))
      drive.mount(gdrive_path)
      if not os.path.isdir(gdrive_dir):
        raise Exception("Sorry, cannot find the Google drive directory %s" %(
           gdrive_dir))

  if input_directory and (not have_input_directory):  # get it
    if not os.path.isdir(input_directory):
      input_directory = os.path.join(gdrive_dir, input_directory)
    print("Input files will be taken from the folder %s" %(
        input_directory), file = log)
    if not os.path.isdir(input_directory):
      raise Exception("Sorry, cannot find the Google drive directory %s" %(
           input_directory))
    input_directory = Path(input_directory)

  if create_output_dir:
    # Check for case where output_dir exists but we really want it in gdrive

    if os.path.isdir(output_dir) and (not save_outputs_in_google_drive):
      full_output_dir = output_dir
    elif (save_outputs_in_google_drive) and \
        gdrive_dir and os.path.isdir(gdrive_dir):
      full_output_dir = os.path.join(gdrive_dir, output_dir)
    else:  # make it in content_dir
      full_output_dir = os.path.join(content_dir, output_dir)
    if not os.path.isdir(full_output_dir):
      os.mkdir(full_output_dir)
    print("Output files will be copied to %s " %(output_dir), file = log)

    full_output_dir = Path(full_output_dir)
  else:
    full_output_dir = Path(os.getcwd())

  params['input_directory'] = os.path.abspath(input_directory)
  params['output_directory'] = os.path.abspath(full_output_dir)
  print("Input directory: ",input_directory, file = log)
  print("Output directory: ",full_output_dir, file = log)
  return params

def add_hash(x,y):
  return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

def clear_directories(all_dirs, log = sys.stdout):

  for d in all_dirs:
    if d and d not in [".", Path("."),"/content",Path("/content")] \
         and d.exists():
      print("Clearing %s" %(d), file = log)
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

def save_sequence(jobname, query_sequence, log = sys.stdout):
  # save sequence as text file
  filename = "{}.fasta".format(jobname)
  with open(filename, "w") as text_file:
    text_file.write(">1\n%s" % query_sequence)
  print("Saved sequence in %s: %s" %(filename, query_sequence), file = log)

def upload_templates(params):
  manual_templates_uploaded = []
  maps_uploaded = []
  msas_uploaded = []
  from google.colab import files
  upload_dir = params.get("upload_dir")
  assert upload_dir is not None and os.path.isdir(upload_dir)
  with redirect_stdout(StringIO()) as out:
    uploaded = files.upload()
    for filename,contents in uploaded.items():
      sys.stdout.flush()
      if params.get('upload_maps',None) and \
           (str(filename).lower().endswith(".ccp4") or
           str(filename).lower().endswith(".map") or
           str(filename).lower().endswith(".mrc")):
        filepath = Path(upload_dir,filename)
        ff = open(filepath, 'wb')
        ff.write(contents)
        maps_uploaded.append(filepath)

      elif params.get('use_msa',None) and \
          params.get('upload_msa_file',None) and \
           str(filename).endswith(".a3m"):
        filepath = Path(upload_dir,filename)
        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          msas_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".cif"):
        filepath = Path(upload_dir,make_four_char_name(filename))
        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".pdb"):
        cif_dir = params.get("cif_dir")
        if not cif_dir or not os.path.isdir(cif_dir):
          exit("Could not set up cif_dir in %s" %(cif_dir))
        pdb_filepath = Path(cif_dir,make_four_char_name(filename))
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = run_pdb_to_cif(pdb_filepath,
           content_dir = params.get("maxit_path") if
           params.get("maxit_path") else params.get("content_dir"))
        manual_templates_uploaded.append(cif_filepath)


  if params.get('upload_msa_file') and params.get('use_msa'):
    print("MSA files uploaded: %s" %(msas_uploaded))
  else:
    assert not msas_uploaded

  if params.get('upload_maps'):
    print("Maps uploaded: %s" %(maps_uploaded))

  print("Templates uploaded: %s" %(manual_templates_uploaded))
  if (not params.get("upload_maps")) and (
      not manual_templates_uploaded):
    print(
      "\n*** WARNING: no templates uploaded...Please use only .cif files ***\n")
  return manual_templates_uploaded, maps_uploaded, msas_uploaded

def get_templates_from_drive(params):
  manual_templates_uploaded = []
  maps_uploaded = []
  msas_uploaded = []

  input_directory = params.get('input_directory',None)
  print("Input directory:",input_directory)
  if input_directory is None:
    exit(
      "No Google drive folder available. Please specify input_directory")

  jobname = params.get('jobname',None) # if None we are going to take everything
  cif_dir = params.get('cif_dir',None)
  if cif_dir is None:
    exit("Need a cif directory...")

  filename_list = os.listdir(input_directory)

  for filename in filename_list:
      sys.stdout.flush()
      useful_files = select_matching_template_files([Path(filename)], jobname)
      if not useful_files: # skip this one
        continue
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

      elif params.get('use_msa',None) and \
            params.get('upload_msa_file',None) and \
           str(filename).lower().endswith(".a3m"):
        filepath = Path(cif_dir,filename)
        ff = open(filepath, 'wb')
        ff.write(contents)
        msas_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".cif"):

        filepath = Path(cif_dir,make_four_char_name(filename))

        with filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
          manual_templates_uploaded.append(filepath)

      elif params.get('upload_manual_templates',None) and \
           str(filename).endswith(".pdb"):

        pdb_filepath = Path(cif_dir,make_four_char_name(filename))
        with pdb_filepath.open("w") as fh:
          fh.write(contents.decode("UTF-8"))
        cif_filepath = run_pdb_to_cif(pdb_filepath,
           content_dir = params.get("maxit_path") if
           params.get("maxit_path") else params.get("content_dir"))
        manual_templates_uploaded.append(cif_filepath)

  if manual_templates_uploaded:
    manual_templates_uploaded = select_matching_template_files(
      manual_templates_uploaded, jobname)
  if maps_uploaded:
    maps_uploaded = select_matching_template_files(
      maps_uploaded, jobname)
  if msas_uploaded:
    msas_uploaded = select_matching_template_files(
      msas_uploaded, jobname)

  if params.get('upload_maps',None):
    print("Maps taken from Google drive: %s" %(len(maps_uploaded)))

  if params.get('upload_msa_file',None) and len(msas_uploaded) > 0:
    print("MSA files taken from Google drive: %s" %(len(msas_uploaded)))

  print("Templates taken from Google drive: %s" %(manual_templates_uploaded))
  if (not params.get('upload_maps',None)) and (not manual_templates_uploaded):
    print("\n*** WARNING: no templates uploaded...***\n")

  return manual_templates_uploaded, maps_uploaded, msas_uploaded

def select_matching_template_files(uploaded_template_files,
    jobname):
  if not jobname:
    return uploaded_template_files

  matching_files = []
  jobname_four_char = make_four_char_name(jobname).replace(".cif","")
  for file_name in uploaded_template_files:
    if file_name.parts[-1].startswith(jobname.split("_")[0][:4]) and \
       (file_name.parts[-1].find(jobname) > -1 or
       file_name.parts[-1].find(jobname_four_char) > -1):
      # has entire jobname present
      matching_files.append(file_name)
  return matching_files

class job_file:
  def __init__(self, file_name):
    self.file_name = file_name
    if os.path.isfile(file_name):
      self.contents = open(file_name).read().encode("UTF-8")
    else:
      self.contents = ""
  def items(self):
    return [(self.file_name,self.contents),]


def get_jobnames_sequences_from_file(params):

  from io import StringIO
  print("Ready...",
       params.get('file_with_jobname_resolution_sequence_lines',None))
  if params.get('file_with_jobname_resolution_sequence_lines',None):
    uploaded_job_file = job_file(
       params.get('file_with_jobname_resolution_sequence_lines',None))
  else:  #usual
    from google.colab import files
    print("Upload file with one jobname, a space, resolution, space,"+
      " and one sequence on each line")
    uploaded_job_file = files.upload()

  if params.get('upload_manual_templates',None) or params.get(
      'upload_maps', None) or params.get('upload_msa_file', None):
    input_directory = params.get('input_directory',None)
    if input_directory and input_directory != params.get('content_dir'):
      uploaded_template_files, uploaded_maps, uploaded_msa_files= \
        get_templates_from_drive(params)
    else:
      print("\nUpload your templates now, all at once. " +\
           "NOTE: template file names must start with your job names")
      uploaded_template_files, uploaded_maps, uploaded_msa_files = \
        upload_templates(params)
    print("Total of %s template files uploaded: %s" %(
          len(uploaded_template_files), uploaded_template_files))
    print("Total of %s map files uploaded: %s" %(
          len(uploaded_maps), uploaded_maps))
    print("Total of %s MSA files uploaded: %s" %(
          len(uploaded_msa_files), uploaded_msa_files))

  else:
    uploaded_template_files = []
    uploaded_maps = []
    uploaded_msa_files = []

  s = StringIO()
  query_sequences = []
  jobnames = []
  resolutions = []
  include_templates_from_pdb_list = []
  include_side_in_templates_list = []
  cif_filename_dict = {}
  map_filename_dict = {}
  msa_filename_dict = {}
  for filename,contents in uploaded_job_file.items():
    print(contents.decode("UTF-8"), file = s)
    text = s.getvalue()
    for line in text.splitlines():
      # Allow a few variables
      if line.find('include_templates_from_pdb') > -1:
        line = line.replace("include_templates_from_pdb","")
        include_templates_from_pdb  = True
      else:
        include_templates_from_pdb  = None
      if line.find('include_side_in_templates') > -1:
        line = line.replace("include_side_in_templates","")
        include_side_in_templates= True
      else:
        include_side_in_templates= None

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
          include_templates_from_pdb_list.append(include_templates_from_pdb)
          include_side_in_templates_list.append(include_side_in_templates)
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
          if uploaded_msa_files:
              msa_filename_dict[jobname] = \
                select_matching_template_files(
                    uploaded_msa_files,
                     jobname)

  params['jobnames'] = jobnames
  params['resolutions'] = resolutions
  params['include_side_in_templates_list'] = include_side_in_templates_list
  params['include_templates_from_pdb_list'] = include_templates_from_pdb_list
  params['msa_filename_dict'] = msa_filename_dict
  params['map_filename_dict'] = map_filename_dict
  params['cif_filename_dict'] = cif_filename_dict
  params['query_sequences'] = query_sequences
  return params


def get_parent_dir(content_dir,
   parent_dir_name = "manual_templates"):
  return Path(os.path.join(content_dir,parent_dir_name))

def get_cif_dir(content_dir, jobname,
    cif_name = "mmcif"):
  parent_dir = get_parent_dir(content_dir)
  return Path(parent_dir,jobname,cif_name)

def set_up_input_files(params,
    require_map = None,
    convert_to_params = True,
    log = sys.stdout):

  from pathlib import Path

  if type(params) == type ({}):
    params_is_dict = True
    original_params = params
  else: # make it one
    params_is_dict = False
    original_params = params
    params = params_as_dict(original_params)  # makes a dict

  content_dir = params.get('content_dir',None)

  # Default working directory
  if not content_dir:
    params['content_dir'] = os.getcwd()
    content_dir = params['content_dir']

  # Set working directory
  os.chdir(content_dir)

  # Make sure params has cif_dir in it, even if None
  params['cif_dir'] = params.get('cif_dir',None)

  # Clear out directories
  parent_dir = get_parent_dir(content_dir)

  # get input and output directories
  params = get_input_output_dirs(params, log = log)
  # Initialize
  query_sequences = []
  jobnames = []
  resolutions = []
  cif_filename_dict = {}
  map_filename_dict = {}
  msa_filename_dict = {}
  dirs_to_clear = []

  if params.get(
     'upload_file_with_jobname_resolution_sequence_lines',None) or \
       params.get('file_with_jobname_resolution_sequence_lines',None):
    params = set_upload_dir(params)
    params = get_jobnames_sequences_from_file(params)
    jobnames = params['jobnames']
    resolutions = params['resolutions']
    msa_filename_dict = params['msa_filename_dict']
    map_filename_dict = params['map_filename_dict']
    cif_filename_dict = params['cif_filename_dict']
    query_sequences = params['query_sequences']
    include_templates_from_pdb_list = params['include_templates_from_pdb_list']
    include_side_in_templates_list = params['include_side_in_templates_list']
  else: # usual
    jobname = params.get('jobname',None)
    query_sequence = params.get('query_sequence',None)
    resolution = params.get('resolution',None)
    upload_manual_templates = params.get('upload_manual_templates',None)
    upload_maps = params.get('upload_maps',None)
    include_side_in_templates_list = None
    include_templates_from_pdb_list = None

    jobname = clean_jobname(jobname, query_sequence)
    query_sequence = clean_query(query_sequence)
    if query_sequence and not jobname:
      print("Please enter a job name and rerun")
      exit("Please enter a job name and rerun")

    if jobname and not query_sequence:
      print("Please enter a query_sequence and rerun")
      exit("Please enter a query_sequence rerun")

    # Add sequence and jobname if new

    if (jobname and query_sequence) and (
         not query_sequence in query_sequences) and (
         not jobname in jobnames):
        query_sequences.append(query_sequence)
        jobnames.append(jobname)
        resolutions.append(resolution)
        params['cif_dir'] = get_cif_dir(params['content_dir'], jobname)
        clear_directories([params['cif_dir']], log = log)

        if upload_manual_templates or upload_maps \
           or params.get('upload_msa_file',None):
          input_directory = params.get('input_directory', None)
          if input_directory and input_directory != params.get('content_dir'):
            cif_filename_dict[jobname], map_filename_dict[jobname],\
              msa_filename_dict[jobname] = get_templates_from_drive(params)
          else:
            params['upload_dir'] = params['cif_dir']
            print("\nPlease upload %s for %s" %(
              "template and map" if upload_manual_templates and upload_maps
              else "template" if upload_manual_templates
              else "map",
              jobname))
            sys.stdout.flush()
            cif_filename_dict[jobname], map_filename_dict[jobname], \
              msa_filename_dict[jobname] = upload_templates(params)

  # Save sequence
  for i in range(len(query_sequences)):
    # save the sequence as a file with name jobname.fasta
    save_sequence(jobnames[i], query_sequences[i], log = log)

  if params.get('upload_maps'):
    if include_templates_from_pdb_list or include_side_in_templates_list:
      print("\nCurrent jobs, resolutions, sequences, templates, "+
       "maps and msas, include_templates include_side_in_templates:",
        file = log)
    else:
      print("\nCurrent jobs, resolutions, sequences, templates, maps and msas:",
       file = log)
  else:
    print("\nCurrent jobs,  sequences templates and msas:", file = log)
  if not include_templates_from_pdb_list:
    include_templates_from_pdb_list = len(resolutions) * [None]
  if not include_side_in_templates_list:
    include_side_in_templates_list = len(resolutions) * [None]
  for qs,jn,res,inclt,incls in zip(query_sequences, jobnames, resolutions,
       include_templates_from_pdb_list, include_side_in_templates_list):
    template_list = []
    for t in cif_filename_dict.get(jn,[]):
      template_list.append(os.path.split(str(t))[-1])
    map_list = []
    for t in map_filename_dict.get(jn,[]):
      map_list.append(os.path.split(str(t))[-1])
    msa_list = []
    for t in msa_filename_dict.get(jn,[]):
      msa_list.append(os.path.split(str(t))[-1])
    if inclt or incls:
      print(jn, res, qs, template_list, map_list, msa_list, inclt, incls,
         file = log)
    else:
      print(jn, res, qs, template_list, map_list, msa_list, file = log)

    if len(qs) < 20:
      print("\n\nMinimum sequence length is 20 residues...\n\n",
            "Please enter a longer sequence and \n",
            "run again\n\n")
      sys.stdout.flush()
      exit("Sequence must be 20 residues or more")
    if not map_list and params.get('upload_maps',None):
      print("\n\nNeed a map for each sequence...\n\n",
        "Please be sure input_directory contains your map file or "+
        "input_directory is not set and run again\n\n")
      sys.stdout.flush()
      exit("Map file needed for job %s" %(jn))
    if not msa_list and params.get('upload_msa_file',None) and \
          params.get('use_msa',None):
      print("\n\nNeed an msa for each sequence...\n\n",
        "Please be sure input_directory contains your msa file or "+
        "input_directory is not set or upload_msa_file is not set "+
        "and run again\n\n")
      sys.stdout.flush()
      exit("MSA file needed for job %s" %(jn))


  if not query_sequences:
    print("Please supply a query sequence and run again")
    exit("Need a query sequence")

  params['jobnames'] = jobnames
  params['resolutions'] = resolutions
  params['map_filename_dict'] = map_filename_dict
  params['msa_filename_dict'] = msa_filename_dict
  params['cif_filename_dict'] = cif_filename_dict
  params['query_sequences'] = query_sequences
  params['include_side_in_templates_list'] = include_side_in_templates_list
  params['include_templates_from_pdb_list'] = include_templates_from_pdb_list

  if params_is_dict and not convert_to_params:
    return params  # return dict version (does not require phenix)

  elif params_is_dict: # set values in get_alphafold_with_density_map
    # params and return
    return get_alphafold_with_density_map_params(params)

  else: # came in with params; set values and return

    for key in params.keys():
      setattr(original_params,key,params[key])
    return original_params

def get_alphafold_with_density_map_params(params):
  if type(params) != type({}):
    return params # already set

  try:
    from phenix.programs.alphafold_with_density_map import master_phil_str
    import iotbx.phil
    full_params = iotbx.phil.parse(master_phil_str).extract()
  except Exception as e:
    # Here if no phenix...just make a group_args object
    full_params = group_args(
      group_args_type = 'dummy parameters',
      )


  for key in params.keys():
    try:
      setattr(full_params,key,params[key])
    except Exception as e:
      print("Note: not setting parameter %s" %(key))
  return full_params

def set_upload_dir(params):
    params['upload_dir'] = Path(params.get('working_directory',os.getcwd()),
       "upload_dir")
    if not os.path.isdir(params['upload_dir']):
      params['upload_dir'].mkdir(parents=True)
    print("Upload dir will be: %s" %(params['upload_dir']))
    return params

class group_args:
  """
  Class to build an arbitrary object from a list of keyword arguments.
  Copied from cctbx_project/iotbx/__init__.py

  Examples
  --------
  >>> from libtbx import group_args
  >>> obj = group_args(a=1, b=2, c=3)
  >>> print(obj.a, obj.b, obj.c)
  1 2 3

  Once stop_dynamic_attributes is called, adding new attributes won't be
  possible, that is this:

  obj.tmp=10

  will fail.
  """

  def __init__(self, **keyword_arguments):
    self.__dict__.update(keyword_arguments)

  def __call__(self):
    return self.__dict__

  def get(self,kw):
    return self.__dict__.get(kw)

  def keys(self):
    return self.__dict__.keys()

  def __repr__(self):
    outl = "group_args"
    for attr in sorted(self.__dict__.keys()):
      tmp=getattr(self, attr)
      if str(tmp).find("ext.atom ")>-1:
        outl += "\n  %-30s : %s" % (attr, tmp.quote())
      else:
        outl += "\n  %-30s : %s" % (attr, tmp)
    return outl

  def merge(self, other):
    """ To merge other group_args into self.
    Overwrites matching fields!!!"""
    self.__dict__.update(other.__dict__)

  def add(self,key=None,value=None):
    self.__dict__[key]=value

  def copy(self):
    """ produce shallow copy of self by converting to dict and back"""
    return group_args(**self().copy())
