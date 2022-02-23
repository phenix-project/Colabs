from __future__ import division
from __future__ import print_function

import os, sys
import shutil

from pathlib import Path
import numpy as np

from phenix_alphafold_utils import clear_directories
from phenix_colab_utils import runsh

try:
  from libtbx import group_args
except Exception as e:  # we don't have phenix, use dummy version
  from phenix_alphafold_utils import group_args

from alphafold_utils import (mk_mock_template,
   predict_structure,
   show_pdb,
   get_msa,
   get_cif_file_list,
   get_template_hit_list)

def run_jobs(params):

  # RUN THE JOBS HERE
  os.chdir(params.content_dir)
  print("Overall working directory: %s" %(os.getcwd()))

  result_list = []
  for query_sequence, jobname, resolution in zip(
    params.query_sequences, params.jobnames, params.resolutions):
    os.chdir(params.content_dir)
    print("\n","****************************************","\n",
         "RUNNING JOB %s with sequence %s at resolution of %s\n" %(
      jobname, query_sequence, resolution),
      "****************************************","\n")
    # GET TEMPLATES AND SET UP FILES
    from copy import deepcopy
    working_params = deepcopy(params)
    working_params.query_sequence = query_sequence
    working_params.jobname = jobname
    working_params.resolution =resolution

    # We are going to work in a subdirectory
    working_params.working_directory= os.path.join(params.content_dir,
      jobname)
    if not os.path.isdir(working_params.working_directory):
      os.mkdir(working_params.working_directory)
    os.chdir(working_params.working_directory)
    print("Working directory for job %s: %s" %(
       jobname, os.getcwd()))

    # User input of manual templates
    working_params.manual_templates_uploaded = \
        working_params.cif_filename_dict.get(working_params.jobname,[])
    if working_params.manual_templates_uploaded:
      print("Using uploaded templates %s for this run" %(
          working_params.manual_templates_uploaded))
    working_params.maps_uploaded = working_params.map_filename_dict.get(
      working_params.jobname,[])
    if working_params.maps_uploaded:
      print("Using uploaded maps %s for this run" %(
          working_params.maps_uploaded))
      assert len(working_params.maps_uploaded) == 1

    if working_params.debug:
      result = run_job(params = working_params)
    else: # usual
      try:
        result = run_job(params = working_params)
        if result and result.filename:
          filename = result.filename
          print("FINISHED JOB (%s) %s with sequence %s and map-model CC of %.2f\n" %(
          filename, working_params.jobname, working_params.query_sequence,
          result.cc if result.cc is not None else 0.0),
          "****************************************","\n")
        else:
          print("NO RESULT FOR JOB %s with sequence %s\n" %(
        working_params.jobname, working_params.query_sequence),
        "****************************************","\n")

      except Exception as e:
        print("FAILED: JOB %s with sequence %s\n\n%s\n" %(
        working_params.jobname, working_params.query_sequence, str(e)),
        "****************************************","\n")
        result = None
    if result:
      result_list.append(result)
  return result_list

def run_one_af_cycle(params):

  from alphafold.data.templates import (_get_pdb_id_and_chain,
                                    _process_single_hit,
                                    _build_query_to_hit_index_mapping,
                                    _extract_template_features,
                                    SingleHitResult,
                                    TEMPLATE_FEATURES)

  os.chdir(params.working_directory)
  if params.template_hit_list:
    #process hits into template features
    from dataclasses import replace
    params.template_hit_list = [[replace(hit,**{"index":i+1}),mmcif]
        for i,[hit,mmcif] in enumerate(params.template_hit_list)]

    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    # Select only one chain from any cif file
    unique_template_hits = []
    pdb_text_list = []

    for i,[hit,mmcif] in enumerate(sorted(params.template_hit_list,
          key=lambda xx: xx[0].sum_probs, reverse=True)):

      pdb_text = hit.name.split()[0].split("_")[0]
      if pdb_text in pdb_text_list:
        continue # skip dups from same PDB entry
      pdb_text_list.append(pdb_text)
      unique_template_hits.append(hit)

      # modifications to alphafold/data/templates.py _process_single_hit
      hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)
      mapping = _build_query_to_hit_index_mapping(
      hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,
      params.query_sequence)
      template_sequence = hit.hit_sequence.replace('-', '')

      try:
        features, realign_warning = _extract_template_features(
          mmcif_object=mmcif,
          pdb_id=hit_pdb_code,
          mapping=mapping,
          template_sequence=template_sequence,
          query_sequence=params.query_sequence,
          template_chain_id=hit_chain_id,
          kalign_binary_path="kalign")
      except Exception as e:
        continue
      features['template_sum_probs'] = [hit.sum_probs]
      single_hit_result = SingleHitResult(
        features=features, error=None, warning=None)
      for k in template_features:
        template_features[k].append(features[k])

    for name in template_features:
      template_features[name] = np.stack(
          template_features[name], axis=0).astype(TEMPLATE_FEATURES[name])



    print("\nIncluding templates:")
    for hit in unique_template_hits:
      print("\t",hit.name.split()[0])

    if len(unique_template_hits) == 0:
      print("No templates found...quitting")
      raise AssertionError("No templates found...quitting")

    for key,value in template_features.items():
      if np.all(value==0):
        print("ERROR: Some template features are empty")
  else:  # no templates
    print("Not using any templates")
    template_features = mk_mock_template(
       params.query_sequence * params.homooligomer)

  print("\nPREDICTING STRUCTURE")
  print("\nMaximum randomization tries: %s " %(params.random_seed_iterations))

  from alphafold.model import data
  from alphafold.model import config
  from alphafold.model import model

  # collect model weights
  use_model = {}
  model_params = {}
  model_runner_1 = None
  model_runner_3 = None
  for model_name in ["model_1","model_2","model_3",
      "model_4","model_5"][:params.num_models]:
    use_model[model_name] = True
    if model_name not in list(model_params.keys()):
      model_params[model_name] = data.get_model_haiku_params(
        model_name=model_name+"_ptm", data_dir = params.data_dir)
      if model_name == "model_1":
        model_config = config.model_config(model_name+"_ptm")
        model_config.data.eval.num_ensemble = 1
        model_runner_1 = model.RunModel(model_config, model_params[model_name])
      if model_name == "model_3":
        model_config = config.model_config(model_name+"_ptm")
        model_config.data.eval.num_ensemble = 1
        model_runner_3 = model.RunModel(model_config, model_params[model_name])
  if params.homooligomer == 1:
    msas = [params.msa]
    deletion_matrices = [params.deletion_matrix]
  else: # ZZZ THis won't work with msa_is_msa_object = True yet ZZ
    # make multiple copies of msa for each copy
    # AAA------
    # ---AAA---
    # ------AAA
    #
    # note: if you concat the sequences (as below), it does NOT work
    # AAAAAAAAA
    msas = []
    deletion_matrices = []
    Ln = len(params.query_sequence)
    for o in range(params.homooligomer):
      L = Ln * o
      R = Ln * (params.homooligomer-(o+1))
      msas.append(["-"*L+seq+"-"*R for seq in params.msa])
      deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in params.deletion_matrix])

  # gather features
  from alphafold.data import pipeline
  if params.msa_is_msa_object:
    feature_dict = {
      **pipeline.make_sequence_features(
                  sequence=params.query_sequence*params.homooligomer,
                   description="none",
                   num_res=len(params.query_sequence)*params.homooligomer),
      **pipeline.make_msa_features(msas),
      **template_features
     }
  else: # original version
    feature_dict = {
      **pipeline.make_sequence_features(
                  sequence=params.query_sequence*params.homooligomer,
                   description="none",
                   num_res=len(params.query_sequence)*params.homooligomer),
      **pipeline.make_msa_features(
         msas=msas,deletion_matrices=deletion_matrices),
      **template_features
     }



  outs = predict_structure(params.jobname, feature_dict,
                           Ls=[len(params.query_sequence)]*params.homooligomer,
                           model_params=model_params,
                           use_model=use_model,
                           model_runner_1=model_runner_1,
                           model_runner_3=model_runner_3,
                           do_relax=False,
                           msa_is_msa_object = params.msa_is_msa_object,
                       random_seed = params.random_seed,
                       random_seed_iterations =
                         params.random_seed_iterations if params.cycle == 1 else
                         params.minimum_random_seed_iterations,
                       big_improvement = params.big_improvement if
                        hasattr(params, 'big_improvement') else 5,)
  if outs: # ok
    print("Done with prediction in",os.getcwd())
  else: # failed
    print("Prediction failed...probably ran out of memory...quitting")
    return None

  os.chdir(params.working_directory)
  model_file_name = "%s_unrelaxed_model_1.pdb" %(params.jobname)
  if os.path.isfile(model_file_name):
    print("Model file is in %s" %(model_file_name))
    cycle_model_file_name = "%s_unrelaxed_model_1_%s.pdb" %(
        params.jobname, params.cycle)
    check_and_copy(model_file_name, cycle_model_file_name)
    check_and_copy(model_file_name, get_af_file_name(params))
    if params.output_directory is not None:
      check_and_copy(model_file_name,
         os.path.join(params.output_directory, cycle_model_file_name))
  else:
    print("No model file %s found for job %s" %(model_file_name,
      params.jobname))
    cycle_model_file_name = None

  #@title Making plots...
  try:
    import matplotlib.pyplot as plt
    from alphafold_utils import plot_plddt_legend, plot_confidence, write_pae_file
  except Exception as e: # No matplotlib
    plt = None

  for n,(model_name,value) in enumerate(outs.items()):
    if n > 0: break # only do one

    # Write pae file
    pae_file = get_pae_file_name(params)
    write_pae_file(value["pae"], pae_file)

  if plt and (not params.msa_is_msa_object):
    # gather MSA info # ZZZ This won't work with msa_is_msa_object
    deduped_full_msa = list(dict.fromkeys(params.msa))
    msa_arr = np.array([list(seq) for seq in deduped_full_msa])
    seqid = (np.array(list(params.query_sequence)) == msa_arr).mean(-1)
    seqid_sort = seqid.argsort() #[::-1]
    non_gaps = (msa_arr != "-").astype(float)
    non_gaps[non_gaps == 0] = np.nan
    ##################################################################
    plt.figure(figsize=(14,4),dpi=100)
    ##################################################################
    plt.subplot(1,2,1); plt.title("Sequence coverage")
    plt.imshow(non_gaps[seqid_sort]*seqid[seqid_sort,None],
               interpolation='nearest', aspect='auto',
               cmap="rainbow_r", vmin=0, vmax=1, origin='lower')
    plt.plot((msa_arr != "-").sum(0), color='black')
    plt.xlim(-0.5,msa_arr.shape[1]-0.5)
    plt.ylim(-0.5,msa_arr.shape[0]-0.5)
    plt.colorbar(label="Sequence identity to query",)
    plt.xlabel("Positions")
    plt.ylabel("Sequences")

    ##################################################################
    plt.subplot(1,2,2); plt.title("Predicted lDDT per position")
    for model_name,value in outs.items():
      plt.plot(value["plddt"],label=model_name)
    if params.homooligomer > 0:
      for n in range(params.homooligomer+1):
        x = n*(len(params.query_sequence)-1)
        plt.plot([x,x],[0,100],color="black")
    plt.legend()
    plt.ylim(0,100)
    plt.ylabel("Predicted lDDT")
    plt.xlabel("Positions")
    plt.savefig(get_plddt_png_file_name(params))
    ##################################################################
    plt.show()

    print("Predicted Alignment Error")
    ##################################################################
    plt.figure(figsize=(3*params.num_models,2), dpi=100)
    for n,(model_name,value) in enumerate(outs.items()):
      if n > 0: break # only do one
      plt.subplot(1,params.num_models,n+1)
      plt.title(model_name)
      plt.imshow(value["pae"],label=model_name,cmap="bwr",vmin=0,vmax=30)
      plt.colorbar()
    plt.savefig(get_pae_png_file_name(params))
    plt.show()
    ##################################################################
    #@title Displaying 3D structure... {run: "auto"}
    model_num = 1
    color = "lDDT"
    show_sidechains = False
    show_mainchains = False



    show_pdb(params.jobname, model_num,show_sidechains,
        show_mainchains, color).show()
    if color == "lDDT": plot_plddt_legend().show()
    plot_confidence(params.homooligomer,
       params.query_sequence, outs, model_num).show()

  return group_args(
    group_args_type = 'result for one cycle',
    cycle_model_file_name = cycle_model_file_name,
    )

def get_rebuilt_file_names(params):
  af_model_file = os.path.abspath(
      params.cycle_model_file_name.as_posix())
  rebuilt_model_name = af_model_file.replace(".pdb","_rebuilt.pdb")
  rebuilt_model_stem = rebuilt_model_name.replace(".pdb","")
  return group_args(group_args_type = ' rebuilt file names',
    af_model_file = af_model_file,
    rebuilt_model_name = rebuilt_model_name,
    rebuilt_model_stem = rebuilt_model_stem)


def rebuild_model(params,
        nproc = 4):

  assert len(params.maps_uploaded) == 1  # just one map
  map_file_name = params.maps_uploaded[0]

  file_name_info = get_rebuilt_file_names(params)
  af_model_file = file_name_info.af_model_file
  rebuilt_model_name = file_name_info.rebuilt_model_name
  rebuilt_model_stem = file_name_info.rebuilt_model_stem

  if params.previous_final_model_name:
    previous_model_file = os.path.abspath(
      params.previous_final_model_name.as_posix())
  else:
    previous_model_file = None
  output_file_name = os.path.abspath(
      "%s_rebuilt_%s.pdb" %(params.jobname,params.cycle))
  print("Rebuilding %s %s with map in %s at resolution of %.2f" %(
       af_model_file,
        " with previous model of %s" %(previous_model_file) \
        if previous_model_file else "",
       map_file_name,
      params.resolution,))

  for ff in (map_file_name,af_model_file, previous_model_file):
    if ff and not os.path.isfile(ff):
      print("\nMissing the file: %s" %(ff))
      return None


  # run phenix dock_and_rebuild here
  from phenix_colab_utils import run_command # run and get output to terminal
  text = "phenix.dock_and_rebuild fragments_model_file=%s nproc=%s resolution=%s previous_model_file=%s model=%s full_map=%s output_model_prefix=%s " %(
     params.mtm_file_name,nproc,params.resolution,previous_model_file,
     af_model_file,map_file_name,
      rebuilt_model_stem,
      )
  result = run_command(text)

  if os.path.isfile(rebuilt_model_name):
    print("Rebuilding successful")
    from iotbx import data_manager
    return group_args(
      group_args_type = 'rebuilt model',
      final_model_file_name = rebuilt_model_name,
      cc = get_map_model_cc(map_file_name = map_file_name,
        model_file_name = rebuilt_model_name,
        resolution = params.resolution))
  else:
    print("Rebuilding not successful")
    return None

def get_map_model_cc(map_file_name, model_file_name, resolution):
  from iotbx.data_manager import DataManager
  dm = DataManager()
  dm.set_overwrite(True)
  if hasattr(map_file_name,'as_posix'):
    map_file_name = map_file_name.as_posix()
  if hasattr(model_file_name,'as_posix'):
    model_file_name = model_file_name.as_posix()
  mmm = dm.get_map_model_manager(map_files = map_file_name,
      model_file = model_file_name)
  mmm.set_resolution(resolution)
  return mmm.map_model_cc()

def get_rmsd(fn_1, fn_2):
  # Get rmsd between these models
  from iotbx.data_manager import DataManager
  dm = DataManager()
  if hasattr(fn_1,'as_posix'):
    fn_1 = fn_1.as_posix()
  if hasattr(fn_2,'as_posix'):
    fn_2 = fn_2.as_posix()
  m1 = dm.get_model(fn_1)
  m2 = dm.get_model(fn_2)
  mmm = m1.as_map_model_manager()
  mb = mmm.model_building()
  rmsd_info = mb.ca_rmsd_after_lsq_superpose(
    fixed_model = m1,
    moving_model = m2)
  if rmsd_info:
    return rmsd_info.rmsd
  else:
    return None

def get_map_to_model(map_file_name,
    resolution,
    seq_file,
    output_file_name = None,
    nproc = 4):

  runsh(
   "phenix.map_to_model nproc=%s seq_file=%s resolution=%s %s pdb_out=%s" %(
     nproc, seq_file, resolution, map_file_name, output_file_name) )
  return output_file_name

def run_job(params = None,
   max_alphafold_attempts = 3):

  from Bio.SeqRecord import SeqRecord
  from Bio.Seq import Seq
  from Bio import SeqIO

  if not params.content_dir:
    params.content_dir = "/content/"

  if params.random_seed is None:
    params.random_seed = 717217

  os.chdir(params.working_directory)

  #standard values of parameters
  params.num_models = 1
  params.homooligomer = 1
  params.use_env = True
  params.use_custom_msa = False
  params.use_templates = True


  #Get the MSA
  params.msa, params.deletion_matrix, params.template_paths, \
    params.msa_is_msa_object = get_msa(params)

  #Process templates
  print("PROCESSING TEMPLATES")

  jobname = params.jobname

  other_cif_dir = Path(os.path.join(params.working_directory,
     params.template_paths))
  from phenix_alphafold_utils import get_parent_dir
  parent_dir = get_parent_dir(params.content_dir)
  from phenix_alphafold_utils import get_cif_dir
  cif_dir = get_cif_dir(params.content_dir, jobname)

  fasta_dir = Path(parent_dir,jobname,"fasta")
  hhDB_dir = Path(parent_dir,jobname,"hhDB")
  msa_dir = Path(hhDB_dir,"msa")
  clear_directories([fasta_dir,hhDB_dir,msa_dir])
  if not params.cif_dir:
    params.cif_dir = cif_dir
  assert params.cif_dir == cif_dir

  if params.uploaded_templates_are_map_to_model and \
      params.manual_templates_uploaded: # mtm
    print("Uploaded tempates are map to model")
    params.mtm_file_name = params.manual_templates_uploaded[0]
    params.manual_templates_uploaded = []
  else:
    print("Uploaded templates are actual templates")
    params.mtm_file_name = "None"

  pdb_cif_file_list = get_cif_file_list(
    include_templates_from_pdb = params.include_templates_from_pdb,
    manual_templates_uploaded = None,
    cif_dir = cif_dir,
    other_cif_dir = other_cif_dir)
  print("CIF files from PDB to include:",pdb_cif_file_list)

  manual_cif_file_list = get_cif_file_list(
    include_templates_from_pdb = False,
    manual_templates_uploaded = params.manual_templates_uploaded,
    cif_dir = cif_dir)
  print("Uploaded CIF files to include:",manual_cif_file_list)

  query_seq = SeqRecord(Seq(params.query_sequence),id="query",
    name="",description="")
  query_seq_path = Path(fasta_dir,"query.fasta")
  with query_seq_path.open("w") as fh:
      SeqIO.write([query_seq], fh, "fasta")
  shutil.copyfile(query_seq_path,Path(msa_dir,"query.fasta"))

  # Removed 2022-02-23 because usually a manual template is not placed in map
  previous_final_model_name = None
  #   params.manual_templates_uploaded[0] if  params.manual_templates_uploaded else None

  if params.upload_maps:
    assert len(params.maps_uploaded) == 1  # just one map
    map_file_name = params.maps_uploaded[0]
  else:
    map_file_name = None

  seq_file = "%s.seq" %(jobname)
  ff = open(seq_file,'w')
  print(params.query_sequence, file = ff)
  ff.close()

  # Run cycles

  rmsd_from_previous_cycle_list = []
  previous_cycle_model_file_name = None
  final_model_file_name = None
  cycle_model_file_name = None

  os.chdir(params.working_directory)

  for cycle in range(1, max(1,params.maximum_cycles) + 1):

    # Decide if it is time to quit
    if change_is_small(params, rmsd_from_previous_cycle_list):
      print("Ending cycles as changes are small...")
      break


    params.cycle = cycle
    print("\nStarting cycle %s" %(cycle))
    if params.cycle == 2 and params.skip_all_msa_after_first_cycle:
      print("Getting dummy msa for cycles after the first")
      #Get dummy msa
      params.use_msa = False
      params.msa, params.deletion_matrix, params.template_paths, \
        params.msa_is_msa_object = get_msa(params)

    working_cif_file_list = list(manual_cif_file_list)
    if params.cycle == 1:
      working_cif_file_list +=  \
       list(pdb_cif_file_list)[:params.maximum_templates_from_pdb]

    print("Templates used in this cycle: %s" %(
        " ".join([w.as_posix() for w in working_cif_file_list])))

    params.template_hit_list = get_template_hit_list(
      cif_files = working_cif_file_list,
      fasta_dir = fasta_dir,
      query_seq = query_seq,
      hhDB_dir = hhDB_dir,
      content_dir = params.content_dir)
    os.chdir(params.working_directory) # REQUIRED


    expected_cycle_model_file_name = "%s_unrelaxed_model_1_%s.pdb" %(
        jobname, params.cycle)
    if params.carry_on and params.output_directory:
      expected_cycle_model_file_name_in_output_dir = os.path.join(
        params.output_directory,
         expected_cycle_model_file_name)

    else:
      expected_cycle_model_file_name_in_output_dir = None

    if params.carry_on and params.output_directory and os.path.isfile(
         expected_cycle_model_file_name_in_output_dir):
      print("Reading in AlphaFold model from %s" %(
        expected_cycle_model_file_name_in_output_dir))
      result = group_args(group_args_type = 'af model read in directly',
        cycle_model_file_name = expected_cycle_model_file_name)

      check_and_copy(expected_cycle_model_file_name_in_output_dir,
        expected_cycle_model_file_name)
      check_and_copy(os.path.join(params.output_directory,
         get_pae_file_name(params)),get_pae_file_name(params))
      check_and_copy(os.path.join(params.output_directory,
         get_pae_png_file_name(params)),get_pae_png_file_name(params))
      check_and_copy(os.path.join( params.output_directory,
         get_plddt_png_file_name(params)),get_plddt_png_file_name(params))
      check_and_copy(os.path.join( params.output_directory,
         get_af_file_name(params)),get_af_file_name(params))


    else: # Get AlphaFold model here
      if params.debug:
        result = run_one_af_cycle(params)
      else:  # usual
        from copy import deepcopy
        template_hit_list = deepcopy(params.template_hit_list)
        for attempt in range(max_alphafold_attempts):
          try:
            result = run_one_af_cycle(params)
          except Exception as e:
            result = None
            print("AlphaFold modeling failed...trying again (#%s)" %(
              attempt))
            # get the hit list again
            params.template_hit_list = deepcopy(template_hit_list)
          if result:
            break

      check_and_copy(get_pae_file_name(params),
         os.path.join(params.output_directory,
         get_pae_file_name(params)))
      check_and_copy(get_pae_png_file_name(params),
         os.path.join(params.output_directory,
         get_pae_png_file_name(params)))
      check_and_copy(get_plddt_png_file_name(params),
         os.path.join( params.output_directory,
         get_plddt_png_file_name(params)))
      check_and_copy(get_af_file_name(params),
         os.path.join( params.output_directory,
         get_af_file_name(params)))

    if (not result) or (not result.cycle_model_file_name) or (
         not os.path.isfile(result.cycle_model_file_name)):
      print("Modeling failed at cycle %s" %(cycle))
      print("You might try checking the 'carry_on' box and rerunning to go on")
      return None


    cycle_model_file_name = result.cycle_model_file_name

    print("\nFinished with cycle %s of AlphaFold model generation" %(cycle))
    cycle_model_file_name = Path(cycle_model_file_name)
    print("Current AlphaFold model is in %s" %(
        cycle_model_file_name.as_posix()))

    if not map_file_name: # we are done (no map).  Just copy AF model
      break

    if previous_cycle_model_file_name and \
      os.path.isfile(previous_cycle_model_file_name):
      rmsd_from_previous = get_rmsd(cycle_model_file_name.as_posix(),
        previous_cycle_model_file_name.as_posix())
      if rmsd_from_previous is not None:
        rmsd_from_previous_cycle_list.append(rmsd_from_previous)
        print("RMSD of predicted model from last cycle: %.2f A" %(
          rmsd_from_previous))
    previous_cycle_model_file_name = cycle_model_file_name

    if params.output_directory is not None:
      cycle_model_file_name_in_output_dir = Path(
        os.path.join(params.output_directory,cycle_model_file_name.name))
      check_and_copy(cycle_model_file_name,
         cycle_model_file_name_in_output_dir)
      print("Copied AlphaFold model to %s" %(
          cycle_model_file_name_in_output_dir))

    print("\nGetting a new rebuilt model at a resolution of %.2f A" %(
        params.resolution))
    # Now get a new rebuilt model
    params.cycle_model_file_name = cycle_model_file_name
    params.previous_final_model_name = previous_final_model_name

    # Decide if we can just read it in...or really build it
    file_name_info = get_rebuilt_file_names(params)

    if params.carry_on and params.output_directory:
       final_model_file_name_in_output_dir = os.path.join(
         params.output_directory,os.path.split(
         file_name_info.rebuilt_model_name)[-1])
    if params.carry_on and params.output_directory and \
          os.path.isfile(final_model_file_name_in_output_dir):
      print("Reading rebuilt model from %s" %(
         final_model_file_name_in_output_dir))
      final_model_file_name = os.path.split(
        file_name_info.rebuilt_model_name)[-1]
      check_and_copy(final_model_file_name_in_output_dir,
         final_model_file_name)
      rebuild_result = group_args(group_args_type = 'rebuild result read in',
        final_model_file_name  = final_model_file_name,
        cc = get_map_model_cc(map_file_name = map_file_name,
          model_file_name = final_model_file_name,
          resolution = params.resolution),)

    else: # usual
      rebuild_result = rebuild_model(params)

    if not rebuild_result or not rebuild_result.final_model_file_name:
      print("No rebuilt model obtained")
      final_model_file_name = None
      cc = None
    else:
      print("Rebuilt model with map-model cc of %s obtained" %(
         rebuild_result.cc if rebuild_result.cc is not None else 0))
      final_model_file_name = rebuild_result.final_model_file_name
      cc = rebuild_result.cc

    if not final_model_file_name:
      print("\nEnding cycles as no rebuilt model obtained")
      break

    final_model_file_name = Path(final_model_file_name)

    final_model_file_name_in_cif_dir = Path(
        os.path.join(cif_dir,final_model_file_name.name))
    check_and_copy(final_model_file_name, final_model_file_name_in_cif_dir)

    if params.output_directory is not None:
      final_model_file_name_in_output_dir = Path(
        os.path.join(params.output_directory,final_model_file_name.name))
      if os.path.isfile(final_model_file_name):
        check_and_copy(final_model_file_name,
          final_model_file_name_in_output_dir)
        print("Copied rebuilt model to %s" %(
          final_model_file_name_in_output_dir))
      else:
        print("Rebuilt model is %s" %(
          final_model_file_name_in_output_dir))

    # Superpose AF model on rebuilt model and write rebuilt model to std name

    if os.path.isfile(expected_cycle_model_file_name) and \
        os.path.isfile(final_model_file_name):
      # Copy final (rebuilt) model to standard name
      rebuilt_model_name = os.path.join(
        params.working_directory, "%s_REBUILT_cycle_%s.pdb" %(
        jobname, params.cycle))
      check_and_copy(final_model_file_name, rebuilt_model_name)
      print("Rebuilt model is in %s" %(rebuilt_model_name))

      print("Superposing AF model %s on rebuilt model (%s)" %(
         expected_cycle_model_file_name,final_model_file_name))
      superposed_af_model_name = os.path.join(
        params.working_directory, get_af_file_name(params))

      runsh("phenix.superpose_and_morph morph=False trim=False "+
         "fixed_model=%s moving_model=%s superposed_model=%s > super.log" %(
         final_model_file_name, expected_cycle_model_file_name,
         superposed_af_model_name))
      if os.path.isfile(superposed_af_model_name):
        print("Current superposed AF model is in %s" %(
          superposed_af_model_name))
      if params.output_directory is not None:
        superposed_af_model_name_in_output_dir = Path(
          os.path.join(params.output_directory,superposed_af_model_name))
        check_and_copy(superposed_af_model_name,
            superposed_af_model_name_in_output_dir)

    from phenix_colab_utils import run_pdb_to_cif
    final_model_file_name_as_cif_in_cif_dir = run_pdb_to_cif(
       final_model_file_name_in_cif_dir, content_dir = params.content_dir)
    manual_cif_file_list = get_cif_file_list(
      include_templates_from_pdb = False,
      manual_templates_uploaded = [
        final_model_file_name_as_cif_in_cif_dir.name],
      cif_dir = cif_dir)
    previous_final_model_name = final_model_file_name
    final_model_file_name = os.path.abspath(final_model_file_name)

  # All done with cycles here

  # Get final zip file
  try:
      runsh(
      "zip -FSr %s'.result.zip'  %s*ALPHAFOLD*.pdb  %s*REBUILT*.pdb %s*PAE*.jsn  %s*PAE*.png %s*lDDT*.png" %(
         jobname,jobname,jobname,jobname,jobname,jobname))
      zip_file_name = f"{jobname}.result.zip"
  except Exception as e:
      zip_file_name = None

  filename = zip_file_name
  if filename and os.path.isfile(filename):
    print("About to download %s" %(filename))
    try:
      print("Downloading zip file %s" %(filename))
      from google.colab import files
      files.download(filename)
      print("Start of download successful (NOTE: if the download symbol does "+
        "not go away it did not work. Download it manually using the folder "+
        "icon to the left)")
      check_and_copy(filename, os.path.join(params.content_dir,filename))
    except Exception as e:
      print("Unable to download zip file %s" %(filename))


  os.chdir(params.content_dir)
  if filename and os.path.isfile(filename):
    print("\nZIP file with results for %s is in %s" %(
      jobname, filename))

  if final_model_file_name:
    print("Returning final model")
    return group_args(
      group_args_type = 'rebuilding result',
      filename = os.path.abspath(final_model_file_name),
      cc = get_map_model_cc(map_file_name = map_file_name,
        model_file_name = final_model_file_name,
        resolution = params.resolution))
  elif cycle_model_file_name:
    return group_args(
      group_args_type = 'alphafold result',
      filename = cycle_model_file_name,
      cc = None,)
  else:
    print("No final model or alphafold model obtained")
    return None

def same_file(f1,f2):
  if not f1 or not f2 or not os.path.isfile(f1) or not os.path.isfile(f2):
    return False
  return os.path.samefile(os.path.abspath(f1),os.path.abspath(f2))

def check_and_copy(a,b):
  if (not a) or (not os.path.isfile(a)):
    return # Nothing to do
  if not same_file(a, b):
     shutil.copyfile(a,b)

def change_is_small(params, rmsd_from_previous_cycle_list, n = 2):
  if len(rmsd_from_previous_cycle_list) < n:
    return False
  biggest_recent_rmsd = 0
  for x in rmsd_from_previous_cycle_list[-n:]:
    if x > biggest_recent_rmsd:
      biggest_recent_rmsd = x
  if x <= params.resolution * params.cycle_rmsd_to_resolution_ratio:
    return True
  else:
    return False
def get_pae_file_name(params):
  return params.jobname+"_PAE_cycle_%s.jsn" %(params.cycle)
def get_pae_png_file_name(params):
  return params.jobname+"_PAE_cycle_%s.png" %(params.cycle)
def get_plddt_png_file_name(params):
  return params.jobname+"_plDDT_cycle_%s.png" %(params.cycle)
def get_af_file_name(params):
  return params.jobname+"_ALPHAFOLD_cycle_%s.pdb" %(params.cycle)
