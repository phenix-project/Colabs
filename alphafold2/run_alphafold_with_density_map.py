from __future__ import division
from __future__ import print_function

import os, sys
import shutil

from pathlib import Path
import numpy as np

from phenix_alphafold_utils import clear_directories
from phenix_colab_utils import runsh

from alphafold_utils import (mk_mock_template,
   predict_structure,
   show_pdb,
   get_msa,
   get_cif_file_list,
   get_template_hit_list)




def run_one_cycle(params):

  # Add all params values to locals()
  print("ZZA",params)
  params_dict = params()
  for key in params_dict.keys():
    locals()[key] = params_dict[key]
    print("ZZ paramsA",key,params_dict[key])

  from alphafold.data.templates import (_get_pdb_id_and_chain,
                                    _process_single_hit,
                                    _build_query_to_hit_index_mapping,
                                    _extract_template_features,
                                    SingleHitResult,
                                    TEMPLATE_FEATURES)

  os.chdir(content_dir)
  if template_hit_list:
    #process hits into template features
    from dataclasses import replace
    template_hit_list = [[replace(hit,**{"index":i+1}),mmcif]
        for i,[hit,mmcif] in enumerate(template_hit_list)]

    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    for i,[hit,mmcif] in enumerate(sorted(template_hit_list,
          key=lambda xx: xx[0].sum_probs, reverse=True)):
      # modifications to alphafold/data/templates.py _process_single_hit
      hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)
      mapping = _build_query_to_hit_index_mapping(
      hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,
      query_sequence)
      template_sequence = hit.hit_sequence.replace('-', '')

      try:
        features, realign_warning = _extract_template_features(
          mmcif_object=mmcif,
          pdb_id=hit_pdb_code,
          mapping=mapping,
          template_sequence=template_sequence,
          query_sequence=query_sequence,
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
    #overwrite template data
    template_paths = cif_dir.as_posix()


    # Select only one chain from any cif file
    unique_template_hits = []
    pdb_text_list = []
    for hit, mmcif in template_hit_list:
      pdb_text = hit.name.split()[0].split("_")[0]
      if not pdb_text in pdb_text_list:
        pdb_text_list.append(pdb_text)
        unique_template_hits.append(hit)
    template_hits = unique_template_hits

    print("\nIncluding templates:")
    for hit,mmcif in template_hit_list:
      print("\t",hit.name.split()[0])
    if len(template_hit_list) == 0:
      print("No templates found...quitting")
      raise AssertionError("No templates found...quitting")


    for key,value in template_features.items():
      if np.all(value==0):
        print("ERROR: Some template features are empty")
  else:  # no templates
    print("Not using any templates")
    template_features = mk_mock_template(query_sequence * homooligomer)

  print("\nPREDICTING STRUCTURE")

  from alphafold.model import data
  from alphafold.model import config
  from alphafold.model import model

  # collect model weights
  use_model = {}
  model_params = {}
  model_runner_1 = None
  model_runner_3 = None
  for model_name in ["model_1","model_2","model_3",
      "model_4","model_5"][:num_models]:
    use_model[model_name] = True
    if model_name not in list(model_params.keys()):
      model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=".")
      if model_name == "model_1":
        model_config = config.model_config(model_name+"_ptm")
        model_config.data.eval.num_ensemble = 1
        model_runner_1 = model.RunModel(model_config, model_params[model_name])
      if model_name == "model_3":
        model_config = config.model_config(model_name+"_ptm")
        model_config.data.eval.num_ensemble = 1
        model_runner_3 = model.RunModel(model_config, model_params[model_name])
  if homooligomer == 1:
    msas = [msa]
    deletion_matrices = [deletion_matrix]
  else:
    # make multiple copies of msa for each copy
    # AAA------
    # ---AAA---
    # ------AAA
    #
    # note: if you concat the sequences (as below), it does NOT work
    # AAAAAAAAA
    msas = []
    deletion_matrices = []
    Ln = len(query_sequence)
    for o in range(homooligomer):
      L = Ln * o
      R = Ln * (homooligomer-(o+1))
      msas.append(["-"*L+seq+"-"*R for seq in msa])
      deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])

  # gather features
  from alphafold.data import pipeline
  feature_dict = {
    **pipeline.make_sequence_features(sequence=query_sequence*homooligomer,
                                      description="none",
                                      num_res=len(query_sequence)*homooligomer),
    **pipeline.make_msa_features(msas=msas,deletion_matrices=deletion_matrices),
    **template_features
  }
  outs = predict_structure(jobname, feature_dict,
                           Ls=[len(query_sequence)]*homooligomer,
                           model_params=model_params, use_model=use_model,
                           model_runner_1=model_runner_1,
                           model_runner_3=model_runner_3,
                           do_relax=False)
  print("DONE WITH STRUCTURE in",os.getcwd())

  os.chdir(content_dir)
  print(os.listdir("."))
  model_file_name = "%s_unrelaxed_model_1.pdb" %(jobname)
  if os.path.isfile(model_file_name):
    print("Model file is in %s" %(model_file_name))
    cycle_model_file_name = "%s_unrelaxed_model_1_%s.pdb" %(jobname, cycle)
    shutil.copyfile(model_file_name,cycle_model_file_name)
    if output_directory is not None:
      shutil.copyfile(model_file_name,os.path.join(output_directory, cycle_model_file_name))
  else:
    print("No model file %s found for job %s" %(model_file_name, jobname))
    cycle_model_file_name = None

  #@title Making plots...
  import matplotlib.pyplot as plt
  from alphafold_utils import plot_plddt_legend, plot_confidence, write_pae_file

  # gather MSA info
  deduped_full_msa = list(dict.fromkeys(msa))
  msa_arr = np.array([list(seq) for seq in deduped_full_msa])
  seqid = (np.array(list(query_sequence)) == msa_arr).mean(-1)
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
  if homooligomer > 0:
    for n in range(homooligomer+1):
      x = n*(len(query_sequence)-1)
      plt.plot([x,x],[0,100],color="black")
  plt.legend()
  plt.ylim(0,100)
  plt.ylabel("Predicted lDDT")
  plt.xlabel("Positions")
  plt.savefig(jobname+"_coverage_lDDT.png")
  ##################################################################
  plt.show()

  print("Predicted Alignment Error")
  ##################################################################
  pae_file_list = []
  plt.figure(figsize=(3*num_models,2), dpi=100)
  for n,(model_name,value) in enumerate(outs.items()):
    plt.subplot(1,num_models,n+1)
    plt.title(model_name)
    plt.imshow(value["pae"],label=model_name,cmap="bwr",vmin=0,vmax=30)
    plt.colorbar()
    # Write pae file
    pae_file = jobname+"_"+model_name+"_PAE.jsn"
    write_pae_file(value["pae"], pae_file)
    pae_file_list.append(pae_file)
  plt.savefig(jobname+"_PAE.png")
  plt.show()
  ##################################################################
  #@title Displaying 3D structure... {run: "auto"}
  model_num = 1
  color = "lDDT"
  show_sidechains = False
  show_mainchains = False



  show_pdb(jobname, model_num,show_sidechains, show_mainchains, color).show()
  if color == "lDDT": plot_plddt_legend().show()
  plot_confidence(homooligomer, query_sequence, outs, model_num).show()
  #@title Packaging and downloading results...

  #@markdown When modeling is complete .zip files with results will be downloaded automatically.

  return cycle_model_file_name

def rebuild_model(params,
        nproc = 4):

  params_dict = params()
  for key in params_dict.keys():
    locals()[key] = params_dict[key]

  assert len(maps_uploaded) == 1  # just one map
  map_file_name = maps_uploaded[0]
  af_model_file = os.path.abspath(
      cycle_model_file_name.as_posix())
  if previous_final_model_name:
    previous_model_file = os.path.abspath(
      previous_final_model_name.as_posix())
  else:
    previous_model_file = None
  output_file_name = os.path.abspath(
      "%s_rebuilt_%s.pdb" %(jobname,cycle))
  print("Rebuilding %s %s with map in %s at resolution of %.2f" %(
       af_model_file,
        " with previous model of %s" %(previous_model_file) \
        if previous_model_file else "",
       map_file_name,
      resolution,))

  for ff in (map_file_name,af_model_file, previous_model_file):
    if ff and not os.path.isfile(ff):
      print("\nMissing the file: %s" %(ff))
      return None

  rebuilt_model_name = af_model_file.replace(".pdb","_rebuilt.pdb")
  rebuilt_model_stem = rebuilt_model_name.replace(".pdb","")

  # run phenix dock_and_rebuild here
  from phenix_colab_utils import run_command # run and get output to terminal
  text = "phenix.dock_and_rebuild fragments_model_file=%s nproc=%s resolution=%s previous_model_file=%s model=%s full_map=%s output_model_prefix=%s " %(
     mtm_file_name,nproc,resolution,previous_model_file,
     af_model_file,map_file_name,
      rebuilt_model_stem,
      )
  result = run_command(text)

  if os.path.isfile(rebuilt_model_name):
    print("Rebuilding successful")
    return rebuilt_model_name
  else:
    print("Rebuilding not successful")
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

def run_job(params = None):

  from Bio.SeqRecord import SeqRecord
  from Bio.Seq import Seq
  from Bio import SeqIO

  if not params.content_dir:
    params.content_dir = "/content/"

  os.chdir(params.content_dir)

  #standard values of parameters
  params.num_models = 1
  params.homooligomer = 1
  params.use_env = True
  params.use_custom_msa = False
  params.use_templates = True


  #Get the MSA
  params.msa, params.deletion_matrix, params.template_paths = get_msa(params)

  #Process templates
  print("PROCESSING TEMPLATES")

  other_cif_dir = Path(os.path.join(params.content_dir,params.template_paths))
  parent_dir = Path(os.path.join(params.content_dir,"manual_templates"))
  cif_dir = Path(parent_dir,"mmcif")
  fasta_dir = Path(parent_dir,"fasta")
  hhDB_dir = Path(parent_dir,"hhDB")
  msa_dir = Path(hhDB_dir,"msa")
  clear_directories([fasta_dir,hhDB_dir,msa_dir])
  params.cif_dir = cif_dir

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

  previous_final_model_name = params.manual_templates_uploaded[0] if \
      params.manual_templates_uploaded else None

  assert len(params.maps_uploaded) == 1  # just one map
  map_file_name = params.maps_uploaded[0]

  seq_file = "%s.seq" %(params.jobname)
  ff = open(seq_file,'w')
  print(params.query_sequence, file = ff)
  ff.close()

  # Run cycles

  for cycle in range(1, params.maximum_cycles + 1):

    params.cycle = cycle
    print("\nStarting cycle %s" %(cycle))
    if params.cycle == 2 and params.skip_all_msa_after_first_cycle:
      print("Getting dummy msa for cycles after the first")
      #Get dummy msa
      params.use_msa = False
      params.msa, params.deletion_matrix, params.template_paths = get_msa(
        params)

    working_cif_file_list = \
     list(manual_cif_file_list) + \
     list(pdb_cif_file_list)[:params.maximum_templates_from_pdb]

    print("Templates used in this cycle: %s" %(
        " ".join([w.as_posix() for w in working_cif_file_list])))

    params.template_hit_list = get_template_hit_list(
      cif_files = working_cif_file_list,
      fasta_dir = fasta_dir,
      query_seq = query_seq,
      hhDB_dir = hhDB_dir,
      content_dir = params.content_dir)

    os.chdir(params.content_dir)

    print("ZZB paramsbefore",params)
    cycle_model_file_name = run_one_cycle(params)

    if (not cycle_model_file_name) or (
         not os.path.isfile(cycle_model_file_name)):
      print("Modeling failed cycle %s" %(cycle))
      return None

    print("\nFinished with cycle %s of AlphaFold model generation" %(cycle))
    cycle_model_file_name = Path(cycle_model_file_name)
    print("Current AlphaFold model is in %s" %(
        cycle_model_file_name.as_posix()))


    print("\nGetting a new rebuilt model at a resolution of %.2f A" %(
        resolution))
    # Now get a new rebuilt model
    params.cycle_model_file_name = cycle_model_file_name
    params.previous_final_model_name = previous_final_model_name

    final_model_file_name = rebuild_model(params)

    jobname = params.jobname
    try:
      runsh(
      "zip -FSr %s'.result.zip'  %s*.pdb %s*.j* %s*.png %s*.bibtex %s*.jsn" %(
         jobname,jobname,jobname,jobname,jobname,jobname))
      zip_file_name = f"{jobname}.result.zip"
    except Exception as e:
      zip_file_name = None

    if not final_model_file_name:
      print("\nEnding cycles as no rebuilt model obtained")
      break

    # now update template_hit_list
    final_model_file_name = Path(final_model_file_name)

    final_model_file_name_in_cif_dir = Path(
        os.path.join(cif_dir,final_model_file_name.name))
    shutil.copyfile(
      final_model_file_name,
      final_model_file_name_in_cif_dir)

    if output_directory is not None:
      final_model_file_name_in_output_dir = Path(
        os.path.join(output_directory,final_model_file_name.name))
      shutil.copyfile(
        final_model_file_name,
        final_model_file_name_in_output_dir)
      print("Copied rebuilt model to %s" %(
          final_model_file_name_in_output_dir))
    from phenix_colab_utils import run_pdb_to_cif
    final_model_file_name_as_cif_in_cif_dir = run_pdb_to_cif(
       final_model_file_name_in_cif_dir, content_dir = content_dir)
    manual_cif_file_list = get_cif_file_list(
      include_templates_from_pdb = False,
      manual_templates_uploaded = [final_model_file_name_as_cif_in_cif_dir.name],
      cif_dir = cif_dir)
    previous_final_model_name = final_model_file_name

  filename = zip_file_name
  if filename and os.path.isfile(filename):
    print("About to download %s" %(filename))

    try:
      print("Downloading zip file %s" %(filename))
      files.download(filename)
      print("Start of download successful (NOTE: if the download symbol does not go away it did not work. Download it manually using the folder icon to the left)")
      return filename
    except Exception as e:
      print("Unable to download zip file %s" %(filename))
      return None
  else:
    print("No .zip file %s created" %(filename))
    return None
