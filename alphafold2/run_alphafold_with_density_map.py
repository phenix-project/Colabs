from __future__ import division
from __future__ import print_function

import os, sys

from alphafold_imports import *
from alphafold_utils import *

from alphafold.data.templates import (_get_pdb_id_and_chain,
                                    _process_single_hit,
                                    _build_query_to_hit_index_mapping,
                                    _extract_template_features,
                                    SingleHitResult,
                                    TEMPLATE_FEATURES)


def run_one_cycle(cycle, template_hit_list,
        query_sequence,
        jobname,
        maps_uploaded,
        maximum_cycles,
        resolution,
        num_models,
        msa, deletion_matrix, template_paths,
        mtm_file_name,
        cif_dir,
        output_directory,
        homooligomer,
        use_msa,
        use_env,
        use_templates):


  os.chdir("/content/")
  if template_hit_list:
    #process hits into template features
    from dataclasses import replace
    template_hit_list = [[replace(hit,**{"index":i+1}),mmcif] for i,[hit,mmcif] in enumerate(template_hit_list)]

    template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
      template_features[template_feature_name] = []

    for i,[hit,mmcif] in enumerate(sorted(template_hit_list, key=lambda xx: xx[0].sum_probs, reverse=True)):
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
      single_hit_result = SingleHitResult(features=features, error=None, warning=None)
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

  # collect model weights
  use_model = {}
  model_params = {}
  model_runner_1 = None
  model_runner_3 = None
  for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
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

  os.chdir("/content/")
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

  citations = {
  "Mirdita2021":  """@article{Mirdita2021,
  author = {Mirdita, Milot and Ovchinnikov, Sergey and Steinegger, Martin},
  doi = {10.1101/2021.08.15.456425},
  journal = {bioRxiv},
  title = {{ColabFold - Making Protein folding accessible to all}},
  year = {2021},
  comment = {ColabFold including MMseqs2 MSA server}
  }""",
    "Mitchell2019": """@article{Mitchell2019,
  author = {Mitchell, Alex L and Almeida, Alexandre and Beracochea, Martin and Boland, Miguel and Burgin, Josephine and Cochrane, Guy and Crusoe, Michael R and Kale, Varsha and Potter, Simon C and Richardson, Lorna J and Sakharova, Ekaterina and Scheremetjew, Maxim and Korobeynikov, Anton and Shlemov, Alex and Kunyavskaya, Olga and Lapidus, Alla and Finn, Robert D},
  doi = {10.1093/nar/gkz1035},
  journal = {Nucleic Acids Res.},
  title = {{MGnify: the microbiome analysis resource in 2020}},
  year = {2019},
  comment = {MGnify database}
  }""",
    "Jumper2021": """@article{Jumper2021,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'{i}}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A. A. and Ballard, Andrew J. and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W. and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  doi = {10.1038/s41586-021-03819-2},
  journal = {Nature},
  pmid = {34265844},
  title = {{Highly accurate protein structure prediction with AlphaFold.}},
  year = {2021},
  comment = {AlphaFold2 + BFD Database}
  }""",
    "Mirdita2019": """@article{Mirdita2019,
  author = {Mirdita, Milot and Steinegger, Martin and S{\"{o}}ding, Johannes},
  doi = {10.1093/bioinformatics/bty1057},
  journal = {Bioinformatics},
  number = {16},
  pages = {2856--2858},
  pmid = {30615063},
  title = {{MMseqs2 desktop and local web server app for fast, interactive sequence searches}},
  volume = {35},
  year = {2019},
  comment = {MMseqs2 search server}
  }""",
    "Steinegger2019": """@article{Steinegger2019,
  author = {Steinegger, Martin and Meier, Markus and Mirdita, Milot and V{\"{o}}hringer, Harald and Haunsberger, Stephan J. and S{\"{o}}ding, Johannes},
  doi = {10.1186/s12859-019-3019-7},
  journal = {BMC Bioinform.},
  number = {1},
  pages = {473},
  pmid = {31521110},
  title = {{HH-suite3 for fast remote homology detection and deep protein annotation}},
  volume = {20},
  year = {2019},
  comment = {PDB70 database}
  }""",
    "Mirdita2017": """@article{Mirdita2017,
  author = {Mirdita, Milot and von den Driesch, Lars and Galiez, Clovis and Martin, Maria J. and S{\"{o}}ding, Johannes and Steinegger, Martin},
  doi = {10.1093/nar/gkw1081},
  journal = {Nucleic Acids Res.},
  number = {D1},
  pages = {D170--D176},
  pmid = {27899574},
  title = {{Uniclust databases of clustered and deeply annotated protein sequences and alignments}},
  volume = {45},
  year = {2017},
  comment = {Uniclust30/UniRef30 database},
  }""",
    "Berman2003": """@misc{Berman2003,
  author = {Berman, Helen and Henrick, Kim and Nakamura, Haruki},
  booktitle = {Nat. Struct. Biol.},
  doi = {10.1038/nsb1203-980},
  number = {12},
  pages = {980},
  pmid = {14634627},
  title = {{Announcing the worldwide Protein Data Bank}},
  volume = {10},
  year = {2003},
  comment = {templates downloaded from wwPDB server}
  }""",
  }

  to_cite = [ "Mirdita2021", "Jumper2021" ]
  if use_msa:       to_cite += ["Mirdita2019"]
  if use_msa:       to_cite += ["Mirdita2017"]
  if use_env:       to_cite += ["Mitchell2019"]
  if use_templates: to_cite += ["Steinegger2019"]
  if use_templates: to_cite += ["Berman2003"]

  with open(f"{jobname}.bibtex", 'w') as writer:
    for i in to_cite:
      writer.write(citations[i])
      writer.write("\n")

  return cycle_model_file_name

def rebuild_model(
        cycle_model_file_name,
        previous_final_model_name,
        mtm_file_name,
        cycle,
        jobname,
        maps_uploaded,
        resolution,
        nproc = 4):
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
  shell("phenix.dock_and_rebuild fragments_model_file=%s nproc=%s resolution=%s previous_model_file=%s model=%s full_map=%s output_model_prefix=%s" %(
     mtm_file_name,nproc,resolution,previous_model,af_model_file,map_file_name,
      rebuilt_model_stem))

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


  shell(
   "phenix.map_to_model nproc=%s seq_file=%s resolution=%s %s pdb_out=%s" %(
     nproc, seq_file, resolution, map_file_name, output_file_name) )
  return output_file_name

def run_job(query_sequence,
        jobname,
        upload_manual_templates,
        manual_templates_uploaded,
        maps_uploaded,
        maximum_cycles,
        resolution,
        maximum_templates_from_pdb,
        use_msa,
        include_templates_from_pdb,
        uploaded_templates_are_map_to_model,
        output_directory,
        skip_all_msa_after_first_cycle):

  from Bio.SeqRecord import SeqRecord
  from Bio.Seq import Seq
  from Bio import SeqIO

  os.chdir("/content/")

  #standard values of parameters
  num_models = 1
  homooligomer = 1
  use_env = True
  use_custom_msa = False
  use_templates = True


  #Get the MSA
  msa, deletion_matrix, template_paths = get_msa(
      query_sequence, jobname, use_env,
      use_templates,
      homooligomer,
      use_msa)

  #Process templates
  print("PROCESSING TEMPLATES")

  other_cif_dir = Path("/content/%s" %(template_paths))
  parent_dir = Path("/content/manual_templates")
  cif_dir = Path(parent_dir,"mmcif")
  fasta_dir = Path(parent_dir,"fasta")
  hhDB_dir = Path(parent_dir,"hhDB")
  msa_dir = Path(hhDB_dir,"msa")
  clear_directories([fasta_dir,hhDB_dir,msa_dir])

  if uploaded_templates_are_map_to_model and \
      manual_templates_uploaded: # mtm
    print("Uploaded tempates are map to model")
    mtm_file_name = manual_templates_uploaded[0]
    manual_templates_uploaded = []
  else:
    print("Uploaded templates are actual templates")
    mtm_file_name = "None"

  pdb_cif_file_list = get_cif_file_list(
    include_templates_from_pdb = include_templates_from_pdb,
    manual_templates_uploaded = None,
    cif_dir = cif_dir,
    other_cif_dir = other_cif_dir)
  print("CIF files from PDB to include:",pdb_cif_file_list)

  manual_cif_file_list = get_cif_file_list(
    include_templates_from_pdb = False,
    manual_templates_uploaded = manual_templates_uploaded,
    cif_dir = cif_dir)
  print("Uploaded CIF files to include:",manual_cif_file_list)

  query_seq = SeqRecord(Seq(query_sequence),id="query",
    name="",description="")
  query_seq_path = Path(fasta_dir,"query.fasta")
  with query_seq_path.open("w") as fh:
      SeqIO.write([query_seq], fh, "fasta")
  shutil.copyfile(query_seq_path,Path(msa_dir,"query.fasta"))

  previous_final_model_name = manual_templates_uploaded[0] if \
      manual_templates_uploaded else None

  assert len(maps_uploaded) == 1  # just one map
  map_file_name = maps_uploaded[0]

  seq_file = "%s.seq" %(jobname)
  ff = open(seq_file,'w')
  print(query_sequence, file = ff)
  ff.close()

  # Run first cycle

  for cycle in range(1, maximum_cycles + 1):
    print("\nStarting cycle %s" %(cycle))

    if cycle == 2 and skip_all_msa_after_first_cycle:
      print("Getting dummy msa for cycles after the first")
      #Get dummy msa
      use_msa = False
      msa, deletion_matrix, template_paths = get_msa(
        query_sequence, jobname, use_env,
        use_templates,
        homooligomer,
        use_msa)


    working_cif_file_list = \
     list(manual_cif_file_list) + \
     list(pdb_cif_file_list)[:maximum_templates_from_pdb]

    print("Templates used in this cycle: %s" %(
        " ".join([w.as_posix() for w in working_cif_file_list])))

    template_hit_list = get_template_hit_list(
      cif_files = working_cif_file_list,
      fasta_dir = fasta_dir,
      query_seq = query_seq,
      hhDB_dir = hhDB_dir)

    os.chdir("/content/")

    cycle_model_file_name = run_one_cycle(
        cycle, template_hit_list,
        query_sequence,
        jobname,
        maps_uploaded,
        maximum_cycles,
        resolution,
        num_models,
        msa, deletion_matrix, template_paths,
        mtm_file_name,
        cif_dir,
        output_directory,
        homooligomer,
        use_msa,
        use_env,
        use_templates)

    cycle_model_file_name = Path(cycle_model_file_name)


    print("\nFinished with cycle %s of AlphaFold model generation" %(cycle))
    if not os.path.isfile(cycle_model_file_name.as_posix()):
      print("No AlphaFold model obtained...quitting")
      return None
    print("Current AlphaFold model is in %s" %(
        cycle_model_file_name.as_posix()))




    print("\nGetting a new rebuilt model at a resolution of %.2f A" %(
        resolution))
    # Now get a new rebuilt model
    final_model_file_name = rebuild_model(
        cycle_model_file_name,
        previous_final_model_name,
        mtm_file_name,
        cycle,
        jobname,
        maps_uploaded,
        resolution)

    try:
      shell(
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
       final_model_file_name_in_cif_dir)
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

def run_alphafold_with_density_map_jobs(params):

  # Set locals from params:
  print("Values of parameters:")
  for key in params.keys():
    locals()[key] = params[key]
    print("Set ",key,":",params[key])


  # RUN THE JOBS HERE

  for query_sequence, jobname, resolution in zip(query_sequences, jobnames, resolutions):
    print("\n","****************************************","\n",
         "RUNNING JOB %s with sequence %s at resolution of %s\n" %(
      jobname, query_sequence, resolution),
      "****************************************","\n")
    # GET TEMPLATES AND SET UP FILES

    # User input of manual templates
    manual_templates_uploaded = cif_filename_dict.get(
      jobname,[])
    if manual_templates_uploaded:
      print("Using uploaded templates %s for this run" %(
          manual_templates_uploaded))
    maps_uploaded = map_filename_dict.get(
      jobname,[])
    if maps_uploaded:
      print("Using uploaded maps %s for this run" %(
          maps_uploaded))
      assert len(maps_uploaded) == 1

    try:
      filename = run_job(query_sequence,
        jobname,
        upload_manual_templates,
        manual_templates_uploaded,
        maps_uploaded,
        maximum_cycles,
        resolution,
        maximum_templates_from_pdb,
        num_models,
        homooligomer,
        use_msa,
        use_env,
        use_custom_msa,
        use_templates,
        include_templates_from_pdb,
        uploaded_templates_are_map_to_model,
        output_directory,
        skip_all_msa_after_first_cycle)
      if filename:
        print("FINISHED JOB (%s) %s with sequence %s\n" %(
        filename, jobname, query_sequence),
        "****************************************","\n")
      else:
        print("NO RESULT FOR JOB %s with sequence %s\n" %(
      jobname, query_sequence),
      "****************************************","\n")

    except Exception as e:
      print("FAILED: JOB %s with sequence %s\n\n%s\n" %(
      jobname, query_sequence, str(e)),
      "****************************************","\n")


  print("\nDOWNLOADING FILES NOW:\n")
  for query_sequence, jobname in zip(query_sequences, jobnames):
    filename = f"{jobname}.result.zip"
    if os.path.isfile(filename):
      print(filename)

  print("\nALL DONE\n")

