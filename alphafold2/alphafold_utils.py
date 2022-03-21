from __future__ import division
from __future__ import print_function

import os, sys
from pathlib import Path
import os.path
import re
import hashlib

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import shutil
from string import ascii_uppercase
from phenix_colab_utils import runsh, exit

#  Make sure we can import phenix_alphafold_utils and phenix_colab_utils
local_path = os.path.split(os.path.abspath(__file__))[0]
if not local_path in sys.path:
  sys.path.append(local_path)

try:
  from alphafold.data import templates
except Exception as e:
  print("Sorry, AlphaFold is not available...")
  raise AssertionError(e)

try:
  import matplotlib.pyplot as plt
except Exception as e:
  plt = None

def set_up_alphafold_logging():
  warnings.filterwarnings('ignore')
  logging.set_verbosity("error")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.get_logger().setLevel('ERROR')


def mk_mock_template(query_sequence):
  import numpy as np
  # since alphafold's model requires a template input
  # we create a blank example w/ zero input, confidence -1
  ln = len(query_sequence)
  output_templates_sequence = "-"*ln
  output_confidence_scores = np.full(ln,-1)
  templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
  templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
  templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                    templates.residue_constants.HHBLITS_AA_TO_ID)
  template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                       'template_all_atom_masks': templates_all_atom_masks[None],
                       'template_sequence': [f'none'.encode()],
                       'template_aatype': np.array(templates_aatype)[None],
                       'template_confidence_scores': output_confidence_scores[None],
                       'template_domain_names': [f'none'.encode()],
                       'template_release_date': [f'none'.encode()]}
  return template_features

def mk_template(a3m_lines, template_paths):
  from alphafold.data import pipeline
  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=template_paths,
      max_template_date="2100-01-01",
      max_hits=20,
      kalign_binary_path="kalign",
      release_dates_path=None,
      obsolete_pdbs_path=None)

  from alphafold.data.tools import hhsearch

  hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[f"{template_paths}/pdb70"])

  hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                       query_pdb_code=None,
                                                       query_release_date=None,
                                                       hits=hhsearch_hits)
  return templates_result.features

def set_bfactor(pdb_filename, bfac, idx_res, chains):
  import numpy as np
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()

def predict_structure(prefix, feature_dict, Ls, model_params,
  use_model,
  model_runner_1,
  model_runner_3,
  do_relax=False,
  random_seed=0,
  msa_is_msa_object = None,
  random_seed_iterations = 5,
  big_improvement = 5,
  confidence_dict = None,):

  """Predicts structure using AlphaFold for the given sequence."""
  import numpy as np

  if confidence_dict is None: # if we have looked at 5 tries only skip
    #   if prob of success is very very low, moderate for 10 tries
    confidence_dict = {
        5: 0.0001,
       10: 0.01,
       20: 0.02,}

  # Minkyung's code
  # add big enough number to residue index to indicate chain breaks
  idx_res = feature_dict['residue_index']
  L_prev = 0
  # Ls: number of residues in each chain
  for L_i in Ls[:-1]:
      idx_res[L_prev+L_i:] += 200
      L_prev += L_i
  chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
  feature_dict['residue_index'] = idx_res

  # Run the models.
  plddts,paes = [],[]
  unrelaxed_pdb_lines = []
  relaxed_pdb_lines = []
  from alphafold.common import protein
  for model_name, local_params in model_params.items():
    if model_name in use_model:
      print(f"running {model_name}")
      # swap params to avoid recompiling
      # note: models 1,2 have diff number of params compared to models 3,4,5
      if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
      if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
      model_runner.params = local_params

      best_value = None
      for i in range(max(1,random_seed_iterations)):
        import random
        random.seed(random_seed)
        random_seed = random.randint(0,1000000)
        processed_feature_dict = model_runner.process_features(feature_dict,
           random_seed=random_seed)
        print("Running prediction ",i+1,"of",random_seed_iterations,"...")
        try:
          if msa_is_msa_object:
            prediction_result = model_runner.predict(processed_feature_dict,
              random_seed = random_seed)
          else:
            prediction_result = model_runner.predict(processed_feature_dict)
        except Exception as e:
          print("Prediction failed...\n%s\n...skipping" %(str(e)))
          continue

        unrelaxed_protein = protein.from_prediction(
           processed_feature_dict,prediction_result)
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        plddts.append(prediction_result['plddt'])
        paes.append(prediction_result['predicted_aligned_error'])

        # Try to estimate what plddt we could achieve and whether it is
        #  worth getting more tries
        lddt_rank = np.mean(plddts,-1).argsort()[::-1]
        try:
          from scitbx.array_family import flex
          values = flex.double()
        except Exception as e:
          values = flex_double()  # something that we can use
        for n,r in enumerate(lddt_rank):
          value = np.mean(plddts[r])
          values.append(value)
        mmm = values.min_max_mean()
        if values.size() >= 5:
          sd = values.standard_deviation_of_the_sample()
          if values.size() < 10:
            sd = max(sd, mmm.max - mmm.mean) # use lower bound for small samples
        else:
          sd = None
        if (best_value is None) or mmm.max > best_value:
          best_value = mmm.max
          nn = list(lddt_rank)[0]
          unrelaxed_pdb_path = f'{prefix}_current_best_{nn+1}.pdb'
          with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdb_lines[nn])
          print("New maximum plDDT (try %s): %.2f, saved as %s" %(
          nn + 1, best_value, unrelaxed_pdb_path))

        if sd is not None: # estimate SD and see if we want to keep going
           # Let's say big_improvement = 5 is worth getting in
           #   random_seed_iterations tries.  If we have mean = a and current
           # best of best_value -> z = (best_value - mean)/sd
           # We want a pick that gives a Z-score of z + BI/sd. About how many
           # tries will it take? p(Z) ~ exp - Z**2/2
          good_z = (best_value + big_improvement - mmm.mean)/sd
          import math
          p_good_z = 0.5 * math.exp(-0.5*min(good_z,20.)**2)
          n_remaining = max(0,random_seed_iterations - i)
          p_get_good_z_in_n = 1 - (1-p_good_z)**n_remaining
          keys = list(confidence_dict.keys())
          keys.sort()
          confidence_level = 0
          for key in keys:
            if n_remaining > key:
              confidence_level = confidence_dict[key]
        else:
          p_get_good_z_in_n = None
        if p_get_good_z_in_n is not None and \
           p_get_good_z_in_n < confidence_level:
          # forget it
          print("Ending randomization as it is unlikely we will improve by\n",
          "%.2f more than current best value of %.2f (mean = %.2f, sd= %.2f)" %(
           big_improvement,best_value,mmm.mean,sd))
          break




  # rerank models based on predicted lddt
  lddt_rank = np.mean(plddts,-1).argsort()[::-1]
  out = {}
  print("reranking models based on avg. predicted lDDT")
  for n,r in enumerate(lddt_rank):
    print(f"model_{n+1} {r} {np.mean(plddts[r])}")

    unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n+1}.pdb'
    with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
    set_bfactor(unrelaxed_pdb_path, plddts[r], idx_res, chains)

    if do_relax:
      relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
      with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
      set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)

    out[f"model_{n+1}"] = {"plddt":plddts[r], "pae":paes[r]}
  return out



def hh_process_seq(
  query_seq = None,
  template_seq = None,
  content_dir = None,
  hhDB_dir = None,
  db_prefix="DB"):
  """
  This is a hack to get hhsuite output strings to pass on
  to the AlphaFold template featurizer.

  Note: that in the case of multiple templates, this would be faster to build one database for
  all the templates. Currently it builds a database with only one template at a time. Even
  better would be to get an hhsuite alignment without using a database at all, just between
  pairs of sequence files. However, I have not figured out how to do this.

  Update: I think the hhsearch can be replaced completely, and we can just do a pairwise
  alignment with biopython, or skip alignment if the seqs match. TODO
  """

  from alphafold.data import pipeline
  from Bio import SeqIO
  from alphafold.data.tools import hhsearch

  # set up directory for hhsuite DB.
  #  Place one template fasta file to be the DB contents
  if not hasattr(hhDB_dir,'exists'):
    hhDB_dir = Path(hhDB_dir)
  if hhDB_dir.exists() and \
        not hhDB_dir.as_posix()  in ["/content", "/content/"]:
    shutil.rmtree(hhDB_dir)

  msa_dir = Path(hhDB_dir,"msa")
  if msa_dir.exists() and \
        not msa_dir.as_posix()  in ["/content", "/content/"]:
    shutil.rmtree(msa_dir)

  msa_dir.mkdir(parents=True)
  assert os.path.isdir(hhDB_dir)
  msa_dir = Path(os.path.abspath(msa_dir))
  hhDB_dir = Path(os.path.abspath(hhDB_dir))
  content_dir = Path(os.path.abspath(content_dir))

  template_seq_path = Path(msa_dir,"template.fasta")
  with template_seq_path.open("w") as fh:
    SeqIO.write([template_seq], fh, "fasta")
  print("MSA DIR",msa_dir)
  # make hhsuite DB
  with redirect_stdout(StringIO()) as out:
    os.chdir(msa_dir)
    import subprocess
    runsh("ffindex_build -s ../DB_msa.ffdata ../DB_msa.ffindex .")
    os.chdir(hhDB_dir)
    runsh(" ffindex_apply DB_msa.ffdata DB_msa.ffindex  -i DB_a3m.ffindex -d DB_a3m.ffdata  -- hhconsensus -M 50 -maxres 65535 -i stdin -oa3m stdout -v 0")
    runsh(" rm DB_msa.ffdata DB_msa.ffindex")
    runsh(" ffindex_apply DB_a3m.ffdata DB_a3m.ffindex -i DB_hhm.ffindex -d DB_hhm.ffdata -- hhmake -i stdin -o stdout -v 0")
    # This one needs subprocess.call:
    subprocess.call(['cstranslate','-f','-x','0.3','-c','4','-I','a3m','-i','DB_a3m','-o','DB_cs219'])
    runsh(" sort -k3 -n -r DB_cs219.ffindex | cut -f1 > sorting.dat")
    runsh(" ffindex_order sorting.dat DB_hhm.ffdata DB_hhm.ffindex DB_hhm_ordered.ffdata DB_hhm_ordered.ffindex")
    runsh(" mv DB_hhm_ordered.ffindex DB_hhm.ffindex")
    runsh(" mv DB_hhm_ordered.ffdata DB_hhm.ffdata")
    runsh(" ffindex_order sorting.dat DB_a3m.ffdata DB_a3m.ffindex DB_a3m_ordered.ffdata DB_a3m_ordered.ffindex")
    runsh(" mv DB_a3m_ordered.ffindex DB_a3m.ffindex")
    runsh(" mv DB_a3m_ordered.ffdata DB_a3m.ffdata")
    os.chdir(content_dir)

  # run hhsearch
  db_dir = hhDB_dir.as_posix()+"/"+db_prefix
  if not os.path.isdir(db_dir):
    os.mkdir (db_dir)

  hhsearch_runner = hhsearch.HHSearch(binary_path="hhsearch",
      databases=[hhDB_dir.as_posix()+"/"+db_prefix])
  with StringIO() as fh:
    SeqIO.write([query_seq], fh, "fasta")
    seq_fasta = fh.getvalue()
  hhsearch_result = hhsearch_runner.query(seq_fasta)

  # process hits
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  if len(hhsearch_hits) >0:
    from dataclasses import replace
    hit = hhsearch_hits[0]
    hit = replace(hit,**{"name":template_seq.id})
  else:
    hit = None
  return hit

def plot_plddt_legend():
  if not plt:
    return # no plotting
  thresh = ['plDDT:','Very low (<50)','Low (60)','OK (70)','Confident (80)','Very high (>90)']
  plt.figure(figsize=(1,0.1),dpi=100)
  ########################################
  for c in ["#FFFFFF","#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF"]:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5,)
  plt.axis(False)
  return plt

def plot_confidence(homooligomer,query_sequence, outs, model_num=1):
  if not plt:
    return # No plotting
  model_name = f"model_{model_num}"
  plt.figure(figsize=(10,3),dpi=100)
  """Plots the legend for plDDT."""
  #########################################
  plt.subplot(1,2,1); plt.title('Predicted lDDT')
  plt.plot(outs[model_name]["plddt"])
  for n in range(homooligomer+1):
    x = n*(len(query_sequence))
    plt.plot([x,x],[0,100],color="black")
  plt.ylabel('plDDT')
  plt.xlabel('position')
  #########################################
  plt.subplot(1,2,2);plt.title('Predicted Aligned Error')
  plt.imshow(outs[model_name]["pae"], cmap="bwr",vmin=0,vmax=30)
  plt.colorbar()
  plt.xlabel('Scored residue')
  plt.ylabel('Aligned residue')
  #########################################
  return plt

def show_pdb(jobname, model_num=1, show_sidechains=False, show_mainchains=False, color="lDDT"):
  import py3Dmol
  model_name = f"model_{model_num}"

  pdb_filename = f"{jobname}_unrelaxed_{model_name}.pdb"
  view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
  view.addModel(open(pdb_filename,'r').read(),'pdb')

  if color == "lDDT":
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
  elif color == "rainbow":
    view.setStyle({'cartoon': {'color':'spectrum'}})
  elif color == "chain":
    for n,chain,color in zip(range(homooligomer),list("ABCDEFGH"),
                     ["lime","cyan","magenta","yellow","salmon","white","blue","orange"]):
       view.setStyle({'chain':chain},{'cartoon': {'color':color}})
  if show_sidechains:
    BB = ['C','O','N']
    view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
    view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
  if show_mainchains:
    BB = ['C','O','N','CA']
    view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

  view.zoomTo()
  return view

def write_pae_file(pae_matrix, file_name):
  shape=tuple(pae_matrix.shape)
  n,n = shape
  # Write out array to text file as json
  residues_1 = []
  residues_2 = []
  distances = []
  for i in range(n):
    ii = i + 1
    for j in range(n):
      jj= j + 1
      residues_1.append(ii)
      residues_2.append(jj)
      distances.append(float("%.2f" %(pae_matrix[i][j])))

  residue_dict = {"residue1":residues_1,
                   "residue2":residues_2,
                  "distance":distances,
                  "max_predicted_aligned_error":0}
  values = [residue_dict]
  text = str(values).replace(" ","").replace("'",'"')

  f = open(file_name, 'w')
  print(text, file = f)
  f.close()
  print("Wrote pae file to %s" %(file_name))


def get_msa(params):
  import colabfold as cf
  from alphafold.data import pipeline
  template_paths = None # initialize

  # Do we already have an msa file:
  if params.upload_msa_file and params.use_msa:
    assert len(params.msas_uploaded) == 1  # just one msa
    msa_file_name = params.msas_uploaded[0]
  else:
    msa_file_name = None

  #@title Get MSA and templates

  if msa_file_name:
    template_paths = None
    print("Reading MSA from %s" %(msa_file_name))
    a3m_lines = open(msa_file_name).read()
  else:
    a3m_lines = None


  if params.include_templates_from_pdb:
    if not hasattr(params, 'template_search_method') or \
        params.template_search_method == 'mmseqs2':
      print("Getting templates from PDB using mmseqs2 server...")
      new_a3m_lines, template_paths = cf.run_mmseqs2(params.query_sequence,
        params.jobname, params.use_env, use_templates=True)
      if not a3m_lines:
        a3m_lines = new_a3m_lines
        print("Using MSA from mmseqs2")
    else:
      print("Getting templates using structure_search")
      template_paths = get_templates_with_structure_search(params)
    print("Templates are in %s" %(template_paths))

  elif params.use_msa and not a3m_lines:
    print("Getting MSA from mmseqs2 server")
    a3m_lines = cf.run_mmseqs2(params.query_sequence,
       params.jobname, params.use_env)

  if (not params.use_msa) or not a3m_lines:
    a3m_lines = ">query sequence \n%s" %(params.query_sequence)
    print("Not using any MSA information")

  # File for a3m
  a3m_file = f"{params.jobname}.a3m"

  with open(a3m_file, "w") as text_file:
      text_file.write(a3m_lines)

  # parse MSA; allow both versions of return from parse_a3m
  msa = pipeline.parsers.parse_a3m(a3m_lines)
  if type(msa) in [ type([1,2,3]), type((1,2,3))]:
    msa, deletion_matrix = msa
    msa_is_msa_object = False
  else:
    deletion_matrix = msa.deletion_matrix
    msa_is_msa_object = True

  print("Done with MSA and templates")
  return msa, deletion_matrix, template_paths, msa_is_msa_object

def get_templates_with_structure_search(params):
  # Run structure_search
  from cctbx.development.create_models_or_maps import generate_model
  m = generate_model()
  build = m.as_map_model_manager().model_building()
  nproc = params.nproc if hasattr(params, 'nproc') else 4
  build.set_defaults(nproc = nproc)
  model_info = build.structure_search(
    number_of_models_per_input_model = params.maximum_templates_from_pdb,
    sequence_list = [params.query_sequence],
    )
  if model_info and model_info.model_list:
    # convert to cif and write them into our directory
    template_paths = params.template_paths if params.template_paths else \
       "TEMPLATES_FROM_PDB"
    other_cif_dir = Path(os.path.join(params.working_directory,
        template_paths))
    if not os.path.isdir(other_cif_dir):
      other_cif_dir.mkdir(parents=True)
    i = 0
    from phenix_colab_utils import run_pdb_to_cif
    for m in model_info.model_list:
      i += 1
      pdb_id = "%s_%s" %(m.info().pdb_id,m.info().chain_id) if \
          m.info() and m.info().get('pdb_id') and m.info().get('chain_id')\
          else 'model_%s' %(i)
      file_name = os.path.join(other_cif_dir,"%s.pdb" %(pdb_id))
      f = open(file_name, 'w')
      print(m.model_as_pdb(), file = f)
      f.close()
      cif_filename = run_pdb_to_cif(file_name,
           content_dir = params.content_dir)
      os.remove(file_name)
    return other_cif_dir


def get_cif_file_list(
    include_templates_from_pdb = None,
    manual_templates_uploaded = None,
    cif_dir = None,
    other_cif_dir = None):

  if cif_dir is not None:
    cif_files = list(cif_dir.glob("*"))
  else:
    cif_files = []
  # Only include the cif_files in manual_templates_uploaded
  manual_files_as_text = []
  if not manual_templates_uploaded:
    manual_templates_uploaded = []
  for f in manual_templates_uploaded:
    manual_files_as_text.append(
        os.path.split(str(f))[-1])
  cif_files_to_include = []
  for cif_file in cif_files:
    text = os.path.split(str(cif_file))[-1]
    if text in manual_files_as_text:
      cif_files_to_include.append(cif_file)
  cif_files = cif_files_to_include

  if include_templates_from_pdb and other_cif_dir is not None:
    other_cif_files = []
    for file_name in list(other_cif_dir.glob("*")):
      if str(file_name).endswith(".cif"):
        other_cif_files.append(file_name)
    cif_files += other_cif_files

  return cif_files

def get_template_hit_list(
    cif_files = None,
    fasta_dir = None,
    query_seq = None,
    hhDB_dir = None,
    content_dir = None):
  assert content_dir is not None
  from alphafold.data import mmcif_parsing
  from Bio.SeqRecord import SeqRecord
  from Bio.Seq import Seq
  from Bio import SeqIO
  template_hit_list = []
  for i,filepath in enumerate(cif_files):
    if not str(filepath).endswith(".cif"): continue
    print("CIF file included:",i+1,str(filepath))
    with filepath.open("r") as fh:
      filestr = fh.read()
      mmcif_obj = mmcif_parsing.parse(file_id=filepath.stem,mmcif_string=filestr)
      mmcif = mmcif_obj.mmcif_object
      if not mmcif:
        print("...No CIF object obtained...skipping...")
        continue

      for chain_id,template_sequence in mmcif.chain_to_seqres.items():
        template_sequence = mmcif.chain_to_seqres[chain_id]
        seq_name = filepath.stem.upper()+"_"+chain_id
        seq = SeqRecord(Seq(template_sequence),id=seq_name,name="",description="")

        with  Path(fasta_dir,seq.id+".fasta").open("w") as fh:
          SeqIO.write([seq], fh, "fasta")

        """
        At this stage, we have a template sequence.
        and a query sequence.
        There are two options to generate template features:
          1. Write new code to manually generate template features
          2. Get an hhr alignment string, and pass that
            to the existing template featurizer.

        I chose the second, implemented in hh_process_seq()
        """
        SeqIO.write([seq], sys.stdout, "fasta")
        SeqIO.write([query_seq], sys.stdout, "fasta")
        try:
          hit = hh_process_seq(
           query_seq = query_seq,
           template_seq = seq,
           hhDB_dir = hhDB_dir,
           content_dir = content_dir)

        except Exception as e:
          print("Failed to process %s" %(filepath),e)
          hit = None
        if hit is not None:
          template_hit_list.append([hit,mmcif])
          print("Template %s included" %(filepath))
        else:
          print("Template %s not included (failed to process)" %(filepath))

  return template_hit_list

class flex_double:
  #  just a holder that allows min_max_mean and standard_deviation_of_the_sample
  def __init__(self):
    self.values = []

  def append(self,x):
    self.values.append(x)

  def min_max_mean(self):
    from phenix_alphafold_utils import group_args
    min_value = None
    max_value = None
    mean_value = 0.
    for x in self.values:
      if min_value is None or x < min_value: min_value = x
      if max_value is None or x > max_value: max_value = x
      mean_value += x
    if len(self.values) > 0:
      mean_value = mean_value/len(self.values)
    return group_args(
      min = min_value,
      max = max_value,
      mean = mean_value)

  def standard_deviation_of_the_sample(self):
    sum = 0
    sum2 = 0
    sumn = 0
    for x in self.values:
      sum += x
      sum2 += x*x
      sumn += 1
    if sumn < 2:
      return None
    else:
      sum = sum/sumn
      sum2 = sum2/sumn
      return max(0.,sum2 - sum*sum)**0.5
  def size(self):
    return len(self.values)
