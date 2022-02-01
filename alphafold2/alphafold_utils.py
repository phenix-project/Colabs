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
from phenix_colab_utils import shell

def set_up_alphafold_logging():
  warnings.filterwarnings('ignore')
  logging.set_verbosity("error")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tf.get_logger().setLevel('ERROR')


def mk_mock_template(query_sequence):
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
  do_relax=False, random_seed=0):
  """Predicts structure using AlphaFold for the given sequence."""

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

  for model_name, params in model_params.items():
    if model_name in use_model:
      print(f"running {model_name}")
      # swap params to avoid recompiling
      # note: models 1,2 have diff number of params compared to models 3,4,5
      if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
      if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
      model_runner.params = params

      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
      prediction_result = model_runner.predict(processed_feature_dict)
      unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
      unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
      plddts.append(prediction_result['plddt'])
      paes.append(prediction_result['predicted_aligned_error'])



  # rerank models based on predicted lddt
  lddt_rank = np.mean(plddts,-1).argsort()[::-1]
  out = {}
  print("reranking models based on avg. predicted lDDT")
  for n,r in enumerate(lddt_rank):
    print(f"model_{n+1} {np.mean(plddts[r])}")

    unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n+1}.pdb'
    with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
    set_bfactor(unrelaxed_pdb_path, plddts[r], idx_res, chains)

    if do_relax:
      relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
      with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
      set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)

    out[f"model_{n+1}"] = {"plddt":plddts[r], "pae":paes[r]}
  return out



def hh_process_seq(query_seq,template_seq,hhDB_dir,db_prefix="DB"):
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

  # set up directory for hhsuite DB. Place one template fasta file to be the DB contents
  if hhDB_dir.exists():
    shutil.rmtree(hhDB_dir)

  msa_dir = Path(hhDB_dir,"msa")
  msa_dir.mkdir(parents=True)
  template_seq_path = Path(msa_dir,"template.fasta")
  with template_seq_path.open("w") as fh:
    SeqIO.write([template_seq], fh, "fasta")
  print("MSA DIR",msa_dir)
  # make hhsuite DB
  with redirect_stdout(StringIO()) as out:
    os.chdir(msa_dir)
    shell(" ffindex_build -s ../DB_msa.ff{data,index} .")
    os.chdir(hhDB_dir)
    shell(" ffindex_apply DB_msa.ff{data,index}  -i DB_a3m.ffindex -d DB_a3m.ffdata  -- hhconsensus -M 50 -maxres 65535 -i stdin -oa3m stdout -v 0")
    shell(" rm DB_msa.ff{data,index}")
    shell(" ffindex_apply DB_a3m.ff{data,index} -i DB_hhm.ffindex -d DB_hhm.ffdata -- hhmake -i stdin -o stdout -v 0")
    shell(" cstranslate -f -x 0.3 -c 4 -I a3m -i DB_a3m -o DB_cs219 ")
    shell(" sort -k3 -n -r DB_cs219.ffindex | cut -f1 > sorting.dat")
    shell(" ffindex_order sorting.dat DB_hhm.ff{data,index} DB_hhm_ordered.ff{data,index}")
    shell(" mv DB_hhm_ordered.ffindex DB_hhm.ffindex")
    shell(" mv DB_hhm_ordered.ffdata DB_hhm.ffdata")
    shell(" ffindex_order sorting.dat DB_a3m.ff{data,index} DB_a3m_ordered.ff{data,index}")
    shell(" mv DB_a3m_ordered.ffindex DB_a3m.ffindex")
    shell(" mv DB_a3m_ordered.ffdata DB_a3m.ffdata")
    os.chdir("/content/")

  # run hhsearch
  hhsearch_runner = hhsearch.HHSearch(binary_path="hhsearch",
      databases=[hhDB_dir.as_posix()+"/"+db_prefix])
  with StringIO() as fh:
    SeqIO.write([query_seq], fh, "fasta")
    seq_fasta = fh.getvalue()
  hhsearch_result = hhsearch_runner.query(seq_fasta)

  # process hits
  hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
  if len(hhsearch_hits) >0:
    hit = hhsearch_hits[0]
    hit = replace(hit,**{"name":template_seq.id})
  else:
    hit = None
  return hit

def plot_plddt_legend():
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

def plot_confidence(outs, model_num=1):
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

def show_pdb(model_num=1, show_sidechains=False, show_mainchains=False, color="lDDT"):
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


def get_msa(
      query_sequence, jobname, use_env,
      use_templates,
      homooligomer,
      use_msa):
  import colabfold as cf
  from alphafold.data import pipeline
  template_paths = None # initialize

  #@title Get MSA and templates
  print("Getting MSA and templates...")
  if use_templates:
    a3m_lines, template_paths = cf.run_mmseqs2(query_sequence, jobname, use_env, use_templates=True)
  elif use_msa:
    a3m_lines = cf.run_mmseqs2(query_sequence, jobname, use_env)



  if (not use_msa):
    a3m_lines = ">query sequence \n%s" %(query_sequence)
    print("Not using any MSA information")

  # File for a3m
  a3m_file = f"{jobname}.a3m"

  with open(a3m_file, "w") as text_file:
      text_file.write(a3m_lines)

  # parse MSA
  msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

  print("Done with MSA and templates")
  return msa, deletion_matrix, template_paths

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

def get_template_hit_list(cif_files = None, fasta_dir = None,
    query_seq = None,
    hhDB_dir = None):
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
        if 1: # ZZtry:
          hit = hh_process_seq(query_seq,seq,hhDB_dir)
        if 0: # ZZexcept Exception as e:
          print("Failed to process %s" %(filepath),e)
          hit = None
        if hit is not None:
          template_hit_list.append([hit,mmcif])
          print("Template %s included" %(filepath))
        else:
          print("Template %s not included (failed to process)" %(filepath))

  return template_hit_list

