# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os,sys
from libtbx.utils import Sorry
from libtbx import group_args
from phenix.program_template import ProgramTemplate

# =============================================================================
class Program(ProgramTemplate):
  program_name = 'phenix.alphafold_with_density_map'
  description = '''
Run AlphaFold and iteratively improve modeling with density map

Inputs:
  Sequence
  Directory containing density map
  job name
  '''

  datatypes = ['phil', 'model']

  master_phil_str = """

  input_directory = ColabInputs
      .type = path
      .help = Input directory containing density map.  The map filename must\
               start with the same characters as the jobname (only including \
               characters before the first underscore)
      .short_caption = Input directory

  output_directory = ColabOutputs
      .type = path
      .help = Output directory. Copy outputs to output_directory.  Used \
               to restart with carry on.
      .short_caption = Output directory

  save_outputs_in_google_drive = False
      .type = bool
      .help = If run on Colab, copy outputs to output_directory in Google drive
      .short_caption = Save outputs in Google Drive

  content_dir = None
      .type = path
      .help = Content directory. Default is working directory
      .short_caption = Content directory

  data_dir = /mnt
      .type = path
      .help = Data directory (location of AlphaFold parameters)
      .short_caption = Content directory

  upload_file_with_jobname_resolution_sequence_lines = None
      .type = bool
      .help = Upload a file with a set of jobs (Colab only). \
              Each line in the file is a jobname, resolution, and sequence
      .short_caption = Upload batch file

  maximum_cycles = 10
      .type = int
      .help = Maximum cycles to carry out
      .short_caption = Maximum cycles

  cycle_rmsd_to_resolution_ratio = 0.1
      .type = float
      .help = Stop iteration if rmsd between subsequent AlphaFold models \
         is less than cycle_rmsd_to_resolution_ratio times the resolution \
          for two cycles in a row
      .short_caption = Cycle RMSD to resolution ratio

  password = None
      .type = str
      .help = Phenix download password (Colab only). The \
                password used to download \
                Phenix at your institution. Updated weekly, so you may \
                need to request a new one frequently.
      .short_caption = Download password

  version = dev-4502
      .type = str
      .help = Version of Phenix to run (Colab only)
      .short_caption = Phenix version 

  query_sequence = 'ITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGP'
      .type = str
      .help = Sequence
      .short_caption = Sequence

  resolution = 3.71
      .type = float
      .help = Resolution of map (A)
      .short_caption = Resolution

  jobname = '7mlz_50'
      .type = str
      .help = Name of this job.  The first characters before any underscore \
               must be unique and will define the first characters of the \
               corresponding map file in the input directory
      .short_caption = Job name

  use_msa = True
      .type = bool
      .help = Use multiple sequence alignments at some point
      .short_caption = Use MSA

  skip_all_msa_after_first_cycle = False
      .type = bool
      .help = Skip multiple sequence alignments after first cycle
      .short_caption = Skip all MSA after first cycle

  include_templates_from_pdb = False
      .type = bool
      .help = Include templates from PDB
      .short_caption = Include templates from PDB

  maximum_templates_from_pdb = 20
      .type = int
      .help = Maximum templates from PDB to include
      .short_caption = Maximum templates from PDB

  upload_manual_templates = False
      .type = bool
      .help = Supply templates for AlphaFold prediction.  Used in the same \
                way as templates from the PDB unless \
                uploaded_templates_are_map_to_model is set. \
                 May be .cif or .pdb files. \
                Templates must start with characters matching the first \
                characters in the jobname (before the first underscore), the\
                remainder of the file name can be anything but just end in \
                .cif or .pdb.  The files must be in the input_directory or \
                be uploaded (in Colab only).
      .short_caption = Include manual templates

  uploaded_templates_are_map_to_model = False
      .type = bool
      .help = The manual templates are models that may or may not have the\
              sequence of the alphafold models.  These are used only as \
              suggestions for placement of the main chain.
      .short_caption = Templates are suggestions for main-chain placement

  upload_maps = True
      .type = bool
      .help = Use maps (required to be True)
      .short_caption = Use map

  random_seed = 7231771
      .type = int
      .help = Random seed
      short_caption = Random seed

  random_seed_iterations = 100
      .type = int
      .help = Random seed iterations of AlphaFold in first cycle.  The \
               model with the highest plDDT will be used
      .short_caption = Random seed iterations

  minimum_random_seed_iterations = 5
      .type = int
      .help = Random seed iterations of AlphaFold after first cycle.  The \
               model with the highest plDDT will be used
      .short_caption = Random seed iterations

  big_improvement = 5
      .type = float
      .help = How much improvement in plDDT is worth going through all \
              randomization cycles
      .short_caption = Big improvement value

  debug = False
      .type = bool
      .help = Debugging run (print traceback error messages)
      .short_caption = Debug run

  carry_on = False
      .type = bool
      .help = Carry on from where previous run ended. Used (usually in Colab) \
              to go on after a crash or timeout.  Requires that files are \
              saved in the output directory
      .short_caption = Carry On

  cif_dir = None
      .type = path
      .help = Location of templates (normally set automatically)
      .short_caption = CIF directory
      .style = hidden

  template_hit_list = None
      .type = strings
      .help = List of templates (normally set automatically)
      .style = hidden

  jobnames = None
      .type = strings
      .help = List of jobnames (normally set automatically)
      .short_caption = Jobname list
      .style = hidden

  resolutions = None
      .type = floats 
      .help = List of resolutions (normally set automatically)
      .short_caption = Resolution list
      .style = hidden

  map_filename_dict = None
      .type = strings
      .help = Map filename dict (used internally only)
      .short_caption = Map filename dict
      .style = hidden
  
  cif_filename_dict = None
      .type = strings
      .help = CIF filename dict (used internally only)
      .short_caption = CIF filename dict
      .style = hidden
  
  query_sequences = None
      .type = strings
      .help = List of query sequences (one per jobname) (used internally only)
      .short_caption = Query sequences 
      .style = hidden

  manual_templates_uploaded = None
      .type = strings
      .help = List of manual templates (used internally only)
      .short_caption = Manual templates
      .style = hidden

  maps_uploaded = None
      .type = strings
      .help = List of maps (used internally only)
      .short_caption = Maps
      .style = hidden

  num_models = 1
      .type = int
      .help = Number of models (used internally only)
      .short_caption = Number of models 
      .style = hidden

  homooligomer = 1
      .type = int
      .help = Number of copies (used internally only)
      .short_caption = Number of copies 
      .style = hidden

  cycle = None
      .type = int 
      .help = Cycle number (internal use only)
      .style = hidden


  use_env = None
      .type = bool
      .help = Use env (internal use only)
      .style = hidden

  use_custom_msa = None
      .type = bool
      .help = Use custom msa (internal use only)
      .style = hidden

  use_templates = None
      .type = bool
      .help = Use templates (internal use only)
      .style = hidden

  template_paths = None
      .type = bool
      .help = Template paths (internal use only)
      .style = hidden

  mtm_file_name = None
      .type = str
      .help = map_to_model file name (internal use only)
      .style = hidden

  cycle_model_file_name = None
      .type = str
      .help = cycle_model_file_name (internal use only)
      .style = hidden

  previous_final_model_name = None
      .type = str
      .help = previous_final_model_name (internal use only)
      .style = hidden

  msa = None
      .type = str
      .help = MSA (internal use only)
      .style = hidden

  msa_is_msa_object = None
      .type = bool 
      .help = MSA info (internal use only)
      .style = hidden

  deletion_matrix = None
      .type = str
      .help = Deletion matrix (internal use only)
      .style = hidden

  
  """

  def run(self):

    # print version and date
    self.print_version_info()
    self.data_manager.set_overwrite(True)

    #
    self.get_data_inputs()  # get any file-based information
    self.print_params()


    print("Data_dir will be: ",self.params.data_dir, file = self.logger)
    print("Content_dir will be: ",self.params.content_dir, file = self.logger)
    print("Random seed will be:",self.params.random_seed, 
         "with ",self.params.random_seed_iterations,
      "iterations, decreasing to ",self.params.minimum_random_seed_iterations,
        file = self.logger)
    print("Carry-on will be: %s" %(self.params.carry_on), file = self.logger)


    from phenix_alphafold_utils import set_up_input_files
    from phenix_colab_utils import import_tensorflow
    self.params = set_up_input_files(self.params)

    from run_alphafold_with_density_map import run_jobs

    # Get tensorflow import before installation
    if not locals().get('tf'):
      tf = import_tensorflow()

    # Working directory
    os.chdir(self.params.content_dir)
    result = run_jobs(self.params)

    self.result = result
    if not result:
      print("Unable to create model",
         file = self.logger)
      return

    print ('\nFinished with alphafold_with_density_map', file=self.logger)
  # ---------------------------------------------------------------------------
  def get_results(self):

    return self.result

# =============================================================================
#    Custom operations
# =============================================================================
#


  def set_defaults(self):
    # set params for files identified automatically and vice versa
    params=self.params
    self.titles=[]
    for d in ["/app","/app/alphafold","/app/phenix/modules/Colabs/alphafold2"]:
      if d not in sys.path:
         sys.path.append(d)

    if not params.content_dir:
      params.content_dir = os.getcwd()

    if params.output_directory is None:
      params.output_directory = params.content_dir

    if params.upload_file_with_jobname_resolution_sequence_lines:
      params['jobname'] = None
      params['resolution'] = None
      params['sequence'] = None

  def get_data_inputs(self):  # get any file-based information
    self.set_defaults()

  def validate(self):  # make sure we have files
    return True

  def print_params(self):
    import iotbx.phil
    master_phil = iotbx.phil.parse(master_phil_str)
    print ("\nInput parameters for alphafold_with_density_map:\n",
          file = self.logger)
    master_phil.format(python_object = self.params).show(out = self.logger)

  def print_version_info(self):

    # Print version info
    import time
    print ("\n"+60*"*"+"\n"+10*" "+"PHENIX alphafold_with_density_map" +\
      "  "+str(time.asctime())+"\n"+60*"*"+"\n",file=self.logger)
    print ("Working directory: ",os.getcwd(),"\n",file=self.logger)
    print ("PHENIX VERSION: ",os.environ.get('PHENIX_VERSION','svn'),"\n",
     file=self.logger)

# =============================================================================
# for reference documentation keywords
master_phil_str = Program.master_phil_str
