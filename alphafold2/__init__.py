# -*- coding: utf-8 -*-
from __future__ import division, print_function
from libtbx import adopt_init_args
from six.moves import StringIO
import mmtbx.model
from scitbx.matrix import col
from mmtbx.model import manager as model_manager
from iotbx.map_model_manager import map_model_manager as MapModelManager
from libtbx.utils import null_out, Sorry
from libtbx import group_args
from scitbx.array_family import flex
import sys, os
import random
from phenix.model_building.morph_info import apply_lsq_fit_to_model, apply_lsq_fit
from phenix.model_building.morph_info import get_lsq_fit, apply_lsq_fit
'''
   ====================================================================
   =========         CLASS LOCAL_MODEL_BUILDING          ==============
   =========    HIGH LEVEL METHODS FOR MODEL BUILDING    ==============
   ====================================================================
   This class holds a map_manager (and model) and user preferences such as
   which methods to use for each type of model-building.  It also holds
   preferences such as chain_type, resolution, number of processors.
   This class has two kinds of model-building methods:
   1. Simple methods (e.g., combine_models) that have just one approach
   For simple methods, this class contains the methods themselves, or a simple
   call to the actual method
   2. Generic methods (e.g., fit_loop) with more than one approach
       (fit_loop_with_resolve, fit_loop_with_trace_chain)
   For generic methods, this class contains short methods for calling
   each type of model-building with any desired method appropriate for
   that type of model-building. For example, ab-initio building into
   density (build) can be done with resolve, find_helices_strands, etc.
   The actual methods for carrying out model-building are located in
   files named by the type of model-building (i.e, fit_ligand.py
   is ligand fitting, build.py is ab-initio building etc.)
   The specific methods are named by the type of model-building and the
   approach used. For example, fitting a ligand with resolve is in the
   file fit_ligand.py with the method name fit_ligand_with_resolve.
   All generic methods for a particular type of model-building have exactly the
   same arguments and all return the same group_args object with scoring
   defined in the file containing those model-building methods (i.e.,
   fit_ligand is scored by map-model CC scaled by (fraction built)**0.5).
   This class contains methods for running an arbitrary subset of the
   allowed methods for any particular type of model-building with
   multiprocessing. The multiprocessing can be terminated if any
   job achieves a sufficient score as defined by good_enough_score.  The
   highest-scoring job is returned as the best result and is accessible with
   get_best_result(). You can also get all the results and their scores
   with get_last_results()
      ------             ADDING NEW METHODS              --------
   GENERIC METHODS:
   TO ADD A NEW MODEL-BUILDING METHOD FOR AN EXISTING TYPE OF MODEL-BUILDING:
   1.  Duplicate the template code provided in the appropriate model-building
      methods file (for example, fit_ligand.py for fitting a ligand).
   2.  Edit the code to do what you want and replace "TEMPLATE" with your
      approach (like "resolve").
   3.  Add your approach to the list of allowed methods at the top of that file.
   4. Now your approach will be included by default in all runs requesting that
      type of model-building.  To select only your method, use the
      set_defaults() method of local_model_building. For example,
      build.set_defaults(allowed_fit_ligand_methods=['resolve'])
   TO ADD A NEW TYPE OF MODEL-BUILDING:
   1. Make a new methods file (i.e., copy fit_ligand.py and change the
      calling arguments in the TEMPLATE to be what is needed for this
      type of model-building, then remove all the ligand-specific code).
   2. Add the new type of model-building (my_model_building_type) to this file:
     A.  Add an import ALLOWED_my_model_building_type_METHODS
     B   Add an entry in this file in the __init__ method like:
        self.info.sequence_from_map_methods= ALLOWED_SEQUENCE_FROM_MAP_METHODS
     C.  Add an entry in set_defaults() in this file with allowed methods
     D.  Add a top-level method (copy and edit fit_ligand in this file)
   SPECIFIC METHODS:
   You can add a method directly to this class.
'''
from phenix.model_building.build import ALLOWED_BUILD_METHODS
from phenix.model_building.extend import ALLOWED_EXTEND_METHODS
from phenix.model_building.fit_loop import ALLOWED_FIT_LOOP_METHODS, \
         ALLOWED_REPLACE_SEGMENT_METHODS, SUGGESTED_FIT_LOOP_METHODS,\
         SUGGESTED_REPLACE_SEGMENT_METHODS, \
         SUGGESTED_FIX_INSERTION_DELETION_METHODS, \
         SUGGESTED_FIX_INSERTION_DELETION_SEQUENCE_METHODS
from phenix.model_building.sequence_from_map \
    import ALLOWED_SEQUENCE_FROM_MAP_METHODS
from phenix.model_building.regularize_model \
    import ALLOWED_REGULARIZE_MODEL_METHODS
from phenix.model_building.fit_ligand import ALLOWED_FIT_LIGAND_METHODS
from phenix.model_building.morph import ALLOWED_MORPH_METHODS
ALLOWED_TABLES = ['n_gaussian','wk1995','it1992','neutron','electron']
ALLOWED_EXPERIMENT_TYPES= ['xray','cryo_em','neutron']
ALLOWED_THOROUGHNESS = ['quick','medium', 'thorough', 'extra_thorough']
ALLOWED_CHAIN_TYPES = ['protein','dna', 'rna']
ONLY_PROTEIN_METHODS = ['structure_search','loop_lib','trace_chain','trace_and_build']
ONLY_RNA_METHODS = ['build_rna_helices']
ONLY_DNA_METHODS = []
class local_model_building:
  def __init__(self,
    map_model_manager = None,
    model_id = 'model',
    normalize = True,
    soft_zero_boundary_mask = True,
    soft_zero_boundary_mask_radius = None,
    remove_atoms_outside_map = False,
    mask_atoms_atom_radius = 1.5,
    nproc = 1,
    log = sys.stdout,
    ):
    '''
     Set up a map for local model-building
     Normally supply map_model_manager,  nproc where map_model_manager
      already knows about resolution and experiment_type
     All other parameters can be default (see below)
     This class is intended for building in a small part of a big map. You
     should initialize with a map_model_manager containing just the boxed
     small region that you want to work with and your full model (or just the
      part of the model inside that region if you want).
     The map_model_manager should contain your working model (if any).
     Note that the map and model are automatically shifted in map_model_manager
       to place the origin of the map at (0,0,0).  The coordinate shift to
       get the model in its original location is the negative of
       map_model_manager.get_model_by_id(model_id).shift_cart().
       If you write the model with model.model_as_pdb() it will be written in
       its original location.
     As the map_manager is usually boxed, its symmetry is normally P1 and the
     map will usually not be wrapped (not an infinite map)
     The map can be a cryo-EM map (default) or an X-ray map (specify
        experiment_type='xray';
     fitting side-chains uses different targets for xray)
     The map_manager can be None.  If so, experiment_type and resolution
      can be missing as well
     Optional parameters:
     remove_atoms_outside_map:  remove atoms outside map to make
       the model smaller. Skipped if map is wrapped
     nproc:  number of processors to use
     normalize : Normalize the map (mean=0, SD=1) before using it
     soft_zero_boundary_mask: Apply a soft boundary mask around the map before
       using it
     soft_zero_boundary_mask_radius: Optional radius for zero_boundary mask
To use these tools you can do something like this:
from phenix.model_building import local_model_building
build=local_model_building(
   map_model_manager=mmm, # map_model manager
   nproc=4,   # more processors mean more tries also
  )
# set any defaults
build.set_defaults(
   allowed_fit_ligand_methods=["resolve"],  # choose a specific method
   scattering_table='electron',  # for a cryo-em map
   thoroughness='thorough',  # quick/medium/thorough/extra_thorough
   )
# run ligand fitting (NOTE: you can run again and get a different answer)
fitted_ligand_model=build.fit_ligand(
   ligand_model=atp_model,           #ligand model object
   restraints_object=atp_restraints,   # CIF object with restraints
   good_enough_score=0.75,           # stop looking if this is achieved
  )
NOTES:
  You can run again and get a different answer
  You can run a different method now without resetting build or you can
    set defaults to be different and run again.
    '''
    # Save the list of arguments to this __init__ so we can use them in
    #  deep_copy, skip arguments that are handled separately
    self.init_arg_list = self._get_init_arg_list(locals(),
       special_args = ['log','map_model_manager'])
    # Check input data types and values
    assert isinstance(nproc, int)
    assert isinstance(map_model_manager, MapModelManager)
    # Save the input arguments as a group_args object and set defaults
    self.log = log
    self.info_from_last_result = None
    if map_model_manager.get_model_by_id(model_id) and \
        map_model_manager.get_model_by_id(model_id).crystal_symmetry() is None:
      map_model_manager.get_model_by_id(
         model_id).add_crystal_symmetry_if_necessary()
    if map_model_manager.get_model_by_id(model_id) and \
        map_model_manager.get_model_by_id(model_id).get_xray_structure(
          ).scattering_type_registry().last_table():
        scattering_table = map_model_manager.get_model_by_id(
           model_id).get_xray_structure(
          ).scattering_type_registry().last_table()
    elif map_model_manager.map_manager():
     scattering_table = map_model_manager.map_manager().scattering_table()
    else:
     scattering_table = None
    if map_model_manager.map_manager():
      mm = map_model_manager.map_manager()
      resolution = mm.resolution()
      experiment_type = mm.experiment_type()
      d_min_ratio = 0.833 if (mm.experiment_type()=='xray') else 1.0
                                     # ratio of resolution
                                     # for Fourier coeffs to nominal resolution
    else:
      resolution = None
      experiment_type = None
      d_min_ratio = 1
    self.info = group_args(
      group_args_type = 'local_model_building info object',
      model_id = model_id,
      resolution = resolution,
      experiment_type = experiment_type,
      normalize = normalize,
      soft_zero_boundary_mask = soft_zero_boundary_mask,
      soft_zero_boundary_mask_radius = soft_zero_boundary_mask_radius,
      remove_atoms_outside_map = remove_atoms_outside_map,
      mask_atoms_atom_radius = mask_atoms_atom_radius,
      nproc = nproc,
      thoroughness = 'medium',   # === Setting defaults starting here ====
      temp_dir = 'temp_dir_mb',
      d_min_ratio = d_min_ratio,
      residues_on_each_end = 3,   # Residues on each end of loops (working)
      residues_on_each_end_extended = 15, # Residues on each end
                                          #for structure_search runs
      scattering_table = scattering_table,
      sequence = None,
      chain_type = None,
      random_seed = 74321,
      debug = False,
      replace_segment_methods = SUGGESTED_REPLACE_SEGMENT_METHODS,
      fix_insertion_deletion_methods = \
         SUGGESTED_FIX_INSERTION_DELETION_METHODS,
      fix_insertion_deletion_sequence_methods = \
         SUGGESTED_FIX_INSERTION_DELETION_SEQUENCE_METHODS,
      build_methods = ALLOWED_BUILD_METHODS,
      extend_methods = ALLOWED_EXTEND_METHODS,
      fit_loop_methods = SUGGESTED_FIT_LOOP_METHODS,
      fit_ligand_methods = ALLOWED_FIT_LIGAND_METHODS,
      sequence_from_map_methods = ALLOWED_SEQUENCE_FROM_MAP_METHODS,
      regularize_model_methods = ALLOWED_REGULARIZE_MODEL_METHODS,
      morph_methods = ALLOWED_MORPH_METHODS,
    )
    # Set up the map and model to be used
    self._map_manager = map_model_manager.map_manager()
    self._model = map_model_manager.get_model_by_id(model_id)
    # Remove excess model if desired
    if remove_atoms_outside_map and not self.map_manager().wrapping():
      # Creates new model (no longer a pointer to original)
      map_model_manager.remove_model_outside_map(
        boundary=mask_atoms_atom_radius)
    # Initialize masked_map_manager and model and restraints
    self._masked_map_manager = None
    self._second_masked_map_manager = None
    self.restraints_dict = {}
    self.score_sequence_dict = {}
    # Initialize useful_results (results from last run of group of methods)
    self.useful_results = []
    # Initialize fitted_loop (last fitted loop)
    self.fitted_loop_info = None
    # Initialize extension (last extension)
    self.extension_info = None
    # Set defaults
    if not self.info.scattering_table:
      if self.info.experiment_type=='xray':
        self.info.scattering_table = 'n_gaussian'
      else:
        self.info.scattering_table = 'electron'
    copied = False
    if normalize and self.map_manager():
      self._map_manager = self.map_manager().deep_copy()
      self._map_manager.set_mean_zero_sd_one()
      copied = True
      mmm = self._map_manager.map_data().as_1d().min_max_mean()
      if mmm.min == mmm.max and not self._map_manager.is_dummy_map_manager():
        print("WARNING: Map is flat...cannot do anything ", file = self.log)
    if self.map_manager() and (not self.map_manager().is_dummy_map_manager()) \
         and soft_zero_boundary_mask:
      if not copied:
        self._map_manager = self.map_manager().deep_copy()
      if not soft_zero_boundary_mask_radius:
        soft_zero_boundary_mask_radius = self.info.resolution
      self.map_manager().create_mask_around_edges(
        boundary_radius = soft_zero_boundary_mask_radius)
      self.map_manager().soft_mask(
        soft_mask_radius = soft_zero_boundary_mask_radius)
      self.map_manager().apply_mask()
      self.map_manager().delete_mask()
    # Ready to go
  def __repr__(self):
    print("\nLocal model-building object:\n%s\n%s" %(
      self.model(),self.map_manager()), file = self.log)
  def deep_copy(self):
    '''
     Deep copy this local_model_building object.  Keep track of what needs
     to be copied based on the input args for the __init__ method and all the
     values in self.info.
     Uses customized_copy
    '''
    return self.customized_copy()
  def as_map_model_manager(self, skip_model = False):
    '''
     Return a map_model_manager with working model and map_manager
    '''
    if skip_model:
      model = None
    else:
      model = self.model()
    mmm = MapModelManager(
        map_manager = self.map_manager(),
        model = model)
    mmm.set_resolution(self.info.resolution)
    mmm.set_experiment_type(self.info.experiment_type)
    if model:
      # Set the scattering table if it was not set before
      last_scattering_table = mmm.model().get_xray_structure(
        ).scattering_type_registry().last_table()
      if self.info.scattering_table and (last_scattering_table is None):
        mmm.set_scattering_table(self.info.scattering_table)
    return mmm
  def customized_copy(self, model = None, map_manager = None):
    '''
     Deep copy this local_model_building object.  Keep track of what needs
     to be copied based on the input args for the __init__ method and all the
     values in self.info.
     If model or map_manager are specified, use them instead of current model
     or map_manager. They must be compatible (same gridding and shift_cart()).
     If model/map_manager specified, they are not deep_copied. If you don't
     want them changed, deep_copy them before using them in the call
    '''
    if model:
      assert self.map_manager().is_compatible_model(model)
    elif self.model():
      model = self.model().deep_copy()
    else:
      model = None
    if map_manager:
      assert self.map_manager().is_similar(map_manager)
    else:
      map_manager = self.map_manager().deep_copy()
    default_keys = list(self.info().keys())  # all the keys for current values
    keys_to_skip = ['group_args_type'] # don't want the type of group args
    for key in default_keys:  # get rid of unset defaults
      if self.info()[key] is None:
        keys_to_skip.append(key)
    kw = {}
    for key in self.init_arg_list:
      kw[key] = getattr(self.info,key)  # put init args here
      keys_to_skip.append(key)
    new_build = local_model_building(
     map_model_manager = MapModelManager(
        map_manager = map_manager,
        model = model),
     log = self.log,
     **kw)
    # Set up the default kw:
    default_kw = {}
    for key in list(self.info.keys()):
      if not key in keys_to_skip:
        default_kw[key] = self.info()[key]
    new_build.set_defaults(**default_kw)
    return new_build
  def _get_init_arg_list(self, locals, special_args=None):
    init_args=group_args()
    adopt_init_args(init_args,locals)
    init_args_as_dict =  init_args()
    # remove special ones that are handled by themselves
    for arg in special_args:
      del init_args_as_dict[arg]
    return list(init_args_as_dict.keys())
  def set_log(self, log):
    ''' Set the output log stream'''
    self.log = log
  def set_defaults(self,     # NOTE: If you add an argument, add to self.info
    thoroughness = None,
    temp_dir = None,
    scattering_table = None,
    experiment_type = None,
    resolution = None,
    sequence = None,
    chain_type = None,
    nproc = None,
    random_seed = None,
    d_min_ratio = None,
    residues_on_each_end = None,
    residues_on_each_end_extended = None,
    debug = None,
    replace_segment_methods = None,
    fix_insertion_deletion_methods = None,
    fix_insertion_deletion_sequence_methods = None,
    build_methods = None,
    fit_loop_methods= None,
    extend_methods= None,
    fit_ligand_methods= None,
    sequence_from_map_methods= None,
    regularize_model_methods= None,
    morph_methods= None,
   ):
    '''
     Default parameters for model_building
     NOTE: If you add a default parameter then also
     add it to self.info object in __init__
    thoroughness: thoroughness (quick/medium/thorough/extra_thorough)
    temp_dir:     directory to put any temporary files. NOTE:
                       if you run multiple instances of model_building, give
                       them separate temp_dir directories
    scattering_table: scattering table for refinement (from ALLOWED_TABLES)
    experiment_type:  xray neutron cryo_em
    resolution:      nominal resolution of map
    sequence:      sequence of the molecule
    chain_type:      chain type (PROTEIN/DNA/RNA)
    random_seed:      random seed for refinement and model-building
    nproc:      number of processors
    d_min_ratio:      ratio of resolution to use when calculating Fourier
                       coefficients to the nominal resolution.  Normally
                        1 if x-ray and 0.833 if cryo-EM.
    residues_on_each_end:  Residues on end of a loop std (usually 3)
    residues_on_each_end_extended:  Residues on each end of a loop or
                      replacement when using structure_search (usually 15)
    debug:            debug run (write out traceback)
    build_methods:    build methods from ALLOWED_BUILD_METHODS
    replace_segment_methods: methods from ALLOWED_REPLACE_SEGMENT_METHODS
    fix_insertion_deletion_methods: methods from ALLOWED_FIX_INSERTION_DELETION_METHODS
    fix_insertion_deletion_sequence_methods: methods from ALLOWED_FIX_INSERTION_DELETION_SEQUENCE_METHODS
    fit_loop_methods:  fit_loop methods from ALLOWED_FIT_LOOP_METHODS
    extend_methods:  fit_loop methods from ALLOWED_EXTEND_METHODS
    fit_ligand_methods: fit_ligands methods from ALLOWED_FIT_LIGANDS_METHODS
    sequence_from_map_methods: sequence_from_map_methods from
                          ALLOWED_SEQUENCE_FROM_MAP_METHODS
    regularize_model_methods: regularize_model_methods methods from
                          ALLOWED_REGULARIZE_MODEL METHODS
    morph_methods  : morph_methods from
                          ALLOWED_MORPH_METHODS
    '''
    if debug is not None:
      assert isinstance(debug, bool)
      self.info.debug = debug
    if temp_dir is not None:
      self.info.temp_dir = temp_dir
    if thoroughness is not None:
      assert thoroughness in ALLOWED_THOROUGHNESS
      self.info.thoroughness = thoroughness
    if replace_segment_methods is not None:
      assert type(replace_segment_methods) == type([1,2,3])
      for x in replace_segment_methods:
        assert x in ALLOWED_REPLACE_SEGMENT_METHODS # replace_segment methods
      self.info.replace_segment_methods = replace_segment_methods
    if fix_insertion_deletion_methods is not None:
      assert type(fix_insertion_deletion_methods) == type([1,2,3])
      for x in fix_insertion_deletion_methods:
        assert x in ALLOWED_REPLACE_SEGMENT_METHODS # replace_segment methods
      self.info.fix_insertion_deletion_methods = fix_insertion_deletion_methods
    if fix_insertion_deletion_sequence_methods is not None:
      assert type(fix_insertion_deletion_sequence_methods) == type([1,2,3])
      for x in fix_insertion_deletion_sequence_methods:
        assert x in ALLOWED_REPLACE_SEGMENT_METHODS # replace_segment methods
      self.info.fix_insertion_deletion_sequence_methods = fix_insertion_deletion_sequence_methods
    if build_methods is not None:
      assert type(build_methods) == type([1,2,3])
      for x in build_methods:
        assert x in ALLOWED_BUILD_METHODS # allowed build methods
      self.info.build_methods = build_methods
    if extend_methods is not None:
      assert type(extend_methods) == type([1,2,3])
      for x in extend_methods:
        assert x in ALLOWED_EXTEND_METHODS # allowed extend methods
      self.info.extend_methods = extend_methods
    if fit_loop_methods is not None:
      assert type(fit_loop_methods) == type([1,2,3])
      for x in fit_loop_methods:
        assert x in ALLOWED_FIT_LOOP_METHODS # allowed fit_loop methods
      self.info.fit_loop_methods = fit_loop_methods
    if fit_ligand_methods is not None:
      assert type(fit_ligand_methods) == type([1,2,3])
      for x in fit_ligand_methods:
        assert x in ALLOWED_FIT_LIGAND_METHODS # allowed fit_ligand methods
      self.info.fit_ligand_methods = fit_ligand_methods
    if sequence_from_map_methods is not None:
      assert type(sequence_from_map_methods) == type([1,2,3])
      for x in sequence_from_map_methods:
        assert x in ALLOWED_SEQUENCE_FROM_MAP_METHODS # allowed methods
      self.info.sequence_from_map_methods = sequence_from_map_methods
    if regularize_model_methods is not None:
      assert type(regularize_model_methods) == type([1,2,3])
      for x in regularize_model_methods:
        assert x in ALLOWED_REGULARIZE_MODEL_METHODS # allowed methods
      self.info.regularize_model_methods = regularize_model_methods
    if morph_methods is not None:
      assert type(morph_methods) == type([1,2,3])
      for x in morph_methods:
        assert x in ALLOWED_MORPH_METHODS # allowed methods
      self.info.morph_methods = morph_methods
    if scattering_table is not None:
      assert scattering_table in ALLOWED_TABLES
      self.map_manager().set_scattering_table(scattering_table)
      self.info.scattering_table = scattering_table
    if experiment_type is not None:
      assert experiment_type in ALLOWED_EXPERIMENT_TYPES
      self.map_manager().set_experiment_type(experiment_type)
      self.info.experiment_type = experiment_type
    if residues_on_each_end is not None:
      self.info.residues_on_each_end = residues_on_each_end
    if residues_on_each_end_extended is not None:
      self.info.residues_on_each_end_extended = \
          residues_on_each_end_extended
    if sequence is not None:
      self.info.sequence = sequence
    if chain_type is not None:
      assert chain_type.lower() in ALLOWED_CHAIN_TYPES
      self.info.chain_type = chain_type
    if resolution is not None:
      self.map_manager().set_resolution(resolution)
      self.info.resolution = resolution
    if random_seed is not None:
      self.info.random_seed = random_seed
    if nproc is not None:
      self.info.nproc = nproc
    if d_min_ratio is not None:
      self.info.d_min_ratio = d_min_ratio
  def show_defaults(self):
    '''
      Show current values of all defaults
    '''
    print ("\n Current model for local_model_building:\n%s\n" %(
       self.model()), file = self.log)
    print ("\n Current map_manager for local_model_building:\n%s\n" %(
      self.map_manager()), file = self.log)
    defaults_as_dict = self.info()
    keys = list(list(defaults_as_dict.keys()))
    keys.sort()
    print ("\n Current defaults for local_model_building:\n", file = self.log)
    print ("   PARAMETER                     VALUE", file = self.log)
    for key in keys:
      print ("%30s: %s" %(key+max(1,30-len(key))*" ", str(defaults_as_dict[key])), file = self.log)
  def quick(self):
    ''' Set thoroughness to quick'''
    self.info.thoroughness = 'quick'
  def medium(self):
    ''' Set thoroughness to medium'''
    self.info.thoroughness = 'medium'
  def thorough(self):
    ''' Set thoroughness to thorough'''
    self.info.thoroughness = 'thorough'
  def extra_thorough(self):
    ''' Set thoroughness to extra_thorough'''
    self.info.thoroughness = 'extra_thorough'
  def _create_temp_dir(self,):
    # Create the temporary directory (normally done in a run method).
    if not os.path.isdir(self.info.temp_dir):
      os.mkdir(self.info.temp_dir)
  def set_model(self, model, quiet = False,
    change_model_in_place = None):
    '''
     Replace working model with supplied model (must be compatible)
     If change_model_in_place the modify existing model in place
    '''
    if model is self._model: # nothing to do
      return
    if model:
      assert self.map_manager().is_compatible_model(model)
    if model and change_model_in_place and self._model is not None:
      self._model.replace_model_hierarchy_with_other(model)
      if not quiet:
        print("Changing working model in place as %s" %(text), file = self.log)
    else:
      self._model = model
      if not quiet:
        print ("Replacing working model:\n%s\nwith supplied model:\n%s" %(
         self.model(),model),file = self.log)
  def refine(self,
    model = None,
    refine_cycles = None,
    return_model = None,
    restrain_ends = None,
    reference_model = None,
    restraint_selection_string = None,
    iterative_refine = False,
    iterative_refine_start_resolution = 6,
    quick = False,
    ):
    '''
      Refine working model, or supplied model.  If working model, replace
      it with refined model unless return_model is True.
      If supplied model, return refined model.
      if return_model=True, return new model and do not modify existing model
      If quick, skip restraint and reference model and run quick refinement
      If iterative_refine, run at low to high resolution
    '''
    model,return_model = self._set_and_print_return_info(
      model=model,text='Refining',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    mmm= MapModelManager(model = model.deep_copy(),
         map_manager= self.map_manager().deep_copy())
         # it is going to get normalized
    if mmm.model().get_sites_cart().size()<1:
      print("Refinement was not successful (no atoms to refine)",
         file = self.log)
      return None # nothing to do
    if restraint_selection_string:
      selection = restraint_selection_string
      print("Restraint selection: '%s' " %(selection), file = self.log)
    elif restrain_ends:
      from mmtbx.secondary_structure.find_ss_from_ca import \
        get_first_resno,get_last_resno
      ph = mmm.model().get_hierarchy()
      selection = "resseq %s or resseq %s and name ca" %(
        get_first_resno(ph), get_last_resno(ph))
    else:
      selection = None
    if iterative_refine:
      map_data = mmm.map_manager().map_data().deep_copy()
      print("Selection string: %s" %(selection), file = self.log)
      print(
       "\nRunning quick refinement of unselected residues varying resolution",
          file = self.log)
      res_start = int(0.5+iterative_refine_start_resolution)
      import math
      res_end = min(res_start, int(math.ceil(self.info.resolution)) if \
         self.info.resolution else 3)
      for res in range(res_start, res_end - 1, -1):
        mmm.map_manager().set_map_data(map_data.deep_copy())
        print("\nRefining at resolution of %s (CC at %.2f A = %.2f)" %(res,
           self.info.resolution,
           mmm.map_model_cc(model = model, resolution = self.info.resolution)),
            file = self.log)
        mmm.map_manager().resolution_filter(d_min=res)
        mmm.set_resolution(res)
        build=mmm.model_building()
        model = build.refine(
           restraint_selection_string=selection,
           model = model, return_model =True, refine_cycles=1,
           quick=True)
      mmm.map_manager().set_map_data(map_data.deep_copy())
      mmm.set_resolution(self.info.resolution)
      print("\nRefining (full refinement) at resolution of %s (CC= %.2f)" %(
        self.info.resolution,
        mmm.map_model_cc(model = model, resolution = self.info.resolution)),
          file = self.log)
      build=mmm.model_building()
      model = build.refine( model = model, return_model =True,
        refine_cycles = refine_cycles,
        quick=False)
      print("Final CC at resolution of %s: CC= %.2f" %(
        self.info.resolution,
        mmm.map_model_cc(model = model, resolution = self.info.resolution)),
        file = self.log)
      return self._set_or_return_model(model,
         text="refined model",return_model=return_model,
         change_model_in_place = True)
    from phenix.autosol.iterative_ss_refine import run_real_space_refine
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    sys_stdout_sav = sys.stdout
    if self.info.debug:
      local_log = self.log
    else:
      local_log = null_out()
      sys.stdout = null_out()
      model.set_log(null_out())
    if quick:
      pdb_input_full_chain = model.get_hierarchy().as_pdb_input()
      high_resolution = self.info.resolution
      unit_cell_object = self.model().crystal_symmetry().unit_cell()
      space_group_object = self.model().crystal_symmetry().space_group()
      density_map = dummy_density_map(map_data = mmm.map_manager().map_data(),
        crystal_symmetry = self.map_manager().crystal_symmetry(),
        d_min = self.info.resolution)
      import mmtbx.monomer_library.server
      mon_lib_srv = mmtbx.monomer_library.server.server()
      ener_lib = mmtbx.monomer_library.server.ener_lib()
      from phenix.utilities.rs_refine import rs_refine
      rs=rs_refine()
      try:
        best_refined_final,best_rstw,refined_processed_pdb_file=\
           rs.run(pdb_inp=pdb_input_full_chain,
                    mon_lib_srv=mon_lib_srv,ener_lib=ener_lib,
                    fft_map=density_map,d_min=high_resolution,
                    unit_cell=unit_cell_object,
                    space_group=space_group_object, # 2013-08-07 use actual SG
                    verbose=self.info.debug, out=local_log)
        ph = refined_processed_pdb_file.all_chain_proxies.pdb_hierarchy
        assert ph.is_similar_hierarchy(model.get_hierarchy())
        new_model = mmm.model_from_hierarchy(ph, return_as_model = True)
        sys.stdout = sys_stdout_sav
      except Exception as e:
        if self.info.debug:
          raise AssertionError(e)
        else:
          sys.stdout = sys_stdout_sav
          print(
            "Quick refinement failed:\n%s\n...using model without refinement",
            file = self.log)
          new_model = model
      return self._set_or_return_model(new_model,
         text="refined model",return_model=return_model,
         change_model_in_place = True)
    refined = run_real_space_refine(
      d_min = self.info.resolution,
      refine_cycles = refine_cycles,
      scattering_table = self.info.scattering_table,
      map_model_manager = mmm,
      return_as_refined = True,
      restrain_to_original_selection = selection,
      reference_model = reference_model,
      debug = self.info.debug,
      out = local_log)
    if not self.info.debug:
      sys.stdout = sys_stdout_sav
    if not refined:
      print("Refinement was not successful",file = self.log)
      sys.stdout = sys_stdout_sav
      return
    new_model = refined.structure_monitor.model
    print("Refinement successful with cc_mask=%.3f" %(
        refined.structure_monitor.cc_mask),file = self.log)
    sys.stdout = sys_stdout_sav
    return self._set_or_return_model(new_model,
       text="refined model",return_model=return_model,
       change_model_in_place = True)
  def model_completion(self,
      model = None,
      mmm = None,  # supply model or map_model_manager with map and model
      thoroughness = None,  # quick, medium, thorough
      max_fraction_verylow = None, # max bad residues in a loop to keep it
      allow_reverse = None, # Allow reverse direction of fragments
      convincing_placement_score = None, # min sequence Z-score to keep junction
      refine_cycles = None,
      read_map_and_model = None, # read previously-written data
      minimum_score_tries = None, # Number of times to lower minimum score
       ):
    '''
    Break up model into fragments, removing loops. Reassemble into optimal
     order and add loops
    '''
    if mmm is None:
      if model:
        mmm = model.as_map_model_manager()
      else:
        mmm = self.as_map_model_manager()
    if not mmm.model():
      return None # nothing to do
    mmm.model().add_crystal_symmetry_if_necessary()
    if thoroughness == "quick":
      superquick = True
      quick = False
    elif thoroughness == "medium":
      superquick = False
      quick = True
    elif thoroughness == "thorough":
      superquick = False
      quick = False
    else:
      assert thoroughness in ["quick", "medium", "thorough"]
    from phenix.model_building.model_completion import run, get_params
    params = get_params(
     sequence_as_string = self.info.sequence,
     nproc = self.info.nproc,
     resolution = self.info.resolution,
     max_fraction_verylow = max_fraction_verylow,
     allow_reverse = allow_reverse,
     convincing_placement_score = convincing_placement_score,
     refine_cycles = refine_cycles,
     quick = quick,
     superquick = superquick,
     read_map_and_model = read_map_and_model,
     )
    new_model_info = run(
       params,
       mmm = mmm,
       log = self.log)
    if not new_model_info or not new_model_info.full_models:
      print("No result for model_completion", file = self.log)
      return None
    return new_model_info
  def ssm_match_to_map(self, model = None,
    helices_strands_model = None,
    ok_brute_force_cc = None):
   """
    Find secondary structure in map and use it to ssm superpose moving_model
    Optionally supply fragmentary model matching map as target
   """
   moving_model = model
   assert moving_model is not None
   # Check the chain type
   chain_type = self.info.chain_type
   if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = moving_model,
         return_protein_if_present = True)
   # Get ss
   if not helices_strands_model:
     self.set_defaults(build_methods = ['find_helices_strands'])
     ss_model = self.build(chain_type = chain_type,
       build_in_boxes = True,
       return_model=True)
     if not ss_model:
       print("Unable to get ss model...", file = self.log)
       return None
   else:  # use supplied
     ss_model = helices_strands_model
   # Now we have a target model.  Use superpose and morph to match it
   self.ssm_superpose(
    fixed_model = ss_model,
    moving_model = moving_model,
    superpose_brute_force = True,
    score_superpose_brute_force_with_cc = True,
    ok_brute_force_cc = ok_brute_force_cc,
    morph = False,
    trim = False,
    )
   result_list = self.get_results()
   if result_list and result_list[0].superposed_model:  # return top one
     best_result = result_list[0]
     superposed_model = best_result.superposed_model
     return superposed_model
   else:
     print("Unable to SSM match", file = self.log)
     # Set result_list to have ss_model
     ga = group_args(group_args_type = 'superposed model and lsq_fit',
        moving_model = moving_model,
        fixed_model = ss_model,
        superposed_model = None,
        lsq_fit = None,
        cc = None,)
     self.useful_results = [ga]
     return None
  def superpose_and_morph(self,
    fixed_model = None,
    moving_model = None,
    superpose = True,
    morph = True,
    trim = True,
    superpose_brute_force = False,
    score_superpose_brute_force_with_cc = False,
    include_swapped_match = None,
    allow_reverse_connectivity = None,
    shift_field_distance = None,
    include_ssm = True,
    include_lsq = True,
    try_lsq_first = True,
    starting_distance_cutoff = None,
    try_with_fragments_if_poor_fit = False,
    target_coverage = 0.5,
    good_enough_fraction = 0.9,
    direct_morph = False,
    ssm_match_to_map = False,
    ok_brute_force_cc = None,
    ):
    ''' Use lsq_superpose and ssm_superpose to try and superpose moving_model
       on fixed_model.
     Typical use:
       superposed_model = build.superpose_and_morph(moving_model = moving_model)
     Optionally morph moving_model after ssm matching and trimmed to
       match region
     If using lsq fitting and target coverage is less than target_coverage and
       try_with_chains_if_poor_fit=True, run with fragments
       and take highest coverage
     Optionally (direct_morph):  directly morph moving model onto fixed model
       Requires that moving and fixed model have same chain ID and residue numbers
       (fixed model can have missing residues though).
     Optionally use brute force method (superpose_brute_force), optionally
      scoring with cc and stopping if ok_brute_force_cc is obtained
    '''
    if ssm_match_to_map:
      print("Superposing model on map using SSM", file = self.log)
      self.ssm_match_to_map(model = moving_model)
      if self.get_results():
        superposed_model_info = self.get_results()[0]
        superposed_model = superposed_model_info.superposed_model
        ss_model = superposed_model_info.fixed_model
      else:
        superposed_model_info = None
        superposed_model = None
      if superposed_model:
        return superposed_model
      else:
        print("Unable to superpose...", file = self.log)
        return None # nothing to do
    elif superpose:
      print("Superposing model on fixed model", file = self.log)
    if morph:
      print("Morphing model to match fixed model", file = self.log)
    if direct_morph:
      print("Directly morphing moving model on to fixed model", file = self.log)
      trim = False
    if trim:
      print("Trimming model to match fixed model", file = self.log)
    if superpose_brute_force:
      print("Using brute force superposition method", file = self.log)
      include_lsq = False
      include_ssm = True
      direct_morph = False
      trim = False
      morph = False
      if score_superpose_brute_force_with_cc and (
          not self.map_manager().is_dummy_map_manager()):
        print("Scoring with map cc after superposition", file = self.log)
      elif score_superpose_brute_force_with_cc:
        score_superpose_brute_force_with_cc = False
        print("Not scoring with cc as no map is available", file = self.log)
    assert (fixed_model is not None) or (ssm_match_to_map)
    if (not superpose):
      superposed_model = moving_model
      found_ok_model = True
    else:
      superposed_model = None
      found_ok_model = False
      # Check the chain type
      chain_type = self.info.chain_type
      if not chain_type:
        from iotbx.bioinformatics import get_chain_type
        chain_type = get_chain_type(model = moving_model,
           return_protein_if_present = True)
      if chain_type.upper() != 'PROTEIN':
        include_ssm = False # cannot use for RNA/DNA
    params = self.get_ssm_params(
     allow_reverse_connectivity = allow_reverse_connectivity,
     shift_field_distance = shift_field_distance,
     starting_distance_cutoff = starting_distance_cutoff,
     trim = trim,
     superpose_brute_force= superpose_brute_force,
     score_superpose_brute_force_with_cc= score_superpose_brute_force_with_cc,
     ok_brute_force_cc = ok_brute_force_cc,
     morph = morph)
    mtmm = None
    if direct_morph:
      search_order = ['direct_morph']
    elif try_lsq_first:
      search_order = ['None','lsq','ssm']
    else:
      search_order = ['None','ssm','lsq']
    for method in search_order:
      found_something = False
      if method == 'direct_morph':
        superposed_model =  self.morph_matching_model(
           fixed_model = fixed_model,
           moving_model = moving_model,
           shift_field_distance = shift_field_distance,
           starting_distance_cutoff = starting_distance_cutoff,
           )
        if superposed_model:
          found_something = True
      if method == 'None' and found_ok_model:  # not doing any superposition
        found_something = True
      if method == 'lsq' and include_lsq and (not found_ok_model):
        print("\nTrying lsq superposition", file = self.log)
        test_model = self.lsq_superpose(
          fixed_model = fixed_model,
          moving_model = moving_model,
          morph = morph,
          trim = trim,
          starting_distance_cutoff = starting_distance_cutoff,
          try_with_fragments_if_poor_fit = try_with_fragments_if_poor_fit,
          target_coverage = target_coverage,
          quit_if_poor_match = True)
        if test_model:
          found_something = True
          if not superposed_model or (
               test_model.get_hierarchy().overall_counts().n_residues >
               superposed_model.get_hierarchy().overall_counts().n_residues):
            superposed_model = test_model
            if superposed_model.get_hierarchy().overall_counts().n_residues >= \
                (moving_model.get_hierarchy().overall_counts().n_residues *
                    good_enough_fraction) :
              found_ok_model = True
      if method == 'ssm' and include_ssm and (not found_ok_model):
        print("\nTrying ssm superposition", file = self.log)
        test_model = self.ssm_superpose(
          fixed_model = fixed_model,
          moving_model = moving_model,
          morph = morph,
          trim = trim,
          superpose_brute_force = superpose_brute_force,
          score_superpose_brute_force_with_cc = \
              score_superpose_brute_force_with_cc,
          include_swapped_match = include_swapped_match,
          allow_reverse_connectivity = allow_reverse_connectivity,
          starting_distance_cutoff = starting_distance_cutoff,
          )
        if test_model:
          found_something = True
          if not superposed_model or (
               test_model.get_hierarchy().overall_counts().n_residues >
               superposed_model.get_hierarchy().overall_counts().n_residues):
            superposed_model = test_model
            if superposed_model.get_hierarchy().overall_counts().n_residues >= \
                (moving_model.get_hierarchy().overall_counts().n_residues *
                    good_enough_fraction) :
              found_ok_model = True
      if found_something:
        mtmm = self.print_similarity(params = params,
            fixed_model = fixed_model, moving_model = moving_model,
          superposed_model = superposed_model,
          prefix = superposed_model.info().get('pdb_text'))
    if superposed_model and trim:
      # trim the model
      selection_string = mtmm.get_matching_parts_of_model_pair(
          matching_model = superposed_model, target_model = fixed_model)
      if selection_string:
        superposed_model = superposed_model.apply_selection_string(
          selection_string)
        print("Selecting matching parts: %s" %(selection_string),
          file = self.log)
        from phenix.model_building.morph_info import morph_info
        mtmm = morph_info(
          params = params,
          original_model = moving_model,
          target_model = fixed_model,
          morphed_model = superposed_model)
        rmsd_info = mtmm.rms_from_start(
          superposed_model, target_model = fixed_model,
          as_group_args=True)
        if rmsd_info:
          print("Trimmed RMSD: %.2f N: %s of %s" %(
            rmsd_info.rmsd,rmsd_info.rms_n,rmsd_info.all_n),
            file = self.log)
          superposed_model.info().rmsd_info = rmsd_info
        else:
          print("Unable to identify trimmed parts",
              file = self.log)
          mtmm = None
      else:
        print("Unable to identify matching parts...keeping everything",
            file = self.log)
        mtmm = None
    if superposed_model:
      return superposed_model
    else:
      print("Unable to superpose", file = self.log)
      return None
  def print_similarity(self, params = None,
      fixed_model = None, moving_model = None,
      superposed_model = None,
    prefix = 'Initial'):
    if moving_model is None:
      moving_model = superposed_model
    from phenix.model_building.morph_info import \
      morph_info
    mtmm = morph_info(
      params = params,
      original_model = moving_model,
      target_model = fixed_model,
      morphed_model = superposed_model)
    if superposed_model:
      superposed_model.info().matching_selection= \
          mtmm.get_selection_string_from_model(superposed_model)
      print("\nMatching selection for %s: %s" %(prefix,
         superposed_model.info().matching_selection), file = self.log)
      rmsd_info = mtmm.rms_from_start(
        superposed_model, target_model = fixed_model,
        as_group_args=True)
      if rmsd_info:
        print("%s RMSD: %.2f N: %s of %s" %(
         prefix, rmsd_info.rmsd, rmsd_info.rms_n, rmsd_info.all_n),
         file = self.log)
        found_ok_model = True
        superposed_model.info().rmsd_info = rmsd_info
      else:
        print("\nUnable to estimate initial RMSD by standard method",
          file = self.log)
        rmsd_info = self.get_ca_rmsd_with_indel(fixed_model, superposed_model,
          as_group_args = True)
        print("Initial RMSD: %.2f N: %s of %s" %(
           rmsd_info.rmsd,rmsd_info.rms_n, rmsd_info.all_n),
          file = self.log)
        mtmm.rmsd_info = rmsd_info
        superposed_model.info().rmsd_info = rmsd_info
    return mtmm
  def ca_rmsd_after_lsq_superpose(self,
    fixed_model = None,
    moving_model = None,):
    ''' Superpose moving model on fixed then get CA rmsd of closest
        pairs of CA atoms '''
    try:
      superposed_moving = self.superpose_and_morph(
        fixed_model = fixed_model,
        moving_model = moving_model,
        superpose = True,
        morph = False,
        trim = False)
    except Exception as e:
      superposed_moving = None

    if not superposed_moving:
      return None # nothing to do
    # Get ca rmsd
    rmsd_info = self.get_ca_rmsd_with_indel(fixed_model, superposed_moving,
      as_group_args = True)
    return rmsd_info
  def lsq_superpose(self,
    fixed_model = None,
    moving_model = None,
    morph = True,
    trim = True,
    quit_if_poor_match = False,
    return_lsq_fit = False,
    shift_field_distance = None,
    starting_distance_cutoff = None,
    try_with_fragments_if_poor_fit = False,
    target_coverage = 0.5,
     ):
    '''
     Use superpose_pdbs (lsq fitting) to superpose moving_model on fixed_model
     Optionally morph moving_model after ssm matching
     Optionally return moving_model trimmed to matching region only
     Optionally return None if poor match
     Optionally just return the transformation lsq_fit
     If target coverage is less than target_coverage and
       try_with_chains_if_poor_fit=True, run with fragments
       and take highest coverage
     NOTE: Assumes moving model and fixed model are in the same reference frame.
       Superposed model is in same reference frame as fixed model
     Typical use:
       superposed_model = build.lsq_superpose(moving_model = moving_model)
       # Matches moving_model to build.model() and trims it and morphs it
    '''
    if not fixed_model:
       fixed_model = self.model()
    if not fixed_model or not moving_model:
      return None # nothing to do
    fixed_model.add_crystal_symmetry_if_necessary()
    moving_model.add_crystal_symmetry_if_necessary()
    from phenix.command_line import superpose_pdbs
    superpose_params = superpose_pdbs.master_params.extract()
    # Check the chain type
    chain_type = self.info.chain_type
    if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = fixed_model,
         return_protein_if_present = True)
    if (not chain_type) or (chain_type.upper() != 'PROTEIN'): # can't do much
      trim = False
      morph = False
      try_with_fragments_if_poor_fit = False
    # Run on full model and pieces if full does not work
    if try_with_fragments_if_poor_fit:
      run_list = get_sample_selections(moving_model) # starts with 'all'
    else:
      run_list = ['all']
    best_x = None
    for run_info in run_list:
      if run_info == 'all':
        ph = moving_model.get_hierarchy().deep_copy()
        moving_hierarchy = ph
      else:
        ph = moving_model.get_hierarchy().deep_copy()
        asc1 = ph.atom_selection_cache()
        moving_hierarchy = ph.select(
          asc1.selection(string = run_info))
      try:
        x = superpose_pdbs.manager(
          superpose_params,
          log = null_out(),
          write_output = False,
          save_lsq_fit_obj = True,
          pdb_hierarchy_fixed = fixed_model.get_hierarchy(),
          pdb_hierarchy_moving = moving_hierarchy)
      except Exception as e:
        x = None
      if x is None:  # nothing to do
        continue
      # apply this to everything and see how good it is
      superposed_model = apply_lsq_fit_to_model(x.lsq_fit_obj, moving_model)
      x.rmsd_info = self.get_ca_rmsd_with_indel(fixed_model, superposed_model,
          chain_type = chain_type, as_group_args = True)
      if x.rmsd_info:
          x.coverage = x.rmsd_info.rms_n/max(2,x.rmsd_info.all_n)
          if x.rmsd_info and (not best_x or
            best_x.rmsd_info.rms_n < x.rmsd_info.rms_n):
            best_x = x
      if best_x and best_x.coverage >= target_coverage: # good enouogh
        break
      if (not try_with_fragments_if_poor_fit): # don't do anything more
        break
    if best_x is not None:
      x = best_x
      if hasattr(best_x,'rmsd_info') and best_x.rmsd_info.all_n:
        print("Replacement RMSD: %.2f N: %s of %s" %(
          best_x.rmsd_info.rmsd,best_x.rmsd_info.rms_n,best_x.rmsd_info.all_n),
            file = self.log)
    else:
      return None
    if return_lsq_fit:
      return x.lsq_fit_obj
    # Now use lsq fit to trim and morph reverse match if desired
    ssm_params = self.get_ssm_params(
     shift_field_distance = shift_field_distance,
     starting_distance_cutoff = starting_distance_cutoff,
     morph = morph)
    moving_model = apply_lsq_fit_to_model(x.lsq_fit_obj, moving_model)
    if moving_model and (moving_model.shift_cart() != fixed_model.shift_cart()):
      moving_model.set_shift_cart(fixed_model.shift_cart())
    if not morph:
      return moving_model
    else: # usual
      morphed_model = self.morph_to_match_model(
       fixed_model = fixed_model,
       trim = trim,
       moving_model = moving_model)
      if morphed_model:
        return morphed_model
      else:
        print("Trimming and morphing not successful", file = self.log)
        print("Unable to match superposed fixed_model to target model",
          file = self.log)
        if quit_if_poor_match:
          return None
        else:
          return moving_model
  def morph_matching_model(self,
    fixed_model = None,
    moving_model = None,
    shift_field_distance = None,
    starting_distance_cutoff = None,
     ):
    '''  superimpose matching parts of model (must be identical residue numbering
       and chain IDs) and morph '''
    if not fixed_model:
       fixed_model = self.model()
    if not fixed_model or not moving_model:
      return None # nothing to do
    fixed_model.add_crystal_symmetry_if_necessary()
    moving_model.add_crystal_symmetry_if_necessary()
    # Check the chain type
    chain_type = self.info.chain_type
    if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = fixed_model,
         return_protein_if_present = True)
    if chain_type.upper() != 'PROTEIN':
      print("Cannot run ssm_superpose on %s" %(chain_type), file = self.log)
      return None
    params = self.get_ssm_params(
     allow_reverse_connectivity = False,
     shift_field_distance = shift_field_distance,
     starting_distance_cutoff = starting_distance_cutoff,
     trim = False,
     morph = morph)
    # Use shift_cart == 0,0,0  as ssm does not keep track of it
    fixed_model_shift_cart = fixed_model.shift_cart()
    fixed_model.set_shift_cart((0,0,0))
    moving_model.set_shift_cart((0,0,0))
    from phenix.model_building.morph_info import morph_info
    mtmm = morph_info(
          params = params,
          original_model = moving_model,
          target_model = fixed_model,
          trimmed_morphed_model = fixed_model)
    try:
      mtmm.deduce_missing_info() # generates morphed_model from original_model
    except Exception as e:
      print(
        "Unable to morph model...may not match exactly in chain ID or residue numbers")
      return None
    return_model = mtmm.morphed_model
    if return_model and (return_model.shift_cart() != fixed_model_shift_cart):
      return_model.set_shift_cart(fixed_model_shift_cart)
    return return_model
  def ssm_superpose(self,
    fixed_model = None,
    moving_model = None,
    morph = True,
    trim = True,
    superpose_brute_force = None,
    score_superpose_brute_force_with_cc = None,
    include_swapped_match = None,
    allow_reverse_connectivity = None,
    shift_field_distance = None,
    starting_distance_cutoff = None,
    ok_brute_force_cc = None,
     ):
    '''
     Use ssm matching to superpose moving_model on fixed_model
     Normal use:
       superposed_model = build.ssm_superpose(moving_model = moving_model)
       # Matches moving_model to build.model() and trims it and morphs it
     Optionally morph moving_model after ssm matching and trim to overlap
     Optionally force use of fixed_model to moving_model instead
       of moving_model to fixed model (default is to do this
       only if standard fails)
     Optionally allow connectivity in fixed_model to have segments
      that are reversed
     Optionally use brute force method (superpose_brute_force), optionally
      scoring with cc and stopping if ok_brute_force_cc is obtained
     Optionally do not trim model at the end
    '''
    if not fixed_model:
       fixed_model = self.model()
    if not fixed_model or not moving_model:
      return None # nothing to do
    if not fixed_model.crystal_symmetry():
      if self.map_manager().is_dummy_map_manager():
        fixed_model.add_crystal_symmetry_if_necessary()
      else:
        fixed_model.set_crystal_symmetry(self.map_manager().crystal_symmetry())
    if not moving_model.crystal_symmetry():
      moving_model.set_crystal_symmetry(fixed_model.crystal_symmetry())
    self.useful_results = []
    # Check the chain type
    chain_type = self.info.chain_type
    if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = fixed_model,
         return_protein_if_present = True)
    if chain_type.upper() != 'PROTEIN':
      print("Cannot run ssm_superpose on %s" %(chain_type), file = self.log)
      return None
    params = self.get_ssm_params(
     allow_reverse_connectivity = allow_reverse_connectivity,
     shift_field_distance = shift_field_distance,
     starting_distance_cutoff = starting_distance_cutoff,
     trim = trim,
     superpose_brute_force = superpose_brute_force,
     score_superpose_brute_force_with_cc = score_superpose_brute_force_with_cc,
     ok_brute_force_cc = ok_brute_force_cc,
     morph = morph)
    params.search_sequence_neighbors = 0
    params.max_final_triples = 1
    params.refine_cycles = 0   # no map involved
    # Use shift_cart == 0,0,0  as ssm does not keep track of it
    fixed_model_shift_cart = fixed_model.shift_cart()
    fixed_model.set_shift_cart((0,0,0))
    moving_model.set_shift_cart((0,0,0))
    if score_superpose_brute_force_with_cc:
      mmm = self.as_map_model_manager()
      map_manager = mmm.map_manager().deep_copy()
      map_manager.set_original_origin_and_gridding((0,0,0),None) # no shift cart
      from iotbx.map_model_manager import map_model_manager
      mmm_for_scoring = map_model_manager(map_manager = map_manager,
        model = fixed_model, wrapping = map_manager.wrapping())
    else:
      mmm_for_scoring = None
    from phenix.model_building.ssm import run as ssm
    from phenix.model_building.ssm import get_sss_info, \
       identify_structure_elements, compare_model_pair, morph_with_shift_field
    # First get sss and cluster info for each model
    sss_info = get_sss_info(params, model = fixed_model.deep_copy(),
        log = self.log)
    cluster_info = identify_structure_elements(params, sss_info, log = self.log)
    other_sss_info = get_sss_info(params,
        model = moving_model.deep_copy(),
       log = self.log)
    other_cluster_info = identify_structure_elements(params, other_sss_info,
       log = self.log)
    # First try standard run
    from copy import deepcopy
    local_params = deepcopy(params)
    print("Running forward match", file = self.log)
    result = ssm(local_params,
      mmm_for_scoring = mmm_for_scoring,
      model = fixed_model,
      other_model = moving_model,
      cluster_info = cluster_info,
      other_cluster_info = other_cluster_info,
      sss_info = sss_info,
      other_sss_info = other_sss_info,
      log = self.log,
     )
    superposed_full_model = None
    superposed_model = None
    superposed_full_model_list = []
    if result and result.superposed_models:
      print("Forward match successful", file = self.log)
      superposed_full_model = result.superposed_full_models[0]
      superposed_full_model_list = result.superposed_full_models
      superposed_model = result.superposed_models[0]
    if (superposed_full_model is None) and include_swapped_match:
      print("Running swapped match (model and moving_model swapped)",
         file = self.log)
      local_params = deepcopy(params)
      local_params.morph = False
      local_params.superpose_model_on_other = True  # superpose  backwards too
      result = ssm(local_params,
         mmm_for_scoring = mmm_for_scoring,
         model = moving_model,
          other_model = fixed_model,
        cluster_info = other_cluster_info,
        other_cluster_info = cluster_info,
        sss_info = other_sss_info,
        other_sss_info = sss_info,
        log = self.log,
        )
      if not result or not result.superposed_models:
        print("Reverse match also unsuccessful", file = self.log)
        return None # nothing there
      print("Reverse match successful", file = self.log)
      # Now trim and morph reverse match if desired
      moving_model = result.superposed_full_models[0]
      morphed_model = self.morph_to_match_model(
       fixed_model = fixed_model,
       moving_model = moving_model,
       trim = trim,
       try_segment_matching_first = False)
      superposed_full_model = moving_model
      superposed_model = morphed_model
    # Put back shift cart
    fixed_model.set_shift_cart(fixed_model_shift_cart)
    if morph:
      return_model = superposed_model
    else:
      return_model =  superposed_full_model
    if return_model and (return_model.shift_cart() != fixed_model.shift_cart()):
      return_model.set_shift_cart(fixed_model_shift_cart)
    mam = self.as_map_model_manager()
    for ii, full_model in enumerate(superposed_full_model_list):
      if not full_model: continue
      if (full_model.shift_cart() != fixed_model.shift_cart()):
        full_model.set_shift_cart(fixed_model_shift_cart)
      if score_superpose_brute_force_with_cc:
        cc_mask = full_model.info().cc
      else:
        cc_mask = None
      from phenix.model_building.morph_info import get_lsq_fit_from_model_pair
      lsq_fit = get_lsq_fit_from_model_pair(full_model, moving_model)
      ga = group_args(group_args_type = 'superposed model and lsq_fit',
        moving_model = moving_model,
        fixed_model = fixed_model,
        superposed_model = full_model,
        lsq_fit = lsq_fit,
        cc = cc_mask,)
      self.useful_results.append(ga)
      if cc_mask is not None:
        print("Final CC mask for superposed model %s: %.2f" %(ii, cc_mask),
           file = self.log)
    return return_model  # self.useful_results has results
  def ssm_search(self,
    model = None,
    max_models = 1,
    morph = True,
    trim = True,
    quick = None,
    search_sequence_neighbors = None,
    pdb_include_list = None,
    pdb_exclude_list = None,
    exclude_similar_to_this_pdb_id = None,
    maximum_percent_identity_to_this_pdb_id = 30,
    allow_reverse_connectivity = None,
    close_distance = None,
    local_pdb_dir = None,
    shift_field_distance = None,
    starting_distance_cutoff = None,
    max_models_to_fetch = None,
    ):
    '''
     Carry out SSM search against filtered PDB (filtered at 90% seq
      similarity and quality filtered.  Based on Richardson lab
      pdb2018 90% but loops are included (not removed from database).
    Typical use:
      model_list_info = build.ssm_search()
      model_list = model_list_info.model_list
      # find models similar to build.model(), trim and morph them to match
    Optionally morph and trim new model after ssm matching
    Optionally return new model trimmed to matching region only (based on
      close_distance)
    Optionally specify maximum models to return and lists of
    pdb_id (e.g 2DY1) or pdb_id and chain (2DY1_A) to include or not include
    Optionally search sequence neighbors (using PDB100 database)
    If quick is set, do not search full PDB (sequence_neighbors)
    NOTE: setting self.info.nproc > 1 speeds this up a lot
    '''
    if not model:
       model = self.model()
    if not model:
      return None # nothing to do
    model.add_crystal_symmetry_if_necessary()
    # Check the chain type
    chain_type = self.info.chain_type
    if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = model,
         return_protein_if_present = True)
    if chain_type.upper() != 'PROTEIN':
      print("Cannot run ssm-search on %s" %(chain_type), file = self.log)
      return None
    params = self.get_ssm_params(
     allow_reverse_connectivity = allow_reverse_connectivity,
     morph = morph,
     trim = trim,
     close_distance = close_distance,
     shift_field_distance = shift_field_distance,
     max_models_to_fetch = max_models_to_fetch,
     starting_distance_cutoff = starting_distance_cutoff,
     local_pdb_dir = local_pdb_dir)
    if exclude_similar_to_this_pdb_id is not None:
      if not pdb_exclude_list: pdb_exclude_list = []
      pdb_exclude_list += self.get_excluded_pdb_id_list(
        exclude_similar_to_this_pdb_id, maximum_percent_identity_to_this_pdb_id)
    params.pdb_include_list = pdb_include_list
    params.pdb_exclude_list = pdb_exclude_list
    params.max_final_triples = max_models
    params.refine_cycles = 0 # we are not using a map
    params.search_sequence_neighbors = search_sequence_neighbors
    if quick:
      if search_sequence_neighbors is None:
        params.search_sequence_neighbors = 0
    else:
      if search_sequence_neighbors is None:
        params.search_sequence_neighbors = 100
    from phenix.model_building.ssm import run as ssm
    result = ssm(params, model = model, log = self.log)
    if not result or not result.superposed_models:
      return # nothing found
    if trim:
      for m in result.superposed_models: # get rmsd from target
        self.print_similarity(fixed_model = model, superposed_model = m,
          prefix = m.info().get('pdb_text'))
      return group_args(
        group_args_type = 'ssm search results with trimmed superposed models',
         model_list = result.superposed_models,
        )
    else:
      for m in result.superposed_full_models: # get rmsd from target
        self.print_similarity(params = params,
         fixed_model = model, superposed_model = m,
          prefix = m.info().get('pdb_text'))
      return group_args(
        group_args_type = 'ssm search results with full superposed models',
         model_list = result.superposed_full_models,
        )
  def get_excluded_pdb_id_list(self,pdb_id, maximum_percent_identity,
     database = None):
    pdb_info_list = self.structure_search(
      pdb_id = pdb_id, # one pdb to search for
      number_of_models = 10000,  # maximum to return
      number_of_models_per_input_model = None,
      sequence_only = True,
      return_pdb_info_list = True,
      database = database,
      minimum_percent_identity = maximum_percent_identity)
    pdb_exclude_list = []
    for info in pdb_info_list:
      id_string = "%s_%s" %(info.pdb_id,info.chain_id)
      pdb_exclude_list.append(id_string)
    print("Exclude list: %s" %(" ".join(pdb_exclude_list)), file = self.log)
    return pdb_exclude_list
  def get_ssm_params(self,
     allow_reverse_connectivity = None,
     shift_field_distance = None,
     max_models_to_fetch = None,
     max_shift_field_ca_ca_deviation = None,
     starting_distance_cutoff = None,
     close_distance = None,
     morph = None,
     trim = True,
     superpose_brute_force = None,
     score_superpose_brute_force_with_cc = None,
     ok_brute_force_cc = None,
     local_pdb_dir = None):
    from phenix.model_building.ssm import get_ssm_params, set_top2018_filtered
    params = get_ssm_params(shift_field_distance = shift_field_distance,
      max_shift_field_ca_ca_deviation = max_shift_field_ca_ca_deviation,
      starting_distance_cutoff = starting_distance_cutoff,
      trim = trim,
      ok_brute_force_cc = ok_brute_force_cc,
      superpose_brute_force = superpose_brute_force,
      score_superpose_brute_force_with_cc = score_superpose_brute_force_with_cc,
      local_pdb_dir = local_pdb_dir,
      max_models_to_fetch = max_models_to_fetch,
      quick = (self.info.thoroughness == 'quick'))
    params = set_top2018_filtered(params)
    params.nproc = self.info.nproc
    if close_distance is not None:
      params.close_distance = close_distance
    params.reclassify_based_on_distances = None
    params.include_reverse = allow_reverse_connectivity
    params.resolution = self.info.resolution
    params.find_ssm_matching_segments = True
    params.morph = morph
    params.find_loops = False
    params.replace_structure_elements = False
    params.replace_segments_in_model = False # replace everything as specified
    params.write_files = False
    return params
  def fragment_search(self,
      model = None,
      n_res = None,
      max_models = None,
      maximum_insertions_or_deletions = None,
      trim = True,
      pdb_include_list = None,
      pdb_exclude_list = None,
      exclude_similar_to_this_pdb_id = None,
      maximum_percent_identity_to_this_pdb_id = 30,
      max_indel = None,
      n_half_window = None,
      max_gap = None,
      max_segment_matching_ca_ca_deviation = None,
       ):
    '''
    Search pdb_2018_90 for fragment matching specified model and containing
     n_res residues
    If trim is False, return entire matching chain, not just the fragment, and
     do not morph
    Only works for protein chains
    '''
    if not model:
       model = self.model()
    if not model:
      return None # nothing to do
    model.add_crystal_symmetry_if_necessary()
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model,
         return_protein_if_present = True)
    if chain_type.upper() != 'PROTEIN':
      print("Fragment search only works for protein chains", file = self.log)
      return
    if exclude_similar_to_this_pdb_id is not None:
      if not pdb_exclude_list: pdb_exclude_list = []
      pdb_exclude_list += self.get_excluded_pdb_id_list(
        exclude_similar_to_this_pdb_id, maximum_percent_identity_to_this_pdb_id)
    from phenix.model_building.fragment_search import run_find_fragment
    result = run_find_fragment(model = model,
       maximum_insertions_or_deletions = maximum_insertions_or_deletions,
       n_res = n_res,
       pdb_include_list = pdb_include_list,
       pdb_exclude_list = pdb_exclude_list,
       trim = trim,
       nproc = self.info.nproc)
    if not result:
      print("No result for fragment search", file = self.log)
      result = group_args(info_list = [])
    else:
      result.info_list = result.info_list[:max_models]
    model_list = []
    for info in result.info_list:
      if (not trim) and info.full_model:
        m = info.full_model
      else: # usual
        m = info.model
      m.info().rmsd = info.final_rmsd
      model_list.append(m)
    return model_list
  def replace_with_fragments_from_pdb(self,
      model = None,
      mmm = None,  # supply model or map_model_manager with map and model
      sequence = None,
      maximum_insertions_or_deletions = None,
      morph = None,
      search_sequence_neighbors = None,
      max_models_to_fetch = None,
      allow_reverse_connectivity = None,
      refine_cycles = None,
      close_distance = None,
      pdb_include_list = None,
      pdb_exclude_list = None,
      exclude_similar_to_this_pdb_id = None,
      maximum_percent_identity_to_this_pdb_id = 30,
      maximum_residues_to_try_fragment_search = 30,
      try_fragment_search_for_short_models = True,
      shift_field_distance = None,
      starting_distance_cutoff = None,
      local_pdb_dir = None,
       ):
    '''
    Search pdb for fragments matching specified model and build a new model
     consisting of these fragments (and possibly a few from original model)
    '''
    if mmm is None:
      if model:
        mmm = model.as_map_model_manager()
      else:
        mmm = self.as_map_model_manager()
    elif mmm and model:
      mmm = mmm.customized_copy(model_dict = {'model': model})
    if not mmm.model():
      return None # nothing to do
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = mmm.model(),
         return_protein_if_present = True)
    if chain_type.upper() != 'PROTEIN':
      print(
        "Replace with fragments from the PDB only works for protein chains",
         file = self.log)
      return
    mmm.model().add_crystal_symmetry_if_necessary()
    original_model = mmm.model()
    new_model = None
    # First try simple fragment search if model is small
    if try_fragment_search_for_short_models and \
       mmm.model().get_hierarchy().overall_counts().n_residues \
         <= maximum_residues_to_try_fragment_search:
      print("\nTrying simple fragment search as model is short...",
        file = self.log)
      model_list = self.fragment_search(
        model = mmm.model(),
        max_models = 1,
        maximum_insertions_or_deletions = maximum_insertions_or_deletions,
        pdb_include_list = pdb_include_list,
        pdb_exclude_list = pdb_exclude_list,
        exclude_similar_to_this_pdb_id = exclude_similar_to_this_pdb_id,
        maximum_percent_identity_to_this_pdb_id = \
          maximum_percent_identity_to_this_pdb_id,
         )
      if model_list:
        new_model = model_list[0]
        print("\nSimple fragment search successful...", file = self.log)
      else:
        print("\nSimple fragment search unsuccessful...", file = self.log)
    if not new_model:
    # Get a new model
      from phenix.model_building.ssm import run, get_ssm_params
      params = self.get_ssm_params(
       allow_reverse_connectivity = allow_reverse_connectivity,
       morph = morph,
       close_distance = close_distance,
       shift_field_distance = shift_field_distance,
       starting_distance_cutoff = starting_distance_cutoff,
       local_pdb_dir = local_pdb_dir)
      if exclude_similar_to_this_pdb_id is not None:
        if not pdb_exclude_list: pdb_exclude_list = []
        pdb_exclude_list += self.get_excluded_pdb_id_list(
          exclude_similar_to_this_pdb_id,
           maximum_percent_identity_to_this_pdb_id)
      params.search_sequence_neighbors = search_sequence_neighbors
      if max_models_to_fetch is not None:
        params.max_models_to_fetch = max_models_to_fetch
      params.pdb_include_list = pdb_include_list
      params.pdb_exclude_list = pdb_exclude_list
      params.refine_cycles = refine_cycles
      # Set up for replacing everything in model
      params.find_ssm_matching_segments = True # replace with ssm
      params.find_loops = True  # replace loops
      params.replace_structure_elements = True # replace segments
      params.replace_segments_in_model = True # replace everything
      params.write_files = False
      new_model = run(
         params,
         mmm = mmm,
         log = self.log)
      if not new_model or not \
           new_model.get_hierarchy().overall_counts().n_residues:
        print("No result for replace_with_fragments_from_pdb", file = self.log)
        return None
    #  Set the crystal_symmetry in case it was changed.
    mmm.set_model_symmetries_and_shift_cart_to_match_map(new_model)
    #  See if we can replace the side chains (requires a map)
    if not self.map_manager().is_dummy_map_manager():
      model_residues = mmm.model().get_hierarchy().overall_counts().n_residues
      new_residues = new_model.get_hierarchy().overall_counts().n_residues \
        if new_model else 0
      if model_residues == new_residues:
        if not sequence or len(sequence) != model_residues: # use original
          from iotbx.bioinformatics import get_sequence_from_pdb
          sequence = get_sequence_from_pdb(
            hierarchy = mmm.model().get_hierarchy(),
            require_chain_type = False)
      if sequence:
        from mmtbx.secondary_structure.find_ss_from_ca import \
          get_first_resno,get_chain_id
        print("Replacing side chains in rebuilt model with %s residues" %(
          new_model.get_hierarchy().overall_counts().n_residues \
            if new_model else 0),
          file = self.log)
        sequenced_model = self.sequence_from_map(
           model = new_model, sequence = sequence,
           refine_cycles = refine_cycles,
           first_resno = get_first_resno(mmm.model().get_hierarchy()),
           new_chain_id = get_chain_id(mmm.model().get_hierarchy()),
           return_model =True)
        if sequenced_model and \
             sequenced_model.get_hierarchy().overall_counts().n_residues > 0:
          sequenced_model.set_info(new_model.info())
          new_model = sequenced_model
    rmsd_info = self.get_ca_rmsd_with_indel(original_model, new_model,
          as_group_args = True)
    print("Replacement RMSD: %.2f N: %s of %s" %(
            rmsd_info.rmsd,rmsd_info.rms_n,rmsd_info.all_n),
            file = self.log)
    new_model.info().rmsd_info = rmsd_info
    return new_model
  def get_ca_sites(self, model = None):
    ''' Get CA sites from model '''
    if not model:
       model = self.model()
    if not model:
      return None # nothing to do
    ph = model.get_hierarchy()
    from phenix.model_building.fragment_search import \
       get_ca_sites_from_hierarchy
    if self.info.debug:
      return get_ca_sites_from_hierarchy(ph)
    else: # usual
      try:
        return get_ca_sites_from_hierarchy(ph)
      except Exception as e:
        return None # could not get result

  def get_ca_rmsd_with_indel(self, model_x, model_y,
      close_distance = None, chain_type = 'PROTEIN', as_group_args = False):
    ''' Get rmsd between ca sites from x and y, taking closest in each
      assume model_x is fixed model and model_y is moving'''
    if close_distance is None:
      if not chain_type or chain_type.upper() == "PROTEIN":
        close_distance = 3
      else:
        close_distance = 8
    x = self.get_ca_sites(model_x)
    y = self.get_ca_sites(model_y)
    rmsd_info = self.get_rmsd_with_indel(x, y, close_distance = close_distance)
    rmsd = rmsd_info.rmsd_value
    close_n  = rmsd_info.close_n
    rms_n = min(x.size(), y.size())
    if as_group_args:
      return group_args(
        group_args_type = 'rmsd and n',
        rmsd = rmsd,
        rms_n = close_n,
        all_n = rms_n,
        target_n = x.size())
    else:
      return rmsd
  def get_rmsd_with_indel(self, x, y, close_distance = 3):
    ''' Get rmsd between x and y, taking closest in each'''
    from phenix.model_building.fragment_search import get_rmsd_with_indel \
       as get_rmsd
    return get_rmsd(x, y, close_distance = close_distance)
  def superpose_sites_with_indel(self,
        sites_cart = None,
        target_sites = None,
        sites_to_apply_superposition_to = None):
    '''
     Superpose sites_cart onto target_sites, allowing insertions/deletions.
     Then apply result to sites_to_apply_superposition_to
    '''
    from phenix.model_building.fragment_search import \
       superpose_sites_with_indel as superpose
    return superpose(sites_cart = sites_cart,
      target_sites = target_sites,
      sites_to_apply_superposition_to = sites_to_apply_superposition_to)
  def morph_to_match_model(self,
    fixed_model = None,
    moving_model = None,
    try_segment_matching_first = True,
    try_backup_method_on_failure = True,
    max_segment_matching_ca_ca_deviation = None,
    max_shift_field_ca_ca_deviation = None,
    shift_field_distance = None,
    starting_distance_cutoff = None,
    trim = True,
     ):
    '''
    Morph moving_model to match fixed_model
    Typical use:
      morphed_model = build.morph_to_match_model(moving_model = m1,
        fixed_model = m2)
      Returns transformed part of m1 that matches m2
    Typically use segment_matching method; shift_field method allows much
      greater deviations and is suitable for ssm matching
    If trim is False, return entire model, distorted to match (may
      be seriously distorted)
    Methods available:
      segment_matching:  identify shifts along chain that allow superposition
        of fixed_model and moving_model
         Allows limited distortion along chain.  Returns parts that match
         Key parameter: max_segment_matching_ca_ca_deviation = 0.8
            defines maximum CA-CA deviation
      shift_field:  create and apply shift field based on pairwise matching
          of CA atoms.  Allows large distortions along chain (2.75 A) and
          allows more than one part of model to match.
          Key parameters:
            shift_field_distance = 10
              defines fall-off distance for shift field effects
            max_shift_field_ca_ca_deviation = 2.75
              defines final CA-CA deviations allowed (residue pairs with larger
               deviations are simply removed at the end)
      If first method tried fails, try the other if try_backup_method_on_failure
      NOTE: models must be approximately superimposed already.  Use
        ssm_superpose to superpose models that are very different.
      Returns only the segment of moving_model that matches.
      You can increase the part returned by increasing
        max_segment_matching_ca_ca_deviation for segment_matching or
        max_shift_field_ca_ca_deviation for shift_field
    '''
    if not fixed_model:
       fixed_model = self.model()
    if not fixed_model:
      return None # nothing to do
    if not moving_model:
      return None
    fixed_model.add_crystal_symmetry_if_necessary()
    moving_model.add_crystal_symmetry_if_necessary()
    replacement_segment = None
    method_list = ['segment_matching','shift_field']
    if not try_segment_matching_first:
      method_list.reverse()
    if not try_backup_method_on_failure:
      method_list = method_list[:1]  # just try one thing
    for method in method_list:
      if replacement_segment:
        break  # already done
      elif method == 'segment_matching':
        from phenix.model_building.fragment_search import get_distort_params
        distort_params = get_distort_params(close_distance = 3,
         max_segment_matching_ca_ca_deviation = \
            max_segment_matching_ca_ca_deviation)
        from phenix.model_building.fragment_search import \
           morph_with_segment_matching
        replacement_info = morph_with_segment_matching(
          params = distort_params,
          model = moving_model,
          match_model = fixed_model,
          return_replacement_fragments_only = True,
          trim = trim,
          )
        if trim: # usual
          if replacement_info and replacement_info.replacement_info_list and \
             len(replacement_info.replacement_info_list) >= 1:
            from phenix.model_building.fit_loop import catenate_segments
            replacement_segment = \
               replacement_info.replacement_info_list[0].replacement_segment
            for r in replacement_info.replacement_info_list[1:]:
              replacement_segment = catenate_segments(
                 replacement_segment, r.replacement_segment)
        elif replacement_info:
          replacement_segment = replacement_info.full_selected_model
      else:
        ssm_params = self.get_ssm_params(
          morph = True,
          shift_field_distance = shift_field_distance,
          max_shift_field_ca_ca_deviation = max_shift_field_ca_ca_deviation,
          starting_distance_cutoff = starting_distance_cutoff,
          trim = trim,
           )
        from phenix.model_building.ssm import morph_with_shift_field
        mmi= morph_with_shift_field(ssm_params,
          moving_model = moving_model,
          fixed_model = fixed_model,
          log = self.log)
        if mmi:
          replacement_segment = mmi.superposed_moving_model
    return replacement_segment
  def fetch_pdb(self,
    pdb_id = None,
    chain_id = None,
    pdb_info = None,
    pdb_info_list = None,
    local_pdb_dir = None):
    '''
     Get model from PDB  based on pdb_id
     Supply identification as pdb_info group_args object or list of them.
     You can set one up like this:
       from phenix.model_building import get_pdb_info
       pdb_info = get_pdb_info(pdb_id = pdb_id, chain_id = chain_id)
     You can also specify a single pdb_id
    You can supply a local_pdb_dir containing a mirror of the PDB as well
    '''
    if pdb_id:
      pdb_info = get_pdb_info(pdb_id = pdb_id, chain_id = chain_id)
    if pdb_info:
      info = fetch_one_pdb(pdb_info = pdb_info,
        local_pdb_dir = local_pdb_dir,
        log = self.log)
      if info.model_list:
        return info.model_list[0]
      else:
        return None
    elif pdb_info_list:
      info = fetch_group_of_pdb(pdb_info_list = pdb_info_list,
         nproc = self.info.nproc,
         local_pdb_dir = local_pdb_dir,
         log = self.log)
      if info.model_list:
        return info.model_list
      else:
        return None
  def structure_search(self,
    model_list = None,  # list of models to be similar to
    pdb_info_list = None,  # list of pdb info to be similar to
    pdb_id = None, # one pdb to search for
    sequence_only = False,
    structure_only = False,
    number_of_models = None,  # total models
    number_of_models_per_input_model = 10, # models per input model
    return_pdb_info_list = False,
    database = None, # default is 'pdb100aa' (pdb unique chains)
    minimum_percent_identity = None,
    maximum_percent_identity = None,
    local_pdb_dir = None,
    ):
    '''
      Find a list of models similar to model/pdb_id
      If return_pdb_info_list, return list of pdb_info objects suitable for
       supplying to fetch_pdb (do not actually get the files).
    '''
    if pdb_id and not pdb_info_list:
      pdb_info_list = [
         group_args(
           group_args_type = 'pdb info',
           pdb_id = pdb_id,
           chain_id = None,
           selection_string = None,
         )]
    if pdb_info_list and not model_list:
      model_list = self.fetch_pdb(pdb_info_list = pdb_info_list,
      local_pdb_dir = local_pdb_dir)
    if not model_list:
      if return_pdb_info_list:
        return []
      else:
        return group_args(
          group_args_type = 'structure search results',
          model_list = [],
         )
    return run_structure_search(model_list,sequence_only,structure_only,
      number_of_models,
      number_of_models_per_input_model,
      return_pdb_info_list = return_pdb_info_list,
      nproc = self.info.nproc,
      minimum_percent_identity = minimum_percent_identity,
      maximum_percent_identity = maximum_percent_identity,
      database = database,
      log = self.log)
  def morph(self, model = None,
    method_type = 'morph',
    default_selection_method = 'by_segment',
    selection_list = None,
    skip_hetero = False,
    rigid_body = False,
    morph_main = True,
    iterative_morph = False,
    n_window = 6,
    return_model = None,
    ):
    '''
      Morph working model, or supplied model, based on map
      If working model, replace
      it with morphed model unless return_model is True.
      If supplied model, return morphed model.
      if return_model=True, return new model and do not modify existing model
      it with morphed model. If supplied model, return refined model.
      If list of selections supplied, use them for morphing (remainder
      remains fixed)
      Otherwise use default_selection method (by_chain, by_segment, all)
      Default is by_segment (preferable to by_chain because ligands will
      be separated from chains and broken chains will be morphed separately.)
      Waters are always skipped
      If skip_hetero,  hetero atoms (ligands etc but also any hetero atoms
      in a chain) are skipped (not moved).
      If rigid_body, each selection is moved as a fixed unit.
      If morph_main, protein/rna/dna are morphed based on main-chain positions
       and CB/N1'
      Iterative morph tries to morph each segment iteratively starting from both
        ends at once. Uses n_window for smoothing.
    '''
    assert default_selection_method in ['by_chain', 'by_segment','all']
    model,return_model = self._set_and_print_return_info(
      model=model,text='Morphing',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    if not model or not model.get_sites_cart().size():
      print("No model to work with...", file = self.log)
      return
    print("\nMorph model " ,file=self.log)
    map_model_manager = MapModelManager(model = model,
         map_manager= self.map_manager())
    if not selection_list:
      from iotbx.map_model_manager import \
         get_selections_and_boxes_to_split_model
      selection_info = get_selections_and_boxes_to_split_model(
        map_model_manager = map_model_manager,
        skip_waters = True,  # always, as resolve does not recognize them
        skip_hetero = skip_hetero,  # optional
        selection_method = default_selection_method)
      selection_list = selection_info.selection_list
    print ("Selections for groups to morph:", file = self.log)
    for sel in selection_list:
      n_atoms = model.select(sel).get_sites_cart().size()
      print ("   %s atoms" %(n_atoms), file = self.log)
    tries_for_each_method = len(selection_list)
    morph_methods = ['resolve']
    # Ready to morph model
    # Define the arguments that are going to be passed to any fit_ligand method
    common_kw = {
     'map_model_manager':    map_model_manager,  # working map and model
     'resolution':self.info.resolution,            # resolution of map
     'selection_list':selection_list,              # list of selections
     'skip_hetero':skip_hetero,                   # skip hetero atoms
     'rigid_body':rigid_body,              # move each selection as rigid body
     'morph_main':morph_main,              # move based on main-chain+CB/N1'
     'iterative_morph':iterative_morph,     # iteration
     'n_window':n_window,     # iteration window
     'good_enough_score':1.,                     # score to quit (never)
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
      }
    # Define the entries in the group_args result object that any
    #   method is supposed to return
    expected_result_names = [
       'model',                # morphed model , as a model object
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatically generated)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  None  # always do everything
    # Now we are ready to run each method in morph_methods using the kw in
    # common_kw (common to all runs)
    self._run_group_of_methods(
     methods_list = self.info.morph_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
     skip_finding_best = True,  # don't need a best one
      )
    # Now put together all the models
    models_dict = {}
    for result in self.get_results():
      map_model_manager.shift_any_model_to_match(result.model) # shift in place
      models_dict[result.run_id] = result.model
    run_id_list=list(list(models_dict.keys()))
    run_id_list.sort()
    xrs = model.get_xray_structure()
    working_sites_cart = xrs.sites_cart()
    for run_id in run_id_list:
      new_sites_cart = models_dict[run_id].get_sites_cart()
      sel=selection_list[run_id-1]
      working_sites_cart.set_selected(sel,new_sites_cart)
    model.set_sites_cart(working_sites_cart)
    return self._set_or_return_model(model,
       text="morphed model",return_model=return_model,
       change_model_in_place = True)
  def fit_ligand(self,
    model = None,
    ligand_model = None,
    ligand_code= None,
    ligand_smiles_string = None,
    n_template_atom = None,
    restraints_object = None,
    use_existing_masked_map = False,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    conformers = None,
    refine_cycles = None,
    fit_ligand_methods = None,
    remove_waters = True,
    chain_id = 'A',
    method_type = 'fit_ligand',
     ):
    '''
     Fit a ligand into unused density in the map.
     Ligand can be a ligand_model, a 3-character code (ATP), or a smiles string
     Ligand can have multiple conformations. If so, specify n_template_atom
       to indicate the first n_template_atom atoms are the unique ligand
     Map will be masked around the existing or supplied model and then
     ligand found in remaining biggest region of density
     If use_existing_masked_map is set then use that map
     If fit_ligand_methods is set use that instead of
         self.info.fit_ligand_methods
     returns a ligand_model unless return_model = False in which case it is
     inserted in existing model
     If restraints_object is not specified, any already-specified restraints
     objects are used
     Ligand chain_id will be chain_id
    '''
    # Checks
    if fit_ligand_methods:
      for x in fit_ligand_methods:
        assert x in ALLOWED_FIT_LIGAND_METHODS
    else:
      fit_ligand_methods = ALLOWED_FIT_LIGAND_METHODS
    model,return_model = self._set_and_print_return_info(
      model=model,text='ligand',return_model=True,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    # Default for quick:
    if (conformers is None):
      if self.info.thoroughness == 'quick':
        conformers = 1
      else:
        conformers = 20
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    if ligand_model and (n_template_atom is not None) and \
      (ligand_model.get_sites_cart().size() >= 2 * n_template_atom):
      print (
       "Assuming input ligand is multi-conformer ligand with %s unique atoms" %(
          n_template_atom),file=self.log)
      conformers_to_generate = 1  # do not make any more conformers
    else:
      conformers_to_generate = conformers
    if (not restraints_object) and (ligand_model) and self.restraints_dict.get(
     ligand_model,None):
      restraints_object = self.restraints_dict.get(ligand_model,None)
    # Get the ligand and (if needed) restraints object
    print ("\nGetting ligand model with %s conformers..." %(
      conformers_to_generate),file=self.log)
    from phenix.model_building.fit_ligand import get_ligand_model_from_any
    ligand_info = get_ligand_model_from_any(
      ligand_model,
      ligand_code,
      ligand_smiles_string,
      conformers = conformers_to_generate,
      restraints_object = restraints_object,
      chain_id = chain_id)
    ligand_model = ligand_info.model
    restraints_object = ligand_info.restraints_object
    if not ligand_model in list(self.restraints_dict.keys()):
      self.restraints_dict[ligand_model] = restraints_object
    # Number of unique atoms in ligand template (may be multiple copies of
    #   conformers)
    n_template_atom = ligand_model.get_sites_cart().size()/max(1,conformers)
    #  Get a masked map
    if not self.model():
      self._masked_map_manager = self.map_manager()
    elif use_existing_masked_map and \
      self.masked_map_manager():
        pass  # ok as is
    elif self.model():
      model = self.model()
      if remove_waters:
        model = self.remove_waters_from_model(model)
      #  Create map masked to only include density outside existing model
      self._masked_map_manager=self.create_masked_map(
         model = model,
         soft_mask = False)
    if self.info.debug:
      self.masked_map_manager().write_map('masked_map_for_ligand.ccp4')
    # Ready to fit ligand_model into masked_map_manager. Return a new
    #  ligand_model that knows (through shift_cart()) its coordinates in
    #  original reference frame
    # Define the arguments that are going to be passed to any fit_ligand method
    common_kw = {
     'map_manager':self.masked_map_manager(),   # the map
     'model':self.model(),                # existing model
     'ligand_model':ligand_model,             # ligand model, any location
     'n_template_atom':n_template_atom,       # unique atoms in ligand
     'resolution':self.info.resolution,            # resolution of map
     'restraints_object':restraints_object,   # any restraints for ligand
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object that any fit_ligand
    #   method is supposed to return
    expected_result_names = [
       'model',                # fitted ligand, as a model object
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatically generated)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in fit_ligand_methods using the kw in
    # common_kw (common to all runs)
    new_ligand = self._run_group_of_methods(
     methods_list = fit_ligand_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    if not new_ligand:
      print ("No ligand found...skipping", file = self.log)
    print ("Returning ligand", file = self.log)
    return new_ligand
  def set_refine_cycles(self, refine_cycles):
    if refine_cycles is None:
      if self.info.thoroughness == 'quick':
        refine_cycles = 1
      else:
        refine_cycles = 3
    return refine_cycles
  def regularize_model(self,
    model = None,
    sequence = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    refine_cycles = None,
    method_type = 'regularize_model',
    remove_clashes = None,
    allow_insertions_deletions = False,
    return_model = None,
     ):
    '''
     Try to regularize a model keeping close to density in map
     Add side chains based on supplied sequence, otherwise guess them
      If supplied model, return regularized model.
      if return_model=True, return new model and do not modify existing model
     These methods only need one try, so tries_for_each_method will be 1 if
      not set
      NOTE: Changes model in place.
    '''
    # Checks
    # Set default for tries_for_each_method
    if tries_for_each_method is None:
      tries_for_each_method = 1
    model,return_model = self._set_and_print_return_info(
      model=model,text='Regularizing',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    if model is None or model.get_sites_cart().size()<1:
      print("\nNo model available in regularize_model", file = self.log)
      return
    # Set the chain type and make sure there is just one
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    # Ready to regularize a model
    # Define arguments that are going to be passed to regularize_model methods
    common_kw = {
     'map_manager':self.map_manager(),   # the map
     'model':model,                # existing model
     'sequence':sequence,               # sequence (1-letter code)
     'chain_type':chain_type,               # chain_type (rna/dna/protein)
     'resolution':self.info.resolution,            # resolution of map
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'remove_clashes':remove_clashes, # Remove clashing side-chains
     'allow_insertions_deletions':allow_insertions_deletions, # length changes
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object the
    #   method is supposed to return
    expected_result_names = [
       'model',          # fitted regularize_model
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatically generated)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in regularize_model_methods using
    #  the kw in common_kw (common to all runs)
    new_model = self._run_group_of_methods(
     methods_list = self.info.regularize_model_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    if not new_model:
      print ("\nRegularize model failed...nothing done",file = self.log)
      return
    # Now add side chains with sequence_from_map
    sequenced_model = self.sequence_from_map(
      model = new_model,
      sequence = sequence,
      refine_cycles = refine_cycles,
      return_model = True,
     )
    if sequenced_model:
      new_model = sequenced_model
    return self._set_or_return_model(new_model,
       text="regularized model",return_model=return_model,
       change_model_in_place = True)
  def sequence_from_map(self,
    model = None,
    sequence = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    refine_cycles = None,
    remove_clashes = True,
    keep_segment_order = None,
    method_type = 'sequence_from_map',
    split_using_alignment = None,
    first_resno = None,
    new_chain_id = None,
    return_model = None,
     ):
    '''
     Identify the sequence of a chain or a set of chains.
     If sequence is supplied, find best fit of sequence to the chain(s), build
       side chains, refine, and return full model
     If no sequence, guess the sequence, build side chains, refine, return full
       model
      Sequence working model, or supplied model.  If working model, replace
      it with sequenced model unless return_model is True.
      If supplied model, return sequenced model.
      If first_resno is set, start sequence at that residue number
      If new_chain_id is set, start sequence in that chain_id
      if return_model=True, return new model and do not modify existing model
     If keep_segment_order, segments will be kept in original order
     These methods only need one try, so tries_for_each_method will be 1 if
      not set
    '''
    # Checks
    # Set default for tries_for_each_method
    if tries_for_each_method is None:
      tries_for_each_method = 1
    model,return_model = self._set_and_print_return_info(
      model=model,text='Sequencing',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    if not sequence:
      sequence = self.info.sequence
    if sequence:
      print("Running sequence from map with a sequence...fit sequence to model",
        file = self.log)
    else:
      print("Running sequence from map without a sequence: guess the sequence",
        file = self.log)
    if model is None or model.get_sites_cart().size()< 1:
      print("\nNo model available in sequence_from_map", file = self.log)
      return
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    # Set the chain type and make sure there is just one
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    # Ready to sequence a model
    # Define arguments that are going to be passed to sequence_from_map methods
    common_kw = {
     'map_manager':self.map_manager(),   # the map
     'model':model,                # existing model
     'sequence':sequence,               # sequence (1-letter code)
     'chain_type':chain_type,               # chain_type (rna/dna/protein)
     'resolution':self.info.resolution,            # resolution of map
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'remove_clashes':remove_clashes, # Remove clashing side-chains
     'keep_segment_order':keep_segment_order, #Keep order of segments
     'split_using_alignment':split_using_alignment, # break or not
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object the
    #   method is supposed to return
    expected_result_names = [
       'model',          # fitted sequence_from_map
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatically generated)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in sequence_from_map_methods using
    #  the kw in common_kw (common to all runs)
    new_model = self._run_group_of_methods(
     methods_list = self.info.sequence_from_map_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    if not new_model:
      print ("\nSequence from map failed...nothing done",file = self.log)
      return
    # Set the chain_id and sequence if desired
    ph = new_model.get_hierarchy()
    from mmtbx.secondary_structure.find_ss_from_ca import \
      get_first_resno,get_chain_id
    if (new_chain_id is not None and new_chain_id != get_chain_id(ph)) or \
       (first_resno is not None and first_resno != get_first_resno(ph)):
      if new_chain_id:
        print("Setting chain ID to %s" %(new_chain_id), file = self.log)
        from mmtbx.secondary_structure.find_ss_from_ca import set_chain_id
        set_chain_id(ph, chain_id = new_chain_id)
      if first_resno:
        print("Setting first resno to %s" %(first_resno), file = self.log)
        from mmtbx.secondary_structure.find_ss_from_ca import renumber_residues
        renumber_residues(ph, first_resno = first_resno)
      new_model = self.as_map_model_manager().model_from_hierarchy(
          ph, return_as_model = True)
    return self._set_or_return_model(new_model,
       text="sequenced model",return_model=return_model,
       change_model_in_place = True)
  def extend_forward(self,
    chain_id = None,
    last_resno_before_extend = None,
    target_number_of_residues_to_build = 10,
    model = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    extend_methods = None,
    refine_cycles = None,
    guide_model = None):
    '''
      Extend a chain in forward direction
      Start at C-terminal (3') end and extend.  If last_resno_before_extend is
        set, use that residue instead as starting point
      Density will be masked around existing model
      Returns the new residues that have been built plus three existing residues
        on the end
      Scoring will be map_model_cc * sqrt (residues_built/
          target_number_of_residues_to_build)
      extend_methods overrides defaults
      If guide_model are supplied and trace_through_density is
      specified, try using pieces of them to guide the path
    '''
    model,return_model = self._set_and_print_return_info(
      model=model,text='chain in forward direction',return_model=True,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    from mmtbx.secondary_structure.find_ss_from_ca import \
      get_first_resno,get_last_resno
    first_resno_after_extend = None
    if last_resno_before_extend is None:
      last_resno_before_extend = get_last_resno(model.get_hierarchy())
    last_possible_resno_before_extend = get_last_resno(model.get_hierarchy())
    if last_resno_before_extend is not None and (
       last_resno_before_extend > last_possible_resno_before_extend):
      last_resno_before_extend = last_possible_resno_before_extend
    elif last_resno_before_extend is None:
      last_resno_before_extend = last_possible_resno_before_extend
    print("Using residue %s as last_resno_before_extend " %(
      last_resno_before_extend), file = self.log)
    available_sequence = None
    if last_resno_before_extend == last_possible_resno_before_extend and \
        self.info.sequence:
      loop_info_list = get_loop_info_list(model, self.info.sequence,
        min_residues_in_loop = 0,
        max_residues_in_loop = 1000000,
        include_ends = True)
      if loop_info_list and loop_info_list[-1].first_resno_after_loop is None:
        # ok...use it
        end_info = loop_info_list[-1]
        available_sequence = end_info.loop_sequence
    new_extension = self._extend(
      chain_id = chain_id,
      last_resno_before_extend = last_resno_before_extend,
      first_resno_after_extend = first_resno_after_extend,
      target_number_of_residues_to_build = target_number_of_residues_to_build,
      available_sequence = available_sequence,
      model = model,
      tries_for_each_method = tries_for_each_method,
      good_enough_score = good_enough_score,
      refine_cycles = refine_cycles,
      extend_methods = extend_methods,
      guide_model = guide_model)
    if not new_extension:
      print("No extension found...", file = self.log)
    else:
      print("Returning new extension", file = self.log)
      return new_extension
  def extend_reverse(self,
    chain_id = None,
    first_resno_after_extend = None,
    target_number_of_residues_to_build = 10,
    model = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    extend_methods = None,
    refine_cycles = None,
    guide_model = None):
    '''
      Extend a chain in reverse direction
      Start at N-terminal (5') end and extend backwards.  If
      first_resno_after_extend is set, use that residue instead as
        starting point
      Density will be masked around existing model
      Returns the new residues that have been built plus three existing residues
        on the end
      Scoring will be map_model_cc * sqrt (residues_built/
          target_number_of_residues_to_build)
      extend_methods overrides defaults
      If guide_model are supplied and trace_through_density is
      specified, try using pieces of them to guide the path
    '''
    model,return_model = self._set_and_print_return_info(
      model=model,text='chain in reverse direction',return_model=True,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    from mmtbx.secondary_structure.find_ss_from_ca import \
      get_first_resno,get_last_resno
    last_resno_before_extend =None
    first_possible_resno_after_extend = get_first_resno(model.get_hierarchy())
    if first_resno_after_extend is not None and (
       first_resno_after_extend < first_possible_resno_after_extend):
      first_resno_after_extend = first_possible_resno_after_extend
    elif first_resno_after_extend is None:
      first_resno_after_extend = first_possible_resno_after_extend
    print("Using residue %s as first_resno_after_extend " %(
      first_resno_after_extend), file = self.log)
    available_sequence = None
    if first_resno_after_extend == first_possible_resno_after_extend and\
        self.info.sequence:
      loop_info_list = get_loop_info_list(model, self.info.sequence,
        min_residues_in_loop = 0,
        max_residues_in_loop = 1000000,
        include_ends = True)
      if loop_info_list and loop_info_list[0].last_resno_before_loop is None:
        # ok...use it
        end_info = loop_info_list[0]
        available_sequence = end_info.loop_sequence
    new_extension = self._extend(
      chain_id = chain_id,
      last_resno_before_extend = last_resno_before_extend,
      first_resno_after_extend = first_resno_after_extend,
      target_number_of_residues_to_build = target_number_of_residues_to_build,
      available_sequence = available_sequence,
      model = model,
      tries_for_each_method = tries_for_each_method,
      good_enough_score = good_enough_score,
      refine_cycles = refine_cycles,
      extend_methods = extend_methods,
      guide_model = guide_model)
    if not new_extension:
      print("No extension found...", file = self.log)
    else:
      print("Returning new extension", file = self.log)
      return new_extension
  def _extend(self,
    chain_id = None,
    last_resno_before_extend = None,
    first_resno_after_extend = None,
    available_sequence = None,
    target_number_of_residues_to_build = 10,
    model = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    refine_cycles = None,
    guide_model = None,
    extend_methods = None,
    method_type = 'extend',
):
    '''
      Extend a chain
      Normally use extend_forward() and extend_reverse() which call this method.
      Start after residue last_resno_before_extend in chain_id and extend
         forwards, or start before residue first_resno_after_extend and extend
         backwards.  Only one of these can be defined.
      Density will be masked around existing model
      Returns the new residues that have been built plus three existing residues
        on the end
      Scoring will be map_model_cc * sqrt (residues_built/
          target_number_of_residues_to_build)
      If guide_model are supplied and trace_through_density is
      specified, try using pieces of them to guide the path
    '''
    # Check methods
    if extend_methods is None:
      extend_methods = self.info.extend_methods
    else: # use supplied methods
      assert type(extend_methods) == type([1,2,3])
      for x in extend_methods:
        assert x in ALLOWED_EXTEND_METHODS # extend methods
    if model:
      assert self.map_manager().is_compatible_model(model)
    else:
      model = self.model()
    if not model:
      raise Sorry("Need model for extend")
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    original_last_resno_before_extend = last_resno_before_extend
    original_first_resno_after_extend = first_resno_after_extend
    assert (last_resno_before_extend,first_resno_after_extend).count(None)==1
    # Initialize extend object
    self.extension_info = None
    # Set default for tries_for_each_method
    if tries_for_each_method is None:
      tries_for_each_method = 1
    # Set the chain type and make sure there is just one
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    print("\nExtend chain with %s model" %(chain_type),file=self.log)
    if chain_id is None:
      from mmtbx.secondary_structure.find_ss_from_ca import get_chain_ids
      chain_ids = get_chain_ids(model.get_hierarchy())
      if len(chain_ids) != 1:
        raise Sorry("Please supply a chain ID if model has multiple chains")
      chain_id = chain_ids[0]
    # Make sure the selection is going to work:
    from phenix.model_building.fit_loop import get_residues_around_loop
    info = get_residues_around_loop(
      model,
      chain_id,
      last_resno_before_extend,
      first_resno_after_extend,
      residues_on_each_end = self.info.residues_on_each_end,
      residues_on_each_end_extended = self.info.residues_on_each_end_extended)
    # Mask out density that is near the model for everything except the
    #  residues starting with three before and going through three after
    from phenix.model_building.fit_loop import \
       use_remainder_of_chain_if_no_loop_end
    last_resno_before_extend, first_resno_after_extend =\
       use_remainder_of_chain_if_no_loop_end(model.get_hierarchy(),
       last_resno_before_extend, first_resno_after_extend,
       residues_on_each_end = self.info.residues_on_each_end)
    selection="chain %s and resseq %s:%s" %(
       chain_id,last_resno_before_extend- (self.info.residues_on_each_end - 1),
      first_resno_after_extend+(self.info.residues_on_each_end - 1))
    self._masked_map_manager = self.create_masked_map(
      model=model.select(~model.selection(selection)))
    # Ready to extend
    #  Return a new model that knows (through shift_cart()) its coordinates in
    #  original reference frame
    # Define arguments that are going to be passed to extend methods
    common_kw = {
     'map_manager':self.masked_map_manager(),   # the map
     'model':model,                      # the working model
     'guide_model':guide_model,  # any fragments to guide (trace_through_density only)
     'chain_id':chain_id,                     # chain ID
     'last_resno_before_extend':last_resno_before_extend,
     'first_resno_after_extend':first_resno_after_extend,
     'residues_on_each_end':self.info.residues_on_each_end,
     'available_sequence':available_sequence,
     'target_number_of_residues_to_build':target_number_of_residues_to_build,
     'chain_type':chain_type,               # chain_type (rna/dna/protein)
     'resolution':self.info.resolution,            # resolution of map
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object the
    #   method is supposed to return
    expected_result_names = [
       'model',               # extension, including 2 residues on end
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatic)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in extend_methods using
    #  the kw in common_kw (common to all runs)
    self._run_group_of_methods(
     methods_list = extend_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    result = self.get_best_result()
    if not result:
      return None
    # We have an extension
    extension = result.model
    if hasattr(result,'residues_on_each_end') and result.residues_on_each_end:
      residues_on_each_end = result.residues_on_each_end
    else:
      residues_on_each_end = self.info.residues_on_each_end
    # Save the extension object
    self.extension_info = group_args(
      extension_with_n_residues_on_end = extension,
      chain_id = chain_id,  # where the loop goes
      last_resno_before_extend = original_last_resno_before_extend,
      first_resno_after_extend = original_first_resno_after_extend,
      number_of_residues_in_extension =
         extension.get_hierarchy().overall_counts().n_residues,
      residues_on_each_end = residues_on_each_end,
    )
    # Return result (best model)
    return self.get_best_model()
  def match_and_insert_extension(self, model = None,
    return_model = None,
    target_extension = None,
    c_term = False):
    """ Insert last extension, but make it match number of residues in
        target_extension
    """
    if not self.extension_info:
      print ("No extension to insert", file = self.log)
      return
    if (not target_extension): # just do usual
      return self.insert_extension(model = model, return_model = return_model)
    extension = self.extension_info.extension_with_n_residues_on_end
    ph = extension.get_hierarchy()
    n_target = target_extension.get_hierarchy().overall_counts().n_residues
    n_avail = ph.overall_counts().n_residues
    from mmtbx.secondary_structure.find_ss_from_ca import \
      get_first_resno,get_last_resno
    if n_avail == n_target:
      new_insert = extension
    elif n_avail > n_target:  # trim back
      if c_term:  # remove last residues
        first_resno = get_first_resno(ph)
        last_resno = first_resno + n_target - 1
      else:
        last_resno = get_last_resno(ph)
        first_resno = last_resno - n_target + 1
      new_insert = extension.apply_selection_string(
          'resseq %s:%s' %(first_resno, last_resno))
    else:  # add residues on end...superimpose CA of existing residues with
         # extra weight on first/last residue
      ext_ca_sc = extension.apply_selection_string(
          'name ca or name p').get_sites_cart()
      if c_term:  # add after last residue
        target_ca_sc = target_extension.apply_selection_string(
          'name ca or name p').get_sites_cart()[:ext_ca_sc.size()]
        last_ca_list  = flex.vec3_double(100,ext_ca_sc[-1])
        target_last_ca_list  = flex.vec3_double(100,target_ca_sc[-1])
        ext_ca_sc.extend(last_ca_list)
        target_ca_sc.extend(target_last_ca_list)
        target_end_end = get_last_resno(target_extension.get_hierarchy())
        target_end_start = target_end_end - (n_target - n_avail - 1)
      else:
        target_ca_sc = target_extension.apply_selection_string(
          'name ca or name p').get_sites_cart()[-ext_ca_sc.size():]
        first_ca_list  = flex.vec3_double(100,ext_ca_sc[0])
        target_first_ca_list  = flex.vec3_double(100,target_ca_sc[0])
        ext_ca_sc.extend(first_ca_list)
        target_ca_sc.extend(target_first_ca_list)
        target_end_start = get_first_resno(target_extension.get_hierarchy())
        target_end_end = target_end_start + (n_target - n_avail - 1)
      lsq_fit_obj = get_lsq_fit(ext_ca_sc,target_ca_sc)
      # Now apply to end of target
      target_end = target_extension.apply_selection_string('resseq %s:%s'
         %(target_end_start, target_end_end))
      fitted_target_end = apply_lsq_fit_to_model(lsq_fit_obj, target_end)
      # and tack fitted_target_end on to extension
      from phenix.model_building.fit_loop import catenate_segments
      new_insert = catenate_segments(extension, fitted_target_end)
    assert new_insert.get_hierarchy().overall_counts().n_residues == \
       target_extension.get_hierarchy().overall_counts().n_residues
    #  Now make sure the sequence is right
    from iotbx.bioinformatics import get_sequence_from_pdb
    segment_sequence = get_sequence_from_pdb(
        hierarchy = target_extension.get_hierarchy())
    # XXX see if we can move this earlier...
    new_insert = self.sequence_from_map(return_model = True,
      model = new_insert,
      sequence = segment_sequence)
    if new_insert is None:
      return None # failed
    self.extension_info.extension_with_n_residues_on_end = new_insert
    return self.insert_extension(model = model, return_model = return_model)
  def insert_extension(self, model = None,
    return_model = None):
    '''
      Insert the last extension (if any) into supplied model.
      Modifies the supplied model
      If no model is supplied, the working model will be modified
        and nothing is returned (model is modified directly)
    '''
    if not self.extension_info:
      print ("No extension to insert", file = self.log)
      return
    model,return_model = self._set_and_print_return_info(
      model=model,text='Inserting extension',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    from mmtbx.secondary_structure.find_ss_from_ca import get_first_resno, get_last_resno
    from phenix.model_building.extend import insert_extend_in_model
    insert_extend_in_model(
      model = model,
      extension_to_insert_with_n_residues_on_end = \
         self.extension_info.extension_with_n_residues_on_end,
      chain_id = self.extension_info.chain_id,
      last_resno_before_extend = self.extension_info.last_resno_before_extend,
      first_resno_after_extend = self.extension_info.first_resno_after_extend,
      residues_on_each_end = self.extension_info.residues_on_each_end,
     )
    return self._set_or_return_model(model,text='model with extension',
       return_model=return_model,
       change_model_in_place = True)
  def fit_all_loops(self,
    model = None,
    max_residues_in_loop = 30,
    min_residues_in_loop = 5,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    remove_clashes = True,
    refine_cycles = None,
    fit_loop_methods = None,
    return_model = None,
    method_type = 'fit_loop',
    ):
    ''' Try to fit all gaps in supplied model.
       If gap is less than min_residues_in_loop, trim back'''
    if not self.info.sequence:
      print("No sequence...needed for fit_all_loops", file = self.log)
      return
    model,return_model = self._set_and_print_return_info(
      model=model,text='fit_loop',return_model=return_model,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    if model is None or model.get_sites_cart().size()<1:
      print ("No model to work with...", file = self.log)
      return
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    loop_info_list = get_loop_info_list(model, self.info.sequence,
      min_residues_in_loop = min_residues_in_loop,
      max_residues_in_loop = max_residues_in_loop)
    if not loop_info_list or len(loop_info_list) == 0:
      print("No loops to fit", file = self.log)
      return
    n_succeeded = 0
    for loop_info in loop_info_list:
      # Try to fit this loop (should not change numbering)
      fitted_loop = self.fit_loop(model = model,
       chain_id = loop_info.chain_id,
       last_resno_before_loop = loop_info.last_resno_before_loop,
       first_resno_after_loop = loop_info.first_resno_after_loop,
       loop_sequence = loop_info.loop_sequence,
       number_of_residues_in_loop = \
         loop_info.first_resno_after_loop -loop_info.last_resno_before_loop - 1,
       tries_for_each_method = tries_for_each_method,
       good_enough_score = good_enough_score,
       remove_clashes = remove_clashes,
       refine_cycles = refine_cycles,
       fit_loop_methods = fit_loop_methods)
      if fitted_loop:
        model = self.insert_fitted_loop(model = model, return_model = True)
        n_succeeded += 1
    print("Total of %s of %s loops fitted " %(
      n_succeeded, len(loop_info_list)), file = self.log)
    return self._set_or_return_model(model,text="model with inserted loops",
       return_model=return_model,
       change_model_in_place = True)
  def fit_loop(self,
    chain_id = None,
    last_resno_before_loop = None,
    first_resno_after_loop = None,
    model = None,
    guide_model = None,
    number_of_residues_in_loop = None,
    number_of_residues_in_loop_minimum = None,
    loop_sequence = None,
    max_residues_in_loop = None,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    remove_clashes = True,
    refine_cycles = None,
    fit_loop_methods = None,
    method_type = 'fit_loop',
    trial_model = None,
     ):
    '''
    Connect two chains (fit a loop) or replace part of a chain (rebuild)
      chain_id: chain ID of chain to be fit
      last_resno_before_loop:  the residue number of the residue just before
                              the loop to be inserted. There must be at least
                              three residues present before the loop (more
                               if structure_search is used)
      first_resno_after_loop:  the residue number of the residue just after
                              the loop to be inserted. There must be at least
                              three residues present after the loop
      loop_sequence = None:  Use this (1-letter code) sequence and make this
        many residues in the loop
      number_of_residues_in_loop = None:  Make this many residues in loop, any
         sequence
      number_of_residues_in_loop_minimum = None:  Make at least this many
                               residues in loop, any sequence
      max_residues_in_loop = None:  If number_of_residues_in_loop is not
       specified and loop_sequence is not specified, only try up to this
       many residues
     Chain type is determined from existing model
     fit_loop_methods overrides defaults for methods to run
     Returns the new loop plus n residues on each end, refined against
     the map.
     Note that some refinement of the junction may be necessary after inserting
      the loop.
     Normally follow with insert_fitted_loop()
     The trial_model object is a model for the loop that can be tried after
       superposition or morphing. It applies only to trace_loops_with_density
       at present
     If guide_model are supplied try using pieces of them to guide the path
    '''
    # Initialize fitted loop object
    self.fitted_loop_info = None
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    # Check methods
    if fit_loop_methods is None:
      fit_loop_methods = self.info.fit_loop_methods
    else: # use supplied methods
      assert type(fit_loop_methods) == type([1,2,3])
      for x in fit_loop_methods:
        assert x in ALLOWED_FIT_LOOP_METHODS # fit_loop methods
    # Set default for tries_for_each_method
    if tries_for_each_method is None:
      tries_for_each_method = 1
    model,return_model = self._set_and_print_return_info(
      model=model,text='fit_loop',return_model=True,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    if model is None or model.get_sites_cart().size()<1:
      print ("No model to work with...", file = self.log)
      return
    # Set the chain type and make sure there is just one
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    print("\nFit loops with %s model" %(chain_type),file=self.log)
    # Decide on number of residues around the loop. Normally it is
    #   residues_on_each_end but if structure_search is present then
    #   also use residues_on_each_end_extended.  If this many are
    #   not present then skip structure search
    #
    from phenix.model_building.fit_loop import \
       use_remainder_of_chain_if_no_loop_end
    if chain_id is None:
      from mmtbx.secondary_structure.find_ss_from_ca import get_chain_ids
      chain_id_list = get_chain_ids(model.get_hierarchy())
      assert len(chain_id_list) == 1 # if chain_id not specified must be just 1
      chain_id = chain_id_list[0]
    last_resno_before_loop, first_resno_after_loop =\
       use_remainder_of_chain_if_no_loop_end(model.get_hierarchy(),
       last_resno_before_loop, first_resno_after_loop,
       residues_on_each_end = self.info.residues_on_each_end)
    fit_loop_methods = self._check_methods(
      methods_list = fit_loop_methods,
      model = model,
      chain_id = chain_id,
      chain_type = chain_type,
      last_resno_before_loop = last_resno_before_loop,
      first_resno_after_loop = first_resno_after_loop)
    if not fit_loop_methods:
      print ("Not running fit_loop as there are not sufficient"+
         " residues on each end (%s)" %(self.info.residues_on_each_end),
           file = self.log)
      return
    # Mask out density that is near the model for everything except the
    #  2 residues on the ends
    if 'trace_and_build' in fit_loop_methods:  # special case XXX
      if len(fit_loop_methods) > 1:
        raise Sorry("Need to use trace_and_build alone in fit_loops")
      n_to_block = 3
    else:
      n_to_block = 1 # usual
    selection="chain %s and resseq %s:%s" %(
       chain_id,last_resno_before_loop-n_to_block,
       first_resno_after_loop+n_to_block)
    # Determine if this is CA-only model
    if model.get_sites_cart().size() < \
        2 * model.get_hierarchy().overall_counts().n_residues:  # assume CA-only
      rad_mask = max(3, self.info.resolution)
    else:
      rad_mask =None # default
    if self.map_manager():
      self._masked_map_manager = self.create_masked_map(
        model=model.select(~model.selection(selection)),
          mask_atoms_atom_radius = rad_mask)
    # Ready to fit loop in model
    #  Return a new model that knows (through shift_cart()) its coordinates in
    #  original reference frame
    # Define arguments that are going to be passed to fit_loop methods
    common_kw = {
     'map_manager':self.masked_map_manager(),   # the map
     'model':model,                      # the working model
     'trial_model': trial_model,   # optional model to test after superposition
     'guide_model': guide_model,   # optional models to use as guide
     'chain_id':chain_id,                     # chain ID
     'last_resno_before_loop':last_resno_before_loop,
     'first_resno_after_loop':first_resno_after_loop,
     'loop_sequence':loop_sequence,               # sequence (1-letter code)
     'number_of_residues_in_loop':number_of_residues_in_loop,
     'number_of_residues_in_loop_minimum':number_of_residues_in_loop_minimum,
     'residues_on_each_end':self.info.residues_on_each_end,
     'residues_on_each_end_extended':self.info.residues_on_each_end_extended,
     'max_residues_in_loop':max_residues_in_loop,
     'chain_type':chain_type,               # chain_type (rna/dna/protein)
     'resolution':self.info.resolution,            # resolution of map
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'remove_clashes':remove_clashes, # Remove clashing side-chains
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object the
    #   method is supposed to return
    expected_result_names = [
       'model',               # fitted loop, including n residues on either end
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatic)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in fit_loop_methods using
    #  the kw in common_kw (common to all runs)
    self._run_group_of_methods(
     methods_list = fit_loop_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    result = self.get_best_result()
    if not result:
      return None
    if hasattr(result,'residues_on_each_end') and result.residues_on_each_end:
      residues_on_each_end = result.residues_on_each_end
    else:
      residues_on_each_end = self.info.residues_on_each_end
    # We have a loop
    fitted_loop = self.get_best_model()
    number_of_residues_in_loop = \
        fitted_loop.get_hierarchy().overall_counts().n_residues
    # Save the loop object
    self.fitted_loop_info = group_args(
      loop_with_n_residues_on_each_end = fitted_loop,
      number_of_residues_in_loop = number_of_residues_in_loop,
      chain_id = chain_id,  # where the loop goes
      last_resno_before_loop = last_resno_before_loop,
      first_resno_after_loop = first_resno_after_loop,
      residues_on_each_end = residues_on_each_end,
    )
    if not fitted_loop:
      print("No loop found...", file = self.log)
    else:
      print("Returning fitted loop ", file = self.log)
      return fitted_loop
  def insert_fitted_loop(self, model = None,
     fitted_loop_info = None,
     return_model = None):
    '''
      Insert the last fitted loop (if any) into supplied model.
      Modifies the supplied model
      If no model is supplied, the working model will be modified.
      if return_model, model is returned and working model is not modified
    '''
    if not fitted_loop_info:
      fitted_loop_info = self.fitted_loop_info
    if not fitted_loop_info:
      print ("No loop to insert", file = self.log)
      return
    assert self.info.residues_on_each_end is not None
    model,return_model = self._set_and_print_return_info(
      model=model,text='Inserting loop',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    from phenix.model_building.fit_loop import insert_loop_in_model
    insert_loop_in_model(
      model = model,
      loop_to_insert_with_n_residues_on_each_end = \
         fitted_loop_info.loop_with_n_residues_on_each_end,
      chain_id = fitted_loop_info.chain_id,
      last_resno_before_loop = fitted_loop_info.last_resno_before_loop,
      first_resno_after_loop = fitted_loop_info.first_resno_after_loop,
      residues_on_each_end = fitted_loop_info.residues_on_each_end,
     )
    return self._set_or_return_model(model,text="model with inserted loop",
      return_model=return_model,
       change_model_in_place = True)
  def replace_segment(self,
    chain_id = None,
    last_resno_before_replace = None,
    first_resno_after_replace = None,
    segment_sequence = None,
    model = None,
    remove_clashes = True,
    tries_for_each_method = None,
    allow_insertions_and_deletions = None,
    good_enough_score = 0.75,
    refine_cycles = None,
    minimum_residues_to_switch_to_replace = 10,
    replace_segment_methods = None,
    method_type = 'replace_segment',
    fix_insertions_deletions = None,
    fix_insertions_deletions_type = 'split_with_sequence',  # or 'rebuild_loops'
     ):
    '''
     Identical code as fit_loop, but conceptually different. Replace a segment
     with a newly-created segment.   The segment does have to exist
     If fix_insertions_deletions is set, just fix insertions and deletions
    '''
    # Check methods
    if replace_segment_methods is None:
      if fix_insertions_deletions:
        if fix_insertions_deletions_type == 'split_with_sequence':
          replace_segment_methods = \
            self.info.fix_insertion_deletion_sequence_methods
          if segment_sequence is None:
            segment_sequence = self.info.sequence
        else:
          replace_segment_methods = self.info.fix_insertion_deletion_methods
      else:
        replace_segment_methods = self.info.replace_segment_methods
    else: # use supplied methods
      assert type(replace_segment_methods) == type([1,2,3])
      for x in replace_segment_methods:
        assert x in ALLOWED_REPLACE_SEGMENT_METHODS # replace_segment methods
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    print("\nRunning fit_loop to replace segment ("+
     "with removal of existing segment)\n",file = self.log)
    model,return_model = self._set_and_print_return_info(
      model=model,text='replaced_segment',return_model=True,
      return_model_must_be_true_for_supplied_model=False,
      return_model_is_ligand_or_fragment=True)
    if last_resno_before_replace is None: # replace all that we can
      from iotbx.map_model_manager import get_selections_for_segments
      from mmtbx.secondary_structure.find_ss_from_ca import \
         get_first_resno,get_last_resno,get_chain_id
      selection_strings = get_selections_for_segments(model)
      if not selection_strings:
         print("No selections found...skipping", file = self.log)
         return  None
      selected_model = model.apply_selection_string(selection_strings[0])
      ph = selected_model.get_hierarchy()
      first_resno = get_first_resno(ph)
      last_resno = get_last_resno(ph)
      chain_id = get_chain_id(ph)
      n_residues = ph.overall_counts().n_residues
      if n_residues != last_resno - first_resno + 1:
        print("Unable to auto-identify segment as model does not "+
          "have simple numbering", file = self.log)
        return None
      if n_residues < 2 * self.info.residues_on_each_end + 1:
        print("Unable to auto-identify segment as model does not "+
          "have enough residues for loop replacement ", file = self.log)
        return None # not enough residues for replacement
      n_replace =  max(1, n_residues - (2 * self.info.residues_on_each_end + 1))
      n_on_end = max(self.info.residues_on_each_end,
         (n_residues - n_replace)//2)
      # if n_replace short, go with it as is.  If long, try to make n_on_end
      #  at least residues_on_each_end_extended
      if n_replace >= minimum_residues_to_switch_to_replace and \
          'structure_search' in replace_segment_methods and\
          n_residues >= n_replace + 2 * self.info.residues_on_each_end_extended:
        n_on_end = self.info.residues_on_each_end_extended
      last_resno_before_replace = n_on_end + first_resno
      first_resno_after_replace = last_resno - n_on_end + 1
      print("\nSetting segment to replace as chain '%s' " %(chain_id) +
         "\nand last_resno_before_replace = '%s' " %(last_resno_before_replace)+
        "\nand first_resno_after_replace = '%s'" %(first_resno_after_replace),
          file = self.log)
    last_resno_before_loop = last_resno_before_replace
    first_resno_after_loop = first_resno_after_replace
    if last_resno_before_loop + 1  > first_resno_after_loop-1:
      raise Sorry(
      "First residue of loop (%s) must be same as or before last (%s)" %(
       last_resno_before_loop + 1,first_resno_after_loop-1))
    if chain_id is None:  # just use chain_id of segment
      from mmtbx.secondary_structure.find_ss_from_ca import get_chain_id
      chain_id = get_chain_id(model.get_hierarchy())
      print("Using default chain ID of '%s' " %(chain_id), file = self.log)
    try:
      current_segment = model.apply_selection_string(
          "chain %s and resseq %s:%s" %(
           chain_id,last_resno_before_loop+1,first_resno_after_loop-1))
    except Exception as e:
      print (e,file = self.log)
      print ("Existing model must contain the segment to be replaced.",
        file = self.log)
      return
    if current_segment.get_sites_cart().size()<1:
      print ("Existing model must contain the segment to be replaced.",
        file = self.log)
      return
    if segment_sequence is None:
      from iotbx.bioinformatics import get_sequence_from_pdb
      segment_sequence = get_sequence_from_pdb(
        hierarchy = current_segment.get_hierarchy())
    if allow_insertions_and_deletions:
      number_of_residues_in_loop = None
      loop_sequence = None
    else:
      loop_sequence = segment_sequence
      number_of_residues_in_loop = len(loop_sequence)
    return self.fit_loop(
      chain_id,
      last_resno_before_loop,
      first_resno_after_loop,
      model = model,
      remove_clashes = remove_clashes,
      number_of_residues_in_loop = number_of_residues_in_loop,
      loop_sequence = loop_sequence,
      max_residues_in_loop = None,
      tries_for_each_method = tries_for_each_method,
      good_enough_score = good_enough_score,
      refine_cycles = refine_cycles,
      fit_loop_methods = replace_segment_methods)
  def rebuild(self,
    model = None,
    sequence = None,
    selection_string = None,
    fix_insertions_deletions = True,   # fix insertions and deletions
    allow_insertions_deletions = None,  # allow in regularization
    base_methods = None,       # which methods to run
    optimization_sequence = None,   # sequence of optimization
    refine_cycles = None,                   # Refinement cycles for a segment
    macro_cycles = 1,                # number of times to run entire procedure
    segment_length = None,        # if not None, work on this size segment
    segment_length_variability = 5,    # how much to randomly vary the length
    minimum_length = 10,          # minimum length of a segment to work on
    return_model = None,          # return model (vs modifying in place)
    take_anything_over_original = True, # try not to return original model
    dummy_run = None,             # go through steps but do not do anything
     ):
    '''
      Rebuild all or specified part of model, keeping connectivity the same
      If fix_insertions_deletions is set, run fix_insertions_deletions
      if allow_insertions_deletions, number of residues can change in
         regularization
      Changes input model in place.
    '''
    if not sequence:
      sequence = self.info.sequence
    model,return_model = self._set_and_print_return_info(
      model=model,text='Rebuilding',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    self.set_model(model)
    params = group_args(
      map_model_manager = self.as_map_model_manager(),
      selection_string = selection_string,
      allow_insertions_deletions = allow_insertions_deletions,
      fix_insertions_deletions = fix_insertions_deletions,
      base_methods = base_methods,
      optimization_sequence = optimization_sequence,
      refine_cycles = refine_cycles,
      sequence = sequence,
      segment_length = segment_length,
      segment_length_variability = segment_length_variability,
      minimum_length = minimum_length,
      macro_cycles = macro_cycles,
      nproc = self.info.nproc,
      take_anything_over_original = take_anything_over_original,
      verbose = self.info.debug,
      dummy_run = dummy_run,
      quick = (self.info.thoroughness == 'quick'),
     )
    from phenix.model_building.rebuild import run
    info = run(
      log = self.log,
      **params())
    rebuild_info = info.model_result
    if not rebuild_info.ok:
      print("Rebuilding was not successful",file = self.log)
      return
    print("Rebuilding successful with cc_mask=%.3f" %(
        rebuild_info.cc_mask),file = self.log)
    self.info_from_last_result = info
    return self._set_or_return_model(rebuild_info.model,
       text="refined model",return_model=return_model,
       change_model_in_place = True)
  def score_model(self,
    model = None,
    sequence = None,
    include_h_bond_score = True,
    include_molprobity_score = True):
    ''' Score model based on map-model CC, fit to sequence, and H-bonding
      Returns score_info object with summary of scoring information and
       summary score (result.score)
      If sequence is supplied, score will include fit to that sequence.
      If sequence is supplied  or not,  score will include best fit of
        any side chains to density, counting number of X (not fit) or Gly
        residues (over 7.4%) in best match to density as negative score.
    '''
    if model is None:
      model = self.model()
    assert model is not None
    mam = self.as_map_model_manager()
    cc_mask = mam.map_model_cc(model = model)
    from phenix.autosol.trace_and_build import trace_and_build
    from phenix.programs.trace_and_build import master_phil_str
    import iotbx.phil
    tnb_params = iotbx.phil.parse(master_phil_str).extract()
    tnb = trace_and_build(log = null_out() if (
       not self.info.debug) else self.log,
       skip_temp_dir = True,
       params = tnb_params)
    tnb.crystal_symmetry = self.map_manager().crystal_symmetry()
    tnb_params.control.skip_temp_dir = True
    tnb_params.control.verbose = self.info.debug
    tnb_params.control.use_starting_model_ca_directly = False
    tnb_params.crystal_info.sequence = sequence
    tnb_params.crystal_info.resolution = self.info.resolution
    tnb_params.rebuilding.refine_cycles = 0
    tnb_params.strategy.score_cc_mask_offset = 0 # Different than default
    tnb.score_sequence_dict = self.score_sequence_dict
    if (not self.info.debug):
      sys_stdout_sav = sys.stdout
      sys.stdout = null_out()
    tnb.score_model_vs_sequence(model = model,
          map_data = self.map_manager().map_data(),
          use_sequence = True,
          force_rescore = False,
          split = False,
          score_by_segments = True,
          refine = False,)
    score_info = tnb.last_best_scoring_info
    n_atoms = max(1,score_info.n_atoms if score_info.n_atoms else 1)
    score_info.cc_mask = cc_mask # use overall cc here
    score_info.cc_mask_score = cc_mask * n_atoms**0.5 # use overall cc here
    score_info.score = 0
    score_info.score += score_info.seq_score if score_info.seq_score else 0
    score_info.score += score_info.x_gly_score if score_info.x_gly_score else 0
    score_info.score += score_info.cc_mask_score if \
       score_info.cc_mask_score else 0
    if include_h_bond_score:
      from mmtbx.secondary_structure.find_ss_from_ca import \
         find_secondary_structure
      fss=find_secondary_structure(hierarchy=model.get_hierarchy(),
         out=null_out())
      n_residues = model.get_hierarchy().overall_counts().n_residues
      ann=fss.get_annotation()
      good_h_bonds=fss.number_of_good_h_bonds
      h_bond_ratio=min(1.,good_h_bonds/max(1,n_residues))
      score_info.h_bonds = good_h_bonds
      score_info.h_bond_score = h_bond_ratio * n_atoms**0.5
      score_info.score += score_info.h_bond_score
    if include_molprobity_score:
      mp = None
      from phenix.model_building.fit_loop import get_molprobity_summary
      try:
        mp = get_molprobity_summary(model, log = self.log if self.verbose else null_out())
      except Exception as e:
        # First just try it again
        try:
          mp = get_molprobity_summary(model, log = self.log if self.verbose else null_out())
        except Exception as e:
          mp = None
      score_info.rama_outliers = mp.get('rama_outliers'
          ) if mp and mp.get('rama_outliers') else 0 #  low is good 0.002
      score_info.rama_favored = mp.get('rama_favored'
          ) if mp and mp.get('rama_favored') else 0 # high is good 0.98
      score_info.rotamer_outliers = mp.get('rotamer_outliers'
          ) if mp and mp.get('rotamer_outliers') else 0 # low is good, .01
      score_info.c_beta_deviations = mp.get('c_beta_deviations'
          ) if mp and mp.get('c_beta_deviations') else 0 # low is good, 0
      score_info.clashscore = mp.get('clashscore'
          ) if mp and mp.get('clashscore') else 0# low is good, 5
      score_info.mpscore = mp.get('mpscore'
          ) if mp and mp.get('mpscore') else 0# low is good 1
      # scale mp_score by cc_mask.  If cc_mask is low, do not allow mp_score to
      #    dominate
      # base score is zero if mpscore is 4, 1 if mpscore is zero.  Max of 1
      if score_info.mpscore is None:
        full_mpscore_score = 0
      else:
        base_mpscore_score = min(1, 0.25* (4 - score_info.mpscore) )
        scaled_mpscore_score = base_mpscore_score * max(0, cc_mask)
        full_mpscore_score = scaled_mpscore_score * n_atoms**0.5
      score_info.mpscore_score = full_mpscore_score
      #  if mpscore = 4 or less , get a zero or positive score
      score_info.score += score_info.mpscore_score
    else:
      score_info.mpscore = 0
      score_info.mpscore_score = 0
    if (not self.info.debug):
      sys.stdout = sys_stdout_sav
    print("\nScoring: CC: %.2f Gly: %s X: %s H-bonds: %s MPscore %.2f" %(
       score_info.get('cc_mask'),
       score_info.get('gly_count'),
       score_info.get('x_count'),
       score_info.get('h_bonds'),
       score_info.get('mpscore'),
       ), "\nWeighted scores: ",
   "CC %.2f  Seq: %.2f  X-Gly: %.2f  H-bond: %.2f MPscore: %.2f  Total: %.2f" %(
     score_info.get('cc_mask_score') if score_info.get('cc_mask_score') else 0,
     score_info.get('seq_score') if score_info.get('seq_score') else 0,
     score_info.get('x_gly_score') if score_info.get('x_gly_score')  else 0,
     score_info.get('h_bond_score') if score_info.get('h_bond_score') else 0,
     score_info.get('mpscore_score') if score_info.get('mpscore_score') else 0,
     score_info.get('score') if score_info.get('score') else 0,
         ),
        file = self.log)
    return score_info
  def apply_ncs(self,
    model = None,
    ncs_object = None):
    '''
     Apply ncs_object (default is the one in map_manager()) to model (
     default is self.model() and return new model. Always returns new model
    '''
    model,return_model = self._set_and_print_return_info(
      model=model,text='Applying NCS',return_model=True)
    if not ncs_object:
      ncs_object = self.map_manager().ncs_object()
    if not ncs_object:
      print ("No NCS object to work with...returning existing model",
          file=self.log)
      return model
    # Ready to apply ncs
    from phenix.model_building.build_tools import apply_ncs_to_model
    return apply_ncs_to_model(model=model,
        ncs_object=ncs_object,
        log=self.log)
  def combine_models(self,
    model_list,
    merge_type = 'crossover',  # non_overlapping/crossover
    cc_min = None,
    remove_clashes = True,
    sequence = None,
    refine_cycles = None,
    unique_parts_only = True,
    merge_second_model = None,
    keep_all_of_first_model = None,):
    '''
      Combine the models in model_list based on fit to density in
      the map, taking into consideration their chain types
      If unique_parts_only is set then do not allow overlaps that would
        occur from symmetry in the map (defined by the ncs_object specified
        in map_manager)
      merge_type:  non_overlapping means remove overlapping pieces,
                   crossover means try to recombine pieces, then remove
                    overlapping remainder.
      cc_min is lowest map-model cc to keep
      Always returns models (does not modify working_model)
      If merge_second_model, merge remainder of second model after initial
         merge (default True for non-overlapping, False for crossover).
      If remove_clashes is set, try to remove clashing side chains
      if sequence is set, redo sequence at end
      if keep_all_of_first_model try to keep all of first model
    '''
    # Checks
    assert merge_type in ['non_overlapping', 'crossover']
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    print ("\nCombining the following models: \n",file=self.log)
    for model in model_list:
      assert isinstance(model, mmtbx.model.manager)
      if self.info.debug:
        print ("\n%s" %(model), file = self.log)
      mam = MapModelManager(map_manager = self.map_manager())
      mam.add_model_by_id(model=model,model_id = 'model')
      print("Residues: %s  CC_mask: %.3f " %(
      mam.model().get_hierarchy().overall_counts().n_residues,
      mam.map_model_cc(resolution=self.info.resolution) ), file = self.log)
    if merge_type == 'non_overlapping' :
      merge_by_segment_correlation = True
      weight_by_ss = True
      if merge_second_model is None:
        merge_second_model = False
      split_model_at_end = True
      if cc_min is None:
         cc_min = None  # keep it
      combine_models_score_min = None
    elif merge_type == 'crossover':
      merge_by_segment_correlation = False
      weight_by_ss = False
      if merge_second_model is None:
        merge_second_model = True
      split_model_at_end = False
      if cc_min is None:
        cc_min = 0.2
      combine_models_score_min = None
    else:
      raise Sorry("Merge_type must be 'crossover' or 'non_overlapping'")
    from phenix.command_line.combine_models import simple_combine_models
    new_model = simple_combine_models(
      models = model_list,
      map_manager = self.map_manager(),
      resolution = self.info.resolution * self.info.d_min_ratio,
      merge_by_segment_correlation = merge_by_segment_correlation,
      weight_by_ss = weight_by_ss,
      merge_second_model = merge_second_model,
      cc_min = cc_min,
      combine_models_score_min = combine_models_score_min,
      unique_parts_only = unique_parts_only,
      split_model_at_end = split_model_at_end,
      keep_all_of_first_model = keep_all_of_first_model,
      temp_dir = self.info.temp_dir,
      verbose = self.info.debug,
      log = self.log)
    if new_model and (sequence is not None):  # resequence
      sequenced_new_model = self.sequence_from_map(
        model = new_model,
        remove_clashes = remove_clashes,
        sequence = sequence,
        return_model = True,
        keep_segment_order = (not split_model_at_end),
        refine_cycles = refine_cycles)
      if sequenced_new_model:
        new_model = sequenced_new_model
    return new_model
  def trim_overlapping_other_model(self, model = None, other_model = None,
    min_dist = None, minimum_fragment_length = 7, unique_chain_id = None):
    """
     Trim residues in other_model that have CA/P within min_dist of one in
     model
     Remove other_model final fragments shorter than minimum_fragment_length
    """
    model_info = self.get_model_info(model, other_model, min_dist = min_dist)
    pair_dict_info = self.get_closest_ca_in_model_to_ca_in_other(
     model = model_info.ca_model,
     other_model = model_info.other_ca_model,
     unique_chain_id = unique_chain_id)
    keep_list_dict = {}
    for m in model_info.other_ca_model.get_hierarchy().models():
      for chain in m.chains():
        if not chain.id in list(keep_list_dict.keys()):
          keep_list_dict[chain.id] = []
        for rg in chain.residue_groups():
          resseq_int = rg.resseq_as_int()
          matching_info = pair_dict_info.pair_dict.get(
             chain.id,{}).get(resseq_int,None)
          if (not matching_info) or (matching_info.dist is None) or (
               matching_info.dist > model_info.min_dist):
            keep_list_dict[chain.id].append(resseq_int)
    from mmtbx.process_predicted_model import get_unique_values, \
         get_indices_as_ranges
    selection_list = []
    for chain_id in list(keep_list_dict.keys()):
      unique_indices = get_unique_values(keep_list_dict[chain_id])
      indices_as_ranges = get_indices_as_ranges(unique_indices)
      print("Ranges for chain '%s'" %(chain_id), file = self.log)
      for r in indices_as_ranges:
        if (r.end - r.start + 1 ) < minimum_fragment_length: continue
        selection_list.append("(chain %s and resseq %s:%s)" %(
           chain_id, r.start,r.end))
    selection_string = " or ".join(selection_list)
    if not selection_string:
      print("No non-overlapping parts of model found...", file = self.log)
      return None
    else:
      # Now select just the good parts of other_model
      print("Selection string applied: %s" %(selection_string), file = self.log)
      return other_model.apply_selection_string(selection_string)
  def get_model_info(self,model, other_model = None, min_dist = None):
    if not model:
      model = self.model()
    assert model and other_model
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    if min_dist is None:
      if chain_type == "PROTEIN":
        min_dist = 3
      else:
        min_dist = 9
    if chain_type == "PROTEIN":
      ca_name = "CA"
    else:
      ca_name = "P"
    ca_model = model.apply_selection_string("name %s" %(ca_name))
    sites_cart = ca_model.get_sites_cart()
    if other_model:
      other_ca_model = other_model.apply_selection_string("name %s" %(ca_name))
      other_sites_cart = other_ca_model.get_sites_cart()
    else:
      other_ca_model = None
      other_sites_cart = None
    return group_args(
      group_args_type = 'model info',
      model = model,
      min_dist = min_dist,
      other_model = other_model,
      chain_type = chain_type,
      ca_name = ca_name,
      ca_model = ca_model,
      other_ca_model = other_ca_model,
     )
  def get_closest_ca_in_model_to_ca_in_other(self,
     model = None,
     other_model = None,
     unique_chain_id = None):
    """ return a group_args object with attributes:
        pair_dict: dict of chain ID, resseq in other_model with value of
        chain ID, resseq of the closest CA in model and the distance,
        (group_args object for each entry)
        other_chain_id_list:  list of keys in pair_dict
    """
    if not unique_chain_id:
      unique_chain_id = "ZA"
    model_info = self.get_model_info(model, other_model)
    from mmtbx.secondary_structure.find_ss_from_ca import get_chain_ids
    chain_list = get_chain_ids(model_info.model.get_hierarchy())
    other_chain_list = get_chain_ids(model_info.other_model.get_hierarchy())
    if unique_chain_id in chain_list + other_chain_list:
      raise Sorry("Please supply a unique chain id for "+
         "get_closest_ca_in_other_to_ca_in_model as " +
         "'%s' is in supplied models (%s and %s)" %(
         unique_chain_id,str(chain_list),str(other_chain_list)))
    model_ph_list = []
    for chain_id in chain_list:
      model_ph_list.append(model_info.ca_model.apply_selection_string(
        "chain %s" %(chain_id)).get_hierarchy())
    other_model_ph_list = []
    for other_chain_id in other_chain_list:
      other_model_ph_list.append(model_info.other_ca_model.apply_selection_string(
        "chain %s" %(other_chain_id)).get_hierarchy())
    from phenix.model_building.ssm import create_composite_ph, get_pair_dict, \
       get_ssm_params
    params = get_ssm_params()
    pair_dict = {}
    for ph2,chain_id in zip(model_ph_list, chain_list):
      ph2_resseq_list = []
      for at in ph2.atoms():
        resseq_int = at.parent().parent().resseq_as_int()
        ph2_resseq_list.append(resseq_int)
      for ph1,other_chain_id in zip(other_model_ph_list, other_chain_list):
        if not other_chain_id in list(pair_dict.keys()):
          pair_dict[other_chain_id] = {}
        ph1 = ph1.deep_copy() # we are going to change it
        # Compare one chain of ph2 with all chains in ph1
        # First put both in same hierarchy so we can use fast pair generation
        m1_size = ph1.atoms().size() # save number of atoms in ph1
        composite_ph = create_composite_ph(
          ph1, ph2, unique_chain_id)  # NOTE: changes ph1
        pair_dict_info = get_pair_dict(params, composite_ph,
           m1_size = m1_size,
           unique_chain_id = unique_chain_id,
           log = self.log)  # pair_dict[resseq_ph1] = list of resseq in ph2 matching
        # pair_dict_info: pair_dict, m1_xyz, m2_xyz
        #  pair_dict[i] = list of indices in m2_xyz matching m1_xyz[i],
        #    sorted by distance
        ii = -1
        for at in ph1.atoms()[:m1_size]:
          ii += 1
          resseq_int = at.parent().parent().resseq_as_int()
          index_in_ph2 = pair_dict_info.pair_dict[ii][0] \
              if pair_dict_info.pair_dict[ii] else None
          if index_in_ph2 is not None:
            model_resseq = ph2_resseq_list[index_in_ph2]
            dist = (col(pair_dict_info.m1_xyz[ii]) -
                    col(pair_dict_info.m2_xyz[index_in_ph2])).length()
            previous_value = pair_dict[other_chain_id].get(resseq_int,None)
            if (not previous_value) or previous_value.dist >= dist: # keep this one
              pair_dict[other_chain_id][resseq_int] = group_args(
                group_args_type = 'chain_id and resseq of closest residue in model',
                 model_chain_id = chain_id,
                 model_resseq = model_resseq,
                 dist = dist)
    return group_args(
      group_args_type = 'closest residues in model by chain ID and resseq_int',
         pair_dict = pair_dict,
         other_chain_id_list = other_chain_list,
        )
  def remove_waters_from_model(self, model=None):
    ''' remove waters from model '''
    if not model:
      model = self.model()
    if not model:
      return
    sel=model.selection("water")
    print("Removing %s waters from model" %(sel.count(True)), file = self.log)
    return model.select(~sel)
  def create_masked_map(self, model = None,
      mask_atoms_atom_radius = None,
      soft_mask = False):
    '''
     Use model to mask current map, setting all all density near
     atoms in model to zero
     Smooth the mask if soft_mask is True
    '''
    if model:
      print ("Using supplied model to mask map", file = self.log)
    else:
      model = self.model()
      print ("Using working model to mask map", file = self.log)
    assert isinstance(model, mmtbx.model.manager)
    if mask_atoms_atom_radius is None:
      mask_atoms_atom_radius = self.info.mask_atoms_atom_radius
    from cctbx.maptbx.mask import create_mask_around_atoms
    soft_mask_radius=2 * self.info.resolution
    if soft_mask:
      # add soft_mask_radius to atom radius
      mask_atoms_atom_radius = \
         mask_atoms_atom_radius+soft_mask_radius
    cm = create_mask_around_atoms(
      map_manager = self.map_manager(),
      model = model,
      mask_atoms_atom_radius = mask_atoms_atom_radius)
    if soft_mask:
      cm.soft_mask(soft_mask_radius = soft_mask_radius)
    mask_data = cm.map_manager().map_data()
    map_data = self.map_manager().map_data()
    new_map_data = map_data * (1 - mask_data)
    mm=self.map_manager().customized_copy(map_data = new_map_data)
    return mm
  def _run_group_of_methods(self,
     methods_list,
     common_kw,
     method_type,
     tries_for_each_method,
     expected_result_names,
     break_condition,
     nproc,
     skip_finding_best = False):
    '''
      Set up runs of a group of related methods, all with same arguments
      based on common_kw
    '''
    self.info_from_last_result = None
    # Save thoroughness
    thoroughness = common_kw['thoroughness']
    random.seed(self.info.random_seed)
    self.info.random_seed = random.randint(0,1000000)
    # Get list of methods to run and kw for each
    run_info = create_run_methods(
      methods_list = methods_list,
      common_kw = common_kw,
      method_type = method_type,
      tries_for_each_method = tries_for_each_method,
      expected_result_names = expected_result_names,
      random_seed = self.info.random_seed,
      nproc = nproc,
      debug = self.info.debug,
      temp_dir = self.info.temp_dir,
      log = self.log)
    if not run_info:
      return None
    kw_list = run_info.kw_list
    tries_for_each_method = run_info.tries_for_each_method
    nproc_for_parallel_run = run_info.nproc_for_parallel_run
    methods_list = run_info.methods_list
    # Now call easy_mp with run_a_method_as_class from methods list
    #  and return result with score, score_type, and new model
    #  Stop looking if result found that is good enough
    # Remove incompatible methods
    if common_kw.get('chain_type') is not None:
      methods_list = self._remove_incompatible_methods(method_type,
        methods_list,
        common_kw['chain_type'])
    unique_methods = []
    for method in methods_list:
     x = "%s_with_%s" %(method_type,method)
     if not x in unique_methods: unique_methods.append(x)
    print("\nRunning build methods...  "+
      "\nMethods: %s " %(" ".join(unique_methods)) +
      "\nTries per method: %s   Thoroughness: %s   Good enough score: %.3f\n" %(
      tries_for_each_method,thoroughness,
       common_kw['good_enough_score']), file=self.log)
    self.print_result(None, header_only = True)
    # Set up multiprocessing
    # If the number of methods to be run is 1, set nproc=1 for multiprocessing
    #   and pass on nproc to the run method itself
    from libtbx.easy_mp import run_parallel
    # We are going to set up run_a_method_as_class with N sets of kw,
    #  then call it with an index "i" to decide which to run in that call
    index_list=[]
    for i in range(len(kw_list)):
      index_list.append({'i':i})
    results = run_parallel(
     method = 'multiprocessing',
     nproc = nproc_for_parallel_run,
     target_function = run_a_method_as_class(kw_list=kw_list),
     preserve_order=False,
     break_condition=break_condition,
     kw_list = index_list)
    #  Now results is a list of results. Find the good ones and return
    self.useful_results = []
    for result in results:
      if result and result.score is not None:
        self.print_result(result, skip_header=True)
        self.useful_results.append(result)
    if skip_finding_best:
      return
    # Choose best result and return it
    result = get_best_result(self.useful_results)
    if not result:
      print("\nNo result...",file=self.log)
      return None
    self.info_from_last_result = getattr(result,'info', None)
    if result.log_as_text:
      print("\nBest log file\n%s\n" %(result.log_as_text),file=self.log)
    print("\nBest result:\n",file=self.log)
    self.print_result(result)
    return result.model
  def print_result(self,result,
   header_only = None,
   skip_header = None):
    if (not skip_header) or header_only:
      print("  ID    CC_MASK   SCORE     SCORE_TYPE      METHOD NAME  "+
         "      THOROUGHNESS",
        file=self.log)
      if header_only:
        return
    if result and result.score is not None:
      print("  %2s  %8.3f  %8.3f %12s  %22s %12s" %(
        result.run_id,
        zero_if_none(result.cc_mask),
        zero_if_none(result.score),
        result.score_type,
        result.method_name,
        result.thoroughness,
        ),
        file=self.log)
  def model(self):
    '''
      Get the current model
    '''
    return self._model
  def map_manager(self):
    '''
      Get the current map_manager
    '''
    return self._map_manager
  def masked_map_manager(self):
    '''
      Get the current masked_map_manager
    '''
    return self._masked_map_manager
  def get_results(self):
    '''
    Return all the useful results from the last run of group_of_methods
    '''
    return self.useful_results
  def get_best_result(self):
    '''
    Return the highest-scoring result from the last run of group_of_methods
    '''
    return get_best_result(self.useful_results)
  def get_best_model(self):
    '''
    Return the model from the highest-scoring result from the last
    run of group_of_methods
    '''
    best_result = self.get_best_result()
    if best_result and best_result.model:
      return best_result.model
    else:
      return None
  def get_info_from_last_result(self):
    return self.info_from_last_result
  def build_in_boxes(self, kw_args,
     box_cushion = None):
    mmm = self.as_map_model_manager()
    volume = mmm.crystal_symmetry().unit_cell().volume()
    small_volume = kw_args.small_dimension**3
    small_target_boxes = int(1.5+volume/small_volume)
    target_for_boxes = min(small_target_boxes, 2 * self.info.nproc)
    if box_cushion is None:
      box_cushion = 5  # bigger than usual
    box_info = mmm.split_up_map_and_model_by_boxes(
      target_for_boxes = target_for_boxes,
      box_cushion = box_cushion,
      select_final_boxes_based_on_model = False, # required
      )
    print("Working in %s boxes" %(len(box_info.mmm_list)), file = self.log)
    overall_temp_dir = self.info.temp_dir
    if not overall_temp_dir:
      overall_temp_dir = os.path.join(os.getcwd(), 'tmp_helices_strands' )
    if not os.path.isdir(overall_temp_dir):
      os.mkdir(overall_temp_dir)
    build_list = []
    i = -1
    temp_dir_list = []
    for small_mmm in box_info.mmm_list:  # run in small box
      i += 1
      build = small_mmm.model_building()
      temp_dir = os.path.join(overall_temp_dir, 'local_hs_%s' %(i))
      temp_dir_list.append(temp_dir)
      build.set_defaults(temp_dir = temp_dir)
      build.set_defaults(build_methods = self.info.build_methods)
      build.set_defaults(resolution = self.info.resolution)
      build.set_defaults(nproc = 1)
      build_list.append(build)
    iteration_list = range(len(build_list))
    from libtbx.easy_mp import simple_parallel
    hs_model_list = simple_parallel(
      iteration_list = iteration_list,
      function = build_something,
      nproc = self.info.nproc,
      build_list = build_list )
    for m in hs_model_list:
      mmm.shift_any_model_to_match(m)
    new_model = self.combine_models(hs_model_list,
      merge_type = kw_args.merge_type)
    # Remove temp_dirs
    if not self.info.debug:
      from phenix.autosol.delete_dir import delete_dir
      for temp_dir in temp_dir_list:
        delete_dir(temp_dir, clean_up=True)
    # Save the build object
    self.build_info = group_args(
        model = new_model,
        number_of_residues_in_model =
           new_model.get_hierarchy().overall_counts().n_residues \
             if new_model else 0,
    )
    return self._set_or_return_model(new_model,text="new model",
        return_model=kw_args.return_model)
  def build(self,
    chain_type = None,
    sequence = None,
    model = None,
    mask_around_model = False,
    starting_model = None,
    use_starting_model_ca_directly = None,
    use_starting_model_ca_as_guide = None,
    trace_only = None,
    cc_min = None,
    include_strands = True,
    remove_clashes = True,
    target_number_of_residues_to_build = 50,
    use_existing_masked_map = False,
    tries_for_each_method = None,
    good_enough_score = 0.75,
    refine_cycles = None,
    build_in_boxes = None,
    small_dimension = 50,
    merge_type = 'non_overlapping',
    method_type = 'build',
    return_model = None,
     ):
    '''
      Create a chain from density in the current map.
      This tool is intended to build a part of a model in a local region.
      If model is supplied or present, map is first masked around
      existing model to remove density in locations where there is already model
      chain_types allowed; PROTEIN RNA DNA one must be chosen
      cc_min is minimum to keep
      include_strands is for find_helices_strands
      If remove_clashes, remove clashing side-chains
      starting_model is a starting model, typically a ca_model
        Assumed to match map already.
      use_starting_model_ca_directly means use CA from starting_model
      use_starting_model_ca_as_guide means use CA from starting_model as guide
      if trace_only = True then return trace atoms as model (finer than CA)
      Returns new model that has been built.
      If no working_model present, new model is used as working_model unless
        return_model is True
      Scoring will be map_model_cc * sqrt (residues_built/
          target_number_of_residues_to_build)
     If build_in_boxes, build in boxes and combine.  Try to make box dim at
      least small_dimension
    '''
    if build_in_boxes and self.info.nproc > 1:
      kw_args = group_args()
      adopt_init_args(kw_args, locals())
      return self.build_in_boxes(kw_args)
    if not chain_type:
      chain_type = self.info.chain_type
    if not chain_type:  # just guess protein
      chain_type = 'PROTEIN'
    assert chain_type.lower() in ALLOWED_CHAIN_TYPES   # protein rna dna
    # make sure chain type compatible with sequence
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    if not sequence:
     sequence = self.info.sequence
    if not sequence:
      sequence = generate_random_sequence(chain_type = chain_type,
        n_residues = target_number_of_residues_to_build, all_same = True)
    # Initialize build object
    self.build_info = None
    # Set default for tries_for_each_method
    if tries_for_each_method is None:
      tries_for_each_method = 1
    # Set the chain type and make sure it is compatible with sequence
    if sequence:
      from iotbx.bioinformatics import guess_chain_types_from_sequences
      chain_types=guess_chain_types_from_sequences(text=sequence)
      if chain_type is None:
        if len(chain_types) == 1:
          chain_type = chain_types[0]
        elif len(chain_types) > 1:
          raise Sorry("Please set chain type...this sequence can be %s" %(
           " or ".join(chain_types)))
        else:
          raise Sorry("Please set chain type...cannot guess from sequence" )
      else:
        if not chain_type in chain_types:
          raise Sorry("The sequence does not appear to be compatible with a "+
           "chain_type  of '%s'" %(chain_type))
    model,return_model = self._set_and_print_return_info(
      model=model,text='Building',return_model=return_model,
      return_model_must_be_true_for_supplied_model=True)
    #  Get a masked map
    if not model:
      self._masked_map_manager = self.map_manager()
    elif use_existing_masked_map and \
      self._masked_map_manager:
        pass  # ok as is
    elif mask_around_model and model:
      #  Create map masked to only include density outside existing model
      self._masked_map_manager=self.create_masked_map(
         model = model,
         soft_mask = False)
    else:
      self._masked_map_manager = self.map_manager()
    if merge_type == 'non_overlapping':
      merge_by_segment_correlation = True
    elif merge_type == 'crossover':
      merge_by_segment_correlation = False
    else:
      raise Sorry("Merge type must be crossover or non_overlapping")
    if trace_only:
      print("Using trace_and_build as trace_only is set",
        file = self.log)
      self.set_defaults(build_methods=['trace_and_build'])
    if use_starting_model_ca_directly: # only can use trace_and_build
      print("Using trace_and_build as use_starting_model_ca_directly is set",
        file = self.log)
      self.set_defaults(build_methods=['trace_and_build'])
      if use_starting_model_ca_as_guide:
        print("CA will be used as guide only", file = self.log)
    # Ready to build
    #  Return a new model that knows (through shift_cart()) its coordinates in
    #  original reference frame
    # Define arguments that are going to be passed to build methods
    common_kw = {
     'map_manager':self.masked_map_manager(),   # the map
     'target_number_of_residues_to_build':target_number_of_residues_to_build,
     'chain_type':chain_type,               # chain_type (rna/dna/protein)
     'sequence':sequence,            # sequence
     'experiment_type':self.info.experiment_type,    # xray neutron cryo_em
     'merge_by_segment_correlation':merge_by_segment_correlation,
        # method for merging models
     'cc_min':cc_min, # minimum cc for merging models
     'include_strands':include_strands, # for find_helices_strands
     'remove_clashes':remove_clashes, # Remove clashing side-chains
     'starting_model':starting_model, # starting model for building
     'use_starting_model_ca_directly':use_starting_model_ca_directly,
        # use CA from starting model for building
     'use_starting_model_ca_as_guide':use_starting_model_ca_as_guide,
        # use CA from starting model for building
     'trace_only':trace_only,            # return trace for starting_model
     'resolution':self.info.resolution,            # resolution of map
     'thoroughness':self.info.thoroughness,   # quick/medium/thorough/extra_thorough
     'scattering_table':self.info.scattering_table,
     'refine_cycles':refine_cycles,     # cycles of refinement (0 is no refine)
     'good_enough_score':good_enough_score,    # score sufficient for quitting
      }
    # Define the entries in the group_args result object the
    #   method is supposed to return
    expected_result_names = [
       'model',               # extension, including 2 residues on end
       'score_type',            # usually cc_mask * sqrt(fraction_built)
       'cc_mask',               # value of cc_mask
       'score',                 # value of score
       'method_name',           # Method that was run (automatically generated)
       'thoroughness',          # thoroughness of this try (automatic)
       'log_as_text',           # Log for the run (automatic)
       'run_id',                # ID for this run (automatic)
       'good_enough_score',     # score sufficient for quitting (automatic)
      ]
    # Define a condition under which there is no need to continue with more
    #  tries (if any) Can be None.  The way it works is that
    #  if break_condition(result) returns True, then no more tries are carried
    #   out.
    break_condition =  break_if_good_enough_score  # use this method to decide
    # Now we are ready to run each method in build_methods using
    #  the kw in common_kw (common to all runs)
    self._run_group_of_methods(
     methods_list = self.info.build_methods,
     method_type = method_type,
     common_kw = common_kw,
     tries_for_each_method = tries_for_each_method,
     expected_result_names = expected_result_names,
     break_condition = break_condition,
     nproc = self.info.nproc,
      )
    result = self.get_best_result()
    if not result:
      print ("Nothing built...", file = self.log)
      return None
    # We have a model
    new_model= self.get_best_model()
    # Save the build object
    self.build_info = group_args(
      model = new_model,
      number_of_residues_in_model =
         new_model.get_hierarchy().overall_counts().n_residues \
          if new_model else 0,
    )
    return self._set_or_return_model(new_model,text="new model",
      return_model=return_model)
  def find_secondary_structure(self,
    model = None,
    tolerant = False,
    evaluate_sheet_topology = False,
    from_ca = True):
    '''
    Find secondary structure in a model
    Optionally evaluate sheet topology
    '''
    if not model:
       model = self.model()
    from mmtbx.secondary_structure.find_ss_from_ca \
       import find_secondary_structure, get_first_resno
    if tolerant:
      args = ['tolerant=true']  # To get tolerant defaults
    else:
      args = []
    fss = find_secondary_structure(
       hierarchy = model.get_hierarchy(),
       ss_by_chain = False, # required
       args = args,
       out = null_out())
    annotation = fss.get_annotation()
    if evaluate_sheet_topology:
      from mmtbx.secondary_structure.find_ss_from_ca import \
        evaluate_sheet_topology
      sheet_topology = evaluate_sheet_topology(annotation,
        hierarchy = model.get_hierarchy(),
       out = null_out())
    else:
      sheet_topology = None
    #  Things you can do:
    #  selection_string = ann.overall_selection()
    #  text = ann.as_pdb_str()
    #  n_residues = ann.count_residues(hierarchy = model.get_hierarchy())
    return group_args(
      group_args_type = 'secondary_structure analysis',
      annotation = annotation,
      sheet_topology = sheet_topology,
    )
  def _connect_with_trace_away_from_sites(self,
      mmm,
      sites_cart,
      minimum_density = 0,
      ca_distance = 3.8,
      as_ca_model = None,
      reverse_ca_model = None,
      guide_model = None,
      max_tries_to_get_more_residues = 3,
      max_time = None,
      minimum_residues = None,
      exact_number_of_residues_to_return = None,
      n_threshold = 10,
      group_args_type = None,
      avoid_sites_radius= None,
      max_branch_length = None,
      kw_args = None,
      trial_model = None,
      refine_cycles = 1,
      n_overlap = 3):
    """ Trace through density in mmm away from each of sites_cart
        Then try to connect"""
    kw_args.allow_connect_with_trace_away_from_sites = False # REQUIRED
    if not minimum_residues:
      minimum_residues = 10  # just to have something
    i = -1
    trace_model_list = []
    for site_cart in sites_cart:
      i += 1
      sc = flex.vec3_double()
      sc.append(site_cart)
      print("Tracing away from (%.3f, %.3f, %.3f)" %(site_cart),
        file = self.log)
      trace_sites_cart = self._trace_away_from_sites(mmm, sc,
        minimum_density = minimum_density,
        ca_distance = ca_distance,
        as_ca_model = as_ca_model,
        reverse_ca_model = reverse_ca_model,
        guide_model = guide_model,
        max_tries_to_get_more_residues = max_tries_to_get_more_residues,
        max_time = max_time,
        minimum_residues = minimum_residues,
        exact_number_of_residues_to_return = exact_number_of_residues_to_return,
        group_args_type = group_args_type,
        max_branch_length = max_branch_length,
        avoid_sites_radius = avoid_sites_radius,
        return_sites_cart = True,
        )
      if trace_sites_cart.size() > 0:
        ts_model= mmm.model_from_sites_cart(trace_sites_cart,return_model = True)
        trace_model = get_model_from_path(trace_sites_cart,
          as_ca_model = True,
          ca_distance = ca_distance,
          reverse_ca_model = False if i==0 else True,
          mmm = mmm)
        trace_model_list.append(trace_model)
      else:
        return None # unable to do anything here
    if len(trace_model_list) == 2: # found it
      print("Traced from both ends...trying to connect now.", file = self.log)
      # Start back a couple residues
      sc1 = trace_model_list[0].get_sites_cart()
      sc2 = trace_model_list[1].get_sites_cart()
      sc1_model= mmm.model_from_sites_cart(sc1,return_model = True)
      sc2_model= mmm.model_from_sites_cart(sc2,return_model = True)
      for n_to_trim in [0]:  # could try more
        i1 = max(0, max(sc1.size()//2,sc1.size() - n_to_trim - 1))
        i2 = min(sc2.size()-1, min(sc2.size()//2, max(0,n_to_trim - 1)))
        # Now try to connect between these sites_cart
        kw_args.site_cart_1 = sc1[i1]
        kw_args.site_cart_2 = sc2[i2]
        # Mask everything up to these sites...
        avoid_these_sites = sc1[:i1-2]
        avoid_these_sites.extend(sc2[i2+2:])
        new_mmm = mmm.deep_copy() # now mask further around sites
        new_mmm.mask_all_maps_around_atoms(
          invert_mask = True,  # keep outside
          model = mmm.model_from_sites_cart(
           avoid_these_sites, return_model = True),
          mask_atoms_atom_radius = avoid_sites_radius,)
        new_build = new_mmm.model_building()
        new_build.set_log(sys.stdout)
        local_kw_args = kw_args().copy() # dict
        local_kw_args = group_args(**local_kw_args)
        effective_residues = (
            get_path_length(sc1) + get_path_length(sc2))/ca_distance
        local_kw_args.minimum_residues = (max(0,
           exact_number_of_residues_to_return if
           exact_number_of_residues_to_return else minimum_residues) -
             effective_residues)
        local_kw_args.allow_connect_with_trace_away_from_sites = False
        result = new_build.trace_between_sites_through_density(
           **local_kw_args())
        if (not result) or (not result.sites_cart) or (not result.model):
          print("No connection between tracings with trim = ",n_to_trim,
             file = self.log)
        else:
          print("Obtained connection with %s residues" %(
             result.sites_cart.size()), file = self.log)
          break # got it
      if (not result) or (not result.sites_cart) or (not result.model) or (
         result.sites_cart.size() + sc1.size() + sc2.size() < minimum_residues):
        if trial_model and \
          trial_model.get_hierarchy().overall_counts().n_residues >=  \
           2* n_overlap and \
          sc1.size() >= n_overlap and sc2.size() >= n_overlap:
          print("Using superposed connection between tracings ",
             file = self.log)
          trial_ca_sites_cart = trial_model.apply_selection_string(
             'name ca').get_sites_cart()
          n_trial = trial_ca_sites_cart.size()
          n_sc1 = sc1.size()
          n_sc2 = sc2.size()
          full_sc1 = sc1.deep_copy()
          full_sc2 = sc2.deep_copy()
          sc1_base = sc1[:-n_overlap]
          sc1 = sc1[-n_overlap:]
          sc2_base = sc2[n_overlap:]
          sc2 = sc2[:n_overlap]
          n1 = sc1_base.size()
          start_sc1 = n_sc1 - n_overlap
          end_sc1 = start_sc1 + n_overlap - 1
          n2 = sc2_base.size()
          start_sc2 = n_sc2 - n2 - n_overlap
          end_sc2 = start_sc2 + n_overlap - 1
          start_matching_sc2 = n_trial - n2 - n_overlap
          end_matching_sc2 = start_matching_sc2 + n_overlap - 1
          assert end_sc1 + 1 == n_sc1
          assert sc1.size() == sc2.size()
          assert sc1.size() == n_overlap
          trial_matching_sc1 = trial_ca_sites_cart[start_sc1:end_sc1+1]
          trial_matching_sc2 = trial_ca_sites_cart[
             start_matching_sc2:end_matching_sc2+1]
          assert trial_matching_sc1.size() == sc1.size()
          assert trial_matching_sc2.size() == sc2.size()
          # Superpose trial onto sc1 and sc2 as possible
          trial_sites = trial_matching_sc1.deep_copy()
          trial_sites.extend(trial_matching_sc2)
          target_sites = sc1.deep_copy()
          target_sites.extend(sc2)
          from scitbx.math import superpose
          fit_obj = superpose.least_squares_fit(
            reference_sites = target_sites,
            other_sites     =  trial_sites)
          fitted_trial_xyz = apply_lsq_fit(fit_obj, trial_sites)
          rmsd = (fitted_trial_xyz - target_sites).rms_length()
          print("RMSD for fitting trial model on ends: %.2f" %(rmsd),
            file = self.log)
          original_trial_model= mmm.model_from_sites_cart(trial_sites,
           return_model = True)
          trial_model= mmm.model_from_sites_cart(fitted_trial_xyz,
           return_model = True)
          target_model= mmm.model_from_sites_cart(target_sites,
           return_model = True)
          working_ca_model = mmm.model_from_sites_cart(
             apply_lsq_fit(fit_obj, trial_ca_sites_cart), return_model = True)
          # get a morphing field to apply
          shift_field_info = group_args(
             group_args_type = 'shift_field_info',
             centers = target_sites,
             shifts = (target_sites - fitted_trial_xyz))
          # Apply the shift field to all trial sites
          fitted_all_trial_xyz = apply_lsq_fit(fit_obj, trial_ca_sites_cart)
          from phenix.model_building.ssm import get_shifts
          shifts = get_shifts(self.get_ssm_params(),
             shift_field_info.shifts,
            shift_field_info.centers, sites_cart = fitted_all_trial_xyz)
          all_ca_sites = fitted_all_trial_xyz + shifts
          ca_model = mmm.model_from_sites_cart(
             all_ca_sites, return_model = True)
          # Now use the ends in sc1 and sc2:
          new_all_ca_sites = all_ca_sites.deep_copy()
          new_all_ca_sites = full_sc1.deep_copy()
          new_all_ca_sites.extend(all_ca_sites[end_sc1+1:start_matching_sc2])
          new_all_ca_sites.extend(full_sc2)
          all_ca_sites = new_all_ca_sites
          ca_model = mmm.model_from_sites_cart(
             all_ca_sites, return_model = True)
          if self.info.debug:
            print("Morphed spliced model:",ca_model.model_as_pdb())
          # Now get model directly from pulchra and not reconstruction...
          full_segment = mmm.model_building().reverse_one_model(ca_model,
            restrain_ends = True,
            ca_only = False,
            refine_cycles = refine_cycles,
            keep_direction = True)
          test_result = group_args(
            group_args_type = 'dummy result',
            model = full_segment)
          if not result or not result.model or (
            test_result.model.get_hierarchy().overall_counts().n_residues >
               result.model.overall_counts().n_residues):
            print("Using morphed model with %s residues" %(
               test_result.model.get_hierarchy().overall_counts().n_residues),
              file = self.log)
            result = test_result
          else:
            print("Using extended model with %s residues" %(
               result.model.overall_counts().n_residues), file = self.log)
        else: #
          all_ca_sites = sc1
          all_ca_sites.extend(sc2)
          print("Just connecting tracings with %s sites" %(all_ca_sites.size()),
               file = self.log)
          result = None
      else: # usual
        all_ca_sites = sc1[:i1]
        all_ca_sites.extend(result.model.get_sites_cart())
        all_ca_sites.extend(sc2[i2+1:])
        print("Obtained trace_away connection with %s sites" %(
           all_ca_sites.size()),
               file = self.log)
      if result and result.model:
        model = result.model
      else:
        model = get_model_from_path(all_ca_sites,
         as_ca_model = as_ca_model,
        ca_distance = ca_distance,
        mmm = mmm)
      return model
  def _trace_away_from_sites(self,
      mmm,
      sites_cart,
      minimum_density = 0,
      ca_distance = 3.8,
      as_ca_model = None,
      reverse_ca_model = None,
      guide_model = None,
      max_tries_to_get_more_residues = 3,
      max_time = None,
      minimum_residues = None,
      exact_number_of_residues_to_return = None,
      n_threshold = 10,
      group_args_type = None,
      avoid_sites_radius= None,
      max_branch_length = None,
      return_sites_cart = False):
    """ Trace through density in mmm away from sites_cart
       Try to use the last site_cart that is in significant density
      If guide_model is set, consider following a path along one
      of them"""
    map_data = mmm.map_manager().map_data()
    crystal_symmetry = mmm.crystal_symmetry()
    sites_frac = mmm.crystal_symmetry().unit_cell().fractionalize(sites_cart)
    # Find threshold that gives about the target path length
    from phenix.autosol.trace_and_build import trace_and_build, \
      get_bool_region_mask, get_indices_one_d_from_indices
    from iotbx.map_manager import get_indices_from_index,\
      get_sites_cart_from_index
    tnb = trace_and_build( params = group_args(
     crystal_info = group_args(wrapping = False),
     control = group_args(verbose = False),
      ),
      )
    typical_density_list = mmm.map_manager().density_at_sites_cart(sites_cart)
    last_3_density= typical_density_list[-3:]
    max_of_last_3 = last_3_density.min_max_mean().max
    # See which site near end to use
    best_k = None
    for k in range(last_3_density.size()):
      value = last_3_density[-1 - k]
      if value >= 0.5 * max_of_last_3:
         best_k = k
         break
    if best_k is None:
      best_k = 0
    index_to_use = sites_cart.size() - 1 - best_k # position in sites_cart
    if exact_number_of_residues_to_return:
      target_path_length = ca_distance * (
         exact_number_of_residues_to_return - 1)
    else:
      target_path_length = None
    if minimum_residues:
      minimum_path_length = ca_distance * (minimum_residues - 1)
    else:
      minimum_path_length = None
    print("\nTracing away from site %s ..." %(
       str(sites_cart[index_to_use])), file = self.log)
    # See if we can extend directly with guide_model
    if guide_model:
      all_sites_cart = self._get_sites_from_guide_model(guide_model,
        start_point = sites_cart[index_to_use],
        previous_point = sites_cart[max(0, index_to_use - 1)],
        target_path_length = target_path_length,
        minimum_path_length = minimum_path_length,
        ca_distance = ca_distance,
        )
      if all_sites_cart.size() > 0 :
        print("\nObtained sites  top from guide model\n", file = self.log)
      else:
        print("No sites obtained with guide model", file = self.log)
    else: # usual
      all_sites_cart = flex.vec3_double()
    working_target_path_length = target_path_length
    working_minimum_path_length = minimum_path_length
    indices_list = tnb.get_indices_list_from_sites_frac(
      sites_frac = sites_frac,
      map_data = map_data)
    indices_1d = get_indices_one_d_from_indices(indices_list,
       all = map_data.all())
    start_index = indices_1d[index_to_use]
    print("Trying to extend chain with %s sites..." %(
       all_sites_cart.size()), file = self.log)
    for cycle in range(max_tries_to_get_more_residues):
      print("\nCycle %s of trace_density_from_point" %(cycle + 1),
          file = self.log)
      best_path_info = trace_density_from_point(
        threshold_min = minimum_density,
        threshold_max = typical_density_list[index_to_use],
        mmm = mmm,
        target_path_length = working_target_path_length,
        minimum_path_length = working_minimum_path_length,
        n_threshold = n_threshold,
        max_tries_to_get_more_residues = max_tries_to_get_more_residues,
        index_to_use = index_to_use,
        map_data = map_data,
        indices_list = indices_list,
        start_index = start_index,
        ca_distance = ca_distance,
        max_time = max_time,
        max_branch_length = max_branch_length,
        )
      if best_path_info:
        sites_cart = tnb.get_sites_cart_from_index(
          points = best_path_info.path_points,
          map_data = map_data,
          crystal_symmetry = mmm.crystal_symmetry())
        # Make sure there are no overlaps (i.e., it didn't go backwards)
        n_close = count_close_sites(sites_to_count = sites_cart,
           sites_to_match = all_sites_cart)
        if n_close > 2 and n_close >= 0.1 * sites_cart.size(): # skip it
          print("Skipping this cycle due to overlap (%s of %s sites are dups)"%(
           n_close,sites_cart.size()),
            file = self.log)
          break
        if all_sites_cart.size() > 1:
          all_sites_cart = all_sites_cart[:-1]
        all_sites_cart.extend(sites_cart)
        path_length = get_path_length(all_sites_cart)
        if target_path_length is not None and \
           path_length >= target_path_length - ca_distance: # not worth going on
          break # done
      else:
        break  # done
      # Still going on here
      if working_target_path_length:
        working_target_path_length -= path_length
      if working_minimum_path_length:
        working_minimum_path_length -= path_length
      sites_frac = mmm.crystal_symmetry().unit_cell().fractionalize(
         all_sites_cart)
      indices_list = tnb.get_indices_list_from_sites_frac(
         sites_frac = sites_frac,
         map_data = map_data)
      indices_1d = get_indices_one_d_from_indices(indices_list,
         all = map_data.all())
      index_to_use = all_sites_cart.size() - 1 # take last one
      start_index = indices_1d[index_to_use]
      avoid_these_sites = all_sites_cart[:-4] # keep a few
      mmm = mmm.deep_copy() # now mask further around sites
      mmm.mask_all_maps_around_atoms(
        invert_mask = True,  # keep outside
        model = mmm.model_from_sites_cart(
           avoid_these_sites, return_model = True),
        mask_atoms_atom_radius = avoid_sites_radius,)
      typical_density_list = mmm.map_manager().density_at_sites_cart(
         all_sites_cart)
      map_data = mmm.map_data()
    print("End of trace away from sites... %s sites" %(
       all_sites_cart.size()), file = self.log)
    # end of cycles
    if return_sites_cart:
      return all_sites_cart
    model = get_model_from_path(all_sites_cart,
        as_ca_model = as_ca_model,
        ca_distance = ca_distance,
        reverse_ca_model = reverse_ca_model,
        mmm = mmm)
    result = group_args(
        group_args_type = group_args_type,
        model = model,
        model_with_exact_residue_count = None,
        sites_cart = all_sites_cart,
        threshold = None,
        path_length = None,
        region_map = None,
         )
    if exact_number_of_residues_to_return:
        self.add_model_with_exact_residues(result,
          exact_number_of_residues_to_return)
    return result
  def _get_sites_from_guide_model(self,
        guide_model,
        start_point = None,
        previous_point = None,
        target_path_length = None,
        minimum_path_length = None,
        ca_distance = None,
        ):
    fragments_as_sites_list = self._get_guide_sites_from_guide_model(
      guide_model)
    best_info = None
    # Now see if we can use one of these fragments_as_sites
    for sc in fragments_as_sites_list:
      info = self._get_extension_sites(
        sc,
        start_point = start_point,
        previous_point = previous_point,
        target_path_length = target_path_length,
        minimum_path_length = minimum_path_length,
        ca_distance = ca_distance)
      if not info:
        continue
      if best_info is None or  \
         abs(info.path_length - target_path_length) < \
         abs(best_info.path_length - target_path_length):
        best_info = info
    if best_info:
      return best_info.path_as_sites
    else:
      return flex.vec3_double()
  def _get_connection_sites_with_guide(self,
        sc,
        start_point = None,
        end_point = None,
        target_path_length = None,
        minimum_path_length = None,
        ca_distance = None):
     """ Try to start at start_point and go to end_point using points in sc """
     target_sc = flex.vec3_double()
     target_sc.append(start_point)
     target_sc.append(end_point)
     if start_point == end_point: # can't tell which way to go
       return None
     if sc.size() < 3: # nothing to do
       return None
     # find match to start
     dist, id1, id2 = target_sc[0:1].min_distance_between_any_pair_with_id(sc)
     if dist > ca_distance: # too far
       return None
     dist_end, id1_end, id2_end = target_sc[1:2
          ].min_distance_between_any_pair_with_id(sc)
     if dist_end > ca_distance: # too far
       return None
     # Now see which way to go
     sites = sc[id2:id2_end + 1]
     if id2_end < id2:  # reverse it
       sites = list(sites)
       sites.reverse()
       sites = flex.vec3_double(sites)
     return group_args(
       group_args_type = 'path as sites',
       path_as_sites = sites,
       path_length = get_path_length(sites))
  def _get_extension_sites(self,
        sc,
        start_point = None,
        previous_point = None,
        target_path_length = None,
        minimum_path_length = None,
        ca_distance = None):
     """ Try to start at start_point and go in direction of prev -> start
       using points in sc """
     target_sc = flex.vec3_double()
     target_sc.append(previous_point)
     target_sc.append(start_point)
     if start_point == previous_point: # can't tell which way to go
       have_direction = False
     else:
       have_direction = True
     if sc.size() < 3: # nothing to do
       return None
     dist, id1, id2 = target_sc[1:2].min_distance_between_any_pair_with_id(sc)
     if dist > ca_distance: # too far
       return None
     # Now see which way to go
     sites = sc[max(0,id2-1):min(sc.size()-1,id2+1)]
     if sites.size() < 2:
       return None # no sites
     if have_direction:
       dir_sites = col(sites[1]) - col(sites[0])
       dir_target = col(start_point) - col(previous_point)
       dot = dir_sites.dot(dir_target)
       if dot > 0: # forward
         forward = True
       else:
         forward = False
     else:  # test both possibilities
       sites_forward = sc[id2:]
       sites_reverse = sc[:id2+1]
       density_forward = self.map_manager().density_at_sites_cart(sites_forward)
       density_reverse = self.map_manager().density_at_sites_cart(sites_reverse)
       if density_forward.min_max_mean().mean >= \
          density_reverse.min_max_mean().mean:
         forward = True
       else:
         forward = False
     if forward:
       sites_to_consider = sc[id2:]
     else: # reverse
       sites_to_consider = list(sc[:id2+1])
       sites_to_consider.reverse()
       sites_to_consider = flex.vec3_double(sites_to_consider)
     from phenix.model_building.model_completion import insert_sites_between
     path_as_sites = insert_sites_between(sites_to_consider)
     # Now take the length of path we want
     path_as_sites = self._trim_path_to_target_length(path_as_sites,
         target_path_length)
     return group_args(
       group_args_type = 'path as sites',
       path_as_sites = path_as_sites,
       path_length = get_path_length(path_as_sites))
  def _trim_path_to_target_length(self, sites_cart, target_path_length):
    if not target_path_length:
      return sites_cart
    if sites_cart.size()<2:
      return sites_cart
    deltas = sites_cart[1:] - sites_cart[:-1]
    norms = deltas.norms()
    sum_length = 0
    for i in range(norms.size()):
      sum_length += norms[i]
      if sum_length > target_path_length:
        return sites_cart[:i+1]
    return sites_cart
  def _get_guide_sites_from_guide_model(self, model):
    from mmtbx.secondary_structure.find_ss_from_ca import model_info, \
       split_model
    fragments_as_sites_list = []
    model.add_crystal_symmetry_if_necessary(
      crystal_symmetry = self.map_manager().crystal_symmetry())
    mm = model.apply_selection_string('name ca or name p')
    for m in split_model(model_info(hierarchy=mm.get_hierarchy())):
      fragments_as_sites_list.append(m.hierarchy.atoms().extract_xyz())
    return fragments_as_sites_list
  def trace_between_sites_through_density(self,
      site_cart_1 = None,
      site_cart_2 = None,
      sites_cart = None,
      as_ca_model = True,
      reverse_ca_model = False,
      minimum_density = 0,
      ca_distance = 3.8,
      sampling_ratio = 1,
      avoid_these_sites = None,
      guide_model = None,
      avoid_sites_radius = 3,
      return_region_map = False,
      max_iter = 2,
      max_tries_to_get_more_residues = 3,
      max_time = None,
      minimum_residues = None,
      exact_number_of_residues_to_return = None,
      max_branch_length = 15,
      allow_connect_with_trace_away_from_sites = True,
      trial_model = None,
      refine_cycles = 1,
      allow_masking = True,
      ):
    '''
      Return a model or sites_cart tracing through high density
      between two supplied sites_cart.
      Also returns threshold used, path length, and map of the
       density used (map only if return_region_map = True)
      If avoid_these_sites (flex.vec3_double) is supplied, mask around
        those atoms first to avoid tracing through them
      If minimum_residues is set and fewer are found, try again after blocking
       density near the place where the chain turns around, if possible
      If exact_number_of_residues_to_return is set...return that many.
      If site_cart_1 and site_cart_2 are None and sites_cart is set,
      just extend away from one of the last sites in sites_cart
      If allow_connect_with_trace_away_from_sites and trial_model, try
       to trace density from each end, then morph trial_model onto that
       set of chains. trial_model must match residue numbering of loop fragment
       expected.
      If guide_model are supplied try using pieces of them to guide the path
    '''
    kw_args = group_args()
    adopt_init_args(kw_args, locals())
    kw_args.minimum_residues = 0
    kw_args.allow_masking = False
    del kw_args.kw_args
    if minimum_residues is None:
       minimum_residues = 1
    if allow_masking and \
       minimum_residues and not sites_cart:  # require this many residue
      # can't use this method if using sites_cart
      # Try to connect as is, no minimum
      result = self.trace_between_sites_through_density(**kw_args())
      if result and result.sites_cart and \
          get_path_length(result.sites_cart) >= minimum_residues * ca_distance:
        print("Long enough chain obtained...(%s residues)" %(
          get_path_length(result.sites_cart)//ca_distance), file = self.log)
        return result
      elif result and not result.model:
        print("Nothing found...skipping", file = self.log)
        return result
      print("Trying to mask density to get longer tracing...", file = self.log)
      best_result = result
      from phenix.model_building.model_completion import get_gap_list, \
          zero_map_near_bad_sites
      previous_gap_lists = []
      for cy in range(max_tries_to_get_more_residues):
        print("\nCycle %s of getting more residues" %(cy+1), file = self.log)
        new_mmm = self.as_map_model_manager().deep_copy()
        gap_list = get_gap_list(best_result.model,
           new_mmm,
           just_one_gap = True,
           take_furthest_if_no_gap = True,
           log = self.log)
        if not gap_list:
          print("Unable to mask density ", file = self.log)
          return best_result
        gap = gap_list[0]
        zero_map_near_bad_sites(new_mmm, model = gap.bad_sites_model)
        new_build = new_mmm.model_building()
        new_result= new_build.trace_between_sites_through_density(**kw_args())
        if new_result and new_result.sites_cart and \
            get_path_length(new_result.sites_cart) \
                >= minimum_residues * ca_distance:
          print("Long enough chain obtained...(%s residues)" %(
            get_path_length(new_result.sites_cart)//ca_distance),
              file = self.log)
          return new_result
        elif new_result and new_result.sites_cart and \
            new_result.sites_cart.size() > result.sites_cart.size():
          best_result = new_result
          print("Best result (%s residues) found this cycle" %(
           get_path_length(new_result.sites_cart)//ca_distance),
           file = self.log)
        elif not new_result: # didn't work
          print("No result this cycle...stopping", file = self.log)
          break
      # Return the best we have found (not enough)
      print("Long chain not found in ",
         "%s tries...going with %s residues (%s from path length)" %(
         max_tries_to_get_more_residues,
          best_result.model.get_hierarchy().overall_counts().n_residues,
          get_path_length(best_result.sites_cart)//ca_distance),
          file = self.log)
      return best_result
    from phenix.autosol.trace_and_build import trace_and_build, \
      get_bool_region_mask, get_indices_one_d_from_indices
    from iotbx.map_manager import get_indices_from_index,\
       get_sites_cart_from_index
    if sampling_ratio and sampling_ratio != 1:
       # We are going to get rid of shifts and apply only at the end
       mmm = self.deep_copy().as_map_model_manager()
       mmm.set_original_origin_grid_units((0,0,0)) # remove shift
       mmm = mmm.as_map_model_manager_with_resampled_maps(
           sampling_ratio = sampling_ratio)
       saved_mmm = mmm.deep_copy()
    else:
      mmm = self.deep_copy().as_map_model_manager()
      saved_mmm = None
    if avoid_these_sites:  # mask the map around these sites
      mmm.mask_all_maps_around_atoms(
        invert_mask = True,  # keep outside
        model = mmm.model_from_sites_cart(
           avoid_these_sites, return_model = True),
        mask_atoms_atom_radius = avoid_sites_radius,)
    if site_cart_1 and site_cart_2:
      sc12 = flex.vec3_double()
      sc12.append(site_cart_1)
      sc12.append(site_cart_2)
      sites_frac = mmm.map_manager(
          ).crystal_symmetry().unit_cell().fractionalize( sc12)
      adjusted_sites_frac = adjust_sites_frac_to_match_best_grid_point(
        sites_frac, mmm.map_manager().map_data())
      sc12 = mmm.map_manager().crystal_symmetry().unit_cell().orthogonalize(
        adjusted_sites_frac)
      site_cart_1 = sc12[0]
      site_cart_2 = sc12[1]
      group_args_type = \
        'trace through density between '+\
         '(%.2f, %.2f, %.2f) and %.2f, %.2f, %.2f)' %(
         tuple(list(site_cart_1)+list(site_cart_2)))
    else:
      assert sites_cart is not None
      group_args_type = \
        'trace through density away from '+\
         '(%.2f, %.2f, %.2f)' %(
         tuple(sites_cart[-1]))
      return self._trace_away_from_sites(mmm, sites_cart,
        minimum_density = minimum_density,
        ca_distance = ca_distance,
        as_ca_model = as_ca_model,
        reverse_ca_model = reverse_ca_model,
        guide_model = guide_model,
        max_tries_to_get_more_residues = max_tries_to_get_more_residues,
        max_time = max_time,
        minimum_residues = minimum_residues,
        exact_number_of_residues_to_return = exact_number_of_residues_to_return,
        group_args_type = group_args_type,
        max_branch_length = max_branch_length,
        avoid_sites_radius = avoid_sites_radius,
        )
    # Special case: can we just connect these points with a line?
    dist = (col(site_cart_1) - col(site_cart_2)).length()
    if minimum_residues <= 3 and dist <= ca_distance * 1.5:
      value = mmm.map_manager().get_density_along_line(
         site_cart_1,site_cart_2
         ).along_density_values.min_max_mean().min
      if value >= minimum_density:
        # Just connect:
        result = group_args(
          group_args_type = group_args_type,
          model_with_exact_residue_count = None,
          model = self.as_map_model_manager().model_from_sites_cart(
            sc12, return_model = True), # Note self knows shift_cart still
          sites_cart = sc12,
          threshold = value,
          path_length = dist,
          region_map = None,
           )
        if exact_number_of_residues_to_return:
          self.add_model_with_exact_residues(result,
              exact_number_of_residues_to_return)
        return result
    if guide_model:
      connection_sites = self._get_connection_from_guide_model(sc12,
        guide_model = guide_model,
        ca_distance = ca_distance)
      if connection_sites:
        result = group_args(
          group_args_type = group_args_type,
          model_with_exact_residue_count = None,
          model = self.as_map_model_manager().model_from_sites_cart(
            connection_sites, return_model = True),
          sites_cart = connection_sites,
          threshold = None,
          path_length = get_path_length(connection_sites),
          region_map = None,
           )
        if exact_number_of_residues_to_return:
          self.add_model_with_exact_residues(result,
              exact_number_of_residues_to_return)
        print("\nObtained trace_between sites from guide model\n",
            file = self.log)
        return result
    print("Looking for standard connection ",
       "with %s target and %s minimum residues" %(
       exact_number_of_residues_to_return, minimum_residues), file = self.log)
    map_data = mmm.map_manager().map_data()
    crystal_symmetry = mmm.crystal_symmetry()
    sites_frac = mmm.crystal_symmetry().unit_cell().fractionalize(sc12)
    # Find highest threshold that just puts both points in the same region
    threshold_info = find_threshold_to_connect_points(
        mmm, sc12[0],sc12[1],
        minimum_density = minimum_density,
         log = self.log)
    if threshold_info and threshold_info.threshold is not None:
      indices1, indices2 = threshold_info.tnb.get_indices_list_from_sites_frac(
        sites_frac = sites_frac,
        map_data = map_data)
    else:
      threshold_info = None
    if not threshold_info:
      max_iter = 0 # skip
    for iter in range(max_iter):
      print("\nIteration %s of %s" %(iter + 1, max_iter), file = self.log)
      density_values_list = [ map_data[x] for x in
         threshold_info.points_in_region_info.points_in_region]
      density_values = flex.double(density_values_list)
      test_cutoff = density_values.min_max_mean().min
      # Now get points just above threshold
      info = two_points_in_same_region_at_this_threshold(
        tnb = threshold_info.tnb,
        map_data = map_data,
        indices1 = indices1,
        indices2 = indices2,
        threshold = test_cutoff)
      if info.same_region: # did not find it...skip iterations
        break
      low_i = density_values_list.index(test_cutoff)
      low_index = threshold_info.points_in_region_info.points_in_region[low_i]
      # Now we are going to adjust map_data to increase it for all points
      # inside our region that are neighbors of low_index
      neighbor_indices = get_neighbor_indices_inside_region(
         threshold_info = threshold_info,
         map_data = map_data,
         index = low_index,
         depth = 2)
      highest_local_map_values = map_data[low_index]
      for i in neighbor_indices:
        highest_local_map_values = max(highest_local_map_values, map_data[i])
      map_data[low_index] = 0.5 * (
         map_data[low_index] + highest_local_map_values)
      for i in neighbor_indices:
        map_data[i] = 0.5 * (map_data[i] + highest_local_map_values)
      print("Set values near saddle point from %.2f to halfway to %.2f" %(
        test_cutoff,highest_local_map_values), file = self.log)
      # Now make sure these are connected
      info = two_points_in_same_region_at_this_threshold(
        tnb = threshold_info.tnb,
        map_data = map_data,
        indices1 = indices1,
        indices2 = indices2,
        threshold = test_cutoff)
      if not info.same_region:
        break # should not happen
      # Find highest threshold that just puts both points in the same region
      print("Recalculating threshold...", file = self.log)
      threshold_info = find_threshold_to_connect_points(
         mmm.deep_copy(), sc12[0],sc12[1],
         minimum_density = minimum_density,
          log = self.log)
    if (not threshold_info) or (not threshold_info.points_in_region_info) or \
       threshold_info.points_in_region_info.region_map is None:
      print("Unable to connect sites", file = self.log)
      evaluate_path_info = None
    else:
      print("Common region found for %s" %(group_args_type), file = self.log)
      print("Region threshold: %.2f " %(
        threshold_info.threshold), file = self.log)
      evaluate_path_info = get_path_through_points(mmm,
        max_time = max_time,
       points_in_region = threshold_info.points_in_region_info.points_in_region,
        ends_sites_cart = sc12)
    if evaluate_path_info and evaluate_path_info.branch_best_path_info and \
        evaluate_path_info.branch_best_path_info.longest_branch_length \
            <= max_branch_length:
      best_path_info = evaluate_path_info.best_path_info
      longest_branch_length = \
        evaluate_path_info.branch_best_path_info.longest_branch_length
      path_length = best_path_info.path_length
      path_points = best_path_info.path_points
      sites_cart = threshold_info.tnb.get_sites_cart_from_index(
        points = path_points,
        map_data = mmm.map_manager().map_data(),
        crystal_symmetry = mmm.crystal_symmetry())
      print("Connection found with path_length of %.2f A " %(
        path_length), file = self.log)
      model = get_model_from_path(sites_cart, as_ca_model = as_ca_model,
        ca_distance = ca_distance,
        reverse_ca_model = reverse_ca_model,
        mmm = self)
      result = group_args(
        group_args_type = group_args_type,
        model = model,
        model_with_exact_residue_count = None,
        sites_cart = sites_cart,
        threshold = threshold_info.threshold,
        path_length = path_length,
        region_map = (threshold_info.points_in_region_info.region_map
            if return_region_map else None),
         )
      if exact_number_of_residues_to_return:
        residues_found = model.get_hierarchy().overall_counts().n_residues
        if abs(exact_number_of_residues_to_return - residues_found) <= \
             2 + exact_number_of_residues_to_return * 0.75:
          self.add_model_with_exact_residues(result,
            exact_number_of_residues_to_return)
          return result  # got it
        else:
          print("Length of path too different from target... skipping",
            file = self.log)
      else:
        return result # got it
    if sc12.size() == 2 and allow_connect_with_trace_away_from_sites:
      # Try to extend away from each end and see if these can connect
      # Start over with unmodified mmm
      if saved_mmm:
        mmm = saved_mmm
      else:
        mmm = self.deep_copy().as_map_model_manager()
      print(
       "Trying to connect by tracing away from ends (direct connection failed)",
           file = self.log)
      group_args_type = \
        'trace through density by connecting trace away',
      connection_model = self._connect_with_trace_away_from_sites(mmm, sc12,
        minimum_density = minimum_density,
        ca_distance = ca_distance,
        as_ca_model = as_ca_model,
        reverse_ca_model = reverse_ca_model,
        guide_model = guide_model,
        max_tries_to_get_more_residues = 1,
        max_time = max_time,
        minimum_residues = minimum_residues,
        exact_number_of_residues_to_return = exact_number_of_residues_to_return,
        group_args_type = group_args_type,
        max_branch_length = max_branch_length,
        avoid_sites_radius = avoid_sites_radius,
        kw_args = kw_args,
        trial_model = trial_model,
        refine_cycles = refine_cycles,
        )
      if connection_model:
        print("Connected sites with tracing away from ends", file = self.log)
        print("Connection model has %s residues (target of %s) "%(
         connection_model.get_hierarchy().overall_counts().n_residues,
         exact_number_of_residues_to_return), file = self.log)
        group_args_type = ' Connection model'
        result = group_args(
          group_args_type = group_args_type,
          model = connection_model,
          model_with_exact_residue_count = None,
          threshold = None,
          sites_cart = connection_model.apply_selection_string('name ca'
            ).get_sites_cart(),
          path_length = None,
         )
        if exact_number_of_residues_to_return:
           self.add_model_with_exact_residues(result,
             exact_number_of_residues_to_return)
        print("Connected with tracing away from ends", file = self.log)
        return result
      else:
        print("Did not connect with tracing away from ends", file = self.log)
    print("No connection found", file = self.log)
    return group_args(
        group_args_type = group_args_type,
        model = None,
        model_with_exact_residue_count = None,
        sites_cart = None,
        threshold = None,
        path_length = None,
        region_map = None)
  def _get_connection_from_guide_model(self, sc12,
        guide_model = None,
        target_path_length = None,
        minimum_path_length = None,
        ca_distance = None):
    fragments_as_sites_list = self._get_guide_sites_from_guide_model(
      guide_model)
    best_info = None
    # Now see if we can use one of these fragments_as_sites
    for sc in fragments_as_sites_list:
      info = self._get_connection_sites_with_guide(
        sc,
        start_point = sc12[0],
        end_point = sc12[1],
        target_path_length = target_path_length,
        minimum_path_length = minimum_path_length,
        ca_distance = ca_distance)
      if not info: continue
      if best_info is None or  \
         abs(info.path_length - target_path_length) < \
         abs(best_info.path_length - target_path_length):
        best_info = info
    if best_info:
      print("\nObtained connection sites from guide model\n", file = self.log)
      return best_info.path_as_sites
    else:
      print("\nDid not obtain sites from guide model\n", file = self.log)
      return None
  def add_model_with_exact_residues(self,result,
          exact_number_of_residues_to_return):
    sites_cart = result.sites_cart
    if sites_cart.size() == 0:
      result.model_with_exact_residue_count = None
      return # nothing to do
    path_length = get_path_length(sites_cart)
    approx_sites_per_ca = sites_cart.size()/exact_number_of_residues_to_return
    def interpolate(sites_cart, xx):
      import math
      ix = int(math.floor(xx))
      ix1 = ix + 1
      if  ix1 < 1: return sites_cart[0]
      if  ix > sites_cart.size() -2: return sites_cart[-1]
      ax = col(sites_cart[ix])
      ax1 = col(sites_cart[ix1])
      wx = (1 - (xx - ix))
      wx1 = 1 - wx
      return wx * ax + wx1 * ax1
    if not result.path_length:
      result.path_length = path_length
    if result.sites_cart.size() == exact_number_of_residues_to_return:
      result.model_with_exact_residue_count = result.model
      return
    as_ca_sites = flex.vec3_double( [interpolate(sites_cart,
          approx_sites_per_ca * i)
           for i in range(exact_number_of_residues_to_return)])
    result.model_with_exact_residue_count = \
        self.as_map_model_manager().model_from_sites_cart(
        as_ca_sites, return_model = True)
  def ca_to_model(self, model = None,
     sequence = None,
     use_ca_as_guide_only = False,
     trace_only = False,
     refine_cycles = None,
     first_resno = None):
    ''' return a full model based on CA in supplied model'''
    if not model:
      model = self.model()
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    assert chain_type.upper() == 'PROTEIN'  # ca_to_model only protein
    ca_model = model.select(
      model.selection(
      "(name ca) and (not water) and (not hetero) and element c"))
    if not sequence:
      from iotbx.bioinformatics import get_sequence_from_pdb
      sequence = get_sequence_from_pdb(
        hierarchy = ca_model.get_hierarchy())
    assert len(sequence) == ca_model.get_sites_cart().size() # sites and seq
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    self.set_defaults(build_methods=['trace_and_build'])
    full_model = self.build(
     sequence = sequence,
     starting_model = ca_model,
     use_starting_model_ca_directly = True,
     use_starting_model_ca_as_guide = use_ca_as_guide_only,
     trace_only = trace_only,
     remove_clashes = False,
     return_model = True,
     mask_around_model = False,
     refine_cycles = refine_cycles)
    if not full_model:
      return None # nothing to do
    if (not residues_are_sequential(full_model)):
      from mmtbx.secondary_structure.find_ss_from_ca import get_first_resno,\
        merge_and_renumber_everything
      ph = full_model.get_hierarchy()
      target_first_resno = first_resno if first_resno is not None else \
         get_first_resno(ca_model.get_hierarchy())
      ph = merge_and_renumber_everything(ph, current_resno = target_first_resno)
      full_model = self.as_map_model_manager().model_from_hierarchy(
          ph, return_as_model = True)
      # Redo the sequence/
      full_model = self.sequence_from_map(return_model = True,
        model = full_model,
        sequence = sequence)
      if not full_model:
        return None # failed
    # Set the first resno if asked
    if first_resno is not None:
      from mmtbx.secondary_structure.find_ss_from_ca import get_first_resno
      current_first_resno = get_first_resno(full_model.get_hierarchy())
      if first_resno != current_first_resno:
        from mmtbx.secondary_structure.find_ss_from_ca import renumber_residues
        ph = full_model.get_hierarchy()
        renumber_residues(ph, first_resno = first_resno)
        full_model = self.as_map_model_manager().model_from_hierarchy(
          ph, return_as_model = True)
    return full_model
  def reverse_direction(self, model = None,
     restrain_ends = True,
     ca_only = False,
     allow_non_sequential = False,
     new_first_resno = None,
     refine_cycles = None):
    ''' return a model with reversed direction (protein only)'''
    if not model:
       model = self.model()
    if not model:
      return None # nothing to do
    # Set refine cycles based on quick/not quick
    refine_cycles = self.set_refine_cycles(refine_cycles)
    from iotbx.bioinformatics import get_chain_type
    chain_type = get_chain_type(model = model)
    assert chain_type.upper() == 'PROTEIN'  # reverse-direction only protein
    # Determine if model has fragments. If so, run on each
    model_list = self.split_model_by_segments(model, chain_type = chain_type)
    if not model_list:
      return None # nothing to do
    new_model = None
    from phenix.model_building.fit_loop import catenate_segments
    from mmtbx.secondary_structure.find_ss_from_ca import \
        get_first_resno, get_last_resno, get_chain_id
    for model in model_list: # run on each model
      ph = model.get_hierarchy()
      print("\nReversing segment 'chain %s resseq %s:%s' " %(
        get_chain_id(ph), get_first_resno(ph), get_last_resno(ph)),
        file = self.log)
      reverse_segment = self.reverse_one_model(model,
         restrain_ends = restrain_ends,
         ca_only = ca_only,
         allow_non_sequential = allow_non_sequential,
         new_first_resno = new_first_resno,
         refine_cycles = refine_cycles)
      if reverse_segment:
        new_first_resno = get_last_resno(reverse_segment.get_hierarchy()) + 10
        if new_model:
          new_model = catenate_segments(new_model, reverse_segment,
            keep_numbers=True)
        else:
           new_model = reverse_segment
        print("Reversed model with %s residues obtained for this segment" %(
          reverse_segment.get_hierarchy().overall_counts().n_residues),
           file = self.log)
      else:
        print("No reversed model obtained for this segment", file = self.log)
    return new_model
  def reverse_one_model(self, model,
     restrain_ends = True,
     ca_only = False,
     allow_non_sequential = False,
     new_first_resno = None,
     refine_cycles = None,
     keep_direction = False):
    ''' return a model with reversed direction (protein only)
        if keep_direction then just run pulchra'''
    ca_model =model.select(
      model.selection("(name ca) and (not water) and (not hetero)"))
    sites_cart_list = list(ca_model.get_sites_cart())
    if (not keep_direction):
      sites_cart_list.reverse()
    if not sites_cart_list:
      return None # nothing to do
    from mmtbx.secondary_structure.find_ss_from_ca import \
      get_first_resno,get_last_resno
    first_resno = get_first_resno(ca_model.get_hierarchy())
    if new_first_resno is None:
      new_first_resno = first_resno
    last_resno = get_last_resno(ca_model.get_hierarchy())
    if first_resno is None or last_resno is None:
      return None
    if last_resno - first_resno + 1 != len(sites_cart_list):
      if allow_non_sequential:
        pass # just go ahead
    else:
      assert last_resno - first_resno + 1 == len(sites_cart_list) # not sequential
    new_last_resno = new_first_resno + len(sites_cart_list) - 1
    assert new_last_resno <= 9999  # pulchra takes up to 9999 only
    resseq_list =list(range(new_first_resno,new_last_resno+1))
    from mmtbx.model import manager
    reverse_ca_model = manager.from_sites_cart(
     sites_cart = flex.vec3_double(sites_cart_list),
     resseq_list = resseq_list,
     crystal_symmetry = ca_model.crystal_symmetry())
    if not ca_only:
      from solve_resolve.resolve_python.run_local_trace import \
        run_pulchra_one_segment
      # NOTE: pulchra does not use hybrid_36 ... residue numbers <=  9999 only
      segment = reverse_ca_model.model_as_pdb()
      segment = segment.replace("GLY", "ALA")
      try:
        new_hierarchy, end_residue_number = run_pulchra_one_segment(segment,
          return_hierarchy = True, start_residue = new_first_resno,
            out = null_out())
      except Exception as e:
        new_hierarchy = None
    else:
      new_hierarchy = None
    if new_hierarchy:
      from iotbx.data_manager import DataManager
      dm = DataManager()
      dm.process_model_str( 'text',new_hierarchy.as_pdb_string())
      reverse_segment = dm.get_model('text')
    else:
      reverse_segment = reverse_ca_model
    reverse_segment.set_crystal_symmetry(ca_model.crystal_symmetry())
    if ca_model.unit_cell_crystal_symmetry():
      reverse_segment.set_unit_cell_crystal_symmetry(
       ca_model.unit_cell_crystal_symmetry())
    else:
      reverse_segment.set_unit_cell_crystal_symmetry(
       ca_model.crystal_symmetry())
    reverse_segment.set_shift_cart(ca_model.shift_cart())
    if refine_cycles and reverse_segment.get_hierarchy().overall_counts().n_residues > 1:
      reverse_segment = self.refine(model = reverse_segment,
        refine_cycles = refine_cycles,
        restrain_ends = restrain_ends,
        return_model = True)
    return reverse_segment
  def split_model_by_segments(self, model = None, chain_type = None):
    """  Split model up by segments (change of residue numbering or chain
      or large gap) and return a list of models"""
    if not model:
       model = self.model()
    if not model:
      return None # nothing to do
    if not chain_type:
      from iotbx.bioinformatics import get_chain_type
      chain_type = get_chain_type(model = model)
    from mmtbx.secondary_structure.find_ss_from_ca import model_info,\
       split_model
    new_model_list = []
    mmm = self.as_map_model_manager()
    for m in split_model(model_info(hierarchy=model.get_hierarchy())):
      new_model = mmm.model_from_hierarchy(m.hierarchy, return_as_model = True)
      new_model_list.append(new_model)
    return new_model_list
  def _set_and_print_return_info(self, model= None,
    text="",
    return_model=None,
    return_model_must_be_true_for_supplied_model=None,
    return_model_is_ligand_or_fragment=None):
    if return_model_is_ligand_or_fragment: # just getting a piece
      # Always return it
      return_model = True
      if model:
        if self.map_manager():
          assert self.map_manager().is_compatible_model(model)
        if self.info.debug:
           print("Building %s using supplied model" %(text),
           file = self.log)
      else:
        model = self.model()
        if model:
          if self.info.debug:
            print("Building %s using working model" %(text),
             file = self.log)
        else:
          if self.info.debug:
            print("Building without model:\n%s" %(text),
             file = self.log)
      if self.info.debug:
        print("New %s will be returned (existing model not modified)" %(text),
          file = self.log)
    else:  # building or inserting
      if model:
        assert self.map_manager().is_compatible_model(model)
        if return_model is None:
          return_model = True
        if return_model_must_be_true_for_supplied_model:
          assert return_model in [True] # must return model if supplied
        if self.info.debug:
          print("%s supplied model" %(text), file = self.log)
      else:
        if return_model:
          if self.model():
            model = self.model().deep_copy()
          else:
            model = None
        else:
          model = self.model()
        if model:
          print("%s using working model:" %(text), file = self.log)
        else:
          if self.info.debug:
            print("%s without existing model" %(text), file = self.log)
      if return_model and model:
        if self.info.debug:
          print("New model will be returned (existing model not modified)",
            file = self.log)
      elif return_model:
        if self.info.debug:
          print("New model will be returned", file = self.log)
      elif model:
        if self.info.debug:
          print("New model will replace existing model", file = self.log)
      else:
        if self.info.debug:
          print("New model will become the working model ", file = self.log)
    return model, return_model
  def _set_or_return_model(self,new_model,text="",return_model=None,
     change_model_in_place = None):
    # If change_model_in_place: modify existing model
    # Some routines return model relative to working map and without
    #  shift information.
    # Make sure new_model matches map_manager shifts and symmetries
    self.map_manager().set_model_symmetries_and_shift_cart_to_match_map(
       new_model)
    if return_model:
      print("Returning %s" %(text), file = self.log)
      return new_model
    else:
      print("Setting working model as %s" %(text), file = self.log)
      self.set_model(new_model, quiet = True,
         change_model_in_place = change_model_in_place)
  def _remove_incompatible_methods(self,method_type, methods_list, chain_type):
    # Remove methods that are incompatible with chain_type
    # method m looks like "do_something_with_this_method"
    #   We want to check to see if this_method is in ONLY_PROTEIN_METHODS,
    #   ONLY_RNA_METHODS or ONLY_DNA_METHODS
    new_methods = []
    for m in methods_list:
      mm = m.split("_with_")[-1]
      if (chain_type.upper() not in ['PROTEIN']) and (
           mm in ONLY_PROTEIN_METHODS):
        print ("Removing %s method as it is protein-only and this is %s" %(
           mm, chain_type), file = self.log)
      elif (chain_type.upper() not in ['RNA']) and (mm in ONLY_RNA_METHODS):
          print ("Removing %s method as it is RNA-only and this is %s" %(
            mm, chain_type), file = self.log)
      elif (chain_type.upper() not in ['DNA']) and (mm in ONLY_DNA_METHODS):
          print ("Removing %s method as it is DNA-only and this is %s" %(
            mm, chain_type), file = self.log)
      else:
        new_methods.append(m)
    return new_methods
  def _check_methods(self,
      methods_list = None,
      model = None,
      chain_id = None,
      chain_type = None,
      last_resno_before_loop = None,
      first_resno_after_loop = None):
    from phenix.model_building.fit_loop import get_residues_around_loop
    info = get_residues_around_loop(
        model,
        chain_id,
        last_resno_before_loop,
        first_resno_after_loop,
        residues_on_each_end = self.info.residues_on_each_end,
        residues_on_each_end_extended = self.info.residues_on_each_end_extended)
    if  ( (not info.residues_before_and_after) or
        info.residues_before_and_after.get_hierarchy(
          ).overall_counts().n_residues < 1):
      return []  # nothing can run
    if 'structure_search' in methods_list and \
       ( (not info.residues_before_and_after_extended) or
        info.residues_before_and_after_extended.get_hierarchy(
          ).overall_counts().n_residues < 1):
      from copy import deepcopy
      methods_list = deepcopy(methods_list)
      del methods_list[methods_list.index('structure_search')]
      print ("Not using structure_search as there are not \nsufficient"+
         " residues on each end (%s)" %(
        self.info.residues_on_each_end_extended),
           file = self.log)
    return methods_list
#     ==========   class to hold data for a multiprocessing run =======
class run_a_method_as_class:
  def __init__(self,kw_list):
    self.kw_list=kw_list
  def __call__(self,i):
    '''
     Run a model-building method with arguments in kw and expect
     a group_args object with attributes given by expected_result_names
     Returns group_args object that can be pickled
    '''
    # We are going to run with the i'th set of keywords
    kw=self.kw_list[i]
    # Get the method name and expected_result_names and remove them from kw
    method_name = kw['method_name']
    method_type = kw['method_type']
    expected_result_names = kw['expected_result_names']
    good_enough_score = kw['good_enough_score']
    debug = kw['debug']
    del kw['method_name']
    del kw['method_type']
    del kw['expected_result_names']
    del kw['good_enough_score']
    import importlib
    module = importlib.import_module('phenix.model_building.{}'.format(method_type))
    run = getattr(module, method_name)
    # Set up log stream and run either with a try (usual) or directly (debug)
    if (not debug): # usual run
      # Set up log stream
      log = StringIO()
      kw['log']=log
      # Run the method
      try:
        result = run(**kw)  # run is method_name in model_building_wrappers
        # Convert the log to text and save
        if result:
          result.log_as_text = log.getvalue()
      except Exception as e:
        print (log.getvalue())
        print("\n FAILURE IN METHOD %s \n\n ERROR MESSAGE:\n %s\n" %(
           method_name,str(e)))
        result=group_args()
        for x in expected_result_names:
          setattr(result,x,None)
      # ====   end usual run ==========
    else:  # debug run
      log = sys.stdout
      print ("\n DEBUG RUN OF %s ....VALUES OF ALL PARAMETERS:" %(method_name))
      for key in sorted(list(kw.keys())):
        print ("\nKEY: %s   VALUE:%s" %(key,str(kw[key])))
      print ("\n DEBUG RUN OF %s  END OF VALUES OF ALL PARAMETERS:" %(
        method_name))
      kw['log']=log
      result = run(**kw)  # run is method_name in model_building_wrappers
      if result:
        result.log_as_text=""
      # ====   end debug run ==========
    if not result:
      return None
    # Save some information about the run
    result.good_enough_score = good_enough_score
    result.run_id = kw['run_id']
    result.thoroughness= kw['thoroughness']
    # Save the name of method that was run
    result.method_name = method_name
    return result
#  ========  method to decide whether to continue with multiprocessing  =====
def break_if_good_enough_score(result):
  if result and hasattr(result,'score') and hasattr(result,'good_enough_score') and (result.score is not None) and (result.good_enough_score is not None) and (result.score >= result.good_enough_score):
    return True
  else:
    return False
#   =========    misc utilities ===========
def count_close_sites(sites_to_count = None,
           sites_to_match = None, dist_close = 3.8):
  count = 0
  if not sites_to_match or sites_to_match.size()<1:
    return count
  for i in range(sites_to_count.size()):
    dist, id1, id2 = sites_to_count[i:i+1
       ].min_distance_between_any_pair_with_id(sites_to_match)
    if dist is not None and dist <= dist_close:
      count += 1
  return count
def residues_are_sequential(model):
  from mmtbx.secondary_structure.find_ss_from_ca import \
          get_last_resno, get_first_resno,get_chain_ids, split_model
  ph = model.get_hierarchy()
  chain_ids = get_chain_ids(ph)
  if len(chain_ids) > 1:
    return False
  models = split_model(ph,
      use_default_distance_cutoff=False, distance_cutoff = False)
  if len(models) > 1:
    return False
  if ph.overall_counts().n_residues == get_last_resno(ph) - get_first_resno(ph) + 1:
    return True
  else:
    return False
def get_sample_selections(model, minimum_size = 20, fractional_size = 0.1,
     number_of_tries = 10):
  # get selection list that samples the model
  ph = model.get_hierarchy()
  n = ph.overall_counts().n_residues
  from mmtbx.secondary_structure.find_ss_from_ca import get_chain_ids, \
          get_first_resno
  chain_ids = get_chain_ids(ph, unique_only = True)
  size_to_use = int(0.5+max(minimum_size, fractional_size * n))
  ratio = number_of_tries/n
  selection_list = ['all']
  for chain_id in chain_ids:
    asc1 = ph.atom_selection_cache()
    selected_atoms = ph.select(
      asc1.selection(string = "chain '%s'" %(chain_id)))
    n_in_chain = selected_atoms.overall_counts().n_residues
    n_selections_in_chain = max(1,int(0.5+n_in_chain*ratio))
    first_resno = get_first_resno(selected_atoms)
    start_resno = first_resno + n_in_chain//2 - \
       n_selections_in_chain*size_to_use//2
    for i in range(n_selections_in_chain):
      selection_string = "chain '%s' and resseq %s:%s" %(
        chain_id,
          start_resno + i * size_to_use,
          start_resno + (i + 1) * size_to_use,)
      sel = asc1.selection(selection_string)
      if sel.count(True) >= minimum_size//2:
        selection_list.append(selection_string)
  return selection_list
def run_structure_search(model_list,
      sequence_only,
      structure_only,
      number_of_models,
      number_of_models_per_input_model,
      return_pdb_info_list = None,
      nproc = 1,
      overall_temp_dir = 'TEMP_STRUCTURE_SEARCH',
      minimum_percent_identity = None,
      maximum_percent_identity = None,
      database = None,
      local_pdb_dir = None,
      log = sys.stdout):
  working_pdb_info = group_args(
      group_args_type = 'pdb_info_list',
      pdb_info_list = [],
      text_list = [])
  if os.path.isdir(overall_temp_dir):
    remove_temp_dir = False
  else:
    os.mkdir(overall_temp_dir)
    remove_temp_dir = True
  if number_of_models_per_input_model and (len(model_list) > 1
      or number_of_models is None):
    models_to_get = number_of_models_per_input_model
  else:
    models_to_get = number_of_models
  from libtbx.easy_mp import run_jobs_with_large_fixed_objects
  end_number = -1
  n_tot = len(model_list)
  n = n_tot//nproc
  if n * nproc < n_tot:
    n = n + 1
  assert n * nproc >= n_tot
  runs_to_carry_out = []
  for run_id in range(nproc):
    start_number = end_number + 1
    end_number = min(n_tot-1, start_number + n - 1)
    if end_number < start_number: continue
    temp_dir = os.path.join(overall_temp_dir,'temp_%s' %(start_number))
    if not os.path.isdir(temp_dir):
      os.mkdir(temp_dir)
    runs_to_carry_out.append(group_args(
      run_id = run_id,
      start_number = start_number,
      end_number = end_number,
      temp_dir = temp_dir,
      ))
  kw_dict = {
    'model_list':model_list,
    'sequence_only':sequence_only,
    'structure_only':structure_only,
    'number_of_models':models_to_get,
    'minimum_percent_identity':minimum_percent_identity,
    'maximum_percent_identity':maximum_percent_identity,
    'database':database,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = False,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = run_group_of_get_local_pdb_info_list,
    log = log)
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.info_list:
      working_pdb_info = get_unique_pdb_info_list(
          working_pdb_info.pdb_info_list,run_info.result.info_list)
  if number_of_models is not None and \
      len(working_pdb_info.text_list) > number_of_models:
    print("Number of models to keep:",number_of_models, file = log)
    working_pdb_info.text_list = working_pdb_info.text_list[:number_of_models]
    working_pdb_info.pdb_info_list = \
       working_pdb_info.pdb_info_list[:number_of_models]
  if return_pdb_info_list:
    if os.path.isdir(overall_temp_dir) and remove_temp_dir:
      from phenix.autosol.delete_dir import delete_dir
      delete_dir(overall_temp_dir,clean_up=True)
    return working_pdb_info.pdb_info_list
  print("Models to get: %s " %(" ".join(
        working_pdb_info.text_list)),
       file = log)
  new_model_list_info = fetch_group_of_pdb(
       pdb_info_list = working_pdb_info.pdb_info_list,
       nproc = nproc,
       local_pdb_dir = local_pdb_dir,
       log = log)
  new_model_list = new_model_list_info.model_list
  print("Total models: %s" %(len(new_model_list)), file = log)
  if os.path.isdir(overall_temp_dir) and remove_temp_dir:
    from phenix.autosol.delete_dir import delete_dir
    delete_dir(overall_temp_dir,clean_up=True)
  return group_args(
        group_args_type = 'structure search results',
        model_list = new_model_list,
       )
def run_group_of_get_local_pdb_info_list(run_info = None,
   model_list = None,
   sequence_only = None,
   structure_only = None,
   number_of_models= None,
   minimum_percent_identity = None,
   maximum_percent_identity = None,
   database = None,
   log = sys.stdout):
  temp_dir = run_info.temp_dir
  current_dir = os.getcwd()
  os.chdir(temp_dir)
  info_list = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    info = get_local_pdb_info_list(model_list[i],sequence_only,structure_only,
    number_of_models,
    minimum_percent_identity = minimum_percent_identity,
    maximum_percent_identity = maximum_percent_identity,
    database = database,
     )
    if info:
      info_list += info
  os.chdir(current_dir)
  return group_args(
      group_args_type = 'run_group_of_get_local_pdb_info_list',
      info_list = info_list,
     )
def get_local_pdb_info_list(model,sequence_only,structure_only,
     number_of_models,
     minimum_percent_identity = None,
     maximum_percent_identity = None,
     database = None,
  ):
  if not database:
    database = 'full_pdb'
  ca_only = (
       model.get_sites_cart().size() < 2 * model.apply_selection_string(
         "name ca").get_sites_cart().size())
  args= ("trim_ends=false sequence_only=%s structure_only=%s" %(
        (sequence_only or ca_only), (structure_only and (not ca_only))) + \
         #" get_pdb = %s" %(number_of_models) + \
         " get_xml_only=True" +\
         " write_pdb=False write_results=False").split()
  if database == 'pdb100':
    args.append(" use_pdb100aa=True")
  else:
    args.append(" use_pdb100aa=False")
  from phenix.command_line.structure_search import structure_search
  ss = structure_search(args=args, model = model)
  return  get_info_list_from_ssr(ss.ssr,
        sequence_only = sequence_only, structure_only = structure_only,
        number_of_models = number_of_models,
      minimum_percent_identity = minimum_percent_identity,
      maximum_percent_identity = maximum_percent_identity,
   )
def get_unique_pdb_info_list(working_pdb_info_list,local_pdb_info_list):
  unique_list = []
  unique_text = []
  for info in working_pdb_info_list + local_pdb_info_list:
    text = "%s_%s" %(
      info.pdb_id,info.chain_id)
    if not text in unique_text:
      unique_text.append(text)
      unique_list.append(info)
  return group_args(
    group_args_type = 'pdb_info_list',
    pdb_info_list = unique_list,
    text_list = unique_text)
def run_group_of_fetch_pdb(run_info = None,
   pdb_info_list = None,
   local_pdb_dir = None,
   log = sys.stdout):
  model_list = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    model_info = fetch_one_pdb(pdb_info = pdb_info_list[i],
      local_pdb_dir = local_pdb_dir,
      log = log)
    if model_info.model_list:
       model_list += model_info.model_list
  return group_args(
      group_args_type = 'fetch pdb on model or model_list',
      model_list = model_list,
     )
def get_pdb_info(pdb_id = None, chain_id = None,):
  pdb_text = pdb_id
  if chain_id:
    pdb_text+="_%s" %(chain_id)
    selection_string = "chain %s" %(chain_id)
  else:
    selection_string = None
  pdb_info = group_args(
        group_args_type = 'pdb info',
        pdb_id = pdb_id,
        chain_id = chain_id,
        pdb_text = pdb_text,
        selection_string = selection_string,
        )
  return pdb_info
def set_environment_for_local_mirror(local_pdb_dir):
  if not os.path.isdir(local_pdb_dir):
    raise Sorry("Local pdb directory %s is missing" %(local_pdb_dir))
  pdb_divided=os.path.join(local_pdb_dir,'data','structures','divided')
  if not os.path.isdir(pdb_divided):  # try again
    pdb_divided = local_pdb_dir
  pdb_mirror_pdb0=os.path.join(pdb_divided,'pdb')
  if not os.path.isdir(pdb_mirror_pdb0):
    raise Sorry("Could not find PDB directory %s" %(pdb_mirror_pdb0))
  pdb_mirror_sf0=os.path.join(pdb_divided,'structure_factors')
  pdb_mirror_mmcif0=os.path.join(pdb_divided,'mmcif')
  os.environ['PDB_MIRROR_PDB']=pdb_mirror_pdb0
  os.environ['PDB_MIRROR_STRUCTURE_FACTORS']=pdb_mirror_sf0
  os.environ['PDB_MIRROR_MMCIF']=pdb_mirror_mmcif0
def fetch_one_pdb(pdb_id = None, chain_id = None,
       pdb_info = None,
       local_pdb_dir = None,
       log = sys.stdout):
    from iotbx.pdb import fetch,  hierarchy
    if local_pdb_dir:
      set_environment_for_local_mirror(local_pdb_dir)
      print("Set environment for using %s as local PDB directory" %(
        local_pdb_dir), file = log)
    print("Fetching %s from RCSB" %(pdb_info.pdb_id), file = log)
    if pdb_id and not pdb_info:
      pdb_info = get_pdb_info(pdb_id = pdb_id, chain_id = chain_id)
    try:
      data=fetch.fetch(id=pdb_info.pdb_id,data_type='pdb',
       format='cif_or_pdb', mirror='rcsb', log=log)
    except Exception as e:
        print("Cannot fetch %s ...skipping" %(pdb_info.pdb_id), file = log)
        print(e)
        return group_args(
      group_args_type = 'fetch pdb on model or model_list',
      model_list = [],
        )
    data_text = data.read()
    try:  # for python 3.8
      pdb_in = hierarchy.input(pdb_string=data_text.decode(encoding='utf-8'))
      model_input = pdb_in.hierarchy.as_pdb_input()
    except Exception as e: # for python 2.7
      pdb_in = hierarchy.input(pdb_string=data_text)
      model_input = pdb_in.hierarchy.as_pdb_input()

    from mmtbx.model import manager as model_manager
    crystal_symmetry = dummy_crystal_symmetry()
    model = model_manager(
        model_input = model_input,
        crystal_symmetry=crystal_symmetry,
        log = null_out())
    if pdb_info.selection_string:
        model = model.apply_selection_string(pdb_info.selection_string)
    model.set_info(pdb_info)
    return group_args(
      group_args_type = 'fetch pdb on model ',
      model_list = [model],
     )
def fetch_group_of_pdb(pdb_info_list, nproc = 1,
   local_pdb_dir = None,
   log = sys.stdout):
  ''' Get a group of structures from PDB'''
  from libtbx.easy_mp import run_jobs_with_large_fixed_objects
  end_number = -1
  n_tot = len(pdb_info_list)
  n = n_tot//nproc
  if n * nproc < n_tot:
    n = n + 1
  assert n * nproc >= n_tot
  runs_to_carry_out = []
  for run_id in range(nproc):
    start_number = end_number + 1
    end_number = min(n_tot-1, start_number + n - 1)
    if end_number < start_number: continue
    runs_to_carry_out.append(group_args(
      run_id = run_id,
      start_number = start_number,
      end_number = end_number,
      ))
  kw_dict = {
    'pdb_info_list':pdb_info_list,
    'local_pdb_dir':local_pdb_dir,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = False,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = run_group_of_fetch_pdb,
    log = log)
  model_list = []
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.model_list:
      model_list += run_info.result.model_list
  return group_args(
      group_args_type = 'fetch pdb on model or model_list',
      model_list = model_list,
     )
def get_info_list_from_ssr(ssr,
      number_of_models = 1,
      sequence_only = False,
      structure_only = False,
      minimum_percent_identity = None,
      maximum_percent_identity = None,
      ):
  pdb_id_list = []
  search_types = []
  if 'seq' in list(ssr.keys()) and not structure_only:
    search_types.append("seq")
  if 'str' in list(ssr.keys()) and not sequence_only:
    search_types.append("str")
  for search_type in search_types:
    groups = split_into_groups(ssr[search_type],"<Hit_id>")
    for group in groups:
      id_info= None
      identity = None
      align = None
      accession = None
      hit_list = []
      for x in group:
        if x.strip().startswith("<Hit_accession>"):
          accession = x.replace("<Hit_accession>","").replace(
             "</Hit_accession>","")
        if x.strip().startswith("<Hit_id>"):
          id_info= x.replace("<Hit_id>","").replace(
             "</Hit_id>","")
        if x.strip().startswith("<Hsp_identity>"):
          identity= x.replace("<Hsp_identity>","").replace(
             "</Hsp_identity>","")
        if x.strip().startswith("<Hsp_align-len>"):
          align = x.replace("<Hsp_align-len>","").replace(
             "</Hsp_align-len>","")
        if x.strip().startswith("<Hit_def>"):
          hit_list = get_hit_list(x.replace("<Hsp_align-len>","").replace(
             "</Hsp_align-len>",""))
      keep = False
      if identity is not None and align is not None:
        keep = True
        percent_identity = 100* float(identity)/max(1,float(align))
        if minimum_percent_identity is not None and \
           percent_identity < minimum_percent_identity:
          keep = False
        if maximum_percent_identity is not None and \
           percent_identity > maximum_percent_identity:
          keep = False
      if keep:
        for x in [accession] + hit_list:
          if not x in pdb_id_list:
            pdb_id_list.append(x)
  pdb_info_list = []
  for pdb_id in pdb_id_list:
    pdb_id, chain_id = pdb_id.strip().split("_")
    pdb_info_list.append(group_args(
      group_args_type = 'pdb id and chain',
      pdb_id = pdb_id,
      chain_id = chain_id,
      selection_string = "chain %s" %(chain_id),
      ))
  return pdb_info_list[:number_of_models]
def get_hit_list(text):
  hit_list = []
  # bracketed like this: |pdb|2EFG|A
  if text.startswith("|pdb|"):
    spl = text.split("|pdb|")
  else: # usual
    spl = text.split("|pdb|")[1:]
  for x in spl:
    if len(x.split("|")) > 1:
      pdb_id = x.split("|")[0]
      chain_id = x.split("|")[1].split(" ")[0]
      hit_list.append("%s_%s" %(pdb_id,chain_id))
  return hit_list
def split_into_groups(list_of_text_items, text):
  group = []
  groups = [group]
  for x in list_of_text_items:
    if x.strip().startswith(text):
      group = [x]
      groups.append(group)
    else:
      group.append(x)
  return groups
def get_loop_info_list(model,
   sequence,
   min_residues_in_loop = 5,
   max_residues_in_loop = 30,
   include_ends = False):
  ''' Return a list of all gaps in all chains of this model
      Also give info on residues missing at beginning and end if include_ends
      May fail with multi-chain models as sequence is not well defined (normally
      has blank lines between chains but chains may not correspond to model)
      '''
  if not model:
    return []  # nothing to do
  sequence = sequence.strip().upper().replace(" ","")
  loop_info_list = []
  from mmtbx.secondary_structure.find_ss_from_ca import get_chain_ids, \
    get_first_resno, get_last_resno, split_model, model_info
  chain_list = get_chain_ids(model.get_hierarchy(), unique_only=True)
  from iotbx.bioinformatics import get_sequence_from_pdb
  for chain_id in chain_list:
    last_end_res = None
    last_start_res = None
    last_sequence = None
    first = False
    all_m = split_model(
       model_info(
         hierarchy=model.apply_selection_string(
          "chain %s" %(chain_id)).get_hierarchy()
                 ),
         use_default_distance_cutoff = False,
                        )
    for m in all_m:
      first = (m == all_m[0])
      last = (m == all_m[-1])
      start_res = get_first_resno(m.hierarchy)
      end_res = get_last_resno(m.hierarchy)
      current_sequence = get_sequence_from_pdb(
          hierarchy = m.hierarchy)
      if first and include_ends:
        loop_sequence = get_loop_sequence(
            sequence = sequence,
            sequence_before_loop = None,
            sequence_after_loop = current_sequence,
            start_of_sequence_before_loop = None,
            start_of_sequence_after_loop = start_res,
            n_trim = 0)
        if loop_sequence:
          loop_info_list.append(group_args(
            chain_id = chain_id,
            last_resno_before_loop = None,
            first_resno_after_loop = start_res,
            loop_sequence = loop_sequence,
            residues_in_loop = len(loop_sequence),
            sequence_before_loop = None,
            sequence_after_loop = current_sequence,
           )
          )
      if last_end_res is not None and \
         start_res - last_end_res <= max_residues_in_loop:  # loop here
        gap_size = start_res - last_end_res - 1
        if gap_size < min_residues_in_loop:
          n_trim = ((min_residues_in_loop+1-gap_size)//2)
        else:
          n_trim = 0
        loop_sequence = get_loop_sequence(
            sequence = sequence,
            sequence_before_loop = last_sequence,
            sequence_after_loop = current_sequence,
            start_of_sequence_before_loop = last_start_res,
            start_of_sequence_after_loop = start_res,
            n_trim = n_trim)
        if loop_sequence: # only if something did work
          loop_info_list.append(group_args(
            chain_id = chain_id,
            last_resno_before_loop = last_end_res - n_trim,
            first_resno_after_loop = start_res + n_trim,
            loop_sequence = loop_sequence,
            residues_in_loop = len(loop_sequence),
            sequence_before_loop = last_sequence,
            sequence_after_loop = current_sequence,
             )
            )
          r = loop_info_list[-1]
          assert r.first_resno_after_loop - r.last_resno_before_loop - 1 == \
            r.residues_in_loop
      if last and include_ends:
        loop_sequence = get_loop_sequence(
            sequence = sequence,
            sequence_before_loop = current_sequence,
            sequence_after_loop = None,
            start_of_sequence_before_loop = end_res,
            start_of_sequence_after_loop = None,
            n_trim = 0)
        if loop_sequence:
          loop_info_list.append(group_args(
            chain_id = chain_id,
            last_resno_before_loop = end_res,
            first_resno_after_loop = None,
            loop_sequence = loop_sequence,
            residues_in_loop = len(loop_sequence),
            sequence_before_loop = current_sequence,
            sequence_after_loop = None,
           ))
      # done with this one
      last_start_res = start_res
      last_end_res = end_res
      last_sequence = current_sequence
  return loop_info_list
def get_loop_sequence(sequence = None,
   sequence_before_loop = None,
   sequence_after_loop = None,
   start_of_sequence_before_loop = None,
   start_of_sequence_after_loop =  None,
   n_trim = None,):
  # Normally just return letters in sequence between
  #  start_of_sequence_before_loop + len(sequence_before_loop) and
  #   start_of_sequence_after_loop.
  # However if the residues before and after do not match
  #   sequence_before_loop and sequence_after_loop, find them in sequence
  #  Also check for start_of_sequence_before_loop==None (N-term end) or
  #    start_of_sequence_after_loop==None (C-term end)
  # Check to see if sequence_before_loop matches sequence at
  #   start_of_sequence_before_loop
  if start_of_sequence_before_loop is None: # N-terminal end...take everything
      # up to sequence_after_loop
    after_loop = sequence.find(sequence_after_loop)
    if after_loop > -1:
       after_loop += n_trim
       sequence_of_loop = sequence[:after_loop]
       return sequence_of_loop
    else:
      return None
  if start_of_sequence_after_loop is None: # C-term end
    before_loop = sequence.rfind(sequence_before_loop)
    if before_loop > -1:
      before_loop += len(sequence_before_loop)
      before_loop -= n_trim
      sequence_of_loop = sequence[before_loop:]
      return sequence_of_loop
    else:
      return None
  offset = 0
  sequence_of_loop = get_sequence_with_offset(
    sequence = sequence,
    offset = offset,
    start_of_sequence_before_loop = start_of_sequence_before_loop,
    sequence_before_loop = sequence_before_loop,
    n_trim = n_trim,
    start_of_sequence_after_loop = start_of_sequence_after_loop,
    sequence_after_loop = sequence_after_loop,)
  if sequence_of_loop:
    return sequence_of_loop
  possible_before_indices = get_possible_positions(sequence, sequence_before_loop)
  possible_after_indices = get_possible_positions(sequence, sequence_after_loop)
  for index_before in possible_before_indices:
    # Otherwise, try to find it
    # index_before = sequence.find(sequence_before_loop)
    index_after = index_before + (start_of_sequence_after_loop - start_of_sequence_before_loop)
    if not index_after in possible_after_indices:
      continue
    offset_before = index_before - start_of_sequence_before_loop
    offset_after = index_after - start_of_sequence_after_loop
    if offset_before != offset_after:
      continue
    offset = offset_before
    sequence_of_loop = get_sequence_with_offset(
      sequence = sequence,
      offset = offset,
      start_of_sequence_before_loop = start_of_sequence_before_loop,
      sequence_before_loop = sequence_before_loop,
      n_trim = n_trim,
      start_of_sequence_after_loop = start_of_sequence_after_loop,
      sequence_after_loop = sequence_after_loop,)
    if sequence_of_loop:
      return sequence_of_loop
def get_possible_positions(sequence, sequence_before_loop):
  positions = []
  start = 0
  for i in range(len(sequence_before_loop)):
    index_before = sequence.find(sequence_before_loop, start)
    if index_before < 0:
      break
    else:
      start = index_before + 1
      positions.append(index_before)
  return positions
def get_sequence_with_offset(
  sequence = None,
  offset = None,
  start_of_sequence_before_loop = None,
  sequence_before_loop = None,
  n_trim = None,
  start_of_sequence_after_loop = None,
  sequence_after_loop = None,
   ):
  start_of_loop = offset + start_of_sequence_before_loop + \
        len(sequence_before_loop) - n_trim
  end_of_loop = offset + start_of_sequence_after_loop + n_trim - 1
  sequence_before_loop_from_sequence = sequence[
       offset + start_of_sequence_before_loop:
       offset + start_of_sequence_before_loop + len(sequence_before_loop)]
  sequence_after_loop_from_sequence = sequence[
       offset + start_of_sequence_after_loop:
       offset + start_of_sequence_after_loop + len(sequence_after_loop)]
  if sequence_before_loop_from_sequence == sequence_before_loop and \
     sequence_after_loop_from_sequence == sequence_after_loop:
    # sequence of loop is in middle
    return sequence[start_of_loop: end_of_loop + 1]
  else:
    return None
def get_best_result(results):
  best_result = None
  for result in results:
    if (result.model is not None or result.score is not None) and (
       (best_result is None) or
       (best_result.score < result.score)):
      best_result = result
  return best_result
def create_run_methods(
    methods_list = None,
    common_kw = None,
    method_type= None,
    tries_for_each_method = None,
    expected_result_names = None,
    random_seed = None,
    temp_dir = None,
    nproc = None,
    debug = None,
    log = sys.stdout,
     ):
    if not methods_list:
      print ("No methods to run...skipping", file = log)
      return None
    thoroughness = common_kw['thoroughness']
    #  If thoroughness == 'quick' just run nproc jobs, with
    #  methods decided based on their order in methods_list
    # Decide on thoroughness and number of tries for each method
    max_thoroughness = ALLOWED_THOROUGHNESS.index(thoroughness)
    assert max_thoroughness is not None # need to set thoroughness
    if (max_thoroughness == 0) and (nproc < len(methods_list)):
        print ("Using only the first %s methods as 'quick' is set: %s " %(
          nproc, str(methods_list[:nproc])), file = log)
        methods_list=methods_list[:nproc]  # take first nproc only
    if not tries_for_each_method:
      if max_thoroughness == 0:   # 'quick'
        tries_for_each_method = max(1, nproc//len(methods_list))
      else:
        tries_for_each_method = max(1, max(nproc,
          2**max_thoroughness)/len(methods_list))
        #(i.e., 8 for extra_thorough)
    tries_for_each_method = int(0.5+tries_for_each_method)
    # If the number of methods to be run is 1, set nproc=1 for multiprocessing
    #   and pass on nproc to the run method itself
    number_of_methods_to_run = tries_for_each_method * len(methods_list)
    if number_of_methods_to_run == 1: # just one run
      nproc_for_parallel_run = 1
      nproc_inside_run = nproc
    else: # usual
      nproc_for_parallel_run = nproc
      nproc_inside_run = 1
    kw_list = []
    for m in methods_list:
      for i in range(1,tries_for_each_method+1):
        if tries_for_each_method > max_thoroughness: # randomly choose
          t = ALLOWED_THOROUGHNESS[random.randint(0,max_thoroughness)]
        else: # choose most thorough always
          t=ALLOWED_THOROUGHNESS[max_thoroughness]
        # Name of method we are going to run
        method_name = "%s_with_%s" %(method_type,m)
        kw={}
        for x in list(list(common_kw.keys())):
          kw[x]=common_kw[x] # Note no deep copy of map/model as they are fixed
        kw['thoroughness'] = t
        kw['run_id']=i
        kw['nproc']=nproc_inside_run
        kw['random_seed']=random.randint(0,10000000)
        kw['temp_dir']=temp_dir
        kw['debug']=debug
        kw['expected_result_names']=expected_result_names
        kw['method_type']=method_type
        kw['method_name']=method_name
        kw_list.append(kw)
    return group_args(
      kw_list=kw_list,
      tries_for_each_method=tries_for_each_method,
      nproc_for_parallel_run=nproc_for_parallel_run,
      methods_list=methods_list)
def zero_if_none(x):
  if not x:
    return 0
  else:
    return x
def generate_random_sequence(chain_type = None,
        n_residues = None, all_same = None):
  if not n_residues: n_residues = 100
  if not chain_type: chain_type = "PROTEIN"
  import random
  if chain_type.upper() == 'PROTEIN':
    letters = 'ARNDCEQGHILKMFPSTWYV'
  elif chain_type.upper() == 'DNA':
    letters = 'GCTA'
  elif chain_type.upper() == 'RNA':
    letters = 'GCUA'
  if all_same:
    letters = letters[:1]
  seq=""
  for i in range(n_residues):
    t = random.randint(0,len(letters)-1)
    seq+=letters[t]
  return seq
def get_neighbor_indices_inside_region(
     threshold_info = None,
     map_data = None,
     index = None,
     depth = 1):
  if depth > 1:
    # and get their neighbors
    local_neighbor_indices = get_neighbor_indices_inside_region(
         threshold_info = threshold_info,
         map_data = map_data,
         index = index)
    neighbor_indices = []
    for local_index in local_neighbor_indices:
      neighbor_indices+= get_neighbor_indices_inside_region(
         threshold_info = threshold_info,
         map_data = map_data,
         index = local_index)
    return neighbor_indices
  from phenix.autosol.trace_and_build import get_indices_from_index
  from phenix.autosol.trace_and_build import get_index_from_indices
  all = map_data.all()
  i, j, k = get_indices_from_index(index = index, all = all)
  region_id = threshold_info.mask_info.edited_mask[index]
  neighbor_indices = []
  for i_off in [-1, 0, 1]:
    new_i = i+i_off
    if new_i < 0: continue
    if new_i >=  all[0]: continue
    for j_off in [-1, 0, 1]:
      # if i_off !=  0 and j_off !=  0: continue
      new_j = j+j_off
      if new_j < 0: continue
      if new_j >=  all[1]: continue
      for k_off in [-1, 0, 1]:
        if i_off == 0 and j_off == 0 and k_off == 0: continue
        # if j_off !=  0 and k_off !=  0: continue
        new_k = k+k_off
        if new_k < 0: continue
        if new_k >=  all[2]: continue
        new_indices = (new_i, new_j, new_k)
        if threshold_info.mask_info.edited_mask[new_indices] == region_id:
          neighbor_index = get_index_from_indices(new_indices, all = all)
          neighbor_indices.append(neighbor_index)
  return neighbor_indices
def get_cutoff_density(density_values, fraction = None, n_bins_ratio = .1,
    min_n = None):
  n_bins = 1+int(n_bins_ratio * density_values.size())
  mmm = density_values.min_max_mean()
  sel = (density_values != mmm.min)
  for i in range(n_bins+1):
    cutoff = mmm.min + (i/n_bins) * (mmm.max - mmm.min)
    n = (density_values < cutoff).count(True)
    if n >= max(min_n,
      fraction * density_values.size()):
       return group_args(
         cutoff = cutoff,
         n = n)
  # Failed
  return group_args(
     cutoff = mmm.max,
     n = density_values.size())
def get_path_length(sites_cart):
  ''' Return path length along sites_cart
  '''
  if sites_cart.size()<2:
    return 0
  deltas = sites_cart[1:] - sites_cart[:-1]
  n = deltas.size()
  return deltas.norms().min_max_mean().mean * n
def two_points_in_same_region_at_this_threshold(
   tnb = None,
   map_data = None,
   indices1 = None,
   indices2 = None,
   threshold = None):
  mask_info = tnb.get_edited_mask(threshold = threshold,
      masked_map_data = map_data,)
  return two_points_in_same_region(mask_info,indices1,indices2)
def get_all_points_in_region_containing_point(
   tnb = None,
   map_data = None,
   indices1 = None,
   threshold = None):
  mask_info = tnb.get_edited_mask(threshold = threshold,
      masked_map_data = map_data)
  if not mask_info:
    return None
  region_id = mask_info.edited_mask[indices1]
  if region_id == 0: # this was outside all regions...skip
    return None
  points_in_region_info = tnb.get_points_in_region( mask_info = mask_info,
      map_data = map_data,
      id = region_id)
  return points_in_region_info
def find_threshold_to_connect_points(
    mmm,
    sc1,
    sc2,
    minimum_density = 0,
    max_tries = 100,
    tol = 1.e-8,
    log = sys.stdout):
  from phenix.autosol.trace_and_build import trace_and_build, \
    get_bool_region_mask, get_indices_one_d_from_indices
  from iotbx.map_manager import get_indices_from_index,\
    get_sites_cart_from_index
  tnb = trace_and_build( params = group_args(
   crystal_info = group_args(wrapping = False),
   control = group_args(verbose = False),
    ),
    )
  map_data = mmm.map_manager().map_data()
  sc12 = flex.vec3_double([sc1, sc2])
  sites_frac = mmm.crystal_symmetry().unit_cell().fractionalize(sc12)
  indices1, indices2 = tnb.get_indices_list_from_sites_frac(
      sites_frac = sites_frac,
      map_data = map_data)
  den1a = map_data[indices1]
  den2a = map_data[indices2]
  index1,index2 = get_indices_one_d_from_indices(
       [indices1,indices2], all = map_data.all())
  threshold = max(minimum_density,map_data.as_1d().min_max_mean().mean)
  low_bound = None
  high_bound = None
  delta_threshold = map_data.as_1d().standard_deviation_of_the_sample()
  mask_info = tnb.get_edited_mask(threshold = threshold,
      masked_map_data = mmm.map_manager().map_data())
  same_region_info = two_points_in_same_region(mask_info,indices1,indices2)
  if same_region_info.same_region:
    low_bound = threshold  # at low_bound they are in same region
  else:
    high_bound = threshold # at high bound the are not in same region
  if low_bound is None:
    low_bound = find_low_bound(tnb, mmm, indices1, indices2,
      threshold = threshold, delta_threshold = delta_threshold,
      minimum_density = minimum_density,
      max_tries = max_tries)
  else:
    high_bound = find_high_bound(tnb, mmm, indices1, indices2,
      threshold = threshold, delta_threshold = delta_threshold,
      max_tries = max_tries)
  if low_bound is None:
    return group_args(
      tnb = None,
      threshold = None,
      mask_info = None,
      region_id = None,
      points_in_region_info = None,)
  # Now it is between low (both in same region) and high (not in same region)
  #   ...just take mean until we are tired of it
  for i in range(max_tries):
    test_threshold = 0.5 * (high_bound + low_bound)
    if high_bound - low_bound <= tol:
      break
    mask_info = tnb.get_edited_mask(threshold = test_threshold,
      masked_map_data = mmm.map_manager().map_data())
    same_region_info = two_points_in_same_region(mask_info,indices1,indices2)
    if same_region_info.same_region:
      low_bound = test_threshold
    else:
      high_bound = test_threshold
  mask_info = tnb.get_edited_mask(threshold = low_bound,
      masked_map_data = mmm.map_manager().map_data())
  same_region_info = two_points_in_same_region(mask_info,indices1,indices2)
  points_in_region_info = tnb.get_points_in_region( mask_info = mask_info,
      map_data = mmm.map_manager().map_data(),
      id = same_region_info.region_id)
  return group_args(
      threshold = low_bound,
      just_too_high_threshold = high_bound,
      mask_info = mask_info,
      region_id = same_region_info.region_id,
      points_in_region_info = points_in_region_info,
      tnb = tnb)
def find_high_bound(tnb, mmm, indices1, indices2,
    threshold = None, delta_threshold = None,
    max_tries = None):
  ''' Return first high threshold where both are not in same region '''
  high_bound = None
  for i in range(max_tries):
    test_threshold = threshold + i* delta_threshold
    mask_info = tnb.get_edited_mask(threshold = test_threshold,
      masked_map_data = mmm.map_manager().map_data())
    same_region_info = two_points_in_same_region(mask_info,indices1,indices2)
    if not same_region_info.same_region:
      return test_threshold
  return None
def find_low_bound(tnb, mmm, indices1, indices2,
      threshold = None, delta_threshold = None,
      minimum_density = 0.,
      max_tries = None):
  ''' Return first low threshold where both are in same region '''
  low_bound = None
  for i in range(max_tries):
    test_threshold = threshold - i* delta_threshold
    if test_threshold < minimum_density:
      test_threshold = minimum_density
    mask_info = tnb.get_edited_mask(threshold = test_threshold,
      masked_map_data = mmm.map_manager().map_data())
    same_region_info = two_points_in_same_region(mask_info,indices1,indices2)
    if same_region_info.same_region:
      low_bound = test_threshold
      break
    if test_threshold == minimum_density:
      break
  return low_bound
def get_model_from_path(sites_cart, as_ca_model = None,
        ca_distance = None,
        reverse_ca_model = None,
        mmm = None):
      if as_ca_model and sites_cart.size()> 1:
        # sample sites_cart at about 3.8 A intervals
        path_length = get_path_length(sites_cart)
        number_of_ca_atoms = max(2, int (1.5 + path_length/ca_distance))
        approx_sites_per_ca = sites_cart.size()/number_of_ca_atoms
        if reverse_ca_model:
           sites_cart = list(sites_cart)
           sites_cart.reverse()
           sites_cart = flex.vec3_double(sites_cart)
        as_ca_sites = flex.vec3_double( [
         sites_cart[min(sites_cart.size()-1,
          int(0.5 + approx_sites_per_ca * i))]
           for i in range(number_of_ca_atoms)])
      else:
        as_ca_sites = sites_cart
      model = mmm.as_map_model_manager().model_from_sites_cart(
        as_ca_sites, return_model = True)
      return model
def get_path_through_points(mmm,
  max_time = None,
  points_in_region = None,
  ends_sites_cart = None,
  ):
 # Identify path through the points
    import time
    time_info = group_args(
      time = time,
      t0 = time.time(),
      max_time = max_time,
     )
    from phenix.autosol.trace_and_build import trace_and_build
    tnb = trace_and_build( params = group_args(
      crystal_info = group_args(wrapping = False),
      control = group_args(verbose = False),
      strategy = group_args(
        test_threshold = False,
        path_density_weight = 2,
        matching_end_dist = 3,
        max_fract_non_contiguous = 1,
        retry_long_branches = False,),
       ),
     time_info = time_info,
     crystal_symmetry = mmm.crystal_symmetry(),
     )
    evaluate_path_info = tnb.evaluate_path_and_branching(
      fragment_id = 1,
      map_data = mmm.map_manager().map_data(),
      mask_id = 1,
      points_in_region = points_in_region,
      ends_sites_cart = ends_sites_cart,
      weight_path_by_density = True,)
    return evaluate_path_info
def trace_density_from_point(
      threshold_min = None,
      threshold_max = None,
      mmm = None,
      target_path_length = None,
      minimum_path_length = None,
      n_threshold = None,
      max_tries_to_get_more_residues = None,
      index_to_use = None,
      map_data = None,
      indices_list = None,
      start_index = None,
      ca_distance= None,
      max_time = None,
      max_branch_length = None,
     ):
    # Now trace away from sc1 and see how far we can go before getting branching
    from phenix.autosol.trace_and_build import trace_and_build
    tnb = trace_and_build( params = group_args(
      crystal_info = group_args(wrapping = False),
      control = group_args(verbose = False),
      strategy = group_args(
        test_threshold = False,
        path_density_weight = 2,
        matching_end_dist = 3,
        max_fract_non_contiguous = 1,
        retry_long_branches = False,),
       ),
       crystal_symmetry = mmm.crystal_symmetry(),
       )
    def get_score(path_length, target_path_length, scale_below = 1):
      if path_length > target_path_length:
        return path_length - target_path_length
      else:
        return scale_below *(target_path_length - path_length)
    def is_it_better(best_path_info, info, target_path_length):
          if best_path_info is None:
             better = True
          elif target_path_length is not None:
            better = get_score(info.path_length, target_path_length) < \
                     get_score(best_path_info.path_length, target_path_length)
          else:
            better = info.path_length > best_path_info.path_length
          return better
    best_path_info = None
    for cycle in range(max_tries_to_get_more_residues):
      length_list = []
      threshold_list = []
      for j in range(n_threshold + 1):
        threshold = threshold_max + (
          threshold_min - threshold_max) * j/n_threshold
        threshold_list.append(threshold)
        region_info = get_all_points_in_region_containing_point(
          tnb = tnb,
          map_data = map_data,
          indices1 = indices_list[index_to_use],
          threshold = threshold)
        if not region_info:
          length_list.append(None)
          continue
        # Get furthest point out and branching..
        info = tnb.trace_best_path(
           points_in_region = region_info.points_in_region,
           map_data = map_data,
           mask_id = 1,
           start_index = start_index,
           max_fract_non_contiguous = 1,
           weight_path_by_density = True)
        if info and info.path_length and (minimum_path_length is None
             or info.path_length >= minimum_path_length):
          length_list.append(info.path_length)
          path_residue_equiv = int(
            0.5+info.path_length/ca_distance)
          better = is_it_better(best_path_info, info, target_path_length)
          path_points = info.path_points
          start_end_indices = [path_points[0],path_points[-1]]
          ends_sites_cart= tnb.get_sites_cart_from_index(
               points = start_end_indices,
               map_data = mmm.map_manager().map_data(),
               crystal_symmetry = mmm.crystal_symmetry())
          evaluate_path_info = get_path_through_points(mmm,
            max_time = max_time,
            points_in_region = region_info.points_in_region,
            ends_sites_cart = ends_sites_cart,)
          if evaluate_path_info and \
            evaluate_path_info.branch_best_path_info:
            if evaluate_path_info.branch_best_path_info.longest_branch_length \
                 > max_branch_length:
              break
            else: # ok
              info = evaluate_path_info.best_path_info
              longest_branch_length = \
                evaluate_path_info.branch_best_path_info.longest_branch_length
              path_length = info.path_length
              path_points = info.path_points
              sites_cart = tnb.get_sites_cart_from_index(points = path_points,
                map_data = mmm.map_manager().map_data(),
                crystal_symmetry = mmm.crystal_symmetry())
              info.threshold = threshold
              if not best_path_info or better:
                best_path_info = info
              if target_path_length is not None and \
                 info.path_length > target_path_length:
                break # got it
        else:
          length_list.append(None)
      # Did we bracket the length we wanted
      if target_path_length is None:
        break # done
      else:
        just_lower = None
        just_higher = None
        for ii in range(len(length_list)):
          if length_list[ii] is None: continue
          if length_list[ii] <= target_path_length and (
            just_lower is None or length_list[ii] > length_list[just_lower]):
            just_lower = ii
          if length_list[ii] >= target_path_length and (
            just_higher is None or length_list[ii] < length_list[just_higher]):
            just_higher = ii
        if just_lower is not None and just_higher is not None:
          threshold_max = threshold_list[just_lower]
          threshold_min = threshold_list[just_higher]
        else:
          break # done
    return best_path_info
def two_points_in_same_region(mask_info,indices1,indices2):
  if not mask_info or not mask_info.edited_mask:
    return group_args(
     same_region = False,
     region_id = None)
  ii = mask_info.edited_mask[indices1]
  jj = mask_info.edited_mask[indices2]
  if ii == jj and ii != 0 and ii is not None:
    return group_args(
        same_region = True,
        region_id = ii)
  else:
    return group_args(
     same_region = False,
     region_id = None)
def get_neighbor_indices_with_highest_map_data(real_indices_list, map_data):
  neighbor_indices = []
  for real_indices in real_indices_list:
    neighbor_indices.append(
     get_one_neighbor_indices_with_highest_map_data(real_indices, map_data))
  return neighbor_indices
def get_one_neighbor_indices_with_highest_map_data(real_indices, map_data):
  closest_indices = [max(0, min(n - 1, int(x+0.5)))
      for x,n in zip(real_indices,map_data.all())]
  neighbor_in_each_direction = [
     max(0, min(n - 1,
    i + 1 if  x > i else i - 1)) for i,x,n in zip(
      closest_indices, real_indices,map_data.all())]
  neighbor_indices = [closest_indices]
  for k in range(3):
    neighbor_indices.append([n if k == i else c for c,n,i in zip(
     closest_indices, neighbor_in_each_direction,[0,1,2])])
  best_indices = None
  best_value = None
  for ni in neighbor_indices:
    value = map_data[ni]
    if best_value is None or value > best_value:
      best_value = value
      best_indices = ni
  return best_indices
def adjust_sites_frac_to_match_best_grid_point(sites_frac, map_data):
  best_indices_list = get_best_indices_from_sites_frac(sites_frac, map_data)
  best_sites_frac = flex.vec3_double()
  for indices in best_indices_list:
    best_sites_frac.append([ i/o for i,o in zip(indices, map_data.all())])
  return best_sites_frac
def get_best_indices_from_sites_frac(sites_frac, map_data):
  real_indices_list = get_real_indices_from_sites_frac(sites_frac, map_data)
  return get_neighbor_indices_with_highest_map_data(
      real_indices_list, map_data)
def get_real_indices_from_sites_frac(sites_frac, map_data):
  n_xyz = map_data.all()
  assert tuple(map_data.origin()) == (0,0,0)
  real_indices = []
  for site_frac in sites_frac:
     real_indices.append( [x*a for x,a in zip(site_frac, n_xyz)])
  return real_indices
def dummy_crystal_symmetry(cell=100):
    from cctbx import crystal
    from cctbx import sgtbx
    from cctbx import uctbx
    SpaceGroup=sgtbx.space_group_info(symbol=str('p1'))
    UnitCell=uctbx.unit_cell((cell,cell,cell,90.,90.,90.,))
    crystal_symmetry=crystal.symmetry(
      unit_cell=UnitCell,space_group_info=SpaceGroup)
    return crystal_symmetry
class dummy_density_map:
  def __init__(self, map_data, crystal_symmetry, d_min):
    self.map_data = map_data
    self._unit_cell = crystal_symmetry.unit_cell()
    self._space_group = crystal_symmetry.space_group()
    self._d_min = d_min
  def real_map(self):
    return self.map_data
  def unit_cell(self):
    return self._unit_cell
  def space_group(self):
    return self._space_group
  def d_min(self):
    return self._d_min
def build_something(i,
    build_list = None,
       ):
  build = build_list[i]
  return build.build(return_model = True)
