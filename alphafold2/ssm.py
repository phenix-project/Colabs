from __future__ import division, print_function
import sys, os, time
from libtbx import group_args
import libtbx.callbacks # import dependency
from libtbx.utils import Sorry, null_out
from mmtbx.secondary_structure.find_ss_from_ca import \
     get_first_resno,get_last_resno, get_chain_id, get_chain_ids, \
       set_chain_id, split_model, \
     model_info, find_secondary_structure, find_helix, find_beta_strand, \
     merge_hierarchies_from_models
from mmtbx.secondary_structure.regularize_from_pdb \
      import replacement_segment_summary, replace_with_segments_from_pdb,\
      get_and_split_model
from iotbx.bioinformatics import get_sequence_from_hierarchy
from phenix.model_building.fit_loop import catenate_segments
from libtbx.easy_mp import run_jobs_with_large_fixed_objects
from phenix.command_line.scale_and_merge import delete_directory
from libtbx import easy_pickle
import random
import mmtbx.model
from scitbx.array_family import flex
from copy import deepcopy
from scitbx.matrix import col
import math
import numpy as np
from scitbx.math import superpose
from phenix.model_building.compact_hierarchy import get_ph_from_info, \
   get_info_from_ph
def run(params,
    mmm = None,
    mmm_for_scoring = None,
    model = None,
    other_model = None,
    sss_info = None,
    cluster_info = None,
    other_sss_info = None,
    other_cluster_info = None,
    log = sys.stdout):
  ''' Identify super-secondary structure in a map and model'''
  libtbx.call_back(message = 'Starting search and morph', data = None)
  # Cannot use models with a shift here for now XXX
  if model:
    assert model.shift_cart() in [None, [0,0,0],(0,0,0)]
    # We don't keep track of shift cart here...so make sure we come in with
    #  no shift
  if other_model:
    assert other_model.shift_cart() in [None, [0,0,0],(0,0,0)]
  # compare files
  if params.comparison_pdb_file_names:
    compare_coordinates(params, log = log)
    return
  # make database
  if params.make_database:
    make_database(params,mmm = mmm)
    return
  # find replacement for entire model
  if params.replace_segments_in_model:
    return replace_segments_in_model(params, mmm = mmm, log = log)
  if not sss_info:
    # Get segments of structure in the model
    libtbx.call_back(message = 'Identifying secondary structure', data = None)
    sss_info = get_sss_info(params, mmm = mmm, model = model, log = log)
    if not sss_info:
      print("No model found...skipping", file = log)
      return
  if not cluster_info:
    # Identify structure type based on inter-segment distances
    cluster_info = identify_structure_elements(params, sss_info, log = log)
  if params.replace_structure_elements:
    libtbx.call_back(message = 'Replacing secondary structure elements',
         data = None)
    if params.create_strand_lib_entry:
      return
    elif params.strand_library_list:
      # Read strand library
      strand_lib = get_strand_lib(params)
    else:
      strand_lib = None
    # Replace secondary structure elements with ideal segments if possible
    replace_structure_elements(cluster_info, sss_info,
      strand_lib = strand_lib,
      nproc = params.nproc, log = log)
    # Write out replacements
    full_new_model_info = write_replacement_models(
       cluster_info, sss_info, log = log)
    return full_new_model_info
  if params.other_model_file_name or other_model:
    libtbx.call_back(message = 'Analyzing secondary structure in second model',
         data = None)
    params.search_sequence_neighbors = 0  # we are just matching one model
    local_params = deepcopy(params)
    local_params.model_file_name = params.other_model_file_name
    local_params.include_reverse = False
    params.pdb_include_list = None
    params.pdb_exclude_list = None
    other_info = get_other_info(
      local_params = local_params,
      other_model = other_model,
      other_sss_info = other_sss_info,
      other_cluster_info = other_cluster_info,
      log = log
      )
    if not other_info:
      print("Failed to set up model", file = log)
      return
    ss_index_info = other_info.ss_index_info
    other_sss_info = other_info.other_sss_info
    other_cluster_info = other_info.other_cluster_info
  else:
    other_cluster_info = None
    libtbx.call_back(message = 'Setting up database', data = None)
    print("Setting up database", file = log)
    ss_index_info = load_database_info(params,
     os.path.join(params.data_dir,params.segment_database_file),
      log = log)
    params.delta = ss_index_info.index_info.delta
    params.dist_max = ss_index_info.index_info.dist_max
    print("Done setting up database", file = log)
  # create dict with db entries by pdb file name
  ss_index_info.segment_id_by_pdb_file_name = {}
  for key in list(ss_index_info.segment_info_dict.keys()):
    segment_info = ss_index_info.segment_info_dict[key]
    if segment_info:
      ss_index_info.segment_id_by_pdb_file_name[segment_info.pdb_file_name] = key
  libtbx.call_back(message = 'Database entries: %s' %(
      len(list(ss_index_info.segment_info_dict.keys()))), data = None)
  print("Total of %s DB entries" %(
     len(list(ss_index_info.segment_info_dict.keys()))), file = log)
  params.minimum_good_triples = min(params.minimum_good_triples,
   len(list(ss_index_info.segment_info_dict.keys())))
  libtbx.call_back(message = 'Indexing fixed model', data = None)
  print("Indexing target structure...", file= log)
  index_clusters(params,
     None,
     cluster_info,
     include_nearby_indices = params.include_nearby_indices,
     log = log)
  print("Done indexing target structure...", file= log)
  if params.superpose_brute_force:
    match_info = get_brute_force_match_info(params,
       model,  # duplicates model in sss_info
       other_model, # duplicates model in other_cluster_info
       mmm_for_scoring,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       log = log)
    return match_info
  if params.find_ssm_matching_segments:
    libtbx.call_back(message = 'Finding matching secondary structure',
       data = None)
    set_up_ca_sites(cluster_info, sss_info)
    matching_segment_info = find_ssm_matching_segments(params,
       cluster_info,
       sss_info,
       ss_index_info,
       log = log)
    return matching_segment_info
  if params.find_loops:
    set_up_ca_sites(cluster_info, sss_info)
    set_up_neighbor_ca_sites(cluster_info, sss_info)
    libtbx.call_back(message = 'Finding matching loops',
       data = None)
    find_loop_info = find_loops(params,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       log = log)
    return find_loop_info
def add_reverse_to_model(params, mmm):
   ''' Tack on a reverse version of model to mmm'''
   build = mmm.model_building()
   forward_model = build.model().deep_copy()
   reverse_model = build.reverse_direction(ca_only=True,refine_cycles = 0)
   composite_model = catenate_segments(forward_model, reverse_model)
   mmm.set_model(composite_model)
def get_other_info(
      local_params = None,
      other_model = None,
      other_sss_info= None,
      other_cluster_info= None,
      log = sys.stdout):
    if not other_sss_info:
      other_sss_info = get_sss_info(local_params, model = other_model, log =log)
    if not other_sss_info:
      print("Unable to set up model...", file = log)
      return
    if other_model:
      if hasattr(other_model.info(),'pdb_id'):
        local_params.other_model_file_name = "%s_%s" %(
          other_model.info().pdb_id,other_model.info().chain_id)
      elif hasattr(other_model.info(),'file_name'):
        local_params.other_model_file_name = other_model.info().file_name
      elif local_params.model_file_name:
        local_params.other_model_file_name = local_params.model_file_name
      else:
        local_params.other_model_file_name = "supplied other_model"
    if not other_cluster_info:
      other_cluster_info = identify_structure_elements(local_params,
       other_sss_info,log = log)
    set_up_ca_sites(other_cluster_info, other_sss_info)
    set_up_neighbor_ca_sites(other_cluster_info, other_sss_info)
    ss_index_info = get_single_indexing(
      local_params,other_cluster_info,log = log)
    segment_info = group_args(
      group_args_type = 'dummy segment info for other_model',
      ca_sites_cart = other_sss_info.mmm.model().apply_selection_string(
        "name ca " ).get_sites_cart(),
      full_model = other_sss_info.mmm.model().deep_copy(), # XXX required
      pdb_file_name = local_params.other_model_file_name)
    segment_info.neighbor_sites_cart_list = \
      other_cluster_info.neighbor_sites_cart_list
    ss_index_info.db_info = group_args(
      group_args_type = 'dummy db info',
      db = group_args(
       group_args_type = 'dummy db',
      segment_info_dict = {0:segment_info}))
    ss_index_info.segment_info_dict = {0:segment_info}
    return group_args(
     group_args_type = 'other_info',
     ss_index_info =  ss_index_info,
     other_cluster_info = other_cluster_info,
     other_sss_info = other_sss_info,
     )
def replace_segments_in_model(params, mmm = None, log = sys.stdout):
  print("\nReplacing segments in input model using fragments from PDB",
         file = log)
  # Get starting map and model
  if not mmm:
    mmm = get_map_and_model(params, log = log)
    model = mmm.model()
  # Set params
  original_params = deepcopy(params)
  params = set_params_for_replace_segments_in_model(params)
  # Replace secondary structure elements in model (changes model in mmm)
  if original_params.replace_structure_elements and \
      (not params.read_replacement_segments):
    replace_secondary_structure_elements_in_model(params,
      mmm, log = log)
  # Find helices/strands in other structures matching this, superpose
  #   matching residues
  if original_params.find_ssm_matching_segments and \
      (not params.read_replacement_segments):
    ssm_model_info = run_indexed_ssm_matching(params, mmm, log = log)
    if not ssm_model_info and not mmm.model().info().get('warnings'):
      mmm.model().info().warnings = []
      mmm.model().info().warnings.append('No SSM matching fragments found')
      print("Set warnings: %s" %(" ".join(mmm.model().info().warnings)))
  else:
    ssm_model_info = None
  # Find loops in other structures matching this, superpose  matching residues
  if original_params.find_loops and \
      (not params.read_replacement_segments):
    loop_model_info = run_find_loops(params, mmm, log = log)
    if not loop_model_info and not mmm.model().info().get('warnings'):
      mmm.model().info().warnings = []
      mmm.model().info().warnings.append('No loop fragments found')
      print("Set warnings: %s" %(" ".join(mmm.model().info().warnings)))
  else:
    loop_model_info = None
  # Make list of all segments we have now
  replacement_segment_info = get_replacement_segments(params, mmm,
     ssm_model_info = ssm_model_info,
     loop_model_info = loop_model_info,
     log = log)
  if not replacement_segment_info:
    print("No replacement segments obtained", file = log)
    return
  # combine all replacements with original to get best composite
  spliced_model = run_splice_model(
        params, mmm, replacement_segment_info.segments, log = log)
  if not spliced_model:
    print("No spliced model obtained", file = log)
    return
  mmm.add_model_by_id(model= spliced_model, model_id = 'model')
  if not mmm.map_manager().is_dummy_map_manager():
    build = mmm.model_building()
    build.refine(refine_cycles = params.refine_cycles)
  if params.output_file_name and params.write_files:
    mmm.write_model(params.output_file_name)
    print("Wrote composite model to %s" %(params.output_file_name), file = log)
  return mmm.model()
def get_segment_sc_info(params, mmm, template_ca_ph,
     segments, log = sys.stdout):
  ''' get segment sc in order and require that ends match template'''
  template_sc_info_list = []
  for segment in segments:
    segment_ca_ph = segment.apply_selection_string(
      'name ca and protein').get_hierarchy()
    trim_info = trim_ends_back_to_match(params, segment_ca_ph,
       template_ca_ph, max_dist = params.max_crossover_distance,
       min_size = params.matching_min_residues)
    if not trim_info: continue
    trimmed_segment = segment.apply_selection_string(trim_info.selection_string)
    sc = trimmed_segment.apply_selection_string('name ca and protein').get_sites_cart()
    density = mmm.map_manager().density_at_sites_cart(sc)
    template_sc_info_list.append(group_args(
      group_args_type = 'template_sc_info',
      sites_cart = sc,
      segment = trimmed_segment,
      density_values = density,
      mean_density = density.min_max_mean().mean,
      score = density.min_max_mean().mean * sc.size()**0.5,
     ))
  if not template_sc_info_list:
     return None
  template_sc_info_list = sorted(template_sc_info_list,
    key = lambda ts: ts.score, reverse = True)
  return template_sc_info_list
def trim_ends_back_to_match(params,
    segment_ca_ph, template_ca_ph, max_dist = None,
    min_size = None):
  # Find matches
  sc = segment_ca_ph.atoms().extract_xyz()
  template_sc = template_ca_ph.atoms().extract_xyz()
  first_i = None
  last_i = None
  first_resno = get_first_resno(segment_ca_ph)
  for i in range(sc.size()-1): # find first two that are close
    dist,id1,id2 = sc[i:i+1].min_distance_between_any_pair_with_id(template_sc)
    if dist > max_dist: continue
    dist,id1,id2 = sc[i+1:i+2].min_distance_between_any_pair_with_id(
      template_sc)
    if dist > max_dist: continue
    first_i = i
    break
  for j in range(sc.size()-1): # find last two that are close
    i = sc.size() - j - 2
    dist,id1,id2 = sc[i:i+1].min_distance_between_any_pair_with_id(template_sc)
    if dist > max_dist: continue
    dist,id1,id2 = sc[i+1:i+2].min_distance_between_any_pair_with_id(
      template_sc)
    if dist > max_dist: continue
    last_i = i + 1
    break
  if first_i is not None and last_i is not None and \
       last_i - first_i + 1 >= min_size:
    return group_args(
     group_args_type = 'trim info',
       first_res = first_i + first_resno,
       last_res = last_i + first_resno,
       selection_string = "resseq %s:%s" %(first_i + first_resno,last_i + first_resno),
      )
  else:
    return None
def run_splice_model(params, mmm, replacement_segments, log = sys.stdout):
  # Run GA to merge segments.
  # Identify model as template
  mmm.add_model_by_id(model_id = 'template_model', model = mmm.model())
  dummy_info = group_args(
      group_args_type = 'one ca info',
      rmsd = None,
      index = 0,
      first_resno_to_add_to_index = 1,
      sites_cart = None,
      model = None,
      file_name = None,
      segment_id = None,
      final_rmsd = None,
      distance_group_info = None,
     )
  template_model_ca = get_ca_sites(mmm.model().get_hierarchy())
  template_model_ca_info = deepcopy(dummy_info)
  template_model_ca_info.sites_cart = template_model_ca
  template_model_ca_info.model = mmm.model()
  # Create replacements from result
  ca_info = group_args(
    group_args_type = 'ca info',
    template_model_ca = template_model_ca,
    template_model_ca_info = template_model_ca_info,
    mmm = mmm,
    source_models_ca = [],
    source_models_info = [])
  for segment in replacement_segments:
    mmm.shift_any_model_to_match(segment)
    ca_model = segment.apply_selection_string('name ca')
    sc = ca_model.get_sites_cart()
    ca_info.source_models_ca.append(sc)
    info = deepcopy(dummy_info)
    info.sites_cart = sc
    info.model = segment
    ca_info.source_models_info.append(info)
  # Keep track of maximum length of replacement segments
  n = 0
  for source_model_ca in ca_info.source_models_ca:
    n = max(n, source_model_ca.size())
  params.observed_max_source_model_length = n
  from phenix.model_building.splice_models import run_splice, get_splice_params
  splice_params = get_splice_params()
  splice_params.nproc = params.nproc
  splice_params.write_files = params.write_files
  splice_params.ga_total_number_of_cycles_per_gene_unit = 100
  splice_params.weights.template_residues = -100
  splice_params.weights.rms_distance_from_target = -10
  splice_params.weights.rms_junction_distance= -10
  splice_params.splice_directly= params.splice_directly
  splice_params.fill_in_gaps = (not params.quick)
  spliced_model = run_splice(splice_params, ca_info, log = log)
  if mmm.model().info().get('warnings'):
    spliced_model.info().warnings = mmm.model().info().warnings
  return spliced_model
def get_replacement_segments(params, mmm,
     ssm_model_info = None,
     loop_model_info = None,
     log = sys.stdout):
  if params.read_replacement_segments:
    segment_info = easy_pickle.load(params.segment_info_file)
  else:
    segment_info = group_args(
      group_args_type = 'segment  info',
      ssm_model_info = ssm_model_info,
      loop_model_info = loop_model_info,
      mmm = mmm)
  if params.dump_replacement_segments and params.segment_info_file:
    easy_pickle.dump(params.segment_info_file, segment_info)
  if params.splice_directly and params.merge_loops_in_splice_directly and \
      segment_info.loop_model_info:
    # merge loops
    from phenix.model_building.splice_models import merge_sequential_models
    segment_info.loop_model_info.superposed_models = merge_sequential_models(
      segment_info.loop_model_info.superposed_models, log = log)
  models = []
  for group in [segment_info.ssm_model_info, segment_info.loop_model_info]:
    if group and group.superposed_models:
      models += group.superposed_models
  if not models:  # take model from mmm
    print("No replacement segments...", file = log)
    return None
  segments = []
  for model in models:
    mmm.shift_any_model_to_match(model)
    is_composite_segment = model.info().get('is_composite_segment')
    for m in split_model(model_info(hierarchy = model.get_hierarchy())):
      segment = segment_info.mmm.model_from_hierarchy(m.hierarchy,
          return_as_model = True)
      segment.info().is_composite_segment = is_composite_segment
      segments.append(segment)
  print("\nTotal of %s replacement segments" %(len(segments)), file = log)
  segment_info.segments = segments
  return segment_info
def set_params_for_replace_segments_in_model(params):
  params = deepcopy(params)
  params.replace_segments_in_model = False # REQUIRED
  params.find_ssm_matching_segments = False
  params.replace_structure_elements = False
  params.find_loops = False
  params.reclassify_based_on_distances = False
  return params
def replace_secondary_structure_elements_in_model(params,
    mmm, log = sys.stdout):
  params = deepcopy(params)
  print(
   "Replacing secondary structure elements in model with %s residues ..." %(
    mmm.model().get_hierarchy().overall_counts().n_residues) , file = log)
  if params.reclassify_based_on_distances is None:
    params.reclassify_based_on_distances = True
  params.replace_structure_elements = True
  full_new_model_info = run(params, mmm = mmm, log = log)
  if full_new_model_info:
    print("Replacement model with %s residues obtained" %(
      full_new_model_info.model.get_hierarchy().overall_counts().n_residues),
       file = log)
    working_model = full_new_model_info.model
    mmm.add_model_by_id(model = working_model, model_id = 'model')
    build = mmm.model_building()
    build.refine(refine_cycles = params.refine_cycles)
    mmm.write_model(model_id = 'model', file_name = 'full_new_model_refined.pdb')
  else:
    print("No replacement model obtained...", file = log)
def run_indexed_ssm_matching(params, mmm, log = sys.stdout):
  print("\nIdentifying similar structures with indexed ssm matching...",
     file = log)
  params = deepcopy(params)
  params.find_ssm_matching_segments = True
  ssm_model_info = run(params, mmm = mmm, log = log)
  if ssm_model_info.superposed_models:
    print("Total of %s matching structures" %(
       len(ssm_model_info.superposed_models)), file = log)
    return ssm_model_info
  else:
    print("No matching structures", file = log)
    return
def run_find_loops(params, mmm, log = sys.stdout):
  print("\nIdentifying similar loops...", file = log)
  params = deepcopy(params)
  params.find_loops = True
  loop_model_info = run(params, mmm = mmm, log = log)
  if loop_model_info.superposed_models:
    print("Total of %s loops matched" %(
       len(loop_model_info.superposed_models)), file = log)
    return loop_model_info
  else:
    print("No matching loops found", file = log)
    return
def get_segments_to_remove(params,ss_index_info):
  keep_dict_by_segment_id = {}
  index_dict = {}
  pdb_id_dict = {}
  for segment_id in list(list(ss_index_info.segment_info_dict.keys())):
    segment_info = ss_index_info.segment_info_dict[segment_id]
    pdb_id_info = get_pdb_info_from_text(segment_info.pdb_file_name)
    text = pdb_id_info.pdb_text.upper()
    index_dict[text] = segment_id
    pdb_id_dict[pdb_id_info.pdb_id.upper()] = segment_id
  exclude_index_dict = {}
  if params.pdb_exclude_list:
    for text in params.pdb_exclude_list:
      pdb_id_info = get_pdb_info_from_text(text)
      text = pdb_id_info.pdb_text.upper()
      if len(text) == 4:
        index = pdb_id_dict.get(text,None)
      else: # full id
        index = index_dict.get(text,None)
      if index is not None:
        exclude_index_dict[index] = True
  include_index_dict = {}
  if params.pdb_include_list:
    keep_if_specified = True
    for text in params.pdb_include_list:
      pdb_id_info = get_pdb_info_from_text(text)
      text = pdb_id_info.pdb_text.upper()
      if len(text) == 4:
        index = pdb_id_dict.get(text,None)
      else: # full id
        index = index_dict.get(text,None)
      if index is not None:
        include_index_dict[index] = True
  else:
    keep_if_specified = False
  # Now make a single dict with keep/remove
  segments_to_keep = []
  segments_to_remove = []
  for segment_id in list(
     list(ss_index_info.segment_info_dict.keys())):
    if include_index_dict.get(segment_id,None):
      keep_dict_by_segment_id[segment_id] = True
    elif exclude_index_dict.get(segment_id,None):
      keep_dict_by_segment_id[segment_id] = False
    elif keep_if_specified: # only keep if specified
      keep_dict_by_segment_id[segment_id] = False
    else: # keep if not specified
      keep_dict_by_segment_id[segment_id] = True
    if keep_dict_by_segment_id[segment_id]:
      segments_to_keep.append(segment_id)
    else:
      segments_to_remove.append(segment_id)
  print("Total include_index_dict",list(include_index_dict.values()).count(True))
  print("Total exclude_index_dict",list(exclude_index_dict.values()).count(True))
  print("Total to keep: ",list(keep_dict_by_segment_id.values()).count(True))
  return group_args(
    group_args_type = 'keep dict info',
    keep_dict_by_segment_id = keep_dict_by_segment_id,
    segments_to_keep = segments_to_keep,
    segments_to_remove = segments_to_remove,
   )
def vec3_double_to_int_100(sc):
    int_values = []
    for v in sc.parts():
      v_as_float = flex.floor(0.5 + 100*v).as_numpy_array()
      v_as_int = flex.int(list(v_as_float.astype(np.int32)))
      int_values.append(v_as_int)
    return int_values
def vec3_int_to_double_100(sc):
    values = []
    for iv in sc:
      values.append(iv.as_double() * 0.01)
    v1,v2,v3 = values
    return flex.vec3_double(v1,v2,v3)
def load_database_info(params, segment_database_file, log = sys.stdout):
  t0 = time.time()
  ss_index_info = easy_pickle.load(segment_database_file)
  t1 = time.time()
  print("Time to load %s: %.2f " %(segment_database_file, t1 - t0))
  segments_to_remove_info = get_segments_to_remove(
     params,ss_index_info)
  new_segment_info_dict = {}
  for segment_id in segments_to_remove_info.segments_to_keep:
    segment_info = ss_index_info.segment_info_dict[segment_id]
    new_segment_info_dict[segment_id] = segment_info
    if not segment_info.ca_sites_cart: continue
    segment_info.ca_sites_cart = vec3_int_to_double_100(
       segment_info.ca_sites_cart)
    sc = segment_info.ca_sites_cart
    neighbor_sites_cart_list = []
    for info in segment_info.neighbor_sites_cart_info_list:
      neighbor_sites_cart_list.append(sc[info.i_start:info.i_end+1])
    segment_info.neighbor_sites_cart_list = neighbor_sites_cart_list
  ss_index_info.segment_info_dict = new_segment_info_dict
  if segments_to_remove_info and (
      segments_to_remove_info.segments_to_remove or
      segments_to_remove_info.segments_to_keep):
    print("Segments specified to remove: %s  ... specified to keep: %s" %(
      len(segments_to_remove_info.segments_to_remove),
      len(segments_to_remove_info.segments_to_keep),
      ), file = log)
  else:
    segments_to_remove_info = None
  t0 = time.time()
  uncompress_database(ss_index_info, segments_to_remove_info)
  t1 = time.time()
  print("Time to uncompress: %.2f " %(t1 - t0))
  return ss_index_info
def read_fragment_database(params,
     write_file = False,
     database_file = None,
     data_dir = None,
     default_database_file = 'db_top2018.pkl.gz',
     log = sys.stdout):
  if hasattr(params,'data_dir') and os.path.isdir(params.data_dir):
    data_dir = params.data_dir
  if hasattr(params,'database_file'):
    database_file = params.database_file
  print("Reading fragment database with "+
     "default dir of %s and database file of %s" %(data_dir,database_file),
     file = log)
  from phenix.model_building.fragment_search import get_db_info
  db_info = get_db_info(
      data_dir = data_dir,
      database_file = database_file,
      default_database_file = default_database_file,
      log = log)
  # Group everything from one file into 1 entry
  file_name_list = get_file_name_list(db_info, log = log)
  data_dir = db_info.data_dir
  # Just return file name list and directory info
  db_info = group_args(
    group_args_type = 'dummy db info',
    data_dir = data_dir,
    db = group_args(
      group_args_type = 'dummy db',
      segment_info_dict = {},
     ))
  db_info.db.segment_info_dict = {}
  segment_id = -1
  for file_name in file_name_list:
    segment_id += 1
    segment_info = group_args(
      group_args_type = 'dummy segment info with file names only',
      ca_sites_cart = None,
      pdb_file_name = file_name)
    db_info.db.segment_info_dict[segment_id] = segment_info
  return db_info
def compare_coordinates(params, log = sys.stdout):
  from iotbx.data_manager import DataManager
  dm = DataManager()             # Initialize the DataManager and call it dm
  model_list = []
  for file_name in params.comparison_pdb_file_names:
    model_list.append(dm.get_model(file_name))
  print("Total files to compare: ",len(model_list))
  for i in range(len(model_list) - 1):
    for j in range(i+1, len(model_list)):
      compare_model_pair(params, model_list[i],model_list[j], log = log)
def compare_model_pair(params, m1,m2, unique_chain_id = None,log = sys.stdout):
  # compare pair of models. M1 will be fixed, m2 moving
  # Only one chain will be compared in each
  # Compare one chain of m2 with all of m1
  pair_dict_info = get_best_pair_dict_info(params = params,
      m1=m1, m2=m2, unique_chain_id = unique_chain_id, log = log)
  return pair_dict_info
def get_best_pair_dict_info(params = None,
      m1 = None, m2 = None,
      unique_chain_id = None,
      ph1 = None,
      ph2 = None,
      update_coordinates = True,
      shift_field_distance = None,
      max_shift_field_ca_ca_deviation = None,
      starting_distance_cutoff = None,
      log = sys.stdout):
  if not params:
     params = get_ssm_params(shift_field_distance = shift_field_distance,
       max_shift_field_ca_ca_deviation = max_shift_field_ca_ca_deviation,
       starting_distance_cutoff = starting_distance_cutoff)
  if not unique_chain_id:
    unique_chain_id = 'ZA'
  if not ph1:
    m1_chain_id = get_chain_id(m1.get_hierarchy())
    ph1 = m1.apply_selection_string('name ca and chain %s' %(m1_chain_id)
      ).get_hierarchy()
  if not ph2:
    m2_chain_id = get_chain_id(m2.get_hierarchy())
    ph2 = m2.apply_selection_string('name ca and chain %s' %(m2_chain_id)
      ).get_hierarchy()
  m1_chain_id = get_chain_id(ph1)
  m2_chain_id = get_chain_id(ph2)
  ph1_sav = ph1.deep_copy()
  ph2_sav = ph2.deep_copy()
  # Try varying min_sequential_fraction if necessary
  min_sequential_fraction = params.target_sequential_fraction
  for i in range(2):
    ph1 = ph1_sav.deep_copy()
    ph2 = ph2_sav.deep_copy()
    if i == 0:
      pass # already set
    else:
      min_sequential_fraction = params.min_sequential_fraction
    pair_dict_info = compare_model_pair_chains(params, ph1, ph2,
        unique_chain_id,
        min_sequential_fraction = min_sequential_fraction,
        log = log)
    if pair_dict_info: break # got it
  if not pair_dict_info:
     return  # nothing to do
  if update_coordinates:
    # Repeat with new orientation
    print("\nRepeating match with new orientation", file = log)
    ph1 = ph1_sav.deep_copy()
    ph2 = ph2_sav.deep_copy()
    ph2.atoms().set_xyz(
        apply_lsq_fit(pair_dict_info.lsq_fit, ph2.atoms().extract_xyz()))
    pair_dict_info = compare_model_pair_chains(params, ph1, ph2,
      unique_chain_id,
      min_sequential_fraction = min_sequential_fraction,
      log = log)
    if not pair_dict_info:
      return  # nothing to do
    # Apply the shift field
    print("\nRepeating match with shift field", file = log)
    ph1 = ph1_sav.deep_copy()
    ph2 = ph2_sav.deep_copy()
    working_xyz =apply_lsq_fit(pair_dict_info.lsq_fit, ph2.atoms().extract_xyz())
    if params.morph:
      working_xyz = apply_shift_field(params, pair_dict_info.shift_field_info,
        working_xyz)
    ph2.atoms().set_xyz(working_xyz)
    new_pair_dict_info = compare_model_pair_chains(params, ph1, ph2,
        unique_chain_id,
      tolerance_ratio_values = [
       pair_dict_info.sequential_group_info.tolerance_ratio],
       min_sequential_fraction  = \
        pair_dict_info.sequential_group_info.min_sequential_fraction,
      log = log)
    if not new_pair_dict_info:
      return  # nothing to do
    pair_dict_info = new_pair_dict_info
  else:
    working_xyz = ph2.atoms().extract_xyz()
  # Select all residues in ph2 that have CA (smoothed) near ph1
  print("\nRepeating match with reverse", file = log)
  ph1 = ph1_sav.deep_copy()
  ph2 = ph2_sav.deep_copy()
  if params.morph and update_coordinates:
    working_xyz = apply_shift_field(params, pair_dict_info.shift_field_info,
      working_xyz)
    working_xyz =apply_lsq_fit(
        pair_dict_info.lsq_fit, ph2.atoms().extract_xyz())
    ph2.atoms().set_xyz(working_xyz)
  reverse_pair_dict_info = compare_model_pair_chains(params, ph2, ph1,
      unique_chain_id,
      tolerance_ratio_values = [
       pair_dict_info.sequential_group_info.tolerance_ratio],
       min_sequential_fraction  = \
        pair_dict_info.sequential_group_info.min_sequential_fraction,
      log = log)
  if not reverse_pair_dict_info:
    return # nothing to do
  # Now identify all the CA in ph2 that are close to ph1
  ca_sites_1 = ph1.atoms().extract_xyz()
  distances = flex.double()
  for i in range(working_xyz.size()):
    close_sites = reverse_pair_dict_info.pair_dict[i] # close in ph1
    if not close_sites:
      distances.append(params.delta)
    else:
      j = close_sites[0]
      dd = (col(working_xyz[i]) - col(ca_sites_1[j])).length()
      distances.append(dd)
  distances = smooth_values(distances)
  keep = keep_if_close(params, working_xyz, distances, params.close_distance,
      params.max_shift_field_ca_ca_deviation,
      min_residues = params.matching_min_residues)
  residues_to_keep = []
  atoms = ph2.atoms()
  for i in range(working_xyz.size()):
    if keep[i]:
      residues_to_keep.append(resno(atoms[i]))
  chain_id = get_chain_id(ph2)
  chain_dict = {chain_id: residues_to_keep}
  pair_dict_info.match_selection = get_selection_from_chain_dict(chain_dict)
  return pair_dict_info
def get_selection_from_chain_dict(chain_dict):
  ''' Create a selection based on a dict of residues present in each chain'''
  from iotbx.pdb import resseq_encode
  int_type = type(1)
  selection_list = []
  for chain_id in list(chain_dict.keys()):
    residues_to_keep = chain_dict[chain_id]
    for c in compress_indices(residues_to_keep):
      if type(c) == int_type:
        i = resseq_encode(c).replace(" ","")
        selection_list.append(" ( chain %s and resseq %s:%s) " %(chain_id, i,i))
      else:
        i = resseq_encode(c[0]).replace(" ","")
        j = resseq_encode(c[1]).replace(" ","")
        selection_list.append(" (chain %s and resseq %s:%s) " %(chain_id, i,j))
  return " or ".join(selection_list)
def keep_if_ca_ca_distortion_is_low(max_shift_field_ca_ca_deviation,keep,
      working_xyz, ca_ca_dist = 3.8):
  distances = (working_xyz[:-1] - working_xyz[1:]).norms()
  ok_bond = (flex.abs(distances - ca_ca_dist) < max_shift_field_ca_ca_deviation)
  for i in range(ok_bond.size()):
     if not ok_bond[i]:
       keep[i] = False
       keep[i+1] = False
  return keep
def keep_if_close(params,working_xyz, distances,
       close_distance,max_shift_field_ca_ca_deviation,
      min_residues = None,
      near_end_max_ratio_to_mean = 1.0):
  #  Keep residues that are close to target...if end residues very near
  #   to close_distance and others are not...toss them
  keep = (distances <= close_distance)
  if max_shift_field_ca_ca_deviation:
    keep = keep_if_ca_ca_distortion_is_low(max_shift_field_ca_ca_deviation,
      keep, working_xyz)
  # Remove end residues that are a bit low
  average_dist = max(params.delta * params.tolerance_ratio_min,
     distances.select(keep).min_max_mean().mean)
  if average_dist is None:
    return keep # it's empty
  cutoff = average_dist * near_end_max_ratio_to_mean
  for i in range(keep.size()//2):
    if distances[i] <= cutoff:
      break # done
    else:
      keep[i] = False
  for i in range(keep.size()//2):
    ii = keep.size() - i - 1
    if distances[ii] <= cutoff:
      break # done
    else:
      keep[ii] = False
  return remove_singles_in_keep(keep, min_residues)
def remove_singles_in_keep(keep, min_residues):
  # Remove all singles ... up to matching_min_residues
  group = None
  groups = []
  for i in range(keep.size()):
   if keep[i] and group and i == group.end_i + 1:
     group.end_i += 1
   elif keep[i]:
     group = group_args(
       group_args_type = 'range group',
       start_i = i,
      end_i = i,
       )
     groups.append(group)
  for group in groups:
    if group.end_i - group.start_i + 1 < min_residues: # remove
      for i in range(group.start_i, group.end_i + 1):
        keep[i] = False
  # Find all gaps < min_residues and fill in
  group = None
  groups = []
  for i in range(keep.size()):
   if (not keep[i]) and group and i == group.end_i + 1:
     group.end_i += 1
   elif (not keep[i]):
     group = group_args(
       group_args_type = 'range group',
       start_i = i,
      end_i = i,
       )
     groups.append(group)
  for group in groups:
    if group.end_i - group.start_i + 1 < min_residues: # remove
      for i in range(group.start_i, group.end_i + 1):
        keep[i] = True
  return keep
def compare_model_pair_chains(params, ph1, ph2,
      unique_chain_id,
      tolerance_ratio_values = None,
      min_sequential_fraction = None,
      log = sys.stdout):
  # Compare one chain of ph2 with all chains in ph1
  # First put both in same hierarchy so we can use fast pair generation
  m1_size = ph1.atoms().size() # save number of atoms in ph1
  composite_ph = create_composite_ph(ph1, ph2, unique_chain_id)  # NOTE: changes ph1
  pair_dict_info = get_pair_dict(params, composite_ph,
       m1_size = m1_size,
       unique_chain_id = unique_chain_id,
       log = log)  # pair_dict[resseq_ph1] = list of resseq in ph2 matching
     # pair_dict_info: pair_dict, m1_xyz, m2_xyz
     #  pair_dict[i] = list of indices in m2_xyz matching m1_xyz[i],
     #    sorted by distance
  # Now identify matchings of m2  to m1 that are:
  #   1. generally sequential in both, but gaps may exist in either
  #   2. minimize rmsd of matching xyz, after application of local shifts to m2
  sequential_group_info = get_best_sequential_group(params, pair_dict_info,
    tolerance_ratio_values = tolerance_ratio_values,
    min_sequential_fraction = min_sequential_fraction,
    log = log)
  if not sequential_group_info:
     return # nothing to do
  print("Sequential groups size: %s (%.2f)" %(
       sequential_group_info.sequential_group.n_elements,
        sequential_group_info.tolerance_ratio), file = log)
  # Now redo the RT matrices with this group
  target_xyz = flex.vec3_double()
  moving_xyz = flex.vec3_double()
  for group in sequential_group_info.sequential_group.group_list:
    for i in range(group.i_start, group.i_end+1):
      target_xyz.append(pair_dict_info.m1_xyz[i])
    for j in range(group.j_start, group.j_end+1):
      moving_xyz.append(pair_dict_info.m2_xyz[j])
  starting_rmsd = (target_xyz - moving_xyz).rms_length()
  lsq_fit = superpose.least_squares_fit(
      reference_sites = target_xyz,
      other_sites     = moving_xyz)
  fitted_xyz = apply_lsq_fit(lsq_fit,moving_xyz)
  shifts = (target_xyz - fitted_xyz)
  fitted_rmsd = shifts.rms_length()
  print("RMSD before fitting: %.2f   after: %.2f " %(
    starting_rmsd, fitted_rmsd), file = log)
  pair_dict_info.rmsd = fitted_rmsd
  pair_dict_info.lsq_fit = lsq_fit
  pair_dict_info.close_sites = shifts.size()
  pair_dict_info.sequential_group_info = sequential_group_info
  pair_dict_info.shift_field_info = group_args(
    group_args_type = 'shift_field_info',
    centers = fitted_xyz,
    shifts = shifts,)
  return pair_dict_info
def get_best_sequential_group(params, pair_dict_info,
    tolerance_ratio_values = None,
    min_sequential_fraction = None,
    log = sys.stdout):
  # Figure out what tolerance, gap we need to get started
  shift_field_info = None
  if not min_sequential_fraction:
    min_sequential_fraction = params.min_sequential_fraction
  if not tolerance_ratio_values:
    tolerance_ratio_values = []
    for i in range(params.tolerance_ratio_tries):
      w = i/params.tolerance_ratio_tries
      tolerance_ratio_values.append(params.tolerance_ratio_min  + \
         w * (1-params.tolerance_ratio_min))
  for tolerance_ratio in tolerance_ratio_values:
    n_possible = min(pair_dict_info.m1_ca_atoms.size(),
                     pair_dict_info.m2_ca_atoms.size())
    # Apply shift_field info to sites in m2, then find closest matching sites
    local_pair_dict_info = get_working_pair_dict_info(params, shift_field_info,
      pair_dict_info,
      tolerance_ratio = tolerance_ratio,
      )
    sequential_groups = get_match_with_gaps(params,local_pair_dict_info,
       log = log)
    if sequential_groups and sequential_groups[0].n_elements >= \
         min_sequential_fraction * n_possible:
      return group_args(
        group_args_type = 'sequential group',
        sequential_group = sequential_groups[0],
        tolerance_ratio = tolerance_ratio,
        shift_field_info = shift_field_info,
        min_sequential_fraction = min_sequential_fraction,
       ) # good enough
  return None
def resno(ca_atom):
  return ca_atom.parent().parent().resseq_as_int()
def get_match_with_gaps(params,pair_dict_info,
     log = sys.stdout):
  # find alignment of m1 and m2 using the indices allowed to match m1 in
  #   pair_dict
  match_list = get_common_differences(params,
      pair_dict_info, log = log) # list of matching indices
  sequential_groups_list = get_sequential_groups_list(match_list)
  return get_best_sequential_groups(sequential_groups_list,
    max_gap = params.max_gap,
    )
def get_best_sequential_groups(sequential_groups_list, max_gap = None):
  # find set of sequential groups always increasing that contains the most entries
  sequential_groups = []
  n = len(sequential_groups_list)
  for k in range(n): # use this as seed
    sg = group_args(group_args_type = 'sequential groups',
      group_list = [sequential_groups_list[k]],
      n_elements = sequential_groups_list[k].i_end - \
          sequential_groups_list[k].i_start + 1,
      )
    sequential_groups.append(sg)
    for kk in range(k+1,n):
      last_i = sg.group_list[-1].i_end
      last_j = sg.group_list[-1].j_end
      current_i = sequential_groups_list[kk].i_start
      current_j = sequential_groups_list[kk].j_start
      if current_i < last_i + 1 or current_i > last_i + max_gap + 1:
        continue #
      if current_j < last_j + 1 or current_j > last_j + max_gap + 1:
        continue #
      sg.group_list.append(sequential_groups_list[kk])
      sg.n_elements += sequential_groups_list[kk].i_end - \
          sequential_groups_list[kk].i_start + 1
    for kk in range(k-1,-1,-1):
      last_i = sg.group_list[0].i_start
      last_j = sg.group_list[0].j_start
      current_i = sequential_groups_list[kk].i_end
      current_j = sequential_groups_list[kk].j_end
      if current_i > last_i - 1 or current_i < last_i - max_gap - 1:
        continue #
      if current_j > last_j - 1 or current_j < last_j - max_gap - 1:
        continue #
      sg.group_list = [sequential_groups_list[kk]] + sg.group_list
      sg.n_elements += sequential_groups_list[kk].i_end - \
          sequential_groups_list[kk].i_start + 1
  sequential_groups = sorted(sequential_groups,
     key = lambda sg: sg.n_elements, reverse = True)
  return sequential_groups
def get_sequential_groups_list(match_list):
  sequential_groups_list = []
  group = None
  for ii in range(len(match_list)):
    i,j = match_list[ii]
    if group and [i,j] == [group.i_end+ 1, group.j_end + 1]:
       group.i_end = i
       group.j_end = j
    else:
      group = group_args(group_args_type = 'match group',
        i_start = i,
        i_end = i,
        j_start = j,
        j_end = j)
      sequential_groups_list.append(group)
  sequential_groups_list = sorted(sequential_groups_list,
      key = lambda g: g.i_end - g.i_start, reverse = True)
  useful_groups = []
  for group in sequential_groups_list:
    if group.i_end - group.i_start > 1:
      useful_groups.append(group)
    else:
      break # done
  useful_groups = sorted(useful_groups,
      key = lambda g: g.i_start)
  return useful_groups
def get_common_differences(params,pair_dict_info, log = sys.stdout):
  # Method:  for each window of length matching_window in m1, find all the
  #   indices in m2 matching m1 and the differences in index. Save the most
  #    common difference in index.  Then try to  link these up keeping the
  #    same common differences as much as possible
  match_list = []
  for i in range(pair_dict_info.m1_xyz.size()):
    delta_indices = flex.int()
    for ii in range(params.matching_window):
      k = i + ii
      if k >= pair_dict_info.m1_xyz.size(): continue
      matching_indices = pair_dict_info.pair_dict[k] # indices in m2 matching k
      if not matching_indices:
        continue
      delta_indices.extend(matching_indices - k)
    best_count = None
    best_delta_index = None
    for key in list(delta_indices.counts().keys()):
      if best_delta_index is None or delta_indices.counts()[key]> best_count:
        best_count = delta_indices.counts()[key]
        best_delta_index = key
    if best_delta_index is None: continue
    # Make sure that i and j actually match...
    j = i + best_delta_index
    if not j in pair_dict_info.pair_dict[i]: continue
    match_list.append([i,j])
  return match_list
def get_working_pair_dict_info(params, shift_field_info,
    pair_dict_info,
   tolerance_ratio = None):
  if params.morph:
    working_m2_xyz = apply_shift_field(params, shift_field_info,
      pair_dict_info.m2_xyz)
  else:
    working_m2_xyz = pair_dict_info.m2_xyz
  # Now just get closest index in m2 to each xyz in m1
  pair_dict_info = sort_pair_dict(params, pair_dict_info = pair_dict_info,
     n_keep = 1, tolerance_ratio = tolerance_ratio)
  return group_args(
    group_args_type = 'local pair dict',
    pair_dict = pair_dict_info.pair_dict,
    m1_xyz = pair_dict_info.m1_xyz,
    m2_xyz = working_m2_xyz,
    lsq_fit = None,
    sequential_group_info = None,
    shift_field_info = None,
    m1_ca_atoms = pair_dict_info.m1_ca_atoms,
    m2_ca_atoms = pair_dict_info.m2_ca_atoms,
     )
def create_composite_ph(ph1, ph2, unique_chain_id):
  set_chain_id(ph2, unique_chain_id)
  for chain in ph2.models()[0].chains():
    ph1.models()[0].append_chain(chain.detached_copy())
  return ph1
def get_pair_dict(params, ph1,
     m1_size = None,
     unique_chain_id =None, log = sys.stdout):
  # Assumes ph1 has all CA of m1 first (total of m1_size of these), then
  #  in a different chain with id unique_chain_id, all the CA of m2.
  # Create dict of all CA in m1 with chain ID of unique_chain_id close to each
  #  CA in ph1
  # Then sort all the matches by distance
  asu_table = get_asu_table(ph1, distance_cutoff = params.delta )
  ca_atoms = ph1.atoms()
  nn = ca_atoms.size()
  other_ca_atoms = ca_atoms[m1_size:]
  ca_atoms = ca_atoms[:m1_size]
  asc1 = ph1.atom_selection_cache()
  sel = asc1.selection(string = 'chain %s' %(unique_chain_id))
  ph1b = ph1.select(sel)
  other_xyz = ph1b.atoms().extract_xyz()
  pair_dict = {}
  ii = -1
  i_seq_unique_chain = None
  xyz_list = ca_atoms.extract_xyz()
  assert other_ca_atoms.size()+ca_atoms.size() == nn
  for ca_atom in ca_atoms:
    ca_atom_resseq = ca_atom.parent().parent().resseq_as_int()
    ii += 1
    i_seq=ca_atom.i_seq
    assert ii == i_seq  # why should this work...what about gaps?
    asu_dict = asu_table[i_seq]
    pair_dict[i_seq] = []
    for jj , j_sym_groups in asu_dict.items() :
      j_seq = jj - m1_size
      if j_seq < 0 or j_seq >= other_xyz.size(): continue
      other_atom = other_ca_atoms[j_seq]  # this is a close atom
      if other_atom.chain().id != unique_chain_id: continue
      other_atom_resseq = other_atom.parent().parent().resseq_as_int()
      pair_dict[i_seq].append(j_seq)
      assert ca_atom.xyz == xyz_list[i_seq]
      assert other_atom.xyz == other_xyz[j_seq]
  assert ii + 1 == m1_size
  pair_dict_info = group_args(
    group_args_type = 'pair dict',
    pair_dict = pair_dict,
    m1_xyz = xyz_list,
    m2_xyz = other_xyz,
    lsq_fit = None,
    sequential_group_info = None,
    shift_field_info = None,
    m1_ca_atoms = ca_atoms,
    m2_ca_atoms = other_ca_atoms,
     )
  pair_dict_info = sort_pair_dict(params,pair_dict_info)
  return pair_dict_info
def get_asu_table(hierarchy=None,distance_cutoff=None):
  # Construct list of all close atom pairs
  xrs=hierarchy.extract_xray_structure()
  pair_asu_table = xrs.pair_asu_table(
      distance_cutoff=distance_cutoff)
  asu_mappings = pair_asu_table.asu_mappings()
  return pair_asu_table.table()
def sort_pair_dict(params,
   pair_dict_info, n_keep = None, tolerance_ratio = None,):
  # work with shallow copy
  local_pair_dict_info = group_args(**pair_dict_info().copy())
  local_pair_dict_info.pair_dict = {}  # modifies local copy only
  for i in range(pair_dict_info.m1_xyz.size()):
    indices_in_m2 = pair_dict_info.pair_dict[i]
    indices_and_distances = []
    for j in indices_in_m2:
      indices_and_distances.append(
        [j,
      (col(pair_dict_info.m2_xyz[j]) - col(pair_dict_info.m1_xyz[i])).length()])
    indices_and_distances = sorted(indices_and_distances,
      key = lambda iad: iad[1])
    sorted_indices = flex.int()
    for j,dist in indices_and_distances:
      if tolerance_ratio is not None and dist > tolerance_ratio * params.delta:
        break
      sorted_indices.append(j)
      if n_keep and sorted_indices.size() >= n_keep:
        break
    local_pair_dict_info.pair_dict[i] = sorted_indices
  return local_pair_dict_info
def make_database(params,
     mmm = None,
     log = sys.stdout):
  if params.make_database_from_pdb_include_list:
    if mmm is None:
      from iotbx.map_model_manager import map_model_manager
      mmm = map_model_manager()
      mmm.set_resolution(10)
      mmm.generate_map()
    print("Creating mini-database from pdb_include_list", file = log)
    if not params.pdb_include_list:
      print("No PDB IDs selected...skipping", file = log)
      return
    print("PDB IDs to include:"," ".join(params.pdb_include_list), file = log)
    db_info = group_args(
      group_args_type = 'dummy db info',
      data_dir = None,
      db = group_args(
        group_args_type = 'dummy db ',
        segment_info_dict = {},
      ))
    model_mmm = mmm
    pdb_info_list = []
    segment_id = -1
    for pdb_id_text in params.pdb_include_list:
      segment_id += 1
      pdb_info = get_pdb_info_from_text(pdb_id_text)
      segment_info = group_args(
        group_args_type = 'dummy segment info with file names only for pdb include list',
        ca_sites_cart = None,
        pdb_file_name = pdb_id_text)
      db_info.db.segment_info_dict[segment_id] = segment_info
  else:
    print("Creating indexing", file = log)
    db_info = read_fragment_database(params,
      log = log) # used for list of files
    # make up a map_model_manager mmm
    from iotbx.map_model_manager import map_model_manager
    model_mmm = map_model_manager()
    segment_id = list(list(db_info.db.segment_info_dict.keys()))[0]
    ch = easy_pickle.load(os.path.join(db_info.data_dir,
      "%s" %( db_info.db.segment_info_dict[segment_id].pdb_file_name)))
    model = ch.as_model()
    model_mmm.set_model(model)
    model_mmm.set_resolution(10)
    model_mmm.generate_map()
  """
  new_dict = {}
  n = 0
  for key in list(db_info.db.segment_info_dict.keys()):
    new_dict[key] = db_info.db.segment_info_dict[key]
    n+=1
    if n > 20: break
  db_info.db.segment_info_dict = new_dict
  """
  ss_index_info = None
  end_number = -1
  nproc = params.nproc
  n_tot = len((list(db_info.db.segment_info_dict.keys())))
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
    'params':params,
    'db_info':db_info,
    'model_mmm':model_mmm,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_run_indexing,
    log = log)
  ss_index_info = None
  for run_info in runs_carried_out:
    local_ss_index_info = easy_pickle.load(run_info.result.saved_file_name)
    ss_index_info = merge_ss_index_info(ss_index_info, local_ss_index_info)
    os.remove(run_info.result.saved_file_name)
  if not ss_index_info:
    print("No indexing done...", file = log)
    return
  ss_index_info.index_info = base_index_info(params) # save what we used
  # Save list of files in ss_index_info.segment_info_dict .
  #   Save neighbor info:
  for key in list(ss_index_info.segment_info_dict.keys()):
    ss_segment_info = ss_index_info.segment_info_dict[key]
    segment_info = db_info.db.segment_info_dict[key]
    segment_info.neighbor_sites_cart_info_list = \
       ss_segment_info.neighbor_sites_cart_info_list
    segment_info.ca_sites_cart = None # don't save them here
  ss_index_info.segment_info_dict = db_info.db.segment_info_dict
  # Make dummy db_info
  db_info = group_args(
    group_args_type = 'dummy db info',
    data_dir = db_info.data_dir,
    db = group_args(
      group_args_type = 'dummy db',
      segment_info_dict = {},
     ))
  # compress database
  compress_database(ss_index_info)
  # Save params used
  db_info.params = deepcopy(params)
  # Save database
  ss_index_info.db_info = db_info
  easy_pickle.dump(
    os.path.join(params.data_dir,params.segment_database_file),ss_index_info)
  print("Wrote database to %s" %(
   os.path.join(params.data_dir,params.segment_database_file)), file = log)
def get_pdb_info_from_text(pdb_id_text, default_chain_id = None):
      pdb_id_text = pdb_id_text.replace(".pkl","").replace(".gz","").replace(
          '.pdb','').replace('_trimmed','').replace('_pruned_mc','')
      pdb_id = pdb_id_text.split("_")[0]
      if len(pdb_id) != 4:
        pdb_id_text = "" # skip it
        pdb_id = ""
      if len(pdb_id_text.split("_")) > 1:
        chain_id = pdb_id_text.split("_")[1]
        selection_string = "chain %s" %(chain_id)
        pdb_text = "%s_%s" %(pdb_id,chain_id)
      elif default_chain_id:
        chain_id = default_chain_id
        selection_string = "chain %s" %(chain_id)
        pdb_text = "%s_%s" %(pdb_id,chain_id)
      else:
        chain_id = None
        selection_string = None
        pdb_text = "%s" %(pdb_id)
      return group_args(
        group_args_type = 'pdb info',
        pdb_id = pdb_id,
        chain_id = chain_id,
        pdb_text = pdb_text,
        selection_string = selection_string,
        pdb_file_name = None,
        full_file_name = None,
       )
def compress_database(ss_index_info):
  sbsi = ss_index_info.structures_by_ss_and_index
  for ss1 in list(sbsi.keys()):
    for ss2 in list(sbsi[ss1].keys()):
      dd = ss_index_info.structures_by_ss_and_index[ss1][ss2]
      for index in list(dd.keys()):
        values = sorted(dd[index])
        dd[index] = values
        compressed_values = compress_indices(values)
        dd[index] = compressed_values
  # Now compress all the coordinates saved:
  for dd in (ss_index_info.cluster_centers_by_segment_id,
             ss_index_info.cluster_ends_1_by_segment_id,
             ss_index_info.cluster_ends_2_by_segment_id):
    for i in list(list(dd.keys())):
      dd[i] = vec3_double_to_int_100(dd[i])
  #And make smaller group_args:
  for segment_id in list(ss_index_info.segment_info_dict.keys()):
    segment_info = ss_index_info.segment_info_dict[segment_id]
    new_neighbor_list = []
    for neighbor_info in segment_info.neighbor_sites_cart_info_list:
      new_neighbor_list.append( [
      neighbor_info.i_start ,
      neighbor_info.i_end ,
      neighbor_info.original_chain_id ,
      neighbor_info.original_start_resno ,
      neighbor_info.original_end_resno ,
      neighbor_info.c1_start ,
      neighbor_info.c1_end ,
      neighbor_info.c2_start ,
      neighbor_info.c2_end ,
      neighbor_info.c1_known_ss ,
      neighbor_info.c2_known_ss])
    segment_info.neighbor_sites_cart_info_list = new_neighbor_list
def uncompress_database(ss_index_info, segments_to_remove_info):
  sbsi = ss_index_info.structures_by_ss_and_index
  for ss1 in list(sbsi.keys()):
    for ss2 in list(sbsi[ss1].keys()):
      dd = sbsi[ss1][ss2]
      for index in list(dd.keys()):
        compressed_values = dd[index]
        values = uncompress_indices(compressed_values,segments_to_remove_info)
        dd[index] = values
  if segments_to_remove_info:
    ocsi = ss_index_info.other_clusters_by_ss_and_index
    for segment_id in list(list(ocsi.keys())):
      if not segments_to_remove_info.keep_dict_by_segment_id[segment_id]:
        del ocsi[segment_id]  # remove it entirely
        del ss_index_info.cluster_known_ss_by_segment_id[segment_id]
        del ss_index_info.cluster_centers_by_segment_id[segment_id]
        del ss_index_info.cluster_ends_1_by_segment_id[segment_id]
        del ss_index_info.cluster_ends_2_by_segment_id[segment_id]
  # Now uncompress all the coordinates saved:
  for dd in (ss_index_info.cluster_centers_by_segment_id,
             ss_index_info.cluster_ends_1_by_segment_id,
             ss_index_info.cluster_ends_2_by_segment_id):
    for i in list(list(dd.keys())):
      dd[i] = vec3_int_to_double_100(dd[i])
  for segment_id in list(ss_index_info.segment_info_dict.keys()):
    segment_info = ss_index_info.segment_info_dict[segment_id]
    if not hasattr(segment_info,'neighbor_sites_cart_info_list'): continue
    new_neighbor_list = []
    for info in segment_info.neighbor_sites_cart_info_list:
      neighbor_info = group_args()
      new_neighbor_list.append(neighbor_info)
      [neighbor_info.i_start ,
        neighbor_info.i_end ,
        neighbor_info.original_chain_id ,
        neighbor_info.original_start_resno ,
        neighbor_info.original_end_resno ,
        neighbor_info.c1_start ,
        neighbor_info.c1_end ,
        neighbor_info.c2_start ,
        neighbor_info.c2_end ,
        neighbor_info.c1_known_ss ,
        neighbor_info.c2_known_ss] = info
      segment_info.neighbor_sites_cart_info_list = new_neighbor_list
def compress_indices(values):
  # Take sorted list of values and compress sequential indices
  compressed_values = []
  # Save list of ranges or lists of individual values (if not sequential)
  n = len(values)
  n1 = n - 1
  working_group = None
  for i in range(n):
    if working_group and working_group[1] + 1  == values[i]:
      working_group[1] = values[i]
    else:
      working_group = [values[i],values[i]]
      compressed_values.append(working_group)
  new_compressed_values = []
  for c1,c2 in compressed_values:
    if c1==c2:
      new_compressed_values.append(c1)
    else:
      new_compressed_values.append([c1,c2])
  return new_compressed_values
def uncompress_indices(new_compressed_values, segments_to_remove_info):
  values = []
  int_type = type(1)
  for c in new_compressed_values:
    if type(c) == int_type:
      values.append(c)
    else:
      values += list(range(c[0],c[1]+1))
  if not segments_to_remove_info:
    return values
  new_values = []
  for value in values:
    if segments_to_remove_info.keep_dict_by_segment_id[value]:
        new_values.append(value)
  return new_values
def get_file_name_list(db_info, log = sys.stdout):
  file_name_list = []
  if hasattr(db_info.db,'segment_info_dict'):
    sid = db_info.db.segment_info_dict
    nn = len(list(list(sid.keys())))
    keys = list(sid.keys())
  else: # for now
    sid = db_info.db.segment_info_list
    keys = list(range(len(sid)))
    nn = len(keys)
  for i in keys:
    segment_info = sid[i]
    if not segment_info.pdb_file_name in file_name_list:
      file_name_list.append(segment_info.pdb_file_name)
  return file_name_list
def group_of_run_indexing(params, run_info = None, db_info = None,
   model_mmm = None, log = sys.stdout):
  ss_index_info = None
  for segment_id in range(run_info.start_number, run_info.end_number + 1):
    local_ss_index_info = run_one_indexing(params,db_info,model_mmm, segment_id,
      log = log)
    if not local_ss_index_info: continue
    ss_index_info = merge_ss_index_info(ss_index_info, local_ss_index_info)
  saved_file_name = "temp_db_%s_%s.pkl" %(
      run_info.start_number, run_info.end_number)
  easy_pickle.dump(saved_file_name,ss_index_info)
  return group_args(
    group_args_type = 'group of run indexing',
    saved_file_name = saved_file_name,
    )
def merge_ss_index_info(ss_index_info, local_ss_index_info):
  if ss_index_info is None:
    return local_ss_index_info
  for ss1 in ['H','S']:
    for ss2 in ['H','S']:
      dd = ss_index_info.structures_by_ss_and_index[ss1][ss2]
      local_dd = local_ss_index_info.structures_by_ss_and_index[ss1][ss2]
      for index in list(list(local_dd.keys())):
        if not index in list(dd.keys()):
          dd[index] = []
        dd[index] += local_dd[index]
  for segment_id in list(
     list(local_ss_index_info.other_clusters_by_ss_and_index.keys())):
    assert not segment_id in list(ss_index_info.other_clusters_by_ss_and_index.keys())
    ss_index_info.other_clusters_by_ss_and_index[segment_id] =\
        local_ss_index_info.other_clusters_by_ss_and_index[segment_id]
    ss_index_info.cluster_centers_by_segment_id[segment_id] = \
        local_ss_index_info.cluster_centers_by_segment_id[segment_id]
    ss_index_info.cluster_ends_1_by_segment_id[segment_id] = \
        local_ss_index_info.cluster_ends_1_by_segment_id[segment_id]
    ss_index_info.cluster_ends_2_by_segment_id[segment_id] = \
        local_ss_index_info.cluster_ends_2_by_segment_id[segment_id]
    ss_index_info.cluster_known_ss_by_segment_id[segment_id] = \
        local_ss_index_info.cluster_known_ss_by_segment_id[segment_id]
    ss_index_info.n_structures += 1
    ss_index_info.segment_info_dict[segment_id] =\
      local_ss_index_info.segment_info_dict[segment_id]
  return ss_index_info
def get_model_from_ph(mmm,matching_ph):
      # comes in with no origin shift
      if not matching_ph:
        return None
      moving_model = mmtbx.model.manager(
        model_input = matching_ph.as_pdb_input(),
        crystal_symmetry = mmm.crystal_symmetry(),)  # want to keep shift
      moving_model.set_unit_cell_crystal_symmetry(
        mmm.unit_cell_crystal_symmetry())
      moving_model.set_shift_cart(mmm.shift_cart())
      return moving_model
def get_ph_for_segment(mmm, db_info, segment_info_dict, segment_id,
   check_only = False, return_as_model = False,
   local_pdb_dir = None):
  segment_info = segment_info_dict[segment_id]
  ph = None
  model = None
  pdb_info = get_pdb_info_from_text(segment_info.pdb_file_name,
      default_chain_id = "A")
  if db_info.data_dir:
    file_name = os.path.join(db_info.data_dir,"%s.gz" %(
     segment_info.pdb_file_name.replace(".gz","")))
    if os.path.isfile(file_name):
      if check_only:
        return group_args(
          group_args_type = 'check only',
          file_name = file_name,
          pdb_info = None,
        )  # exists
      # Read in the file directly
      ch = easy_pickle.load(file_name)
      ph = ch.as_ph()
  if not ph:
    build = mmm.model_building()
    if check_only:
      return group_args(
          group_args_type = 'check only',
          file_name = None,
          pdb_info = pdb_info,)
    model = build.fetch_pdb(pdb_info = pdb_info,
      local_pdb_dir = local_pdb_dir)
    if model:
      model = clean_model(model)
      ph = model.get_hierarchy()
  if check_only:
    return None
  if return_as_model:
    if model:
      return model
    else:
      from mmtbx.model import manager as model_manager
      model = model_manager(
        model_input = ph.as_pdb_input(),
        crystal_symmetry=dummy_crystal_symmetry(),
        log = null_out())
      model.set_info(pdb_info)
    return model
  return ph
def file_name_contains_pdb_from_list(file_name, pdb_id_list):
  for pdb_id in pdb_id_list:
    spl = pdb_id.split("_")
    if len(spl) >1:
      pdb_id = spl[0]
      chain_id = spl[1]
      if file_name.upper().startswith(pdb_id.upper()) and \
          len(file_name.split("_")) == 2:
        return True
      if file_name.upper().startswith("%s_%s" %(pdb_id,chain_id)):
        return True
    else:
      chain_id = None
      if file_name.upper().startswith(pdb_id.upper()):
        return True
  return False
def run_one_indexing(params,db_info,model_mmm, segment_id, log = sys.stdout):
  segment_info = db_info.db.segment_info_dict[segment_id]
  ph = get_ph_for_segment(model_mmm, db_info,
     db_info.db.segment_info_dict, segment_id,
     local_pdb_dir = params.local_pdb_dir)
  if not ph:
    segment_info.neighbor_sites_cart_info_list = []
    return
  asc1 = ph.atom_selection_cache()
  sel = asc1.selection(string = 'name ca and protein')
  ph = ph.select(sel)
  ca_model = model_mmm.model_from_hierarchy(ph, return_as_model = True)
  other_sss_info = get_sss_info(params, mmm = model_mmm,
     ca_model = ca_model,
     log =log)
  other_cluster_info = identify_structure_elements(params,
     other_sss_info,log = log)
  ss_index_info = increment_indexing(params,
    other_cluster_info,
    other_sss_info,
    segment_id = segment_id,
    log = log)
  ph = ca_model.get_hierarchy()
  atoms = ph.atoms()
  atoms.reset_i_seq()  # make sure
  segment_info.ca_sites_cart = atoms.extract_xyz()
  neighbor_sites_cart_info_list = []
  selection_list = get_neighbor_selections(other_cluster_info.clusters)
  clusters = other_cluster_info.clusters
  for c1,c2 in zip(clusters, clusters[1:]):
    asc1 = ph.atom_selection_cache()
    sel1 = asc1.selection(string = "chain %s and resseq %s:%s" %(
     c1.original_chain_id,c1.original_start_res,c1.original_start_res))
    xyz_start = ph.select(sel1).atoms().extract_xyz()
    if xyz_start.size() < 1: continue
    sc_start = xyz_start[0]
    sel2 = asc1.selection(string = "chain %s and resseq %s:%s" %(
     c2.original_chain_id,c2.original_end_res, c2.original_end_res))
    xyz_end = ph.select(sel2).atoms().extract_xyz()
    if xyz_end.size() < 1: continue
    sc_end = xyz_end[0]
    i_start = list(segment_info.ca_sites_cart).index(sc_start)
    i_end = list(segment_info.ca_sites_cart).index(sc_end)
    neighbor_sites_cart_info_list.append(group_args(
      group_args_type = 'indices in ca_sites_cart for neighbors',
      i_start = i_start,
      i_end = i_end,
      original_chain_id = c1.original_chain_id,
      original_start_resno = c1.original_start_res,
      original_end_resno = c2.original_end_res,
      c1_start = c1.original_start_res,
      c1_end = c1.original_end_res,
      c2_start = c2.original_start_res,
      c2_end = c2.original_end_res,
      c1_known_ss = c1.known_ss,
      c2_known_ss = c2.known_ss,
      ))
  segment_info.neighbor_sites_cart_info_list = neighbor_sites_cart_info_list
  ss_index_info.segment_info_dict= {segment_id: segment_info}
  # Set up structures_by_ss_and_index
  for ss1 in ['H','S']:
    ss_index_info.structures_by_ss_and_index[ss1] = {}
    for ss2 in ['H','S']:
      ss_index_info.structures_by_ss_and_index[ss1][ss2] = {}
      dd = ss_index_info.structures_by_ss_and_index[ss1][ss2]
      for index in \
         ss_index_info.other_clusters_by_ss_and_index[segment_id][ss1][ss2]:
        if not index in list(dd.keys()):
          dd[index] = []
        dd[index].append(segment_id)
  return ss_index_info
def find_loops(params,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       log = sys.stdout):
  print("\nFinding loops matching target (based on ssm matches)", file = log)
  if params.use_fragment_search_in_find_loops:
    # Load the fragment db
    from phenix.model_building.fragment_search import get_db_info
    data_dir = None
    database_file = None
    if hasattr(params,'data_dir') and os.path.isdir(params.data_dir):
      data_dir = params.data_dir
    if hasattr(params,'database_file'):
      database_file = params.database_file
    print("Loading fragment database", file = log)
    db_info = get_db_info(
      data_dir = params.data_dir,
      database_file = database_file,
      log = log)
    print("Done loading fragment database", file = log)
    print("Finding matching loops for each supersecondary-structure element",
       file = log)
    # Make an index of db_info.db.segment_info_list matching our
    #   segment_info_dict
    db_info.db.segment_info_dict_by_pdb_text = {}
    # is it unique?
    for segment_info in db_info.db.segment_info_list:
      pdb_info = get_pdb_info_from_text(segment_info.pdb_file_name)
      text = pdb_info.pdb_text
      if not text in list(db_info.db.segment_info_dict_by_pdb_text.keys()):
        db_info.db.segment_info_dict_by_pdb_text[text] = []
      db_info.db.segment_info_dict_by_pdb_text[text].append(segment_info)
  else:
    db_info = None
  run_info_list = []
  for i in range(len(cluster_info.clusters)-1):
    run_info = group_args(
      group_args_type = 'run info',
      run_id = None,
      i = i)
    run_info_list.append(run_info)
  kw_dict={
    'params':  params,
    'ss_index_info':  ss_index_info,
    'sss_info':  sss_info,
    'cluster_info':  cluster_info,
    'db_info': db_info,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = params.nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = run_info_list,
    job_to_run = get_best_fitted_loop,
    log = log)
  fitted_loop_dict = {}
  for run_info in runs_carried_out:
    if run_info and run_info.result and hasattr(run_info.result,'model') and run_info.result.model is not None:
      fitted_loop_dict[run_info.result.cluster_id] = run_info.result
  # get resulting models and save as simple model object
  model_info_list = []
  model_list = []
  keys = list(list(fitted_loop_dict.keys()))
  keys.sort()
  for i in keys:
    print("Fitted loop:",i,fitted_loop_dict[i].cc_score if hasattr(fitted_loop_dict[i],'cc_score') else fitted_loop_dict[i].final_rmsd, file = log)
    model_list.append(fitted_loop_dict[i].model)
    model_info_list.append(model_info(
       hierarchy = fitted_loop_dict[i].model.get_hierarchy()))
  new_model = merge_hierarchies_from_models(models=model_info_list,
        resid_offset = 10,
        renumber = True,
        remove_break_records = True,
        chain_id = 'A')
  new_ph = new_model.hierarchy
  merged_model = sss_info.mmm.model_from_hierarchy(
      new_ph, return_as_model = True)
  f = open('fitted_loops.pdb', 'w')
  print(merged_model.model_as_pdb(), file = f)
  f.close()
  print("Wrote fitted loops to fitted_loops.pdb")
  return group_args(
    group_args_type = 'fitted loops',
    superposed_models = model_list,
   )
def get_matching_ss(segment_info, expected_ss1_start_resno,
      expected_ss1_end_resno, is_c1 = True):
  if not hasattr(segment_info,'neighbor_sites_cart_info_list'):
    return None
  for sc_info in segment_info.neighbor_sites_cart_info_list:
    # does this cover something...and does it match secondary structure?
    if is_c1:
      first_res = sc_info.c1_start
      last_res = sc_info.c1_end
      known_ss = sc_info.c1_known_ss
    else:
      first_res = sc_info.c2_start
      last_res = sc_info.c2_end
      known_ss = sc_info.c2_known_ss
    overlap = min(expected_ss1_end_resno,last_res) - max(
           expected_ss1_start_resno,first_res) + 1
    actual_n = last_res - first_res + 1
    expected_n = expected_ss1_end_resno - expected_ss1_start_resno + 1
    if overlap >= params.min_overlap_fraction * min(actual_n, expected_n):
      return known_ss
  return None
def get_allowed_starts(params,
    segment_info, sc_index, ss1_list, ss2_list, n_residues):
  ''' Get list of places in this segment that start within secondary
    structure type ss1 and end in ss2 with n_residues in the fragment '''
  start_list = []
  if not hasattr(segment_info,'neighbor_sites_cart_info_list'):
    return start_list
  if sc_index >= len(segment_info.neighbor_sites_cart_info_list):
    return start_list
  sc_info = segment_info.neighbor_sites_cart_info_list[sc_index]
  # possible range of start:i = sc_info.c1_start to sc_info.c1_end
  # This leads to ends at i + n_residues - 1
  # These will be ok if ends inside sc_info.c2_start to sc_info.c2_end
  # NOTE:
  # sc_info.i_start is index of c1_start in original
  #    segment_info.ca_sites_cart
  #  so:    (c1_start is residue number for c1)
  # j >= sc_info.c1_start
  # j <= sc_info.c1_end
  # j + n_residues >= sc_info.c2_start
  # j + n_residues <= sc_info.c2_end
  if n_residues < params.min_overlap_fraction * (
     sc_info.c2_end - sc_info.c1_start +1):
    return []
  j_start = max(sc_info.c1_start, sc_info.c2_start - n_residues + 1)
  j_end = min(sc_info.c1_end ,sc_info.c2_end - n_residues + 1)
  if j_end < j_start:
    return []
  offset = sc_info.i_start - sc_info.c1_start
  known_ss = sc_info.c1_known_ss
  if sc_info.c1_known_ss and (not sc_info.c1_known_ss in ss1_list):
    return []
  if sc_info.c2_known_ss and (not sc_info.c2_known_ss in ss2_list):
    return []
  return list(range(j_start, j_end + 1))
def use_fragment_search_in_find_loops(params,
       c1,c2,
       sc,
       best_fitted_loop_info,
       ss_index_info,
       sss_info,
       db_info,
       models_to_keep = None,
       residue_ratio_helix_strand = 3.3/1.54,
       residue_ratio_fraction = 0.5,
       log = sys.stdout):
  c1_nres = c1.original_end_res - c1.original_start_res + 1
  c2_nres = c2.original_end_res - c2.original_start_res + 1
  # How much of difference between helices and strands to use...
  r = 1 + (residue_ratio_helix_strand - 1) * residue_ratio_fraction
  info_list = []
  allowed_start_info_list = []
  for ss1 in c1.known_ss_list:
    for ss2 in c2.known_ss_list:
      target_n_res = sc.size()
      if ss1 == 'H' and \
        c1.original_helix_sheet_designation.original_sites_ss == 'S':
          # building helix instead of strand for c1_nres:
          target_n_res += (c1_nres-1) * (r - 1)
      elif ss1 == 'S' and \
        c1.original_helix_sheet_designation.original_sites_ss == 'H':
          # building strand instead of helix for c1_nres:
          target_n_res -= (c1_nres - 1) * (1/r - 1)
      if ss2 == 'H' and \
        c2.original_helix_sheet_designation.original_sites_ss == 'S':
          # building helix instead of strand for c2_nres:
          target_n_res += (c2_nres - 1 ) * (r - 1)
      elif ss2 == 'S' and \
        c2.original_helix_sheet_designation.original_sites_ss == 'H':
          # building strand instead of helix for c2_nres:
          target_n_res -= (c2_nres - 1 ) * (1 / r - 1)
      target_n_res = int(0.5 + target_n_res)
      # Find all possible target segments here
      other_clusters_by_ss_and_index = \
           ss_index_info.other_clusters_by_ss_and_index
      structures_by_ss_and_index = \
           ss_index_info.structures_by_ss_and_index
      c12_index = c1.composite_indices_dict[c2.cluster_id][0]
      # all structures that have the index of c1,c2
      structure_id_text_list = []
      for structure_id in structures_by_ss_and_index[ss1][ss2].get(
          c12_index,[]):
        ocbsi = other_clusters_by_ss_and_index[structure_id]
        for other_i,other_j in ocbsi[ss1][ss2].get(c12_index,[]):
          if other_j != other_i + 1: continue
          other_segment_info = ss_index_info.segment_info_dict[structure_id]
          # Here is one place c1,c2 match in structure_id.
          for nn in range(target_n_res - params.max_change_in_residues,
             target_n_res + params.max_change_in_residues + 1):
            allowed_starts = get_allowed_starts(params,
             other_segment_info, other_i, [ss1], [ss2], nn)
            if allowed_starts:
              allowed_start_info_list.append(group_args(
                group_args_type = ' allowed starts',
                segment_id = structure_id,
                n_res = nn,
                allowed_starts = allowed_starts))
              pdb_info = get_pdb_info_from_text(
                other_segment_info.pdb_file_name)
              text = pdb_info.pdb_text
              if not text in structure_id_text_list:
                structure_id_text_list.append(text)
  info_list = []
  match_model = sss_info.mmm.model_from_sites_cart(sc, return_model = True)
  from phenix.model_building.fragment_search import get_distort_params,\
    morph_with_segment_matching, superpose_sites_with_indel, get_rmsd_with_indel
  # Choose most promising matches
  best_start_info_list = []
  for start_info in allowed_start_info_list:
    segment_info = ss_index_info.segment_info_dict[start_info.segment_id]
    # Now get same segment from fragment database:
    pdb_info = get_pdb_info_from_text(segment_info.pdb_file_name)
    text = pdb_info.pdb_text
    si = get_si_from_db_info(db_info,text,
      start_info.allowed_starts,start_info.n_res)
    if not si:
      continue
    # Now see if we can fit these sites well
    best_start = None
    best_delta = None
    for i in start_info.allowed_starts:
      j = i - si.start_resno
      other_sc = si.ca_sites_cart[j:j+start_info.n_res]
      if other_sc.size() < 3:
        continue
      delta = abs( (col(sc[0]) - col(sc[-1])).length() -\
          (col(other_sc[0]) - col(other_sc[-1])).length())
      if best_start is None or (delta < best_delta):
        best_start = i
        best_delta = delta
    if best_start is not None:
      start_info.allowed_start = best_start
      start_info.delta = delta
      best_start_info_list.append(start_info)
      segment_info = ss_index_info.segment_info_dict[start_info.segment_id]
  best_start_info_list = sorted(best_start_info_list,
       key = lambda b: b.delta, )
  ok_found = 0
  info_list = []
  for start_info in best_start_info_list[:params.max_loops_to_try]:
    segment_info = ss_index_info.segment_info_dict[start_info.segment_id]
    if start_info.delta > params.max_loop_delta or \
        ok_found >= params.target_loops_to_find:
      break
    pdb_info = get_pdb_info_from_text(segment_info.pdb_file_name)
    text = pdb_info.pdb_text
    si = get_si_from_db_info(db_info,text,
      start_info.allowed_starts,start_info.n_res)
    assert si is not None
    j = start_info.allowed_start - si.start_resno
    other_sc = si.ca_sites_cart[j:j+start_info.n_res]
    # Try to match si (target) and other_sc (matching). They don't have the
    # same number of atoms...
    new_ca_sites_cart  = superpose_sites_with_indel(
      sites_cart = other_sc,
        target_sites = sc,
      sites_to_apply_superposition_to = other_sc)
    # Now get model from full ph
    model = sss_info.mmm.model_from_sites_cart(new_ca_sites_cart,
       return_model = True)
    # Now just use this already-superimposed model as source of
    #   coordinates and match to match_model and distort as necessary to match
    distort_params = get_distort_params()
    distort_params.close_distance = 3  # we are already close
    replacement_info = morph_with_segment_matching(  # Distortion routine
       params = distort_params,
       model = model,
       match_model = match_model,
       log = null_out())
    # Print out the rmsd for this
    # Should be just 1 replacement segment.  Put it in
    if not replacement_info:
      continue
    replacement = replacement_info.replacement_info_list[0]
    segment = replacement.replacement_segment
    range_start = replacement.range_start
    # where optimized segment starts in match_model
    range_end = replacement.range_end
    first_resno = get_first_resno(model.get_hierarchy())
    last_resno = get_last_resno(model.get_hierarchy())
    residue_start = first_resno + range_start
    residue_end = first_resno + range_end
    replacement_fragment = segment.apply_selection_string("resseq %s:%s" %(
        residue_start,residue_end))
    from phenix.model_building.fit_loop import catenate_segments
    if residue_start > first_resno:
      first_part_of_model = model.apply_selection_string("resseq %s:%s" %(
        first_resno,residue_start-1))
      working_model = catenate_segments(first_part_of_model,
          replacement_fragment)
    else:
      working_model = replacement_fragment
    if residue_end < last_resno:
      last_part_of_model = model.apply_selection_string("resseq %s:%s" %(
        residue_end+1,last_resno))
      working_model = catenate_segments(working_model, last_part_of_model)
    model = working_model # all set
    all_sites_cart = model.get_sites_cart()
    ca_sites_cart = model.apply_selection_string('name ca').get_sites_cart()
    # Final superposition
    new_all_sites_cart = superpose_sites_with_indel(
      sites_cart = ca_sites_cart,
        target_sites = sc,
      sites_to_apply_superposition_to = ca_sites_cart)
    model.set_sites_cart(new_all_sites_cart)
    # Update rmsd:
    final_rmsd = get_rmsd_with_indel(
        new_all_sites_cart,sc).rmsd_value
    if final_rmsd <= params.target_loop_rmsd:
      ok_found += 1
    info = group_args(
      group_args_type = 'best_fit',
      segment_id = start_info.segment_id,
      final_rmsd = final_rmsd,
      cluster_id = c1.cluster_id,
      model = model,
     )
    info_list.append(info)
  if info_list:
    info_list = sorted(info_list, key = lambda i: i.final_rmsd)
    return info_list[0]
  else:
    return group_args(
      group_args_type = 'result',
      result = None,
     )
def get_si_from_db_info(db_info,text,allowed_starts,n_res):
  best_si = None
  best_count = 0
  if not text in list(db_info.db.segment_info_dict_by_pdb_text.keys()):
    text = "%s_A" %(text)
  if not text in list(db_info.db.segment_info_dict_by_pdb_text.keys()):
    for key in list(db_info.db.segment_info_dict_by_pdb_text.keys()):
      if key.startswith(text):
        text = key
        break
  if not text in list(db_info.db.segment_info_dict_by_pdb_text.keys()):
    raise Sorry("Missing key: %s" %(text))
  for si in db_info.db.segment_info_dict_by_pdb_text[text]: # with sites_cart
    count = 0
    for i in allowed_starts:
      if i >= si.start_resno and i + n_res - 1 <= si.end_resno:
        count += 1
    if (best_si is None and count > 0 ) or count > best_count:
      best_si = si
      best_count = count
  return best_si
def refine_model(params, mmm, model):
    if mmm.map_manager().is_dummy_map_manager():
      return group_args(
        group_args_type = 'refined model',
        model = model,
        cc = 1,
        cc_score = 1)
    sys_stdout_sav = sys.stdout
    local_mmm = mmm.customized_copy(
       model_dict={'model':model})
    box_mmm = local_mmm.extract_all_maps_around_model()
    starting_cc = box_mmm.map_model_cc()
    build = box_mmm.model_building()
    new_model = build.model()
    if params.refine_cycles:
      new_model = build.refine(
         refine_cycles = params.refine_cycles, return_model =True)
    elif params.quick_refine:
      new_model = build.refine(quick = True, return_model = True)
      sys.stdout =  sys_stdout_sav
    cc = box_mmm.map_model_cc(model = new_model)
    cc_score = cc * new_model.get_hierarchy().overall_counts().n_residues
    mmm.shift_any_model_to_match(new_model)
    return group_args(
      group_args_type = 'refined model',
      model = new_model,
      cc = cc,
      cc_score = cc_score)
def get_best_fitted_loop(run_info = None,
       params = None,
       ss_index_info = None,
       sss_info = None,
       cluster_info = None,
       i = None,
       verbose = None,
       db_info = None,
       log = sys.stdout):
  i = run_info.i
  other_clusters_by_ss_and_index = ss_index_info.other_clusters_by_ss_and_index
  structures_by_ss_and_index = ss_index_info.structures_by_ss_and_index
  n_structures = ss_index_info.n_structures
  mmm = sss_info.mmm
  neighbor_sites_cart_list = cluster_info.neighbor_sites_cart_list
  j = i + 1
  c1 = cluster_info.clusters[i]
  c2 = cluster_info.clusters[j]
  best_fitted_loop_info = group_args(
      group_args_type = 'No fitted loop',
      fitted_loop = None,
      score = None,
      final_rmsd = None,
      cc_score = None,
      model = None,
      cluster_id = i)
  sc = neighbor_sites_cart_list[i]
  if params.use_fragment_search_in_find_loops:
    return use_fragment_search_in_find_loops(params,
       c1, c2,
       sc,
       best_fitted_loop_info,
       ss_index_info,
       sss_info,
       db_info,
       log = log)
  if not c2.cluster_id in c1.composite_indices_dict:
    return best_fitted_loop_info
  fixed_model = mmm.model_from_sites_cart(sc, return_model = True)
  loop_info_list = []
  c12_index = c1.composite_indices_dict[c2.cluster_id][0]
  for ss1 in c1.known_ss_list:  # typically just one
    for ss2 in c2.known_ss_list:  # typically just one
      if not c12_index in list(structures_by_ss_and_index[ss1][ss2].keys()):
        continue
      for structure_id in structures_by_ss_and_index[ss1][ss2][c12_index]:
        ocbsi = other_clusters_by_ss_and_index[structure_id]
        c12_list = ocbsi[ss1][ss2].get(c12_index,[])
        if not c12_list: continue
        # Here c1-c2  matches something in the other structure. Does the loop
        #  match ?  Run from N-term of c1 to C-term of c2.  same in other...
        segment_info = \
         ss_index_info.segment_info_dict[structure_id]
        for other_i,other_j in c12_list:
          if other_j != other_i + 1: continue
          # Make sure secondary structure matches
          other_sc = segment_info.neighbor_sites_cart_list[other_i]
          # Are they similar...  Superpose based on ends and centers
          centers = ss_index_info.cluster_centers_by_segment_id[structure_id]
          ends_1= ss_index_info.cluster_ends_1_by_segment_id[structure_id]
          ends_2= ss_index_info.cluster_ends_2_by_segment_id[structure_id]
          target_xyz = flex.vec3_double()
          moving_xyz = flex.vec3_double()
          for i1,i2 in ([i,other_i],[j,other_j]):
            # NOTE: order of addition here must match axis/center below
            target_xyz.append(cluster_info.cluster_centers[i1])
            target_xyz.append(cluster_info.clusters[i1].split_sites[0])
            target_xyz.append(cluster_info.clusters[i1].split_sites[-1])
            moving_xyz.append(centers[i2])
            moving_xyz.append(ends_1[i2])
            moving_xyz.append(ends_2[i2])
          lsq_fit = superpose.least_squares_fit(
            reference_sites = target_xyz,
            other_sites     = moving_xyz)
          other_sc_fitted = apply_lsq_fit(lsq_fit, other_sc)
          moving_xyz_fitted = apply_lsq_fit(lsq_fit, moving_xyz)
          # Now translate to move tip of loop to match
          axis = normalize(
                 0.5* (col(target_xyz[2]) + col(target_xyz[4])) -
                 0.5* (col(target_xyz[1]) + col(target_xyz[5])) )
          center =  \
                 0.25* (col(target_xyz[2]) + col(target_xyz[5])) + \
                 0.25* (col(target_xyz[1]) + col(target_xyz[4]))
          other_axis = normalize(
                 0.5* (col(moving_xyz_fitted[2]) + col(moving_xyz_fitted[4]) )-
                 0.5* (col(moving_xyz_fitted[1]) + col(moving_xyz_fitted[5])) )
          other_center =  \
                 0.25* (col(moving_xyz_fitted[2]) + col(moving_xyz_fitted[5]) ) +\
                 0.25* (col(moving_xyz_fitted[1]) + col(moving_xyz_fitted[4]))
          max_dist_up_axis = (sc- center).dot(axis).min_max_mean().max
          other_max_dist_up_axis = (other_sc_fitted - other_center).dot(
             other_axis).min_max_mean().max
          shift_along_axis = (
             max_dist_up_axis - other_max_dist_up_axis) * other_axis
          other_sc_fitted += shift_along_axis
          # Now match it up...
          max_dist = params.delta * 0.5
          closest_sites_cart = flex.vec3_double()
          closest_matching_sites_cart = flex.vec3_double()
          for ii in range(other_sc_fitted.size()):
            dist,id1,id2 = target_xyz.min_distance_between_any_pair_with_id(
              other_sc_fitted[ii:ii+1])
            if dist <= max_dist:
              closest_sites_cart.append(target_xyz[id1])
              closest_matching_sites_cart.append(other_sc[ii])
          if closest_sites_cart.size() < \
               params.minimum_matching_fraction * other_sc_fitted.size():
            continue
          lsq_fit = superpose.least_squares_fit(
            reference_sites = closest_sites_cart,
            other_sites     = closest_matching_sites_cart)
          other_sc_fitted = apply_lsq_fit(lsq_fit, other_sc)
          other_matching_sc_fitted = apply_lsq_fit(lsq_fit,
             closest_matching_sites_cart)
          # Get shift field
          shift_field_info = group_args(
            group_args_type = 'shift_field_info',
            centers = other_matching_sc_fitted,
            shifts = (closest_sites_cart - other_matching_sc_fitted),)
          # Apply shift field
          if params.morph:
            working_xyz = apply_shift_field(params,
             shift_field_info,
             other_sc_fitted)
            other_matching_sc_fitted_shifted= apply_shift_field(params,
             shift_field_info,
             other_matching_sc_fitted)
          else:
            working_xyz = other_sc_fitted
            other_matching_sc_fitted_shifted = other_matching_sc_fitted
          diffs = (closest_sites_cart - other_matching_sc_fitted_shifted)
          rms_diff = diffs.rms_length()
          fitted_loop_info = group_args(
            group_args_type = 'fitted loop',
            cluster_id = i,
            other_cluster_id = other_i,
            other_structure_id = structure_id,
            lsq_fit = lsq_fit,
            rmsd = rms_diff,
            n_match = other_matching_sc_fitted.size(),
            shift_field_info = shift_field_info,
            model = None)
          fitted_loop_info.score = fitted_loop_info.n_match**0.5 * (
             params.delta * 0.5 - fitted_loop_info.rmsd)
          loop_info_list.append(fitted_loop_info)
  if not loop_info_list:
    return best_fitted_loop_info
  # Take top N and refine them if map info is available, then return top one
  loop_info_list = sorted(loop_info_list,
    key = lambda li: li.score, reverse = True)
  loop_info_list = loop_info_list[:max(1,params.max_loops_to_refine)]
  new_loop_info_list = []
  sys_stdout_sav = sys.stdout
  for loop_info in loop_info_list:
    # Now put in full info for best fitted loop
    segment_id = loop_info.other_structure_id
    segment_info = ss_index_info.segment_info_dict[segment_id]
    sc_info =  segment_info.neighbor_sites_cart_info_list[
        loop_info.other_cluster_id]
    ph = get_ph_for_segment(mmm,ss_index_info.db_info,
       ss_index_info.segment_id_dict, segment_id,
       local_pdb_dir = params.local_pdb_dir)
    ph = select_residue_range(ph,
      sc_info.original_start_resno,
      sc_info.original_end_resno,
      chain_id = sc_info.original_chain_id)
    sc = ph.atoms().extract_xyz()
    sc_fitted = apply_lsq_fit(loop_info.lsq_fit, sc)
    if params.morph:
      working_xyz = apply_shift_field(params,
           loop_info.shift_field_info,
           sc_fitted)
    else:
      working_xyz = sc_fitted
    ph.atoms().set_xyz(working_xyz)
    moving_model = mmm.model_from_hierarchy(ph, return_as_model = True)
    # Now continue with compare_model_pair
    pair_dict_info = compare_model_pair(params,
      fixed_model, moving_model, log = log)
    # Make sure that worked
    loop_info.cc_score = None
    if not pair_dict_info or not pair_dict_info.lsq_fit or \
        not pair_dict_info.match_selection:
      continue
    loop_info.final_lsq_fit = pair_dict_info.lsq_fit
    loop_info.final_shift_field_info = pair_dict_info.shift_field_info
    # Make sure that loop is close to target. Apply selections
    full_moving_model = moving_model
    if (not moving_model) or (params.require_complete_model and
       not is_complete_model(moving_model)):
      continue
    n_res_original = \
       full_moving_model.get_hierarchy().overall_counts().n_residues
    moving_model = full_moving_model.apply_selection_string(
      pair_dict_info.match_selection)
    if not moving_model:
       continue
    n_res_trimmed = moving_model.get_hierarchy().overall_counts().n_residues
    if n_res_trimmed < n_res_original *  params.minimum_matching_fraction:
       continue
    matching_xyz = moving_model.get_sites_cart()
    updated_xyz = apply_lsq_fit(pair_dict_info.lsq_fit, matching_xyz)
    if params.morph:
      shifted_xyz = apply_shift_field(params, pair_dict_info.shift_field_info,
      updated_xyz)
    else:
      shifted_xyz = updated_xyz
    moving_model.set_sites_cart(shifted_xyz)
    refined_model_info = refine_model(params, sss_info.mmm, moving_model)
    loop_info.model = refined_model_info.model
    loop_info.cc = refined_model_info.cc
    loop_info.cc_score = refined_model_info.cc_score
    new_loop_info_list.append(loop_info)
  if not new_loop_info_list:
    return best_fitted_loop_info
  loop_info_list = sorted(new_loop_info_list,
     key = lambda li: li.cc_score, reverse = True)
  best_fitted_loop_info = loop_info_list[0]
  return best_fitted_loop_info
def is_complete_model(model):
    # are there missing residues in segment
    ph = model.get_hierarchy()
    n_res = ph.overall_counts().n_residues
    first_res = get_first_resno(ph)
    last_res = get_last_resno(ph)
    if n_res != last_res - first_res + 1:
       return False
    else:
       return True
def get_triple_match_info(params,
       triple_match_info,
       cluster_info,
       sss_info,
       ss_index_info,
       full_range_only = None,
       previous_groups= None,
       tries_tested = None,
       cluster_window = None,
       log = sys.stdout):
  group_tries = get_group_tries(params, sss_info,
     adjacent_only = params.find_loops,
     full_range_only = full_range_only,
     previous_groups = previous_groups,
     cluster_window = cluster_window)
  if not tries_tested:
    tries_tested = []
  if not triple_match_info:
    triple_match_info = group_args(
      group_args_type = 'triple match info',
      good_triples = None,
      triple_list = [],
      previous_groups = group_tries,
      tries_tested = tries_tested,
     )
  else:
    triple_match_info.previous_groups += group_tries
  run_info_list = []
  for i in range(len(group_tries)):
      run_info_list.append(group_args(
        run_id = i,
        group_try = group_tries[i],
        ))
  kw_dict={
      'params':  params,
      'tries_tested':  deepcopy(triple_match_info.tries_tested),
      'cluster_info':  cluster_info,
      'sss_info':  sss_info,
      'ss_index_info':  ss_index_info,
     }
  runs_carried_out = run_jobs_with_large_fixed_objects(
      nproc = params.nproc,
      verbose = params.verbose,
      kw_dict = kw_dict,
      run_info_list = run_info_list,
      job_to_run = run_one_group_try,
      log = log)
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.triple_list:
      for t in run_info.result.triple_list:
        if t and hasattr(t,'pairs'):
          triple_match_info.triple_list.append(t)
          segment_info = ss_index_info.segment_info_dict[t.db_id]
      for tt in run_info.result.tries_tested:
        if not tt in triple_match_info.tries_tested:
          triple_match_info.tries_tested.append(tt)
  # Now each triple_info has number of matches
  triple_match_info.triple_list = sorted(triple_match_info.triple_list,
     key = lambda t: len(t.pairs) if hasattr(t,'pairs') else 0, reverse = True)
  triple_match_info.good_triples = count_good_triples(
      params, triple_match_info.triple_list)
  return triple_match_info
def count_good_triples(params, triple_list):
  good_triples = 0
  for t in triple_list:
    if len(t.pairs) >= params.good_triple_pairs:
      good_triples += 1
  return good_triples
def get_brute_force_match_info(params,
       fixed_model, # duplicates model in sss_info
       moving_model, # duplicates model in other_cluster_info
       mmm_for_scoring,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       log = sys.stdout):
  # Try quick and then slower
  for include_reverse, offset_first, offset_second in zip(
       [False, False, False, False, True],
       [False, True, False, True, True],
       [False, False, True, True, True],):
    result = get_brute_force_match_info_forward_reverse(params,
       fixed_model, # duplicates model in sss_info
       moving_model, # duplicates model in other_cluster_info
       mmm_for_scoring,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       include_reverse,
       offset_first,
       offset_second,
       log = log)
    if result and result.superposed_full_models and \
     result.superposed_full_models[0].info().cc >= params.ok_brute_force_cc:
       print("OK CC found: %.2f" %(result.superposed_full_models[0].info().cc),
        file = log)
       return result
  return result # just return what we have
def get_brute_force_match_info_forward_reverse(params,
       fixed_model, # duplicates model in sss_info
       moving_model, # duplicates model in other_cluster_info
       mmm_for_scoring,
       cluster_info,
       other_cluster_info,
       sss_info,
       ss_index_info,
       include_reverse = False,
       offset_first = False,
       offset_second= False,
       include_without_offset = 10,
       log = sys.stdout):
  print("Getting match info with brute force method %s %s %s" %(
     "including reverse" if include_reverse else "",
     "offsetting first" if offset_first else "",
     "offsetting second" if offset_second else "",
      ), file = log)
  run_info_list = []
  for i in range(len(cluster_info.clusters)):
    end_i = len(cluster_info.clusters)
    if not offset_first:
      end_i = min(i+include_without_offset, end_i)
    for j in range(i+1, end_i):
      run_info = group_args(group_args_type = 'dummy_run_info',
          i = i,
          j = j,
         )
      run_info_list.append(run_info)
  nproc = params.nproc
  end_number = -1
  nproc = params.nproc
  n_tot = len(run_info_list)
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
  kw_dict={
    'params':  params,
    'run_info_list': run_info_list,
    'cluster_info':  cluster_info,
    'other_cluster_info': other_cluster_info,
    'fixed_model': fixed_model.apply_selection_string('name ca or name p'),
    'moving_model': moving_model.apply_selection_string('name ca or name p'),
    'include_reverse': include_reverse,
    'include_without_offset': include_without_offset,
    'offset_second': offset_second,
   }
  print("Running %s jobs to evaluate superpositions" %(
     len(runs_to_carry_out)), file = log)
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = params.nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_get_match_to_cluster_pair,
    log = log)
  match_list = []
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.match_list:
      match_list += run_info.result.match_list
  if not match_list:
    print("No superpositions to check...", file = log)
    return # Nothing to do
  print("Total of %s possible superpositions to check..." %(len(match_list)),
     file = log)
  match_list = sorted(match_list, key = lambda m: m.n, reverse = True)
  iteration_list = []
  match_list = match_list[:params.max_brute_force]
  for i in range(len(match_list)):
    match_list[i].model_id = i
    iteration_list.append(group_args(
      group_args_type = 'iteration',
      i = i))
  # Run this in parallel too
  from libtbx.easy_mp import simple_parallel
  result_list = simple_parallel(
    function = get_map_model_cc,        # function to run
    iteration_list = iteration_list,  # list of N values or objects that vary
    nproc = params.nproc,        # number of processors
    mmm_for_scoring = mmm_for_scoring,
    match_list = match_list,
    moving_model = moving_model,
    log = log,            # pass log stream if used
     )
  superposed_full_models = []
  for r in result_list:
    match_list[r.model_id].superposed_model = r.superposed_model
    r.superposed_model.info().model_id = r.model_id
    r.superposed_model.info().cc = r.cc
    r.superposed_model.info().rmsd = r.rmsd
    r.superposed_model.info().n_residues = r.n
    superposed_full_models.append(r.superposed_model)
  if params.score_superpose_brute_force_with_cc:
    superposed_full_models = sorted(superposed_full_models,
      key = lambda m: m.info().cc, reverse = True)
  else:
    superposed_full_models = sorted(superposed_full_models,
      key = lambda m: m.info().rmsd)
  print("Total of %s superposed models" %(
      len(superposed_full_models)), file = log)
  for i, m in enumerate(superposed_full_models):
    if m.info().cc:
      print("Model %s N=%s  CC=%.2f rmsd = %.2f" %(
        i, m.info().n_residues,
       m.info().cc, m.info().rmsd), file = log)
    else:
      print("Model %s N=%s  rmsd = %.2f" %(
        i, m.info().n_residues,
       m.info().rmsd), file = log)
  return group_args(group_args_type = 'superposed models',
    superposed_full_models = superposed_full_models,
    superposed_models = superposed_full_models)
def get_map_model_cc(run_info,
    mmm_for_scoring = None,
    match_list = None,
    moving_model = None,
    log = sys.stdout):
  i = run_info.i
  match_info = match_list[i]
  assert match_info.model_id == i
  from phenix.model_building.morph_info import apply_lsq_fit_to_model
  superposed_model = apply_lsq_fit_to_model(match_info.lsq_fit, moving_model)
  result = group_args(
    group_args_type = 'map_model_cc',
    cc = mmm_for_scoring.map_model_cc(model = superposed_model) if mmm_for_scoring else 0,
    model_id = i,
    n = match_info.n,
    rmsd = match_info.rmsd,
    superposed_model = superposed_model,
   )
  return result
def group_of_get_match_to_cluster_pair(run_info,
        params,
        run_info_list,
        cluster_info = None, other_cluster_info = None,
        fixed_model = None, moving_model = None,
        include_reverse = None,
        include_without_offset= None,
        offset_second = None,
        log = sys.stdout):
    match_list = []
    for key in range(run_info.start_number, run_info.end_number + 1):
      ri = run_info_list[key]
      i = ri.i
      j = ri.j
      c1 = cluster_info.clusters[i]
      c2 = cluster_info.clusters[j]
      new_match_list = get_match_to_cluster_pair(params,
          c1, c2, other_cluster_info = other_cluster_info,
          fixed_model = fixed_model, moving_model = moving_model,
          include_reverse = include_reverse,
          include_without_offset = include_without_offset,
          offset_second = offset_second,
          log = log)
      if new_match_list:
        match_list += new_match_list
    return group_args(
      group_args_type = 'match_list',
      match_list = match_list,
    )
def group_of_apply_transformation(
     run_info,
     params,
     run_info_list,
     ca_model,
     cluster_info,
     sss_info,
     ss_index_info,
     mmm_for_scoring = None,
     log = sys.stdout):
  new_models = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    ri = run_info_list[i]
    info = apply_transformation(
      ri,
      params,
      None,
      ca_model,
      ss_index_info,
      sss_info,
      verbose = False,
      log = log)
    if not info or not info.triple:
      continue
    new_model = info.triple.superposed_moving_model
    new_model.info().rmsd = ri.info_for_apply.rmsd
    if mmm_for_scoring:
      new_model.info().cc = mmm_for_scoring.map_model_cc(model= new_model)
    else:
      new_model.info().cc = None
    new_model.info().n_residues = \
       new_model.get_hierarchy().overall_counts().n_residues
    new_models.append(new_model)
  return group_args(
     group_args_type = 'new models',
     superposed_full_models = new_models,
    )
def get_match_to_cluster_pair(params,
        c1, c2, other_cluster_info = None,
        fixed_model = None, moving_model = None,
        number_to_return = 1,
        include_reverse = True,
        include_without_offset= None,
        offset_second = None,
        log = sys.stdout):
  # Get best match to this cluster pair
  match_list = []
  for ii in range(len(other_cluster_info.clusters)):
    o1 = other_cluster_info.clusters[ii]
    if o1.known_ss != c1.known_ss: continue
    end_jj = len(other_cluster_info.clusters)
    if not offset_second:
      end_jj = min(ii+include_without_offset, end_jj)
    for jj in range(ii+1,end_jj):
      o2 = other_cluster_info.clusters[jj]
      if o2.known_ss != c2.known_ss: continue
      new_match_list = get_match_info_for_pair(params,
        c1, c2, o1, o2,
        fixed_model, moving_model,
        include_reverse = include_reverse,
        offset_first = True,
        offset_second = True,
        log = log)
      if new_match_list:
        match_list += new_match_list
  if not match_list:
    return # nothing found
  match_list = sorted(match_list, key = lambda m: m.n, reverse = True)
  return match_list[:number_to_return]
def get_match_info_for_pair(params, c1,c2, o1, o2,
    fixed_model, moving_model,
    include_reverse = True,
        offset_first = None,
        offset_second = None,
     log = sys.stdout):
  # Match c1 to o1 and c2 to o2
  n1 = min(o1.original_ca_sites_cart.size(),
        c1.original_ca_sites_cart.size())
  c1_list = get_subfragments(c1.original_ca_sites_cart, n1,
       include_reverse = include_reverse,
       allow_offset = offset_first)
  o1_list = get_subfragments(o1.original_ca_sites_cart, n1,
       allow_offset = offset_second)
  match_list = []
  for c1_sc in c1_list:
    for o1_sc in o1_list:
      assert c1_sc.size() == o1_sc.size()
      n2 = min(o2.original_ca_sites_cart.size(),
          c2.original_ca_sites_cart.size())
      c2_list = get_subfragments(c2.original_ca_sites_cart, n2,
         include_reverse = include_reverse,
         allow_offset = offset_first)
      o2_list = get_subfragments(o2.original_ca_sites_cart, n2,
         allow_offset = offset_second)
      for c2_sc in c2_list:
        for o2_sc in o2_list:
          match = get_match_info_for_ca_pair(params,
             c1_sc, c2_sc, o1_sc, o2_sc, fixed_model, moving_model,
       log = log)
          if match:
            match_list.append(match)
  return match_list
def get_match_info_for_ca_pair(params,
     c1_sc, c2_sc, o1_sc, o2_sc, fixed_model, moving_model, log = sys.stdout):
  # Is superposing c1 on o1 and c2 on o2 plausible:
  assert not params.include_reverse
  n1 = c1_sc.size() - 1
  n2 = c2_sc.size() - 1
  assert c2_sc.size() == o2_sc.size()
  assert c1_sc.size() == o1_sc.size()
  if n1 < 1 or n2 < 1:
    return None
  for i in [0,n1]:
    for j in [0,n2]:
      aa = (col(c1_sc[i]) - col(c2_sc[j])).length()
      bb = (col(o1_sc[i]) - col(o2_sc[j])).length()
      if abs(aa - bb) > params.match_distance_high:
        return None # off
  # Plausible match here.  Superpose them all
  c_sc = c1_sc.deep_copy()
  c_sc.extend(c2_sc)
  o_sc = o1_sc.deep_copy()
  o_sc.extend(o2_sc)
  if c_sc.size() < 3 or o_sc.size() != c_sc.size():
    return None # nothing to do
  lsq_fit = superpose.least_squares_fit(
      reference_sites = c_sc,
      other_sites     = o_sc)
  m_sc = apply_lsq_fit(lsq_fit, o_sc)
  rmsd = (m_sc - c_sc).rms_length()
  if rmsd > params.match_distance_high:
    return None # not too good
  # Now we have a match that might be useful. Try applying to everything in
  #  moving structure
  fixed_sc = fixed_model.get_sites_cart()
  moving_sc = moving_model.get_sites_cart()
  superposed_sc = apply_lsq_fit(lsq_fit, moving_sc)
  # Count how many are close...
  from phenix.model_building.fragment_search import get_rmsd_with_indel
  info = get_rmsd_with_indel(fixed_sc, superposed_sc)
  rmsd = info.rmsd_value
  n = info.close_n
  return group_args(
    group_args_type = 'fit of fragments',
    lsq_fit = lsq_fit,
    rmsd = rmsd,
    n = n)
def get_subfragments(sc, n, include_reverse = None,
    allow_offset = True, include_without_offset = 10):
  assert n <= sc.size()
  frag_list = []
  ranges = list(range(sc.size() - n + 1))
  if not allow_offset:
    ranges = ranges[:include_without_offset]
  for i in ranges:
    frag_list.append(sc[i:i+n])
    if include_reverse:
      frag_list.append(reverse_vec3_double(sc[i:i+n]))
  return frag_list
def find_ssm_matching_segments(params,
       cluster_info,
       sss_info,
       ss_index_info,
       log = sys.stdout):
  print("\nCarrying out ssm matching on target structure...", file = log)
  # first group and if nothing found, run everything
  triple_match_info = None
  previous_groups = []
  min_window = 3
  max_window = params.cluster_window if params.cluster_window else \
      len(cluster_info.clusters)
  for cluster_window in range(min_window, max_window + 2):
    if cluster_window == max_window + 1:
      full_range_only = True
      print("Looking for matches with full range", file = log)
    else:
      full_range_only = False
      print("Looking for matches with cluster window = %s" %(cluster_window),
        file = log)
    triple_match_info = get_triple_match_info(params,
       triple_match_info,
       cluster_info,
       sss_info,
       ss_index_info,
       full_range_only = full_range_only,
       cluster_window = cluster_window,
       previous_groups = previous_groups,
       log = log)
    print("Current good triples: %s (total of %s)"  %(
       triple_match_info.good_triples, len(triple_match_info.triple_list)),
      file = log)
    if triple_match_info.good_triples >= params.minimum_good_triples:
      break
  libtbx.call_back(message = 'Initial matching structures: %s' %(
      len(triple_match_info.triple_list)),
       data = None)
  # See if we can improve by grouping within a single db_id
  # Take top  params.minimum_hits_to_keep
  print("\nGrouping matches within chains...", file = log)
  triples_by_segment_id = {}
  n_kept = 0
  libtbx.call_back(message = 'Grouping triples by PDB ID...', data = None)
  for t in triple_match_info.triple_list:
    if not hasattr(t,'pairs'):
      continue
    if not t.db_id in list(triples_by_segment_id.keys()):
      if n_kept >= params.minimum_hits_to_keep:
        continue
      else:
        triples_by_segment_id[t.db_id] = []
      n_kept += 1
    triples_by_segment_id[t.db_id].append(t)
  print("Total unique triples kept: %s" %(
     len(list(list(triples_by_segment_id.keys())))),
     file = log)
  libtbx.call_back(message = 'Unique triples: %s' %(
     len(list(list(triples_by_segment_id.keys())))), data = None)
  result = group_args(
    group_arg_type = 'matching segments',
    superposed_models = [],
    superposed_full_models = [],
    )
  if len(list(list(triples_by_segment_id.keys()))) < 1:
    if params.return_best_triples:
      return group_args(
        group_args_type = 'best_triples with db info',
        best_triples = [],
        ss_index_info = ss_index_info)
    else:
      return result
  # See if we can combine matches
      #segment_id = keys[i],
  libtbx.call_back(message = 'Combining matches', data = None)
  print("Combining matches...", file = log)
  keys = list(list(triples_by_segment_id.keys()))
  nproc = params.nproc
  end_number = -1
  nproc = params.nproc
  n_tot = len(keys)
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
  kw_dict={
    'params':  params,
    'triples_by_segment_id': triples_by_segment_id,
    'keys': keys,
    'cluster_info':  cluster_info,
    'sss_info':  sss_info,
    'ss_index_info':  ss_index_info,
   }
  print("Running %s jobs to merge triples" %(
     len(runs_to_carry_out)), file = log)
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = params.nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_get_segment_match,
    log = log)
  new_triples = []
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.triple_list:
      new_triples += run_info.result.triple_list
  # sort and remove dups
  print("Sorting %s triples" %(len(new_triples)), file = log)
  triple_list = sorted(new_triples,
     key = lambda t: len(t.pairs), reverse = True)
  best_triples = remove_duplicate_triples(triple_list)
  best_triples = best_triples[:params.minimum_hits_to_keep]
  libtbx.call_back(message = 'Sorted matches: %s' %(len(best_triples)),
      data = None)
  print("Done with %s sorted triples" %(len(best_triples)), file = log)
  if params.return_best_triples:
    return group_args(
      group_args_type = 'best_triples with db info',
      best_triples = best_triples,
      ss_index_info = ss_index_info)
  if params.search_sequence_neighbors:
    libtbx.call_back(message = 'Getting sequence neighbors of top matches:',
      data = None)
    new_best_triples = search_sequence_neighbors(params,
      best_triples,
      cluster_info,
       sss_info,
       ss_index_info,
       log = log)
    if new_best_triples:
      libtbx.call_back(
         message = 'Found %s sequence neighbors of top matches:' %(
       len(new_best_triples)), data = None)
      print("Total of %s best new triples" %(len(new_best_triples)), file = log)
      best_triples += new_best_triples
      triple_list = sorted(best_triples,
          key = lambda t: len(t.pairs), reverse = True)
      best_triples = remove_duplicate_triples(triple_list)
      best_triples = best_triples[:params.minimum_hits_to_keep]
  good_triples = count_good_triples(params, best_triples)
  if good_triples >= params.minimum_good_triples:  #
    best_triples = best_triples[:params.minimum_good_triples]
    print("Selecting the top %s triples " %(len(best_triples)), file = log)
  # Get all the unique models that we may need
  fetch_model_info_list = []
  read_model_info_list = []
  data_dir = None
  print("\nReading and fetching necessary models", file = log)
  for t in best_triples:
    if hasattr(t,'ss_index_info'):
     sii = t.ss_index_info
    else:
     sii = ss_index_info
    pdb_file_name = sii.segment_info_dict[t.db_id].pdb_file_name
    pdb_id_info = get_pdb_info_from_text(pdb_file_name)
    text = pdb_id_info.pdb_text.upper()
    print("TRIPLE RMSD %.2f ID: %s (%s) Matching: %s " %(
         t.rmsd, t.db_id,
         text,
         len(t.pairs),
          ), file = log)
    segment_info = sii.segment_info_dict[t.db_id]
    pdb_file_name = segment_info.pdb_file_name
    if hasattr(segment_info,'full_model') and segment_info.full_model:
      pass # already there
    else:
      mmm = sss_info.mmm
      info = get_ph_for_segment(mmm, sii.db_info,
        sii.segment_info_dict, t.db_id, check_only = True,
        local_pdb_dir = params.local_pdb_dir)
      already_done = False
      if info and info.file_name and not already_done:
        for mi in read_model_info_list:
          if mi.file_name == info.file_name:
            mi.triple_list.append(t)
            assert sii.db_info.data_dir == data_dir
            already_done = True
            break
      if info and info.pdb_info and not already_done:
        for mi in fetch_model_info_list:
          if mi.pdb_info and mi.pdb_info.pdb_text == info.pdb_info.pdb_text:
            mi.triple_list.append(t)
            already_done = True
            break
      if info.file_name and not already_done:
          already_done = True
          if data_dir is not None:
            assert sii.db_info.data_dir == data_dir
          data_dir = sii.db_info.data_dir
          read_model_info_list.append(
           group_args(group_args_type = 'model info',
           triple_list = [t],
           pdb_info = None,
           file_name = info.file_name,
           data_dir = data_dir))
      elif info.pdb_info and info.pdb_info.pdb_text and not already_done:
          already_done = True
          fetch_model_info_list.append(
           group_args(group_args_type = 'model info',
           triple_list = [t],
           pdb_info = info.pdb_info,
           file_name = None))
  if read_model_info_list:
    libtbx.call_back(message = 'Reading %s models from local database:' %(
       len(read_model_info_list)), data = None)
    # Get all the models in parallel
    model_list = read_models_in_parallel(params, read_model_info_list,
     log = log)
    print("Read %s models" %(len(model_list)), file = log)
    for info in read_model_info_list:
      t = info.triple_list[0]
      if hasattr(t,'ss_index_info'):
        sii = t.ss_index_info
      else:
        sii = ss_index_info
      pdb_file_name = info.file_name
      model = None
      for m in model_list:
        if m.info().file_name == pdb_file_name:
          model = m
          break
      if not model: continue
      for t1 in info.triple_list:
        segment_info = sii.segment_info_dict[t1.db_id]
        segment_info.full_model = model
  if fetch_model_info_list:
    if params.max_models_to_fetch:
      fetch_model_info_list = fetch_model_info_list[:params.max_models_to_fetch]
    libtbx.call_back(message = 'Fetching %s models from PDB:' %(
       len(fetch_model_info_list)), data = None)
    print("Fetching %s models" %(len(fetch_model_info_list)), file = log)
    pdb_info_list = []
    pdb_id_list = []
    for info in fetch_model_info_list:
      pdb_info_list.append(info.pdb_info)
      pdb_id_list.append(info.pdb_info.pdb_id)
    # Get them all
    build = sss_info.mmm.model_building()
    build.set_defaults(nproc = params.nproc)
    model_list = build.fetch_pdb(pdb_info_list = pdb_info_list,
      local_pdb_dir = params.local_pdb_dir)
    if not model_list:
      model_list = []
    for model in model_list:
      for info in fetch_model_info_list:
        if model.info().pdb_text == info.pdb_info.pdb_text:
          for t1 in info.triple_list:
            if hasattr(t1,'ss_index_info'):
              sii = t1.ss_index_info
            else:
              sii = ss_index_info
            segment_info = sii.segment_info_dict[t1.db_id]
            segment_info.full_model = model
  libtbx.call_back(message = 'Combining top %s models:' %(
       len(best_triples)), data = None)
  print("\nCombining top %s matches" %(
      len(best_triples)), file = log)
  # Now superpose everything and see what we have
  mmm = sss_info.mmm
  ca_model = mmm.model().apply_selection_string("name ca " )
  # See if we can combine matches
  run_info_list = []
  for i in range(len(best_triples)):
    run_info = group_args(
      group_args_type = 'run info',
      run_id = None,
      triple_id = i)
    run_info_list.append(run_info)
  kw_dict={
    'params':  params,
    'best_triples': best_triples,
    'ca_model':  ca_model,
    'ss_index_info':  ss_index_info,
    'sss_info':  sss_info,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = params.nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = run_info_list,
    job_to_run = apply_transformation,
    log = log)
  final_triples = []
  for run_info in runs_carried_out:
    if run_info.result and run_info.result.triple:
      final_triples.append(run_info.result.triple)
  final_triples = sorted(final_triples,
     key = lambda t: t.score, reverse = True)
  libtbx.call_back(message = 'Ready with %s models:' %(len(final_triples)),
    data = None)
  print("Final matches : %s" %(len(final_triples)), file = log)
  for t in final_triples:
    if hasattr(t,'ss_index_info'):
     sii = t.ss_index_info
    else:
     sii = ss_index_info
    pdb_file_name = sii.segment_info_dict[t.db_id].pdb_file_name
    pdb_id_info = get_pdb_info_from_text(pdb_file_name)
    text = pdb_id_info.pdb_text.upper()
    print("MATCH " +\
     "%s (%s) Matching: %s (%.2f A N=%s, morphing rmsd: %.2f A)" %(
         t.db_id,
         text,
         len(t.pairs),
         t.matching_site_info.rmsd,
         t.matching_site_info.matching_sites.size(),
         getattr(t,'rms_morphing',0)),
         file = log)
  if final_triples:
    if params.write_files:
      print("Writing out superposed models", file = log)
      if (not params.trim) and (params.morph):
        print("Untrimmmed morphed superposed models will be written",
           file = log)
      elif params.morph:
        print("Morphed superposed models will be written", file = log)
    for i in range(min(params.max_final_triples,len(final_triples))):
      t = final_triples[i]
      pdb_text = get_pdb_text_from_triple(t, ss_index_info)
      add_pdb_text_if_needed(final_triples[i].superposed_moving_model,pdb_text)
      add_pdb_text_if_needed(final_triples[i].superposed_full_moving_model,
         pdb_text)
      result.superposed_models.append(
       final_triples[i].superposed_moving_model)
      result.superposed_full_models.append(
       final_triples[i].superposed_full_moving_model)
      if not params.write_files: continue
      file_name = 'superposed_%s_%s.pdb' %(pdb_text,i)
      f = open(file_name, 'w')
      print(final_triples[i].superposed_moving_model.model_as_pdb(),
       file = f)
      f.close()
      print("Wrote model (matching part only) %s to %s" %(i,file_name),
         file = log)
      file_name = 'superposed_full_%s_%s.pdb' %(pdb_text,i)
      f = open(file_name, 'w')
      print(final_triples[i].superposed_full_moving_model.model_as_pdb(),
       file = f)
      print("Wrote full model %s to %s" %(i,file_name), file = log)
      f.close()
  return result
def add_pdb_text_if_needed(model,pdb_text):
  if not hasattr(model.info(),'pdb_text'):
    model.info().pdb_text = pdb_text
def get_segment_info_from_triple(t,ss_index_info):
  if hasattr(t,'ss_index_info'):
    sii = t.ss_index_info
  else:
    sii = ss_index_info
  return sii.segment_info_dict[t.db_id]
def get_pdb_text_from_triple(t, ss_index_info):
  segment_info = get_segment_info_from_triple(t,ss_index_info)
  pdb_info = get_pdb_info_from_text(segment_info.pdb_file_name)
  return pdb_info.pdb_text
def print_triples(best_triples, ss_index_info, text = 'TRIPLE',
     log = sys.stdout):
  for t in best_triples:
    if hasattr(t,'ss_index_info'):
     sii = t.ss_index_info
    else:
     sii = ss_index_info
    print("%s RMSD: %.2f ID: %s (%s) Matching: %s " %(text,
         t.rmsd, t.db_id,
         sii.segment_info_dict[t.db_id].pdb_file_name,
         len(t.pairs),
          ), file = log)
    sc = sii.segment_info_dict[t.db_id].ca_sites_cart
    if sc:
     print("SC:", sc.size(),sc[0])
def read_models_in_parallel(params, read_model_info_list, log = sys.stdout):
  end_number = -1
  nproc = params.nproc
  n_tot = len(read_model_info_list)
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
    'read_model_info_list':read_model_info_list,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = params.verbose,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_read_one_model,
    log = log)
  model_list = []
  for run_info in runs_carried_out:
    if run_info.result.model_list:
      model_list += run_info.result.model_list
  return model_list
def group_of_read_one_model(run_info, read_model_info_list, log = sys.stdout):
  model_list = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    model = read_one_model(read_model_info_list[i])
    if model:
      model_list.append(model)
  return group_args(
    group_args_type = 'group of read one model',
    model_list = model_list)
def read_one_model(info):
  file_name = os.path.join(info.data_dir,"%s.gz" %(
     info.file_name.replace(".gz","")))
  if not os.path.isfile(file_name):
       return
  # Read in the file directly
  ph = easy_pickle.load(file_name).as_ph()
  from mmtbx.model import manager as model_manager
  model = model_manager(
        model_input = ph.as_pdb_input(),
        crystal_symmetry=dummy_crystal_symmetry(),
        log = null_out())
  model.set_info(group_args(
   group_args_type = 'info on read_one_model',
   file_name = file_name))
  return model
def search_sequence_neighbors(params,
      best_triples,
      cluster_info,
       sss_info,
       ss_index_info,
       log = sys.stdout):
  local_params = deepcopy(params)
  local_params.return_best_triples = True
  import libtbx.load_env
  segment_lib_dir = libtbx.env.find_in_repositories(
       relative_path=os.path.join("chem_data","segment_lib"),
       test=os.path.isdir)
  local_params = set_pdb_100(local_params)
  local_params.is_sequence_neighbors_search = True
  if not local_params.pdb_exclude_list: local_params.pdb_exclude_list = []
  for segment_info in ss_index_info.segment_info_dict.values():
    id_as_string = get_id_from_pdb_file_name(segment_info.pdb_file_name)
    local_params.pdb_exclude_list.append(id_as_string) # don't repeat
  print("Total to exclude:",len(local_params.pdb_exclude_list), file = log)
  if local_params.pdb_include_list:
   for x in local_params.pdb_exclude_list:
      if x in local_params.pdb_include_list:
         local_params.pdb_include_list.remove(x)
      y = x.split("_")[0]
      if y in local_params.pdb_include_list:
         local_params.pdb_include_list.remove(y)
   if not local_params.pdb_include_list: # nothing left
     print("Nothing left to include", file = log)
     return None
  if local_params.pdb_include_list:
    print("Total in include_list:",
       len(local_params.pdb_include_list), file = log)
  print("\nFinding sequence neighbors of %s top hits" %(
     len(best_triples)), file = log)
  local_mmm = sss_info.mmm.customized_copy(model_dict={})
  local_mmm.set_log(log)
  local_mmm.set_verbose(params.verbose)
  build = local_mmm.model_building()
  build.set_defaults(nproc = local_params.nproc)
  matching_pdb_list = []
  moving_model_list = []
  for t in best_triples:
    segment_info = ss_index_info.segment_info_dict[t.db_id]
    id_as_string = get_id_from_pdb_file_name(segment_info.pdb_file_name)
    if id_as_string in matching_pdb_list:
      continue
    # List of neighbors to run
    matching_pdb_list.append(id_as_string)
    if len(matching_pdb_list) > local_params.max_triples_for_neighbors:
      break # done
    if hasattr(segment_info,'full_model'): # we read in the matching model
      moving_model = segment_info.full_model
      matching_ph = moving_model.get_hierarchy()
      print("Getting structures similar to %s" %(id_as_string), file = log)
    else:
      matching_ph = get_ph_for_segment(local_mmm,ss_index_info.db_info,
        ss_index_info.segment_info_dict, t.db_id,
        local_pdb_dir = params.local_pdb_dir)
      if not matching_ph:
        continue # nothing to do
      moving_model = get_model_from_ph(local_mmm,matching_ph)
      print("Getting structures similar to %s with %s residues" %(
        segment_info.pdb_file_name,matching_ph.overall_counts().n_residues),
        file = log)
    moving_model_list.append(moving_model)
  pdb_info_list = build.structure_search(model_list = moving_model_list,
    number_of_models_per_input_model = local_params.search_sequence_neighbors,
    sequence_only = True,
    return_pdb_info_list = True)
  print("\nTotal of %s sequence neighbor models to be tested" %(
    len(pdb_info_list)), file = log)
  if not pdb_info_list:
    print("No sequence neighbors to examine...", file = log)
    return []
  local_params.pdb_include_list = []
  for info in pdb_info_list:
    id_string = "%s_%s" %(info.pdb_id,info.chain_id)
    local_params.pdb_include_list.append(id_string)
  new_triples_list = []
  if not local_params.pdb_include_list:
    print("No neighbors to include...", file = log)
    return # nothing to do
  run_result = run(local_params,
    mmm = local_mmm,
    sss_info = sss_info,
    cluster_info = cluster_info,
    log = log)
  new_triples_list = []
  if run_result:
    for t in run_result.best_triples:
      new_triples_list.append(t)
      ss_index_info = run_result.ss_index_info
      segment_info = ss_index_info.segment_info_dict[t.db_id]
      t.ss_index_info = get_dummy_ss_index_info(
        segment_info.ca_sites_cart, segment_info.pdb_file_name,
        db_id = t.db_id) # need to download
  print("Total of %s new triples" %(len(new_triples_list)), file = log)
  return new_triples_list
def get_dummy_ss_index_info(ca_sites_cart, model_file_name,
      db_id = 0):
    segment_info = group_args(
      group_args_type = 'dummy segment info II',
      ca_sites_cart = ca_sites_cart,
      full_model = None,
      pdb_file_name = model_file_name)
    segment_info.neighbor_sites_cart_list = None
    ss_index_info = group_args(
      segment_info_dict = {db_id:segment_info})
    ss_index_info.db_info = group_args(
      group_args_type = 'dummy db info',
      db = None,
      data_dir = None,
    )
    return ss_index_info
def get_id_from_pdb_file_name(pdb_file_name):
  spl = pdb_file_name.split("_")
  if len(spl) == 2:
    return spl[0]
  else:
    return "_".join(spl[:2])
def apply_transformation(
      run_info,
      params,
      best_triples,
      ca_model,
      ss_index_info,
      sss_info,
      verbose = None,
      log = sys.stdout):
    if hasattr(run_info,'info_for_apply'):
      segment_info = run_info.info_for_apply.segment_info
      t = group_args(
        group_args_type = 'dummy_t',
        rmsd = run_info.info_for_apply.rmsd,
       db_id = 0,
       pairs = [],
       lsq_fit = run_info.info_for_apply.lsq_fit,
       )
      sii = None
    else:
      t = best_triples[run_info.triple_id]
      if hasattr(t,'ss_index_info'):
        sii = t.ss_index_info
      else:
        sii = ss_index_info
      segment_info = sii.segment_info_dict[t.db_id]
    fixed_model = ca_model.deep_copy()
    # Read in the full model
    mmm = sss_info.mmm
    if hasattr(segment_info,'full_model') and segment_info.full_model:
      # we read in the matching model
      segment_info.full_model.set_crystal_symmetry(mmm.unit_cell_crystal_symmetry())
      local_mmm = sss_info.mmm.customized_copy( model_dict={})
      segment_info.full_model.set_shift_cart(local_mmm.shift_cart())
      local_mmm.add_model_by_id(model_id = 'model',
         model = segment_info.full_model.deep_copy())
      moving_model = local_mmm.model()
      matching_ph = moving_model.get_hierarchy()
    else:
      matching_ph = get_ph_for_segment(mmm, sii.db_info,
        sii.segment_info_dict, t.db_id,
        local_pdb_dir = params.local_pdb_dir)
      moving_model = get_model_from_ph(mmm,matching_ph)
    if not moving_model:
      return group_args(
        group_args_type = 'finished triple',
         triple = None,
       )
    if params.superpose_model_on_other:  #
      print("Superposing model on other_model instead of other_model on model")
      assert not params.morph #won't work if we are distorting
      lsq_fit = invert_lsq_fit(t.lsq_fit)
      t.lsq_fit = lsq_fit
      m_model = moving_model
      t_model = fixed_model
      moving_model = t_model # swapping
      fixed_model = m_model
    matching_ca_sites_cart = get_ca_sites(moving_model.get_hierarchy())
    superposed_sites_cart = apply_lsq_fit(t.lsq_fit,matching_ca_sites_cart)
    matching_site_info = find_matching_sites(params.close_distance,
     fixed_model.get_sites_cart(), superposed_sites_cart)
    t.matching_site_info = matching_site_info
    print("MATCH %.2f %s (%s) Matching: %s (%.2f)" %(
       t.rmsd, t.db_id,
       sii.segment_info_dict[t.db_id].pdb_file_name if sii else "",
       len(t.pairs),
        t.matching_site_info.rmsd if
            t.matching_site_info.rmsd is not None else 0),
       file = log)
    if not matching_ca_sites_cart.size(): # failed
      return group_args(
        group_args_type = 'finished triple',
         triple = None,
       )
    t.ca_sites_cart = matching_ca_sites_cart  # full model
    t.superposed_sites_cart = apply_lsq_fit(t.lsq_fit,matching_ca_sites_cart)
    if params.morph:
      t.shifted_sites = apply_shift_field(params,
        t.shift_field_info, t.superposed_sites_cart) # full model
    else:
      t.shifted_sites = t.superposed_sites_cart
    # Now match these sites directly
    # Create new full moving_model here that is already close
    full_sc = moving_model.get_sites_cart()
    superposed_full_sc = apply_lsq_fit(t.lsq_fit, full_sc)
    if params.morph:
      shifted_full_sc = apply_shift_field(params,
        t.shift_field_info,superposed_full_sc)
    else:
      shifted_full_sc = superposed_full_sc
    t.rms_morphing = (shifted_full_sc - superposed_full_sc).rms_length()
    moving_model.set_sites_cart(shifted_full_sc)
    # Now we can apply transformation in pair_dict_info
    # Do it for selection and also for full model
    # Note that morph_with_segment_matching and  morph_with_shift_field both
    # require that the moving_model be superimposed already
    moving_model_info = morph_with_shift_field(params,
      mmm = mmm,
      moving_model = moving_model,
      fixed_model = fixed_model,
      log = log)
    if not moving_model_info:
      return group_args(
        group_args_type = 'finished triple',
         triple = None,
       )
    t.superposed_full_moving_model = \
      moving_model_info.superposed_full_moving_model
    t.superposed_moving_model = moving_model_info.superposed_moving_model
    # Now refine this (just the selected matching model)
    local_mmm = sss_info.mmm.customized_copy(
       model_dict={'model':t.superposed_moving_model})
    if not local_mmm.map_manager().is_dummy_map_manager():
      box_mmm = local_mmm.extract_all_maps_around_model()
    else:
      box_mmm = local_mmm
    if mmm.map_manager().is_dummy_map_manager():
      starting_cc = 1
    else: # have map
      starting_cc = box_mmm.map_model_cc()
    build = box_mmm.model_building()
    t.superposed_moving_model = build.model()
    if not t.superposed_moving_model:
      return group_args(
        group_args_type = 'finished triple',
         triple = None,
       )
    if params.refine_cycles and starting_cc < 1:
      new_model = build.refine(
         refine_cycles = params.refine_cycles, return_model =True)
      if new_model:
        t.superposed_moving_model = new_model
    elif params.quick_refine and starting_cc < 1:
      new_model = build.refine(quick = True, return_model = True)
      if new_model:
        t.superposed_moving_model = new_model
      sys.stdout =  sys_stdout_sav
    if box_mmm.map_manager().is_dummy_map_manager():
      cc = 1
    else:
      cc = box_mmm.map_model_cc(model = t.superposed_moving_model)
    t.starting_cc = starting_cc
    t.cc = cc
    t.cc_score = cc * \
       t.superposed_moving_model.get_hierarchy().overall_counts().n_residues
    mmm.shift_any_model_to_match(t.superposed_moving_model)
    return group_args(
      group_args_type = 'finished triple',
      triple = t,
     )
def morph_with_shift_field(params,
      moving_model = None,
      fixed_model = None,
      mmm = None,
      log = sys.stdout):
    '''
      Morph a model to match fixed_model using a shift field
      Select final matching residues that have maximum ca-ca deviation of
         max_shift_field_ca_ca_deviation (typically 2.75 A) and
        matching distance of
        tolerance_ratio * params.delta (typically 0.2 * 10 A)
    '''
    pair_dict_info = compare_model_pair(params, fixed_model, moving_model,
     log = log)
    if not pair_dict_info:
      return None
    # Full model
    working_xyz = moving_model.get_sites_cart()
    working_xyz =apply_lsq_fit(pair_dict_info.lsq_fit, working_xyz)
    if params.morph:
      working_xyz = apply_shift_field(params, pair_dict_info.shift_field_info,
        working_xyz)
    moving_model.set_sites_cart(working_xyz)
    if params.use_density_to_score and mmm and mmm.map_manager() and (not
       mmm.map_manager().is_dummy_map_manager()):
      match_selection = get_match_selection_from_density(
         mmm = mmm,
         original_selection = pair_dict_info.match_selection,
         moving_model = moving_model,
         max_non_superposing_residues = params.max_non_superposing_residues,
         matching_window = params.matching_window,
         matching_min_residues = params.matching_min_residues,
         matching_density_sigma = params.matching_density_sigma,
         )
    else:
      match_selection = pair_dict_info.match_selection
    if params.trim:
      moving_model_sel = moving_model.apply_selection_string(match_selection)
    else:
      moving_model_sel = moving_model
    if not moving_model_sel or not moving_model_sel.get_sites_cart().size():
      return None
    return group_args(
     group_args_type = 'moving_model_info',
     match_selection = match_selection,
     superposed_moving_model = moving_model_sel,
     superposed_full_moving_model = moving_model,
     moving_model_first_resno = get_first_resno(
       moving_model_sel.get_hierarchy()),
     moving_model_last_resno = get_last_resno(
       moving_model_sel.get_hierarchy()),
     moving_model_chain_id = get_chain_id(moving_model_sel.get_hierarchy()),
     moving_model_n_residues = \
        moving_model_sel.get_hierarchy().overall_counts().n_residues,
     moving_model_rmsd = pair_dict_info.rmsd,
     moving_model_lsq_fit = pair_dict_info.lsq_fit,
    )
def get_match_selection_from_density(
         mmm = None,
         original_selection = None,
         moving_model = None,
         max_non_superposing_residues = None,
         matching_window = None,
         matching_min_residues = None,
         matching_density_sigma = None,
         ):
  ''' choose parts of moving_model that are within
      max_non_superposing_residues of a residue in the original selection
      and that have good density in window of matching_window
  '''
  # original selection as residue ranges
  selected_model = moving_model.apply_selection_string("(%s) and (%s)" %(
     original_selection, 'name ca and protein'))
  density = mmm.map_manager().density_at_sites_cart(
      selected_model.get_sites_cart())
  smoothed_density = smooth_values(density)
  mean_smoothed_density = smoothed_density.min_max_mean().mean
  sd_smoothed_density = smoothed_density.standard_deviation_of_the_sample()
  # make a new selection that is everything in original plus max_non_superposing_residues away
  extended_selection = get_extended_selection(selected_model,
      max_non_superposing_residues)
  extension_only_ca_model = moving_model.apply_selection_string(
     "(%s) and not (%s)" %(
     extended_selection,original_selection))
  keep_selections = []
  for ca_extension in get_and_split_model(
       extension_only_ca_model.get_hierarchy()):
    # note ca_extension is model_info object with hierarchy attribute
    ca_sites_cart = ca_extension.hierarchy.atoms().extract_xyz()
    density = mmm.map_manager().density_at_sites_cart(ca_sites_cart)
    smoothed_density = smooth_values(density)
    keep = (smoothed_density > mean_smoothed_density -
       matching_density_sigma * sd_smoothed_density)
    keep = remove_singles_in_keep(keep, matching_min_residues)
    if keep.count(True) >= matching_min_residues: # something to keep
      keep_selections += get_keep_selections(
       start_res = get_first_resno(ca_extension.hierarchy),
       keep = keep)
  if not keep_selections:
    return original_selection
  else:
    new_selection = " or ".join(keep_selections)
    return "(%s) or (%s)" %(original_selection, new_selection)
def get_keep_selections( start_res =None, keep = None):
  keep_selections = []
  for i in range(keep.size()):
    if keep[i]:
      keep_selections.append("(resseq %s:%s)" %(
        i + start_res,
        i + start_res,))
  return keep_selections
def get_extended_selection(model,max_non_superposing_residues):
  selection_list = []
  for m in  get_and_split_model(model.get_hierarchy()):
    first_res = max(1,
          get_first_resno(m.hierarchy) -max_non_superposing_residues)
    last_res = get_last_resno(m.hierarchy) + max_non_superposing_residues
    selection =  "( chain %s and resseq %s:%s)" %(
      get_chain_id(m.hierarchy),
      first_res,
      last_res)
    selection_list.append(selection)
  return  " or ".join(selection_list)
def run_one_group_try(
       run_info,
       params,
       cluster_info,
       sss_info,
       ss_index_info,
       tries_tested,
       log = sys.stdout):
    triple_list_info = get_segments_by_group(params,
       run_info,
       cluster_info,
       sss_info,
       ss_index_info,
       tries_tested,
       log = log)
    triple_list = triple_list_info.triple_list
    tries_tested = triple_list_info.tries_tested
    # See if any of these are good based on lsq fitting of all segments
    if len(triple_list) > 0:
      print("Total initial triples: %s" %(len(triple_list)), file = log)
      new_triple_list = update_triple_matches(params,
       triple_list,
       cluster_info,
       sss_info,
       ss_index_info,
       log = log)
      triple_list += new_triple_list
      print("Total triples after lsq fitting all segments: %s" %(
       len(new_triple_list)), file = log)
    return group_args(
      group_args_type = 'triple matches',
      triple_list = triple_list,
      tries_tested = tries_tested)
def remove_duplicate_triples(triples):
  unique_triples = []
  unique_triples_as_str = []
  for t in triples:
    t_as_string = str((t.c1_id,t.c2_id,t.c3_id,t.c1a_id,t.c2a_id,t.c3a_id))
    ok = True
    for t1, t1_as_string in zip(unique_triples,unique_triples_as_str):
      if t_as_string == t1_as_string:
        ok = False
        break
      if t.pairs and t1.pairs and t.pairs == t1.pairs:
        ok = False
        break
      if t1.used_matching_sites.size() == t.used_matching_sites.size() and\
          (t1.used_matching_sites - t.used_matching_sites).rms_length() == 0:
        ok = False
        break
    if ok:
      unique_triples.append(t)
      unique_triples_as_str.append(t_as_string)
  return unique_triples
def group_of_get_segment_match(
     run_info,
     params,
     triples_by_segment_id,
     keys,
     cluster_info,
     sss_info,
     ss_index_info,
     log = sys.stdout):
  new_triples = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    key = keys[i]
    triples = triples_by_segment_id[key]
    result = get_segment_match(
     params,
     triples,
     cluster_info,
     sss_info,
     ss_index_info,
     log = log)
    if result.triple_list:
      new_triples += result.triple_list
  return group_args(
     group_args_type = 'new triples',
     triple_list = new_triples,
    )
def get_segment_match(
     params,
     triples,
     cluster_info,
     sss_info,
     ss_index_info,
     log = sys.stdout):
  #  See if we can combine information from triples to get one good one
  if len(triples) < 2:
    return group_args(
       group_args_type = 'new triples',
       triple_list = triples,)
  # See if a pair of triples has a similar transformation
  target_sites = flex.vec3_double()
  matching_sites = flex.vec3_double()
  ok_pairs_dict = {}
  original_triple = triples[0]
  for i in range(len(triples)-1):
    t1 = triples[i]
    if not hasattr(t1,'lsq_fit'): continue
    for j in range(i+1, len(triples)):
      t2 = triples[j]
      if t1.used_matching_sites.size() == t2.used_matching_sites.size() and\
          (t1.used_matching_sites - t2.used_matching_sites).rms_length() == 0:
        continue # these are the same
      sites = apply_lsq_fit(t1.lsq_fit, t2.used_matching_sites)
      rmsd1 = (t2.used_target_sites - sites).rms_length()
      if rmsd1 > params.delta: # not similar
        continue
      sites = apply_lsq_fit(t2.lsq_fit, t1.used_matching_sites)
      rmsd2 = (t1.used_target_sites - sites).rms_length()
      if rmsd2 > params.delta: # not similar
        continue
      if not are_pairs_compatible(t1.pairs,t2.pairs):
        continue
      if not i in list(ok_pairs_dict.keys()):
        ok_pairs_dict[i] = []
      if not j in list(ok_pairs_dict.keys()):
        ok_pairs_dict[j] = []
      ok_pairs_dict[i].append(j)
      ok_pairs_dict[j].append(i)
  all_sets = []
  t = triples[0]
  if not hasattr(t,'pairs'):
    return group_args(
     group_args_type = 'new triples',
     triple_list = triples)
  one_set = group_args(
      group_args_type = 'best set',
      triple_list = [t],
      pairs= t.pairs,
     )
  all_sets.append(one_set)
  for i in list(ok_pairs_dict.keys()):
    t = triples[i]
    triple_ids_used = [i]
    one_set = group_args(
      group_args_type = 'best set',
      triple_list = [t],
      pairs = None,
     )
    for j in ok_pairs_dict[i]:
      ok = True
      for k in triple_ids_used:
        if not k in ok_pairs_dict[j]:
          ok = False
      if ok: # j is compatible with i and all previous k
        one_set.triple_list.append(triples[j])
    one_set.pairs = get_pairs_from_compatible_triple_list(one_set.triple_list)
    ok = True
    for a_set in all_sets:
      if a_set.pairs == one_set.pairs:
        ok = False
        break
    if not ok: continue
    all_sets.append(one_set)
  all_sets = sorted(all_sets, key = lambda s: len(s.pairs), reverse = True)
  best_set = all_sets[0]
  new_triples = []
  for one_set in all_sets:  # build and score a set
    triple_info = deepcopy(one_set.triple_list[0])
    centers = ss_index_info.cluster_centers_by_segment_id[triple_info.db_id]
    ends_1 = ss_index_info.cluster_ends_1_by_segment_id[triple_info.db_id]
    ends_2 = ss_index_info.cluster_ends_2_by_segment_id[triple_info.db_id]
    fit_info = get_fit_info(params,
     pairs = one_set.pairs,
     ends_1 = ends_1,
     ends_2 = ends_2,
     cluster_info = cluster_info,
     centers = centers,
      )
    build_match_from_triple(params,
       sss_info.mmm,
       triple_info,
       cluster_info,
       ss_index_info,
       fit_info = fit_info)
    new_triples.append(triple_info)
  return group_args(
     group_args_type = 'new triples',
     triple_list = new_triples,
    )
def get_pairs_from_compatible_triple_list(triple_list):
  pairs = []
  for t in triple_list:
    for p in t.pairs:
      if not p in pairs:
        pairs.append(p)
  return pairs
def intersection(group1, group2):
  inter = []
  for i in group1:
    if i in group2:
       inter.append(i)
  return inter
def find_matching_sites(close_distance, sc1, sc2, good_enough_rmsd = 1): #  XXX library routine
  #  Try to find matching sites in sc2 tolerance close_distance
  sc1_dict = {}
  sc2_dict = {}
  matching_pairs = []
  target_sites = flex.vec3_double()
  matching_sites = flex.vec3_double()
  for i in range(sc1.size()):
    dist,id1,id2 = sc1[i:i+1].min_distance_between_any_pair_with_id(sc2)
    if dist <= close_distance:
      sc1_dict[i] = id2
  for j in range(sc2.size()):
    dist,id1,id2 = sc2[j:j+1].min_distance_between_any_pair_with_id(sc1)
    if dist <= close_distance:
      sc2_dict[j] = id2
  for i in range(sc1.size()):
    if not i in list(sc1_dict.keys()):
      continue
    id2 = sc1_dict[i]
    if not id2 in list(sc2_dict.keys()):
      continue
    j = id2
    if sc2_dict[id2] == i: # these agree
      matching_pairs.append([i,j])
      target_sites.append(sc1[i])
      matching_sites.append(sc2[j])
  if target_sites.size() > 0:
    rmsd = (matching_sites - target_sites).rms_length()
    score = matching_sites.size() / max(good_enough_rmsd, rmsd)
  else:
    rmsd = None
    score = 0
  return group_args(
   group_args_type = 'matching_info',
   target_sites = target_sites,
   matching_sites = matching_sites,
   rmsd = rmsd,
   score = score,
   )
def are_pairs_compatible(p1_list,p2_list):
  #  are all pairs in p1_list and p1_list either matching or not overlapping
  p1_dict = {}
  p2_dict = {}
  for p1 in p1_list:
    p1_dict[p1[0]] = p1[1]
  for p2 in p2_list:
    p2_dict[p2[0]] = p2[1]
  for p1 in p1_list:
    if p1[0] in list(p2_dict.keys()):
      if p1[1] != p2_dict[p1[0]]:
        return False
  for p2 in p2_list:
    if p2[0] in list(p1_dict.keys()):
      if p2[1] != p1_dict[p2[0]]:
        return False
  return True
def update_triple_matches(params,
       triple_list,
       cluster_info,
       sss_info,
       ss_index_info,
       log = sys.stdout):
  # Analyze matches and build full match from triple
  new_matches = []
  for triple_info in triple_list:
    build_match_from_triple(params,
       sss_info.mmm,
       triple_info,
       cluster_info,
       ss_index_info)
    if triple_info.rmsd is not None \
        and triple_info.rmsd <= 0.5* params.delta \
        and len(triple_info.pairs) >= params.minimum_matches:
      new_matches.append(triple_info)
  return new_matches
def build_match_from_triple(params,
     mmm,
     triple_info,
     cluster_info,
     ss_index_info,
     fit_info = None,
     ):
  segment_info = ss_index_info.segment_info_dict[triple_info.db_id]
  centers = ss_index_info.cluster_centers_by_segment_id[triple_info.db_id]
  ends_1= ss_index_info.cluster_ends_1_by_segment_id[triple_info.db_id]
  ends_2= ss_index_info.cluster_ends_2_by_segment_id[triple_info.db_id]
  if not fit_info:
    # centers of the members of this triple in the target:
    target_xyz = flex.vec3_double()
    for i in (triple_info.c1_id,triple_info.c2_id,triple_info.c3_id):
      target_xyz.append(cluster_info.cluster_centers[i])
    # centers in the template we want to match to the target:
    moving_xyz = flex.vec3_double()
    for i in ( triple_info.c1a_id, triple_info.c2a_id, triple_info.c3a_id):
      moving_xyz.append(centers[i])
    # Get initial transformation
    lsq_fit = superpose.least_squares_fit(
      reference_sites = target_xyz,
      other_sites     = moving_xyz)
    # Get the rmsd of these targets and template sites
    fitted_xyz = apply_lsq_fit(lsq_fit,moving_xyz)
    rmsd = (target_xyz - fitted_xyz).rms_length()
    # Now apply to ends
    pairs = []
    for i,j in zip(
      (triple_info.c1_id,triple_info.c2_id,triple_info.c3_id),
      (triple_info.c1a_id, triple_info.c2a_id, triple_info.c3a_id)):
      pairs.append([i,j])
    fit_info = get_fit_info(params,
       lsq_fit = lsq_fit,
       pairs = pairs,
       ends_1 = ends_1,
       ends_2 = ends_2,
       cluster_info = cluster_info,
       centers = centers,
      )
  lsq_fit = fit_info.lsq_fit
  match_dict = fit_info.match_dict
  # Try to bring in more segments now:
  max_dist = max(fit_info.rmsd, 0.5 * params.delta)
  match_segments_info = match_segments(params,
      triple_info,
      match_dict,
      lsq_fit,
      max_dist,
      centers,
      ends_1,
      ends_2,
      cluster_info,
      ss_index_info,)
  triple_info.close_matches = match_segments_info.close_matches
  triple_info.close_sites = match_segments_info.close_sites
  triple_info.rmsd = match_segments_info.rmsd
  triple_info.lsq_fit = lsq_fit
  triple_info.used_target_sites = match_segments_info.used_target_sites
  triple_info.used_matching_sites = match_segments_info.used_matching_sites
  triple_info.shift_field_info = match_segments_info.match_info.shift_field_info
  triple_info.pairs = match_segments_info.match_info.pairs
  triple_info.score = match_segments_info.match_info.score
def get_fit_info(params,
     lsq_fit = None,
     pairs = None,
     ends_1 = None,
     ends_2 = None,
     cluster_info = None,
     centers = None,
     ):
  if not lsq_fit: # get initial lsq fit
    target_xyz = flex.vec3_double()
    moving_xyz = flex.vec3_double()
    for i,j in pairs:
      target_xyz.append(cluster_info.cluster_centers[i])
      moving_xyz.append(centers[j])
    lsq_fit = superpose.least_squares_fit(
      reference_sites = target_xyz,
      other_sites     = moving_xyz)
  fitted_ends_1 = apply_lsq_fit(lsq_fit,ends_1)
  fitted_ends_2 = apply_lsq_fit(lsq_fit,ends_2)
  dist_info_list = flex.double()
  target_xyz = flex.vec3_double()
  moving_xyz = flex.vec3_double()
  match_dict = {}
  for i,j in pairs:
    dist_info = get_min_pair_distance(params,
      cluster_info.clusters[i].split_sites[0],
      cluster_info.clusters[i].split_sites[-1],
      ends_1[j],ends_2[j],
      fitted_ends_1[j],fitted_ends_2[j],
      )
    dist_info_list.append(dist_info.dist)
    target_xyz.append(cluster_info.cluster_centers[i])
    moving_xyz.append(centers[j])
    target_xyz.append(dist_info.x1_use)
    target_xyz.append(dist_info.x2_use)
    moving_xyz.append(dist_info.y1_use)
    moving_xyz.append(dist_info.y2_use)
    match_dict[i] = group_args(
      group_args_type = 'match dict',
      target_index = i,
      matching_index = j,)
  lsq_fit = superpose.least_squares_fit(
    reference_sites = target_xyz,
    other_sites     = moving_xyz)
  fitted_xyz = apply_lsq_fit(lsq_fit,moving_xyz)
  rmsd = (target_xyz - fitted_xyz).rms_length()
  return group_args(
    group_args_type = 'local fit',
    fitted_xyz = fitted_xyz,
    lsq_fit = lsq_fit,
    match_dict = match_dict,
    pairs = pairs,
    rmsd = rmsd)
def apply_shift_field(params,shift_field_info, sites_cart):
  if not shift_field_info:
    return sites_cart  # nothing to do
  shifts = get_shifts(params, shift_field_info.shifts,
     shift_field_info.centers, sites_cart = sites_cart)
  return sites_cart + shifts
def match_segments(params,
      triple_info,
      match_dict,
      lsq_fit,
      max_dist,
      centers,
      ends_1,
      ends_2,
      cluster_info,
      ss_index_info,):
  # Now map all the centers in the template
  fitted_centers = apply_lsq_fit(lsq_fit,centers)
  fitted_ends_1 = apply_lsq_fit(lsq_fit,ends_1)
  fitted_ends_2 = apply_lsq_fit(lsq_fit,ends_2)
  # For each target center find all the fitted centers that are plausible.
  #  check with ends.  Then see if we can find a distortion field that optimally
  #   maps fitted centers on to target center.
  pairs = []
  for i in list(match_dict.keys()):
    k = match_dict[i].matching_index
    pairs.append([i,k])
  match_info = group_args(
    group_args_type = 'match_info',
    match_dict = deepcopy(match_dict),
    offset_info = get_offset_info(params,
       triple_info.db_id,
       cluster_info,
       fitted_centers, fitted_ends_1, fitted_ends_2,
       ss_index_info),
    lsq_fit = lsq_fit,
    pairs = pairs,
    score = None,
    rms_offset = None,
    mean_dd = None,
    n_match = None,
   )
  score_match_info(params,match_info)
  # Now start with offsets based on our original match (i.e., as is)
  test_match_info = optimize_match_info(params,triple_info.db_id,
     cluster_info, match_info,
      fitted_centers,fitted_ends_1, fitted_ends_2,
      ss_index_info)
  if test_match_info.score > match_info.score:
    match_info = test_match_info
  # Get shift field and save also as working_offsets
  match_info = update_match_info_with_shift_field(
     params, cluster_info, match_info, fitted_centers)
  # Set up matched segments including shift field
  close_distances = flex.double()
  close_sites = flex.vec3_double()
  close_matches  = []
  used_target_sites = flex.vec3_double()
  used_matching_sites = flex.vec3_double()
  fitted_ends_1_use = fitted_ends_1 + match_info.offset_info.working_offsets
  fitted_ends_2_use = fitted_ends_2 + match_info.offset_info.working_offsets
  for k in list(match_info.match_dict.keys()):
    i = match_info.match_dict[k].matching_index
    dist = (col(fitted_centers[i]) +
       col(match_info.offset_info.working_offsets[i]) -
       col(cluster_info.cluster_centers[k])).length()
    dist_info = get_min_pair_distance(params,
      cluster_info.clusters[k].split_sites[0],
      cluster_info.clusters[k].split_sites[-1],
      ends_1[i],ends_2[i],
      fitted_ends_1_use[i],fitted_ends_2_use[i],
       )
    close_matches.append( group_args(
        group_args_type = 'split_sites center matches',
        target_cluster_id = k,
        matching_segment_id = triple_info.db_id,
        matching_cluster_id = i,
       ) )
    close_distances.append(dist)
    used_target_sites.append(cluster_info.cluster_centers[k])
    used_matching_sites.append(centers[i])
    used_target_sites.append(dist_info.x1_use)
    used_target_sites.append(dist_info.x2_use)
    used_matching_sites.append(dist_info.y1_use)
    used_matching_sites.append(dist_info.y2_use)
    close_sites.append(fitted_centers[i])
  return group_args(
    group_args_type = 'match segments with lsq obj',
    lsq_fit = lsq_fit,
    close_matches = close_matches,
    used_target_sites = used_target_sites,
    used_matching_sites = used_matching_sites,
    close_sites = close_sites,
    rmsd = close_distances.rms() if close_distances.size() > 0 else 0,
    match_info = match_info)
def update_match_info_with_shift_field(
     params, cluster_info, match_info, fitted_centers):
  # Identify shift field to move matches together locally
  # list of target shifts:
  target_shifts = flex.vec3_double()
  target_centers = flex.vec3_double()
  for i in list(match_info.match_dict.keys()):
    k = match_info.match_dict[i].matching_index
    target_shifts.append(match_info.offset_info.diff_offset_dict[i][k])
    target_centers.append(cluster_info.cluster_centers[i])
  shifts = get_shifts(params, target_shifts,target_centers,
    sites_cart = fitted_centers) # NOTE: can get shifts for any sites_cart using
    #  params, target_shifts,target_centers
  match_info.offset_info.working_offsets = shifts # save them
  match_info.shift_field_info = group_args(
    group_args_type = 'shift_field_info',
    centers = target_centers,
    shifts = target_shifts)
  return match_info
def get_shifts(params, target_shifts,target_centers, sites_cart = None):
  if not sites_cart:
    sites_cart = target_centers
  assert target_shifts.size() == target_centers.size()
  expected_shifts = flex.vec3_double()
  for i in range(sites_cart.size()):
    distances = (target_centers - col(sites_cart[i])).norms()
    weights = flex.exp(-distances*distances*(1./params.shift_field_distance**2))
    weighted_values = target_shifts * weights
    value = col(weighted_values.mean())*(1./max(1.e-10,
      weights.min_max_mean().mean))
    expected_shifts.append(value)
  return expected_shifts
def get_offset_info(params,segment_id,cluster_info, fitted_centers, fitted_ends_1,
    fitted_ends_2, ss_index_info):
  diff_offset_dict = {}
  distance_offset_dict = {}
  indices_offset_dict = {}
  n = cluster_info.cluster_centers.size()
  for i in range(n):
    diffs = fitted_centers  - col(cluster_info.cluster_centers[i])
    diffs = -1. * diffs
    distances = diffs.norms()
    sel = (distances <= 0.5* params.delta)
    # Check here for cluster.known_ss and direction if available
    for k in list(sel.iselection()):
      # now is it ok...
      dot = normalize(col(fitted_ends_2[k]) - col(fitted_ends_1[k])).dot(
             normalize((col(cluster_info.clusters[i].split_sites[-1]) -
              col(cluster_info.clusters[i].split_sites[0] ))))
      known_ss = cluster_info.clusters[i].known_ss
      matching_known_ss = \
          ss_index_info.cluster_known_ss_by_segment_id[segment_id][k]
      if known_ss and matching_known_ss and known_ss != matching_known_ss:
        sel[k] = False
      if sel[k]:
         if dot < 0:
           sel[k] = False
      if abs(dot) < params.min_abs_dot:
        sel[k] = False
    diffs_close = diffs.set_selected(~sel,(0,0,0))
    distances_close = distances.set_selected(~sel,0)
    indices_close = sel.iselection()
    diff_offset_dict[i] = diffs_close
    distance_offset_dict[i] = distances_close
    indices_offset_dict[i] = indices_close
  return group_args(
    group_args_type = 'offset info',
    diff_offset_dict = diff_offset_dict,
    distance_offset_dict = distance_offset_dict,
    indices_offset_dict = indices_offset_dict,
    working_offsets = None,
    n_clusters = n,
    n_fitted = fitted_centers.size())
def optimize_match_info(params,segment_id,cluster_info, match_info,
      fitted_centers, fitted_ends_1, fitted_ends_2,
      ss_index_info):
  best_match_info = match_info
  # First try all the best matches for each fragment. Do not allow dup matches
  # Keep all the matches that are unique
  keep_match_dict = {}
  used_matches = []
  for i in range(match_info.offset_info.n_clusters):
    if i in list(match_info.match_dict.keys()): # take it
      k = match_info.match_dict[i].matching_index
      if not k in used_matches:
        keep_match_dict[i] = k
        used_matches.append(k)
    elif len(match_info.offset_info.indices_offset_dict[i]) == 1:
      k = match_info.offset_info.indices_offset_dict[i][0]
      if not k in used_matches:
        keep_match_dict[i] = k
        used_matches.append(k)
  # Now get best match for all remaining
  for i in range(match_info.offset_info.n_clusters):
    if i in list(keep_match_dict.keys()): continue # already done
    if len(match_info.offset_info.indices_offset_dict[i]):
      best_k = None
      best_dd = None
      for k in match_info.offset_info.indices_offset_dict[i]:
        if k in used_matches: continue
        dd = match_info.offset_info.distance_offset_dict[i][k]
        if best_k is None or dd < best_dd:
          best_k = k
          best_dd = dd
        if best_k:
          keep_match_dict[i] = best_k
          used_matches.append(k)
  test_match_info = deepcopy(best_match_info)
  pairs = []
  for i in list(keep_match_dict.keys()):
    k = keep_match_dict[i]
    test_match_info.match_dict[i] = group_args(
           group_args_type = 'match dict',
           target_index = i,
           matching_index = k,
        )
    pairs.append([i,k])
  test_match_info.pairs = pairs
  test_match_info.offset_info = get_offset_info(params,
       segment_id,
       cluster_info,
       fitted_centers, fitted_ends_1, fitted_ends_2,
       ss_index_info)
  score_match_info(params,test_match_info)
  return test_match_info
def score_match_info(params,match_info):
  dd_values = flex.double()
  for i in list(list(match_info.match_dict.keys())):
    info = match_info.match_dict[i]
    j = info.matching_index
    distance = max(params.match_distance_low,min(params.match_distance_high,
      match_info.offset_info.distance_offset_dict[i][j]))
    w = (params.match_distance_high-distance)/(
       params.match_distance_high-params.match_distance_low)
    dd = w * params.score_distance_low + (1-w) * params.score_distance_high
    dd_values.append(dd)
  # want to increase score if just 1 additional distance_high or less
  #   but 1 additional low dist is better maybe 2 low dist to 1 high
  mean_dd = dd_values.min_max_mean().mean
  match_info.score = mean_dd * dd_values.size()
  match_info.mean_dd = mean_dd
  match_info.n_match = dd_values.size()
def get_working_offsets(match_info):
  working_diffs = flex.vec3_double(match_info.offset_info.n_fitted,(0,0,0))
  working_weights = flex.double(match_info.offset_info.n_fitted,0)
  for i in list(list(match_info.match_dict.keys())):
    info = match_info.match_dict[i]
    j = info.matching_index
    working_diffs[i] = match_info.offset_info.diff_offset_dict[i][j]
    working_weights[i] = 1
  return smooth_diffs(working_diffs,working_weights)
def smooth_diffs(working_diffs,working_weights, window = 6):
  # smooth in region where known, extend beyond
  n = working_diffs.size()
  w2 = window//2
  smoothed_values = flex.vec3_double(n,(0,0,0))
  smoothed_weights = flex.double(n,0)
  for i in range(n):
    if working_weights[i] == 0: continue
    for j in range(max(0,i-w2),min(n,i+w2)):
       w = working_weights[i] * max(0,1 - abs((i-j)/w2))
       smoothed_values[j] =  col(smoothed_values[j])+ w * col(working_diffs[i])
       smoothed_weights[j] += w
  sel = (smoothed_weights == 0)
  smoothed_weights.set_selected(sel,1.e-10)
  smoothed_values = smoothed_values/smoothed_weights
  smoothed_weights.set_selected(sel,0)
  for i in sel.iselection():
    smoothed_values[i] = get_neighboring_values(
      smoothed_values,smoothed_weights,i)
  return smoothed_values
def get_neighboring_values(smoothed_values,smoothed_weights,i):
  for k in range(smoothed_values.size()):
    sum = col((0,0,0))
    sum_n = 0
    for jj in [-k,k]:
      j = i + jj
      if j < 0 or j > smoothed_weights.size() - 1: continue
      if smoothed_weights[j] and smoothed_values[j] != (0,0,0):
        sum += col(smoothed_values[j])
        sum_n += 1
      if sum_n > 0:
        return sum/sum_n
  return col((0,0,0))
def get_min_pair_distance(params,x1,x2,y1_orig,y2_orig,y1,y2):
  # return minimum of x1-y1,x2-y2  or x1-y2,x2-y1
  d1 = (col(x1) - col(y1)).length()
  d2 = (col(x2) - col(y2)).length()
  dd1 = max(d1,d2)
  return group_args(
     group_args_type = 'min pair distance',
     x1_use = x1,
     x2_use = x2,
     y1_use = y1_orig,
     y2_use = y2_orig,
     dist = dd1,)
def get_segments_by_group(params,
       run_info,
       cluster_info,
       sss_info,
       ss_index_info,
       tries_tested = None,
       log = sys.stdout):
  triple_list = []
  triple_list_as_strings = []
  if not tries_tested:
    tries_tested = []
  find_matching_triples_info = find_matching_triples(params,
    run_info.group_try.j_start,
    run_info.group_try.j_end,
    cluster_info,
    ss_index_info,
    sss_info,
    tries_tested,
    log = log)
  local_triple_list = find_matching_triples_info.triple_list
  tries_tested = find_matching_triples_info.tries_tested
  for x in local_triple_list:
    xx = str(x)
    if not xx in triple_list_as_strings:
      triple_list_as_strings.append(xx)
      triple_list.append(x)
  return group_args(
    group_args_type = 'triple list from get segments',
    triple_list = triple_list,
    tries_tested = tries_tested)
def get_group_tries(params, sss_info, adjacent_only = False,
    full_range_only = False,
    previous_groups = None,
    cluster_window = None):
  if cluster_window is None:
    cluster_window = params.cluster_window
  n = len(sss_info.cluster_info_list)
  group_size = 1
  group_tries = []
  if adjacent_only:  # just make list of all adjacent clusters
    for i in range(n - 1):
      group_tries.append(group_args(
        group_args_type = 'group_try',
        j_start = i,
        j_end = i+1,
     ))
    return group_tries
  if full_range_only:
    max_tries = 1
  else:
    max_tries = params.max_tries
  for i in range(max_tries+1):
    if i == params.max_tries:
      if full_range_only:
        group_size = n
      else: #  usual
        group_size = min(n, cluster_window)
      delta = max(1,group_size)
    else:
      group_size += 2
      group_size = min(group_size, cluster_window)
      delta = max(1,group_size//2)
    for j in range(0, n, delta):
      j_end = min(n - 1, j + group_size -1)
      if j_end - j < 2: continue # need 3
      group_try = group_args(
        group_args_type = 'group_try',
        j_start = j,
        j_end = j_end,
        )
      if group_try_is_new(group_try,group_tries + previous_groups):
        group_tries.append(group_try)
    if group_size == n or group_size == cluster_window:
      break # no more
  return group_tries
def group_try_is_new(group_try,group_tries):
  for gt in group_tries:
    if gt.j_start == group_try.j_start and gt.j_end == group_try.j_end:
      return False
  return True
def get_groups_from_distances(params, distances, log = sys.stdout):
  #
  group_info = group_args(
    group_args_type = 'group info',
    groups = [],
  )
  # Try to get params.domain-size groups
  assert distances.is_square_matrix()
  n = distances.all()[0]
  rmsd_values = flex.double()
  for i in range(n):  # try center at i size of params.domain_size
    local_distances = flex.double()
    j_min = max(0,i-params.domain_size//2)
    j_max = min(n,i+params.domain_size//2)
    if j_min > 0:
      j_min = max(0,j_max - params.domain_size)
    if j_max < n:
      j_max = min(n,j_min + params.domain_size)
    for j in range(j_min, j_max):
      for k in range(j_min, j_max):
        local_distances.append(distances[j,k])
    rmsd_values.append(local_distances.rms())
    print("%s  %.2f" %(i,rmsd_values[-1]), file = log)
  return group_info
def get_distances_between_centers(sss_info):
  center_list = flex.vec3_double()
  for c in sss_info.cluster_info_list:
    center_list.append(c.original_ca_sites_cart.mean())
  distances = flex.double()
  n = center_list.size()
  for i in range(center_list.size()):
    for j in range(center_list.size()):
      distances.append( (col(center_list[i]) - col(center_list[j])).length())
  distances.reshape(flex.grid(n,n))
  return distances
def increment_indexing(params,other_cluster_info, other_sss_info,
    segment_id = None,
    log = sys.stdout):
  ss_index_info = group_args(
      group_args_type = 'ss index info',
      structures_by_ss_and_index =  {},
      other_clusters_by_ss_and_index =  {},
      cluster_centers_by_segment_id = {},
      cluster_ends_1_by_segment_id = {},
      cluster_ends_2_by_segment_id = {},
      cluster_known_ss_by_segment_id = {},
      n_structures = 0)
  ss_index_info.cluster_known_ss_by_segment_id[segment_id] = \
    get_known_ss_from_clusters( other_cluster_info.clusters)
  ss_index_info.cluster_centers_by_segment_id[segment_id],\
    ss_index_info.cluster_ends_1_by_segment_id[segment_id],\
    ss_index_info.cluster_ends_2_by_segment_id[segment_id] = \
    get_centers_from_clusters(
      other_cluster_info.clusters, use_reduced_sites = True)
  ss_index_info.other_clusters_by_ss_and_index[segment_id] = index_clusters(
       params,
       params.database_cluster_window,
       other_cluster_info,
       include_nearby_indices = False,
       log = log)
  ss_index_info.n_structures = 1
  return ss_index_info
def get_known_ss_from_clusters(clusters):
  known_ss = []
  for c in clusters:
    known_ss.append(c.known_ss)
  return known_ss
def get_centers_from_clusters(clusters, use_reduced_sites = True):
  centers = flex.vec3_double()
  ends_1 = flex.vec3_double()
  ends_2 = flex.vec3_double()
  if use_reduced_sites:
    for c in clusters:
      centers.append(c.split_sites.mean())
      ends_1.append(c.split_sites[0])
      ends_2.append(c.split_sites[-1])
  else:
    for c in clusters:
      centers.append(c.original_ca_sites_cart.mean())
      ends_1.append(c.original_ca_sites_cart[0])
      ends_2.append(c.original_ca_sites_cart[-1])
  return centers,ends_1,ends_2
def get_single_indexing(params,other_cluster_info, log = sys.stdout):
    other_id = 0
    other_clusters_by_ss_and_index = {}
    other_clusters_by_ss_and_index[other_id] = index_clusters(
       params,
       params.database_cluster_window,
       other_cluster_info,
       include_nearby_indices = False,
       log = log)
    n_structures = 1
    structures_by_ss_and_index = {}
    for ss1 in ['H','S']:
      structures_by_ss_and_index[ss1] = {}
      for ss2 in ['H','S']:
        structures_by_ss_and_index[ss1][ss2] = {}
        dd = structures_by_ss_and_index[ss1][ss2]
        for other_id in range(1):
          for index in other_clusters_by_ss_and_index[other_id][ss1][ss2]:
            if not index in list(dd.keys()):
              dd[index] = []
            dd[index]+= [0]
    segment_id = 0
    cluster_known_ss_by_segment_id = {}
    cluster_known_ss_by_segment_id[segment_id] = \
      get_known_ss_from_clusters( other_cluster_info.clusters)
    cluster_centers_by_segment_id = {}
    cluster_ends_1_by_segment_id = {}
    cluster_ends_2_by_segment_id = {}
    cluster_centers_by_segment_id[segment_id],\
      cluster_ends_1_by_segment_id[segment_id],\
      cluster_ends_2_by_segment_id[segment_id] = \
        get_centers_from_clusters(
         other_cluster_info.clusters, use_reduced_sites = True)
    return group_args(
      group_args_type = 'ss index info',
      structures_by_ss_and_index = structures_by_ss_and_index,
      other_clusters_by_ss_and_index = other_clusters_by_ss_and_index,
      cluster_centers_by_segment_id = cluster_centers_by_segment_id,
      cluster_ends_1_by_segment_id = cluster_ends_1_by_segment_id,
      cluster_ends_2_by_segment_id = cluster_ends_2_by_segment_id,
      cluster_known_ss_by_segment_id = cluster_known_ss_by_segment_id,
      n_structures = n_structures)
def find_matching_triples(params,
    start_cluster_index,
    end_cluster_index,
    cluster_info,
    ss_index_info,
    sss_info,
    tries_tested,
    log = sys.stdout):
  other_clusters_by_ss_and_index = ss_index_info.other_clusters_by_ss_and_index
  structures_by_ss_and_index = ss_index_info.structures_by_ss_and_index
  n_structures = ss_index_info.n_structures
  # find cluster in cluster_info that matches
  #   something in other_cluster_info.
  triple_list = []
  triple_list_as_tuples = []
  if start_cluster_index is None:
    start_cluster_index = 0
  if end_cluster_index is None:
    end_cluster_index = len(cluster_info.clusters) - 1
  for i in range(start_cluster_index, end_cluster_index - 1):
    c1 = cluster_info.clusters[i]
    for j in range(i+1, end_cluster_index):
      c2 = cluster_info.clusters[j]
      for k in range(j+1, end_cluster_index + 1):
        c3 = cluster_info.clusters[k]
        if (i,j,k) in tries_tested: continue
        tries_tested.append((i,j,k))
        for ss1 in c1.known_ss_list:  # typically just one
          for ss2 in c2.known_ss_list:  # typically just one
            triple_list += match_one_triple(params,
              c1,c2,c3,ss1,ss2,
              structures_by_ss_and_index,
              n_structures,
              other_clusters_by_ss_and_index,
              triple_list_as_tuples,
             log = log)
  return group_args(
    group_args_type = 'triple list',
    triple_list = triple_list,
    tries_tested = tries_tested)
def match_one_triple(params,c1,c2,c3,ss1,ss2,
     structures_by_ss_and_index,n_structures,
   other_clusters_by_ss_and_index,
   triple_list_as_tuples,
   log = sys.stdout):
  # Does this set of indices exist in others as a triple?
  triple_list = []
  if not c2.cluster_id in (list(c1.composite_indices_dict.keys())):
   return triple_list
  c12_index = c1.composite_indices_dict[c2.cluster_id][0]
  if not c12_index in list(structures_by_ss_and_index[ss1][ss2].keys()):
    return triple_list
  found_list_12_dict = {}
  reverse_found_list_12_dict = {}
  local_id = -1
  for i in range(n_structures):
    found_list_12_dict[i] = n_structures # max possible
  for structure_id in structures_by_ss_and_index[ss1][ss2][c12_index]:
    local_id += 1
    found_list_12_dict[structure_id] = local_id
    reverse_found_list_12_dict[local_id] = structure_id
  n_working = local_id + 1
  if n_working < 1:
    return triple_list
  for ss3 in c3.known_ss_list:  # typically one
    c13_index_list = []
    c23_index_list = []
    found_list_13_dict = {}
    found_list_23_dict = {}
    for c13_index in c1.composite_indices_dict[c3.cluster_id]: # typically one
      if (not c13_index in list(structures_by_ss_and_index[ss1][ss3].keys())):
        continue
      c13_index_list.append(c13_index)
      found_list_13 = flex.bool(n_working, False)
      for structure_id in structures_by_ss_and_index[ss1][ss3][c13_index]:
        #found_list_13[structure_id] = True
        local_id = found_list_12_dict[structure_id]
        if local_id == n_structures:
          continue
        found_list_13[local_id] = True
      found_list_13_dict[c13_index] = found_list_13
    for c23_index in c2.composite_indices_dict[c3.cluster_id]: # typically one
      if (not c23_index in list(structures_by_ss_and_index[ss2][ss3].keys())):
        continue
      c23_index_list.append(c23_index)
      found_list_23 = flex.bool(n_working, False)
      for structure_id in structures_by_ss_and_index[ss2][ss3][c23_index]:
        local_id = found_list_12_dict[structure_id]
        if local_id == n_structures:
          continue
        found_list_23[local_id] = True
      found_list_23_dict[c23_index] = found_list_23
    for c13_index in c13_index_list:
      for c23_index in c23_index_list:
        all_3 = (found_list_13_dict[c13_index] &
          found_list_23_dict[c23_index])
        if all_3.count(True) < 1:
          continue
        target_structure_id_list = all_3.iselection()
        # And which correspond to a triple...
        for local_id in target_structure_id_list:
          other_id = reverse_found_list_12_dict[local_id]
          ok = False
          ocbsi = other_clusters_by_ss_and_index[other_id]
          c12_list = ocbsi[ss1][ss2].get(c12_index,[])
          c13_list = ocbsi[ss1][ss3].get(c13_index,[])
          c23_list = ocbsi[ss2][ss3].get(c23_index,[])
          for c1a,c2a in c12_list:
            for a,b in c13_list:
              if a in [c1a,c2a]:
                c3a_match = b
                if a == c1a:
                  c1a_match = a
                else:
                  c1a_match = b
              elif b in [c1a,c2a]:
                c3a_match = a
                if b == c1a:
                  c1a_match = b
                else:
                  c1a_match = a
              else:
                continue
              for aa,bb in c23_list:
                if [c2a,c3a_match] == [aa, bb]:
                   c2a_match = aa
                elif [c2a,c3a_match] == [bb,aa]:
                   c2a_match = bb
                else:
                  continue
                # now c1, c2, c3 match with c1a_match,c2a_match,c3a_match
                pairs = [[c1.cluster_id,c1a_match],
                         [c2.cluster_id,c2a_match],
                         [c3.cluster_id,c3a_match]]
                p1,p2,p3= sorted(pairs, key = lambda p: p[0])
                triple = group_args(
                  group_args_type = 'triple',
                  db_id = other_id,
                  c1_id = p1[0],
                  c2_id = p2[0],
                  c3_id = p3[0],
                  c1a_id = p1[1],
                  c2a_id = p2[1],
                  c3a_id = p3[1],
                 )
                if params.require_sequential:
                  if p2[1]<= p1[1] or p3[1] <= p2[1]:
                    continue
                x = (other_id,p1[0],p2[0],p3[0],p1[1],p2[1],p3[1])
                if not x in triple_list_as_tuples:
                  triple_list_as_tuples.append(x)
                  triple_list.append(triple)
  return triple_list
def print_cluster_as_model(x1, text = 'clusters'):
    f = open('%s_%s.pdb' %(text,x1.cluster_id), 'w')
    m1 = mmtbx.model.manager.from_sites_cart(
      resseq_list = list(
         range(100*x1.cluster_id,100*x1.cluster_id+x1.split_sites.size())),
         sites_cart = x1.split_sites,
         crystal_symmetry = dummy_crystal_symmetry())
    print(m1.model_as_pdb(), file = f)
def dummy_crystal_symmetry(cell=1000):
    from cctbx import crystal
    from cctbx import sgtbx
    from cctbx import uctbx
    SpaceGroup=sgtbx.space_group_info(symbol=str('p1'))
    UnitCell=uctbx.unit_cell((cell,cell,cell,90.,90.,90.,))
    crystal_symmetry=crystal.symmetry(
      unit_cell=UnitCell,space_group_info=SpaceGroup)
    return crystal_symmetry
def expand_cluster_group(cluster_pair_info, cluster_group_number,
   cluster_info, other_cluster_info,
   minimum_match_fraction = 1.00, log = sys.stdout):
  cluster_group = cluster_pair_info.matching_cluster_groups[
      cluster_group_number]
  print_cluster_group(cluster_group, log = log)
  # Now expand it. Try every other segment in cluster_info and
  # add if relationship to everything in cluster_info and matching
  #  segment in other_cluster_info matches
  cluster_group.found_something = False
  cluster_group.tried_something = False
  pair_dict = {}
  for third_c in cluster_info.clusters:
    if third_c in cluster_group.group_1: continue
    print("trying third",third_c.cluster_id)
    for other_third_c in other_cluster_info.clusters:
      if other_third_c in cluster_group.group_2: continue
      count_bad = 0
      count_total = 0
      for i in range(len(cluster_group.group_1) - 1):
        existing = cluster_group.group_1[i]
        other_existing = cluster_group.group_2[i]
        if not cluster_pairs_match(existing, third_c,
             other_existing, other_third_c):
          count_bad += 1
        count_total += 1
      f = (count_total - count_bad)/max(1,count_total)
      if f >= minimum_match_fraction:
        if f < 1: print("FRACT:",f)
        cluster_group.found_something = True
        cluster_group.group_1.append(third_c)
        cluster_group.group_2.append(other_third_c)
        print_cluster_group(cluster_group, log = log)
        break  # go on to next third_c
  return cluster_pair_info
def order_cluster_group(cluster_group):
  sort_it = []
  for x1,x2 in zip (cluster_group.group_1,cluster_group.group_2):
   sort_it.append([x1,x2])
  sort_it = sorted(sort_it, key = lambda s: s[0])
  cluster_group.group_1 = []
  cluster_group.group_2 = []
  for x1,x2 in sort_it:
    cluster_group.group_1.append(x1)
    cluster_group.group_2.append(x2)
def print_cluster_group(cluster_group, log = sys.stdout):
  order_cluster_group(cluster_group)
  print("CG:", end = " ", file = log)
  print("Group 1:", end = " ", file = log)
  for x1 in cluster_group.group_1:
    print(x1.cluster_id,  end = " ", file = log)
  print("Group 2:", end = " ", file = log)
  for x2 in cluster_group.group_2:
    print(x2.cluster_id,  end = " ", file = log)
  print("",file = log)
def find_new_cluster_pair(cluster_pair_info,
    cluster_info,other_cluster_info):
  if not cluster_pair_info:
    cluster_pair_info = group_args(
      group_args_type = 'cluster pair info',
      used_cluster_id_pairs = [],
      matching_cluster_groups = [],
      tried_something = False,
      found_something = False,
      )
  cluster_pair_info.found_something = False
  cluster_pair_info.tried_something = False
  for first_c in cluster_info.clusters:
    for other_first_c in other_cluster_info.clusters:
      x =  (first_c.cluster_id,other_first_c.cluster_id)
      if x in cluster_pair_info.used_cluster_id_pairs:  continue
      cluster_pair_info.used_cluster_id_pairs.append(x)
      x1 =  (other_first_c.cluster_id,first_c.cluster_id)
      cluster_pair_info.used_cluster_id_pairs.append(x1)
      cluster_pair_info.tried_something = True
      match_info = clusters_match(first_c, other_first_c)
      if not match_info.ok: continue
      second_c = cluster_info.clusters[match_info.cluster_id]
      other_second_c = other_cluster_info.clusters[
        match_info.other_cluster_id]
      if not cluster_pairs_match(first_c, second_c,
           other_first_c, other_second_c):
        continue
      for third_c in cluster_info.clusters:
        if third_c in [first_c, second_c]: continue
        for other_third_c in other_cluster_info.clusters:
          if other_third_c in [other_first_c, other_second_c]:
            continue
          if not cluster_pairs_match(first_c, third_c,
              other_first_c, other_third_c):
            continue
          if not cluster_pairs_match(second_c, third_c,
              other_second_c, other_third_c):
            continue
          cluster_pair_info.matching_cluster_groups.append(
            group_args(
             group_args_type = 'matching cluster group',
             group_1 = [first_c, second_c, third_c],
             group_2 = [other_first_c, other_second_c,
                other_third_c],)
            )
          cluster_pair_info.found_something = True
          return cluster_pair_info
  return cluster_pair_info
def cluster_pairs_match(c1, c2, other_c1, other_c2):
  # Match if indices for c1/c2 match any indices in other_c1/other_c2
  if c1.known_ss and other_c1.known_ss and c1.known_ss != other_c1.known_ss:
    return False
  if c2.known_ss and other_c2.known_ss and c2.known_ss != other_c2.known_ss:
    return False
  for c1_index in c1.composite_indices_dict.get(c2.cluster_id,{}):
    if c1_index in other_c1.composite_indices_dict.get(other_c2.cluster_id,{}):
      return True
  return False
def clusters_match(c, other_c):
  # Reject if both are typed and they are different
  if c.known_ss and other_c.known_ss and \
      c.known_ss != other_c.known_ss:
    return group_args(
            group_args_type = 'cluster match info',
            ok = False,
            cluster_id = None,
            other_cluster_id = None,
            matching_index = None,
           )
  # Do clusters each match another cluster with similar distances
  for c1_id in list(c.composite_indices_dict.keys()):
    if c1_id == c.cluster_id: continue
    for other_c1_id in list(other_c.composite_indices_dict.keys()):
      if other_c1_id == other_c.cluster_id: continue
      for c1_index in c.composite_indices_dict[c1_id]:
        if c1_index in other_c.composite_indices_dict[other_c1_id]:
          return group_args(
            group_args_type = 'cluster match info',
            ok = True,
            cluster_id = c1_id,
            other_cluster_id = other_c1_id,
            matching_index = c1_index,
           )
  return group_args(
            group_args_type = 'cluster match info',
            ok = False,
            cluster_id = None,
            other_cluster_id = None,
            matching_index = None,
           )
def base_index_info(params):
  info = group_args(
    group_args_type = 'index info',
    n_max = int(0.5+params.dist_max/params.delta),
    dist_max = params.dist_max,
    delta = params.delta,
    )
  return info
def get_composite_indices(params, index_info, dist_values,
     include_nearby_indices = False):
  indices = []
  scale = 1
  composite_index = 0
  for d in dist_values:
    composite_index += get_index(index_info,d) * scale
    scale *= index_info.n_max
  composite_indices = [composite_index]
  if include_nearby_indices and params.include_nearby_indices_n >= 3:
    i_range = 6
    j_range = 6
    k_range = 6
    i_values = [-0.5 * index_info.delta, 0, 0.5 * index_info.delta]
    j_values = [-0.5 * index_info.delta, 0, 0.5 * index_info.delta]
    k_values = [-0.5 * index_info.delta, 0, 0.5 * index_info.delta]
    # Now find nearby indices that
    for i in range(i_range):
     for delta_dist_i in i_values:
      for j in range(j_range):
       for delta_dist_j in j_values:
        for k in range(k_range):
         for delta_dist_k in k_values:
          if i == j: continue
          if i == k: continue
          if j == k: continue
          dist_values_local = deepcopy(dist_values)
          dist_values_local[i] += delta_dist_i
          dist_values_local[k] += delta_dist_k
          dist_values_local[j] += delta_dist_j
          for composite_index in get_composite_indices(params,
             index_info, dist_values_local):
            if not composite_index in composite_indices:
              composite_indices.append(composite_index)
  elif include_nearby_indices:
    i_range = 6
    i_values = [-0.5 * index_info.delta, 0, 0.5 * index_info.delta]
    # Now find nearby indices that
    for i in range(i_range):
     for delta_dist_i in i_values:
          dist_values_local = deepcopy(dist_values)
          dist_values_local[i] += delta_dist_i
          for composite_index in get_composite_indices(params,
             index_info, dist_values_local):
            if not composite_index in composite_indices:
              composite_indices.append(composite_index)
  return composite_indices
def get_index(index_info,d):
  return max(0,min(index_info.n_max - 1, int(0.5 + d/index_info.delta)))
def index_clusters(params,
      cluster_window,
      cluster_info,
      include_nearby_indices = False,
      log = sys.stdout):
  index_info = base_index_info(params)
  clusters_by_ss_and_index = {}
  for ss1 in ['H','S']:
    clusters_by_ss_and_index[ss1] = {}
    for ss2 in ['H','S']:
      clusters_by_ss_and_index[ss1][ss2] = {}
  if cluster_window is None:
    cluster_window = len(cluster_info.clusters)
  for c in cluster_info.clusters:
    if c.known_ss:
      c.known_ss_list = [c.known_ss]
    elif params.use_original_ss_for_unknown_in_db:
      if c.original_helix_sheet_designation.original_sites_ss:
        c.known_ss_list = [c.original_helix_sheet_designation.original_sites_ss]
      else:
        c.known_ss_list = []
    else:
      c.known_ss_list = ['H','S']
  total_composite = 0
  for i in range(len(cluster_info.clusters)-1):
    c = cluster_info.clusters[i]
    c.composite_indices_dict = {}
    end_i = min(len(cluster_info.clusters) - 1, i + cluster_window - 1)
    for j in range(i+1, end_i + 1):
      other_c = cluster_info.clusters[j]
      c.composite_indices_dict[other_c.cluster_id] =  []
      other_sc1 = other_c.split_sites[0]
      other_sc2 = other_c.split_sites[-1]
      total_composite += 1
      sc1 = c.split_sites[0]
      sc2 = c.split_sites[-1]
      dist_values = []
      for s1,s2 in (  # 6 items here, that is the 6 in get_composite_indices
         [sc1,sc2],
         [other_sc1,other_sc2],
         [sc1,other_sc1],
         [sc1,other_sc2],
         [sc2,other_sc1],
         [sc2,other_sc2]):
        dist_values.append((col(s1) - col(s2)).length())
      cc = c.composite_indices_dict[other_c.cluster_id]
      for index in get_composite_indices(params, index_info, dist_values,
          include_nearby_indices = include_nearby_indices):
        if not index in cc:
          cc.append(index)
      for ss1 in c.known_ss_list:
        for ss2 in other_c.known_ss_list:
          dd = clusters_by_ss_and_index[ss1][ss2]
          for index in c.composite_indices_dict[other_c.cluster_id]:
            if not index in list(dd.keys()):
              dd[index] = []
            if c.cluster_id<other_c.cluster_id: # smaller, bigger
              dd[index].append([c.cluster_id,other_c.cluster_id])
            else:
              dd[index].append([other_c.cluster_id,c.cluster_id])
  #print("Total composite: %s" %(total_composite), file = log)
  return clusters_by_ss_and_index
def cluster_dist(c, other_c):
  return 0.5 * (col(c.split_sites[0]) + col(c.split_sites[-1]) -
        col(other_c.split_sites[0]) - col(other_c.split_sites[-1])).length()
def get_strand_lib(params):
  from libtbx import easy_pickle
  library_entries = []
  for x in params.strand_library_list:
    if not os.path.isfile(x): continue
    if x.endswith(".pdb"):
      x = x.replace(".pdb",".pkl")
    library_entries += easy_pickle.load(x)
  return library_entries
def run_replace_strand_in_sheet(sss_info, cluster_info, log = sys.stdout):
  # Replace a strand using density and neighbors
  # Set up ca sites in all clusters
  set_up_ca_sites(cluster_info, sss_info)
  # Get idealized beta dipeptide along x
  ideal_beta = idealized_beta_dipeptide(sss_info.mmm)
  for c in cluster_info.clusters:
    if c.known_ss != 'S': continue
    run_one_replace_strand_in_sheet(c, sss_info, cluster_info,
       ideal_beta, log = log)
def get_path_length(sites_cart):
  pl = 0
  for s1,s2 in zip(sites_cart,sites_cart[1:]):
    pl += (col(s1) - col(s2)).length()
  return pl
def invert_lsq_fit(lsq_fit_obj):
   lsq_fit_obj = deepcopy(lsq_fit_obj)
   r = lsq_fit_obj.r
   t = lsq_fit_obj.t
   r_inv=r.inverse()
   t_inv=-1.*r_inv*t
   lsq_fit_obj.r = r_inv
   lsq_fit_obj.t = t_inv
   return lsq_fit_obj
def apply_lsq_fit(lsq_fit_obj,xyz):
  return lsq_fit_obj.r.elems * xyz + lsq_fit_obj.t.elems
def get_transformation_from_center_and_xy_directions(
    from_center_position = None,
    from_x_direction = None,
    from_y_direction = None,
    to_center_position = None,
    to_x_direction = None,
    to_y_direction = None,):
  from_xyz_values = flex.vec3_double()
  from_xyz_values.append(from_center_position)
  from_xyz_values.append(col(from_center_position) + col(from_x_direction))
  from_xyz_values.append(col(from_center_position) + col(from_y_direction))
  to_xyz_values = flex.vec3_double()
  to_xyz_values.append(to_center_position)
  to_xyz_values.append(col(to_center_position) + col(to_x_direction))
  to_xyz_values.append(col(to_center_position) + col(to_y_direction))
  return superpose.least_squares_fit(
    reference_sites = to_xyz_values,
    other_sites     = from_xyz_values)
def is_even(i):
  if 2 * (i//2) == i:
    return True
  else:
    return False
def normalize(x):
  dd = x.length()
  if dd > 0:
    x = x/dd
  return x
def get_directions(c, other_strand):
  dir_forward = get_direction_from_sites_cart(c.original_ca_sites_cart)
  dir_left = other_strand.direction_from_target
  dir_up = dir_forward.cross(dir_left)
  dir_up = dir_up/max(1.e-10, dir_up.length())
  return group_args(
    group_args_type = 'directions',
    dir_forward = dir_forward,
    dir_up = dir_up,
    dir_left = dir_left,
    dir_other = get_direction_from_sites_cart(
      other_strand.cluster.original_ca_sites_cart)
   )
def print_strand_info(c, up_to_two_nearby_strands, log = sys.stdout):
  print("Found %s strands near strand %s" %(
    len(up_to_two_nearby_strands), c.cluster_id), file = log)
  print("Target strand length: %s " %( c.original_ca_sites_cart.size()),
      "Direction: (%.2f, %.2f, %.2f)" %(
      tuple(get_direction_from_sites_cart(c.original_ca_sites_cart))),
       file = log)
  target_direction = get_direction_from_sites_cart(c.original_ca_sites_cart)
  for strand_info in up_to_two_nearby_strands:
    print("Strand ID: %s   " %( strand_info.cluster.cluster_id),
     "Direction: (%.2f, %.2f, %.2f) " %(
      tuple(strand_info.cluster.direction)),
    "\n  Direction from target: (%.2f, %.2f, %.2f)  " %(
       tuple(strand_info.cluster.direction)),
    "Distance: %.2f" %( strand_info.distance_from_target),
    "\n  Direction dot to target_direction: %.2f " %(
       target_direction.dot(strand_info.cluster.direction)),
     file = log)
    if len(up_to_two_nearby_strands) == 2:
      print("  Direction dot to direction of other nearby strand: %.2f " %(
        up_to_two_nearby_strands[0].cluster.direction.dot(
        up_to_two_nearby_strands[1].cluster.direction)),
        file = log)
def get_up_to_two_nearby_strands(cluster_info, c):
  other_strands_list = get_nearby_strands(cluster_info, c)
  # Now choose up to two of these to left and right
  up_to_two_other_strands = []
  for other_strand_info in other_strands_list:
    if len(up_to_two_other_strands) == 0: # take it
      up_to_two_other_strands.append(other_strand_info)
    elif len(up_to_two_other_strands) == 1:
      existing_direction = up_to_two_other_strands[0].cluster.direction
      new_direction = other_strand_info.cluster.direction
      if existing_direction.dot(new_direction) < 0:  # opposite...ok
        up_to_two_other_strands.append(other_strand_info)
  return up_to_two_other_strands
def idealized_beta_dipeptide(mmm):
  # Located near 0,0,0, CB along y, dipeptide translation 6.74 A along x
  # Residue 11 has CB up and residue 12 has CB down
  text = """
ATOM     51  N   ALA A  11      -1.267   0.361   0.480  1.00 30.00           N
ATOM     52  CA  ALA A  11       0.000   0.830  -0.121  1.00 30.00           C
ATOM     53  C   ALA A  11       1.209   0.227   0.567  1.00 30.00           C
ATOM     54  O   ALA A  11       1.172   0.205   1.784  1.00 30.00           O
ATOM     55  CB  ALA A  11       0.114   2.364  -0.139  1.00 30.00           C
ATOM     56  N   ALA A  12       2.171  -0.207  -0.287  1.00 30.00           N
ATOM     57  CA  ALA A  12       3.435  -0.830   0.121  1.00 30.00           C
ATOM     58  C   ALA A  12       4.608   0.104  -0.268  1.00 30.00           C
ATOM     59  O   ALA A  12       4.676   0.723  -1.359  1.00 30.00           O
ATOM     60  CB  ALA A  12       3.863  -2.178  -0.470  1.00 30.00           C
   """
  m = mmm.model_from_text(text, return_as_model = True)
  # Shift to place CA at exactly 0,0,0
  sites_cart = m.get_sites_cart()
  sites_cart -= (0.000, 0.830,-0.121)
  m.set_sites_cart(sites_cart)
  #print(m.model_as_pdb(do_not_shift_back=True))
  return group_args(
   group_args_type = 'idealized dipeptide CA at 0,0,0 ',
   model = m,
   dipeptide_translation = (6.74,0,0),
   direction = (1,0,0),
   c_beta_direction = (0,1,0),
   )
def set_up_neighbor_ca_sites(cluster_info, sss_info):
  mmm = sss_info.mmm
  ca_model = mmm.model().apply_selection_string("name ca " )
  selection_list = get_neighbor_selections(sss_info.cluster_info_list)
  cluster_info.neighbor_sites_cart_list = []
  for c1,c2, selection in zip(
       sss_info.cluster_info_list,sss_info.cluster_info_list[1:],
        selection_list):
    if selection is None:
      cluster_info.neighbor_sites_cart_list.append(None)
    else:
      cluster_info.neighbor_sites_cart_list.append(
        ca_model.apply_selection_string(
             selection).get_sites_cart())
def set_up_ca_sites(cluster_info, sss_info):
  mmm = sss_info.mmm
  ca_model = mmm.model().apply_selection_string("name ca " )
  selection_list = get_selections(sss_info.cluster_info_list)
  for c, selection in zip(
       sss_info.cluster_info_list, selection_list):
    c.add(key = 'original_ca_sites_cart',
          value = ca_model.apply_selection_string(
             selection).get_sites_cart())
    c.add(key = 'direction',
         value = get_direction_from_sites_cart(c.original_ca_sites_cart))
def get_nearby_strands(cluster_info, c, max_dist = 8):
  # find strands near c on left and right side
  c_sites = c.original_ca_sites_cart
  other_strands_list = []
  for other_ga in cluster_info.all_neighbors_dict[c.cluster_id]: # close to far
    other_c = cluster_info.clusters[other_ga.other_cluster_id]
    assert other_c.cluster_id == other_ga.other_cluster_id
    assert c.cluster_id == other_ga.cluster_id
    if other_c == c:
      continue
    if other_c.known_ss != 'S':
      continue
    other_sites =  other_c.original_ca_sites_cart
    closest_dist,id1,id2 = c_sites.min_distance_between_any_pair_with_id(
       other_sites)
    dist = other_ga.dist
    if dist <= max_dist:
      other_strands_list.append(
          group_args(
          group_args_type = 'other_strand_info',
            cluster = other_c,
            distance_from_target = dist,
            direction_from_target = get_direction_from_two_sites(
              c_sites[id1], other_sites[id2]),
            n_close_sites = get_n_close_sites(c_sites, other_sites,
              max_dist = max_dist),
           ))
  return other_strands_list
def get_n_close_sites(c_sites, other_sites, max_dist = None):
  # Count number of sites in other_sites close to a site in c_sites
  n = 0
  for i in range(other_sites.size()):
    dist, id1,id2 = c_sites.min_distance_between_any_pair_with_id(
       other_sites[i:i+1])
    if dist <= max_dist:
      n += 1
  return n
def find_a_strand(cluster_info):
  # Find a strand
  for c in cluster_info.clusters:
    if c.known_ss == 'S':  # found one
      return c
def replace_strand_in_sheet(mmm, strand_segment, other_strand_segments,
    log = sys.stdout):
  pass
def write_replacement_models(cluster_info, sss_info, log = sys.stdout):
  mmm = sss_info.mmm
  model_info_list = []
  full_model_info_list = []
  full_ph = mmm.model().get_hierarchy()
  last_original_end_res = get_first_resno(full_ph) - 1
  for c in cluster_info.clusters:
    if not c.replacement_model: continue
    m = model_info(hierarchy = c.replacement_model.get_hierarchy())
    model_info_list.append(m)
    # Get anything before this insert
    if c.original_start_res > last_original_end_res + 1:  # start with these
      full_model_info_list.append(model_info(hierarchy = select_residue_range(
       full_ph, last_original_end_res + 1, c.original_start_res - 1)))
    # Get the insert
    full_model_info_list.append(model_info(hierarchy = m.hierarchy))
    last_original_end_res = c.original_end_res
  # Get after the insert
  if get_last_resno(full_ph) > last_original_end_res:
      full_model_info_list.append(model_info(hierarchy = select_residue_range(
        full_ph, last_original_end_res + 1, get_last_resno(full_ph))))
  if not model_info_list:
    return
  new_model = merge_hierarchies_from_models(models=model_info_list,
        resid_offset=None,
        chain_id = c.original_chain_id,
        renumber=False,)
  new_ph = new_model.hierarchy
  local_model = mmm.model_from_hierarchy(new_ph, return_as_model = True)
  mmm.add_model_by_id(model_id = 'temp', model = local_model)
  mmm.write_model(model_id = 'temp', file_name = 'replacement.pdb')
  full_new_model_info = merge_hierarchies_from_models(
        models=full_model_info_list,
        resid_offset=None,
        chain_id = c.original_chain_id,
        renumber=True,
        remove_break_records = True,
        first_residue_number=get_first_resno(full_ph))
  full_new_ph = full_new_model_info.hierarchy
  full_new_model = mmm.model_from_hierarchy(full_new_ph, return_as_model = True)
  mmm.add_model_by_id(model_id = 'full_new_model', model = full_new_model)
  mmm.write_model(model_id = 'full_new_model', file_name = 'full_new_model.pdb')
  return group_args(
    group_args_type = 'full new model',
    model = full_new_model)
def is_straight(ca_sites = None,
     dist_delta = 7.5, maximum_strand_length= None):
  ''' do original coordinates run in a straight line'''
  n = ca_sites.size()
  if n < 9:
    return  group_args( group_args_type = 'worst offset',
         is_straight = True,
         worst_offset = None,
         worst_middle = None)
  worst_offset = 0
  worst_middle = None
  for i in range(n-8):
    start_n = i
    middle_n = i + 4
    end_n = i + 8
    average_of_ends = 0.5*(col(ca_sites[start_n]) + col(ca_sites[end_n]))
    worst_offset = max(worst_offset,
        (col(ca_sites[middle_n]) - average_of_ends).length())
    worst_middle = middle_n
  if worst_offset > dist_delta or (
    maximum_strand_length and n > maximum_strand_length):
      return group_args( group_args_type = 'worst offset',
         is_straight = False,
         worst_offset = worst_offset,
         worst_middle = worst_middle)
  else:
      return  group_args( group_args_type = 'worst offset',
         is_straight = True,
         worst_offset = worst_offset,
         worst_middle = worst_middle)
def replace_strand_with_helix(c, existing_segment, extended_existing_segment,
     sss_info,
     temp_dir = None,
     log = sys.stdout):
  '''
    Use density in sss_info.mmm to find a helix and replace existing segment
  '''
  local_mmm = sss_info.mmm.customized_copy(
     model_dict={'model':extended_existing_segment})
  box_mmm = local_mmm.extract_all_maps_around_model()
  box_mmm.mask_all_maps_around_atoms(mask_atoms_atom_radius = 5)
  build = box_mmm.model_building()
  build.set_model(None)
  build.set_log(log)
  build.set_defaults(build_methods=['find_helices_strands'],
    temp_dir = temp_dir)
  build.build(include_strands = False, cc_min = 0.01, refine_cycles = 0)
  if not build.model():
    return  # did not work
  build.refine(restrain_ends = True)
  new_segment = build.model()
  sss_info.mmm.shift_any_model_to_match(existing_segment)
  sss_info.mmm.shift_any_model_to_match(new_segment)
  # Can we splice it into existing segment...
  existing_ca = get_ca_sites(existing_segment.get_hierarchy())
  new_ca = get_ca_sites(new_segment.get_hierarchy())
  existing_chain_direction = get_direction_from_sites_cart(existing_ca)
  new_chain_direction = get_direction_from_sites_cart(new_ca)
  if existing_chain_direction.dot(new_chain_direction) < 0:
    new_segment = build.reverse_direction(new_segment, refine_cycles = 0,
      allow_non_sequential = True)
    new_ca = get_ca_sites(new_segment.get_hierarchy())
  from iotbx.data_manager import DataManager
  dm = DataManager()
  dm.set_overwrite(True)
  dm.write_model_file(existing_segment,'existing_segment.pdb')
  dm.write_model_file(new_segment,'new_segment.pdb')
  # Find first new_ca in c-term direction from first existing_ca
  start_point_in_new_segment = get_start_point_in_new_segment(existing_ca,
     new_ca)
  if start_point_in_new_segment is None:
    return None  # didn't work
  # Find first new_ca in n-term direction from last existing_ca
  end_point_in_new_segment = get_start_point_in_new_segment(existing_ca,
     new_ca, reverse = True)
  if end_point_in_new_segment is None:
    return None  # didn't work
  # Now select the parts of original and new segment to make a complete segment
  existing_segment = sss_info.mmm.model().apply_selection_string(
       get_selections([c])[0])
  new_model = match_replacement_to_existing(
        mmm = sss_info.mmm,
        replacement_model = new_segment,
        extended_existing_segment = existing_segment,
        existing_segment = existing_segment,  # what we have
        first_res = c.original_start_res,
        last_res = c.original_end_res,
        log = log)
  return new_model
def reverse_vec3_double(new_ca):
    new_ca = list(new_ca)
    new_ca.reverse()
    new_ca = flex.vec3_double(new_ca)
    return new_ca
def get_start_point_in_new_segment(existing_ca, new_ca,
   close_ca = 1.5, dist_ca = 3.8, reverse = False):
  start_point_in_new_segment = None
  if reverse:
    new_ca = reverse_vec3_double(new_ca)
    existing_ca = reverse_vec3_double(existing_ca)
  for i in range(new_ca.size()//2):
    dd = distance_towards_end(site = new_ca[i], sites_cart = existing_ca)
    if dd > 0:
      # it is in the right direction.  Is it somewhat close to a ca in existing?
      dist, id1,id2 = new_ca[i:i+1].min_distance_between_any_pair_with_id(
        existing_ca)
      if dist <= close_ca:  # overlap here...take next residue
        start_point_in_new_segment = i + 1
        break
      elif dist <= close_ca + dist_ca:  # take it from here
        start_point_in_new_segment = i
        break
  return start_point_in_new_segment
def distance_towards_end(site = None, sites_cart = None):
  # Is site towards the end of sites_cart
  chain_direction = get_direction_from_sites_cart(sites_cart)
  return chain_direction.dot(col(site) - col(sites_cart[0]))
def get_direction_from_sites_cart(sites_cart, i_start = None, i_end = None):
  if sites_cart.size() == 0:
    return col((1,0,0))  # just to return something
  if i_start is None:
    i_start = 0
  if i_end is None:
    i_end = sites_cart.size() -1
  dd = col(sites_cart[i_end]) - col(sites_cart[i_start])
  dd = dd/max(1.e-10, dd.length())
  return dd
def get_direction_from_two_sites(s1,s2):
  dd = col(s2) - col(s1)
  dd = dd/max(1.e-10, dd.length())
  return dd
def replace_structure_elements(cluster_info, sss_info,
    strand_lib = None,
    residues_on_end = 2,
    verbose = False,
    nproc = None,
    log = sys.stdout):
  '''
  for each structure element, try to replace it with idealized structure
  May need to break it up or try more than once
  '''
  replace_from_pdb = replace_segment( log = null_out())
  run_info_list = []
  for i in range(len(cluster_info.clusters)):
    run_info_list.append(group_args(
      cluster_id = i,
      ))
  print("\nTotal of %s clusters to replace" %(
     len(run_info_list)), file = log)
  kw_dict={
    'cluster_info': cluster_info,
    'replace_from_pdb':deepcopy(replace_from_pdb),
    'sss_info': sss_info,
    'strand_lib': strand_lib,
    'residues_on_end':residues_on_end
   }
  run_info_list = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = verbose,
    kw_dict = kw_dict,
    run_info_list = run_info_list,
    job_to_run = replace_one_cluster,
    log = log)
  for run_info in run_info_list:
    if run_info.result and getattr(run_info.result,'model',None):
      m = run_info.result.model
      sss_info.mmm.shift_any_model_to_match(m)
      c = cluster_info.clusters[run_info.result.cluster_id]
      print("\nReplacement for ",c.cluster_id,
         m.info().target_length,
         m.info().length, c.known_ss,
         c.original_helix_sheet_designation.original_sites_ss,
          m.info().score, file = log)
      c.replacement_model = m
def replace_with_best_matching_strand_from_pairs(
          sss_info = None,
          strand_lib = None,
          c = None,
          other_c = None,
          log = sys.stdout):
  ''' Try to replace c (and other_c as placement) with something from
       strand_lib'''
  best_fit_info = None
  strands_are_parallel = is_parallel(
      c.original_ca_sites_cart,other_c.original_ca_sites_cart)
  for strand_pair in strand_lib:
    if not (strands_are_parallel == strand_pair.parallel): continue
    for s1_ph,s1,s2 in (
       [strand_pair.s1_ph,strand_pair.s1_ca_sites,strand_pair.s2_ca_sites],
       [strand_pair.s2_ph,strand_pair.s2_ca_sites,strand_pair.s1_ca_sites]):
      if s1.size() < c.original_ca_sites_cart.size(): continue # too small
      fit_info = superpose_strand_pairs(
        s1_ph,
        c.original_ca_sites_cart,
        other_c.original_ca_sites_cart,
        s1,
        s2,
        log = log)
      if fit_info and (
         (best_fit_info is None) or (fit_info.rmsd < best_fit_info.rmsd)):
        best_fit_info = fit_info
  if best_fit_info:
    print("Best fit for strand %s rmsd: %.2f A for %s CA atoms" %(
       c.cluster_id, best_fit_info.rmsd, best_fit_info.n_ca_atoms), file = log)
    # Create model with this fitted strand
    new_model = sss_info.mmm.model_from_hierarchy(
        best_fit_info.fitted_hierarchy, return_as_model=True)
    existing_segment = sss_info.mmm.model().apply_selection_string(
       get_selections([c])[0])
    new_model = match_replacement_to_existing(
        mmm = sss_info.mmm,
        replacement_model = new_model,
        extended_existing_segment = existing_segment,
        existing_segment = existing_segment,  # what we have
        first_res = c.original_start_res,
        last_res = c.original_end_res,
        log = log)
    return new_model
  else:
    return None
def superpose_strand_pairs(
      s1_ph,
      sc,
      other_sc,
      s1,
      s2,
      n_residue_gap = 5,
      max_dist = 6,
      max_dist_at_ends = 2,
      log = sys.stdout):
  # Superpose 1 atom of sc on any atom of s1. Then orient N residues out on
  #   each in same direction.  Then orient nearest atom in other_sc on nearest
  #   atom in s2.  Get closest site pairs, including all atoms in s1.
  #    Then superpose with all the close sites and get rmsd as score
  best_fit_info = None
  # Now we are going to work from the ends only
  i = 0
  ii = sc.size()-1  # end
  middle = 0.5*(sc[0:1]+sc[sc.size()-1:sc.size()])
  dist, id1,id2 = middle.min_distance_between_any_pair_with_id(other_sc)
  other_i = id2
  dist_sc_0_end = (col(sc[0]) -col(sc[-1])).length()
  dist_sc_0 = (col(sc[0]) - col(other_sc[other_i])).length()
  dist_sc_end = (col(sc[-1]) - col(other_sc[other_i])).length()
  n = sc.size()
  for k in range(s1.size()-1):
    for kk in range(k+1,s1.size()):
      # find distance similar to dist_sc_0_end
      dist_k_kk = (col(s1[k]) - col(s1[kk])).length()
      if abs(dist_k_kk - dist_sc_0_end) >  max_dist_at_ends: continue
      s1_close_to_sc_0 = k
      s1_close_to_sc_end = kk
      # Now get third point for s1/s2...a point in s2 that has same distance to
      #  s1_close_to_sc_0 and s1_close_to_sc_end as does other_i in other_sc to
      #   beginning and end of sc
      best_j = None
      best_dist_sq = None
      for j in range(s2.size()):
        dist_s1_0 = (col(s1[s1_close_to_sc_0]) - col(s2[j])).length()
        dist_s1_end = (col(s1[s1_close_to_sc_end]) - col(s2[j])).length()
        dist_sq = (dist_s1_0 - dist_sc_0)**2 + (dist_s1_end - dist_sc_end)**2
        if best_j is None or dist_sq < best_dist_sq:
          best_j = j
          best_dist_sq = dist_sq
      s2_close_to_other_i = best_j
      reference_sites = flex.vec3_double()
      other_sites = flex.vec3_double()
      reference_sites.append(sc[0])
      reference_sites.append(sc[-1])
      reference_sites.append(other_sc[other_i])
      other_sites.append(s1[s1_close_to_sc_0])
      other_sites.append(s1[s1_close_to_sc_end])
      other_sites.append(s2[s2_close_to_other_i])
      # now superpose with these 3 points:
      from scitbx.math import superpose
      fit_obj = superpose.least_squares_fit(
        reference_sites = reference_sites,
        other_sites     = other_sites)
      # And apply to s1 and s2
      fitted_other = apply_lsq_fit(fit_obj,other_sites)
      rmsd_ends =  (reference_sites - fitted_other).rms_length()
      fitted_s1 = apply_lsq_fit(fit_obj,
         s1[s1_close_to_sc_0:s1_close_to_sc_end+1])
      coords_in_s1_matching_sc = get_matching_coords(sc,fitted_s1)
      rmsd = (sc - coords_in_s1_matching_sc).rms_length()
      fitted_s2 = apply_lsq_fit(fit_obj, s2)
      fitted_hierarchy = s1_ph.deep_copy()
      sites_cart = fitted_hierarchy.atoms().extract_xyz()
      fitted_sites_cart = apply_lsq_fit(fit_obj,sites_cart)
      fitted_hierarchy.atoms().set_xyz(fitted_sites_cart)
      fit_info =  group_args(
        group_args_type = 'fit of two sets of coordinates to two strands',
        fitted_hierarchy = fitted_hierarchy,
        original_fitted_working_s1 = fitted_s1,
        fitted_working_s1 = fitted_s1,
        fitted_working_s2 = fitted_s2,
        rmsd = rmsd,
        n_ca_atoms = fitted_s1.size(),
        diff_at_ends = rmsd_ends,
        lsq_fit_obj = fit_obj,
        offsets = None,
         )
      if not fit_info or fit_info.diff_at_ends > max_dist_at_ends: continue
      if (best_fit_info is None) or (fit_info.rmsd < best_fit_info.rmsd):
        best_fit_info = fit_info
  return best_fit_info
def get_fit_matching_one_residue(sc,other_sc,s1,s2,i,j,ii,jj,other_i,other_j,
    weight_ends = 10):
  dir_i = get_direction_from_two_sites(sc[i],sc[ii])
  cross_dir_i = get_direction_from_two_sites(sc[i], other_sc[other_i])
  dir_j = get_direction_from_two_sites(s1[j],s1[jj])
  cross_dir_j = get_direction_from_two_sites(s1[j],s2[other_j])
  lsq_fit_obj = get_transformation_from_center_and_xy_directions(
     from_center_position = s1[j],
     from_x_direction = dir_j,
     from_y_direction = cross_dir_j,
     to_center_position = sc[i],
     to_x_direction = dir_i,
     to_y_direction = cross_dir_j,)
  fitted_s1 = apply_lsq_fit(lsq_fit_obj,s1)
  fitted_s2 = apply_lsq_fit(lsq_fit_obj,s2)
  coords_in_s1_matching_sc = get_matching_coords(sc,fitted_s1)
  coords_in_s2_matching_other_sc = get_matching_coords(other_sc,fitted_s2)
  target_sc = sc.deep_copy()
  target_sc.extend(other_sc)
  working_sc = coords_in_s1_matching_sc.deep_copy()
  working_sc.extend(coords_in_s2_matching_other_sc)
  target_w = target_sc.deep_copy()
  working_w = working_sc.deep_copy()
  for i in range(int(weight_ends)):
    target_w.append(sc[0])
    target_w.append(sc[1])
    target_w.append(sc[-1])
    target_w.append(sc[-2])
    working_w.append(coords_in_s1_matching_sc[0])
    working_w.append(coords_in_s1_matching_sc[1])
    working_w.append(coords_in_s1_matching_sc[-1])
    working_w.append(coords_in_s1_matching_sc[-2])
  # Superpose working on target by lsq
  from scitbx.math import superpose
  new_fit_obj = superpose.least_squares_fit(
    reference_sites = target_w,
    other_sites     = working_w)
  fitted_working_sc = apply_lsq_fit(new_fit_obj,working_sc)
  fitted_working_s1 = fitted_working_sc[:sc.size()]
  fitted_working_s2 = fitted_working_sc[sc.size():]
  diffs = fitted_working_sc - target_sc
  rmsd = diffs.rms_length()
  rmsd_s1 = (fitted_working_s1 - sc).rms_length()
  # Find distortion of  coords_in_s1_matching_sc that optimizes fit
  shifts = sc - fitted_working_s1
  local_chain_direction = (fitted_working_s1[2:] - fitted_working_s1[:-2])
  local_chain_direction =flex.vec3_double(
    [local_chain_direction[0]] +
    list(local_chain_direction) + [local_chain_direction[-1]])
  dd = local_chain_direction.norms()
  dd.set_selected(dd <= 0, 1.e-10)
  local_chain_direction =local_chain_direction/dd  # chain direction
  other_chain_direction = get_direction_from_two_sites(fitted_working_s1[0],
    fitted_working_s2[0])
  local_x_direction = flex.vec3_double()
  local_y_direction = flex.vec3_double()
  for i in range(local_chain_direction.size()):
    local_x_direction.append(
      normalize(col(local_chain_direction[i]).cross(other_chain_direction)))
    local_y_direction.append(
      normalize(
       col(local_chain_direction[i]).cross(col(local_x_direction[-1]))))
  # Remove part of shifts going along direction of chain
  # Fit shifts to simple quadratic in x,y,z
  local_diffs = (fitted_working_s1 - sc)
  x_obs = list(local_diffs.dot(local_x_direction))
  y_obs = list(local_diffs.dot(local_y_direction))
  x = list(range(local_diffs.size()))
  x_coeffs = np.polyfit(x, x_obs, 2)
  p = np.poly1d(x_coeffs)
  x_calc = flex.double(list(p(x)))
  y_coeffs = np.polyfit(x, y_obs, 2)
  p = np.poly1d(y_coeffs)
  y_calc = flex.double(list(p(x)))
  # Now offsets in xyz original coordinate system
  offsets = local_x_direction * x_calc + local_y_direction * y_calc
  # Now apply offsets to coordinates
  original_fitted_working_s1 = fitted_working_s1.deep_copy()
  fitted_working_s1 += offsets
  rms_offset = offsets.rms_length()
  # And recalculate results
  fitted_working_sc = fitted_working_s1.deep_copy()
  fitted_working_sc.extend(fitted_working_s2)
  diffs = fitted_working_sc - target_sc
  rmsd = diffs.rms_length()
  rmsd_s1 = (fitted_working_s1 - sc).rms_length()
  diff_at_ends = max(
     min(col(diffs[0]).length(), col(diffs[1]).length()),
     min(col(diffs[sc.size()-1]).length(), col(diffs[sc.size()-2]).length()))
  return group_args(
    group_args_type = 'fit of two sets of coordinates to two strands',
    original_fitted_working_s1 = original_fitted_working_s1,
    fitted_working_s1 = fitted_working_s1,
    fitted_working_s2 = fitted_working_s2,
    rmsd = rmsd,
    n_ca_atoms = fitted_working_s1.size() + fitted_working_s2.size(),
    diff_at_ends = diff_at_ends,
    lsq_fit_obj = new_fit_obj,
    offsets = offsets,
   )
def get_matching_coords(sc, fitted_s1):
  #  for each site in sc, find closest in fitted_s1 (can be dups)
  matching_sites = flex.vec3_double()
  for i in range(sc.size()):
    dist, id1,id2 = sc[i:i+1].min_distance_between_any_pair_with_id(fitted_s1)
    matching_sites.append(fitted_s1[id2])
  return matching_sites
def replace_one_cluster(run_info,
      cluster_info,
      replace_from_pdb,
      sss_info,
      strand_lib,
      residues_on_end,
      log):
    c = cluster_info.clusters[run_info.cluster_id]
    assert run_info.cluster_id == c.cluster_id
    print("\nWorking on cluster ",c.cluster_id,
      c.original_start_res, c.original_end_res, c.known_ss, file = log)
    existing_segment = sss_info.mmm.model().apply_selection_string(
       get_selections([c])[0])
    extended_existing_segment = sss_info.mmm.model().apply_selection_string(
         get_selections([c],residues_on_end = residues_on_end)[0])
    new_model = None
    # Now replace ss with known_ss. Use density and starting coords
    if strand_lib and (c.known_ss == 'S' or \
       c.original_helix_sheet_designation.original_sites_ss == 'S'):
      # Find its neighbors that are strands
      up_to_two_nearby_strands = get_up_to_two_nearby_strands(cluster_info, c)
      if up_to_two_nearby_strands:
        #print_strand_info(c, up_to_two_nearby_strands, log = log)
        other_strand = up_to_two_nearby_strands[0]
        other_c = other_strand.cluster
        # Look through our library
        test_model = replace_with_best_matching_strand_from_pairs(
          sss_info = sss_info,
          strand_lib = strand_lib,
          c = c,
          other_c = other_c,
          log = log)
      if test_model and ((not new_model) or
           test_model.info().score > new_model.info().score):
         new_model = test_model
    if c.original_helix_sheet_designation.original_sites_ss == 'S' and \
       c.known_ss == 'H':
      # Special case for replacing existing strand with a helix
      #  Cannot use the existing tracing as a template
      print("Replacing strand %s:%s with a helix..." %(
         c.original_start_res, c.original_end_res), file = log)
      # Make a temporary directory
      temp_dir = "temp_cluster_%s" %(run_info.cluster_id)
      test_model = replace_strand_with_helix(c, existing_segment,
          extended_existing_segment,
          sss_info,
          temp_dir,
          log = log)
      # Remove temp dir
      delete_directory(temp_dir)
      if test_model and ((not new_model) or
           test_model.info().score > new_model.info().score):
         new_model = test_model
    if not new_model:
      # Replace with segment from pdb if possible
      print("Replacing segment with segment from pdb...", file = log)
      new_model = replace_from_pdb.run(
         existing_segment = existing_segment,
         extended_existing_segment = extended_existing_segment,
         known_ss = c.known_ss,
         mmm = sss_info.mmm)
    return group_args(
      group_args = 'replace_one_segment',
      model = new_model,
      cluster_id = run_info.cluster_id,
    )
class replace_segment(replace_with_segments_from_pdb):
  def __init__(self,args = None, rss = None, log = sys.stdout):
     if args is None:
       args = []
     self.params = self.get_params(args, out = log)
     self.params.extract_segments_from_pdb.extract = None
     self.log = log
     self.helix_lib,self.strand_lib,self.other_lib=self.get_libraries(
           self.params,out=self.log)
  def get_all_first_last_combinations(self,first_res,last_res,
     min_length = 5, max_length = None):
    # return list of combinations to cover the range with various lengths
    # go from long to short
    if not max_length or max_length > last_res - first_res + 1:
      max_length = last_res - first_res + 1
    run_list = []
    for l in range(max_length, min_length - 1, -1):
       i_range = list(range(first_res, last_res + 1 - (l-1)))
       i_range_use = []
       # alternate ends so we cover it as quickly as possible
       first = True
       for j in range(len(i_range)):
         if first:
           ii = i_range[j]
           first = False
         else:
           ii = i_range[-j]
           first = True
         if not ii in i_range_use:  i_range_use.append(ii)
       for i in i_range_use:
          run_list.append(group_args(
            group_args_type = 'first_res last_res combinations',
            first_res = i,
            last_res = i + l - 1,
           ))
    return run_list
  def run(self,
     existing_segment = None,
     extended_existing_segment = None,
     known_ss = None,
     mmm = None):
    # model coming in is model object not model_info
    # we are going to work with extended_existing_segment but eventually only
    #  return the part matching existing_segment
    target_first_resno = get_first_resno(existing_segment.get_hierarchy())
    target_last_resno = get_last_resno(existing_segment.get_hierarchy())
    extended_first_resno = get_first_resno(
      extended_existing_segment.get_hierarchy())
    extended_last_resno = get_last_resno(
      extended_existing_segment.get_hierarchy())
    target_start_point = target_first_resno - extended_first_resno
    # how far in from end
    target_end_point = extended_last_resno - target_last_resno
    print("Target first resno: %s  last:%s  extended first:%s last %s" %(
     target_first_resno, target_last_resno,
      extended_first_resno, extended_last_resno), file = self.log)
    # Try helices and strands separately
    run_types = []
    if known_ss in [None, 'H']:
       run_types.append('H')
    if known_ss in [None, 'S']:
       run_types.append('S')
    #  Note: if run_type differs from original...check it
    self.params.control.verbose=False
    extended_ph = extended_existing_segment.get_hierarchy()
    first_res = target_first_resno
    last_res = target_last_resno
    n = last_res + 1 - first_res
    successful_segment_dict = {}
    covered_dict = {}
    for run_type in run_types:
      successful_segments = []
      successful_segment_dict[run_type] = successful_segments
      covered_dict[run_type] = False
      # Try all possible sub-segments if necessary. Max length is 30 for strand.
      if run_type == 'S':
         max_length = 30
      else:
         max_length = None
      mask = flex.bool(n, False)
      for info in self.get_all_first_last_combinations(first_res,last_res,
          max_length = max_length):
        if covered_dict[run_type]:
          break
        asc1 = extended_ph.atom_selection_cache()
        sel = asc1.selection(string = 'resseq %s:%s' %(
          info.first_res,info.last_res))
        ph = extended_ph.select(sel)
        models = get_and_split_model(ph, get_start_end_length = True)
        rss_model = models[0]
        rss = replacement_segment_summary(rss_model)
        if run_type == 'H':
          rss.model.find_alpha = find_helix(
            params = self.params.alpha,
            model_as_segment=True,
            model=rss.model,verbose=False)
        else:
          rss.model.find_beta= find_beta_strand(
            params = self.params.beta,
            model_as_segment=True,
            model=rss.model,verbose=False)
        self.model_replacement_segment_summaries = [rss]
        # get segments that might replace structure in this model (1 chain)
        replacement_segments=self.get_replacement_segments(
          self.params,model=rss.model,
          helix_lib=self.helix_lib,strand_lib=self.strand_lib,
          other_lib=self.other_lib,out=self.log)
        if replacement_segments:
          for i in range(
              info.first_res-first_res, info.last_res + 1 - first_res):
            mask[i] = True
          successful_segments.append( group_args(
            group_args_type = 'successful segment for %s' %(run_type),
            first_res = info.first_res,
            last_res = info.last_res,
            replacement_segments = replacement_segments))
          if mask.count(True) == mask.size():
            covered_dict[run_type] = True
    best_new_model = None
    for run_type in run_types:
      print("Summary for %s-%s run type %s:  Covered: %s " %(
       first_res,last_res,run_type, covered_dict[run_type]), file = self.log)
      # Merge segments to make single complete segments if possible
      successful_segments = successful_segment_dict[run_type]
      if not successful_segments: continue
      new_ph = merge_replacement_segments(extended_ph,
        extended_first_resno,extended_last_resno, successful_segments)
      if not new_ph: continue
      # refine and trim down
      local_model = mmm.model_from_hierarchy(new_ph, return_as_model=True)
      new_model = match_replacement_to_existing(
        mmm = mmm,
        replacement_model = local_model,
        extended_existing_segment = extended_existing_segment,
        existing_segment = existing_segment,  # what we have
        first_res = first_res,  # first res in existing to replace
        last_res = last_res,   # last res in existing to replace
        log = self.log)
      if not new_model: continue
      if best_new_model is None or \
           new_model.info().score > best_new_model.info().score:
        best_new_model = new_model
    return best_new_model
def match_replacement_to_existing(
        mmm = None,
        replacement_model = None,
        extended_existing_segment = None,
        existing_segment = None,
        first_res = None,
        last_res = None,
        score_with_length = False,
        log = sys.stdout):
      local_mmm = mmm.customized_copy(
      model_dict={'model':replacement_model})
      build = local_mmm.model_building()
      build.set_log(log)
      model = build.model()
      # Figure out crossover points to match first_res and last_res in original
      new_first_res = find_matching_residue_number(
        target_residue_number = first_res,
        fixed_model=existing_segment,
        model = model)
      new_last_res = find_matching_residue_number(
        target_residue_number = last_res,
        fixed_model=existing_segment,
        model = model)
      if new_first_res is None or new_last_res is None or \
          new_last_res <= new_first_res:
        print("Unable to match final model to starting segment (%s:%s)" %(
        new_first_res,new_last_res),
           file = log)
        return # did not work
      print("Refining segment", file = log)
      build.refine(restrain_ends = True,refine_cycles = 0)
      print("Done refining segment", file = log)
      model = build.model()
      mmm.shift_any_model_to_match(model)
      # Cut down ends
      new_model = model.apply_selection_string(
        "resseq %s:%s" %(new_first_res, new_last_res))
      length = new_model.get_hierarchy().overall_counts().n_residues
      if not mmm.map_manager().is_dummy_map_manager():
        cc = mmm.map_model_cc(model = new_model)
      else:
        cc = 1
      if score_with_length:
        score = cc * length**0.5
      else:
        score = cc
      new_model.set_info(
        group_args(
         length = length,
         target_length = last_res + 1 - first_res,
         replaces_first_res = first_res,
         replaces_last_res = last_res,
         cc = cc,
         score = score))
      return new_model
def find_matching_residue_number(
        target_residue_number = None,
        fixed_model = None,
        model = None,
        max_dist = 3):
  ''' find residue in working that is closest to residue in target '''
  target_sites = get_ca_sites(
     select_residue_range(fixed_model.get_hierarchy(),
     target_residue_number,
    target_residue_number))
  working_sites = get_ca_sites(model.get_hierarchy())
  dist, id1,id2 = working_sites.min_distance_between_any_pair_with_id(
     target_sites)
  if dist <= max_dist:
    return get_first_resno(model.get_hierarchy()) + id1
  else:
    return None # nothing is close
def merge_replacement_segments(full_ph,
    first_res,last_res, successful_segments):
  # Just take parts of original and segments
  #  successful_segment: original first_res, last_res, replacement_segments
  working_ph = None
  successful_segments = sorted(successful_segments,
    key = lambda s: s.first_res)
  for s in successful_segments:
    other_ph = s.replacement_segments[0].hierarchy
    assert s.first_res >= first_res
    other_first_res = get_first_resno(other_ph)
    other_last_res = get_last_resno(other_ph)
    if other_first_res > other_last_res:  # not enough
      continue
    if s.first_res == first_res:
      model_list = [model_info(hierarchy = other_ph)]
    else: # figure out how to crossover with what we have
      if not working_ph:  # cross over at first residue present
        working_ph = full_ph # start with what we have
        crossover_info = group_args(
          group_args_type = 'crossover info',
          end_of_first_resseq = s.first_res - 1,
          start_of_second_resseq = other_first_res)
      else: # usual
        crossover_info = get_crossover_point(
           working_ph, other_ph)
      first_ph = select_residue_range(working_ph,
         get_first_resno(working_ph),
         crossover_info.end_of_first_resseq)
      second_ph = select_residue_range(other_ph,
         crossover_info.start_of_second_resseq,
         get_last_resno(other_ph))
      model_list = []
      model_list.append(model_info(hierarchy = first_ph))
      model_list.append(model_info(hierarchy = second_ph))
      # Now just join these two hierarchies
    new_model = merge_hierarchies_from_models(models=model_list,
        resid_offset=None,
        renumber=True,
        first_residue_number=first_res,
        chain_id=get_chain_id(full_ph))
    working_ph = new_model.hierarchy
    assert get_first_resno(working_ph) == first_res
  return working_ph
def get_crossover_point(working_ph, other_ph):
  # Find CA that is closest match between these
  sc_ph = get_ca_sites(working_ph)
  sc_other_ph = get_ca_sites(other_ph)
  assert sc_other_ph.size() == get_last_resno(other_ph)+1-get_first_resno(other_ph)
  dist, id1,id2 = sc_ph.min_distance_between_any_pair_with_id(sc_other_ph)
  return group_args(
    group_args_type = 'crossover point',
    end_of_first_resseq = get_first_resno(working_ph) + id1,
    start_of_second_resseq = min(get_last_resno(other_ph),
       get_first_resno(other_ph) + id2 + 1,))
def select_residue_range(ph, start_resseq, end_resseq, chain_id = None):
  asc1 = ph.atom_selection_cache()
  if chain_id is not None:
    sel = asc1.selection(string = 'resseq %s:%s and chain %s' %(
       start_resseq, end_resseq, chain_id))
  else:  # usual
    sel = asc1.selection(string = 'resseq %s:%s' %(start_resseq, end_resseq))
  return ph.select(sel)
def get_ca_hierarchy(ph):
  asc1 = ph.atom_selection_cache()
  sel = asc1.selection(string = 'name ca')
  return ph.select(sel)
def get_ca_sites(ph):
  asc1 = ph.atom_selection_cache()
  sel = asc1.selection(string = 'name ca')
  return ph.select(sel).atoms().extract_xyz()
def get_ss_dict(sss_info):
  ss_dict = {}
  i = -1
  mmm = sss_info.mmm
  ca_model = mmm.model().apply_selection_string("name ca " )
  selection_list = get_selections(sss_info.cluster_info_list)
  i = -1
  for cluster_info, selection in zip(
       sss_info.cluster_info_list, selection_list):
    i += 1
    info = cluster_info.original_helix_sheet_designation
    ss_dict[i] = info
    if info.original_sites_ss is None: # try to get it
      original_sites = ca_model.apply_selection_string(
         selection).get_sites_cart()
      info.original_sites_ss = choose_helix_strand_from_distances(
        original_sites)
  return ss_dict
def choose_helix_strand_from_distances(original_sites):
  ''' is the original model helix or strand in this location
     Use quick test of distances i -> i+3 and i-> i+4'''
  if original_sites.size() < 3:
    return None  # cannot tell
  elif original_sites.size() < 5:
    dist = (col(original_sites[0]) - col(original_sites[-1])).length()
    n = original_sites.size() - 1
    dist_per_res = dist/max(1,n)
  else:
    ip3 = (original_sites[:-4] - original_sites[3:-1]).norms()
    ip4 = (original_sites[:-4] - original_sites[4:]).norms()
    ip34 = 0.5 * (ip3 + ip4)
    dist_per_res = (ip34.min_max_mean().mean)/3.5
  helix_strand_info = get_p_helix_strand_from_dist_per_res(dist_per_res)
  return helix_strand_info.ss_type
def get_p_helix_strand_from_inter_segment_distance(dist,
     inter_segment_distance_info = None,
     confidence_threshold = 0.9,
     p_zero_helix = 0.5, p_zero_strand = 0.5):
  return get_p_helix_strand_any_targets(dist,
     confidence_threshold = confidence_threshold,
     strand_value = inter_segment_distance_info.mean_sheets,
     strand_value_sd = inter_segment_distance_info.sd_sheets,
     helix_value = inter_segment_distance_info.mean_helix,
     helix_value_sd = inter_segment_distance_info.sd_helix,
     p_zero_helix = p_zero_helix,
     p_zero_strand = p_zero_strand)
def get_p_helix_strand_from_dist_per_res(dist_per_res,
     confidence_threshold = 0.9,
     strand_rise = 3.3, strand_rise_sd = .75,
     helix_rise = 1.54, helix_rise_sd = 0.25,
     p_zero_helix = 0.5, p_zero_strand = 0.5):
  return get_p_helix_strand_any_targets(dist_per_res,
     confidence_threshold = confidence_threshold,
     strand_value = strand_rise,
     strand_value_sd = strand_rise_sd,
     helix_value = helix_rise,
     helix_value_sd = helix_rise_sd,
     p_zero_helix = p_zero_helix,
     p_zero_strand = p_zero_strand)
def get_p_helix_strand_any_targets(dist,
    confidence_threshold = 0.9,
     strand_value = None, strand_value_sd = None,
     helix_value = None, helix_value_sd = None,
     p_zero_helix = 0.5, p_zero_strand = 0.5):
  helix_value_sd = max(1.e-10, helix_value_sd)
  strand_value_sd = max(1.e-10, strand_value_sd)
  p_value_given_helix = math.exp(
      -((dist - helix_value)/helix_value_sd)**2)
  p_value_given_strand = math.exp(
   - ((dist - strand_value)/strand_value_sd)**2)
  p_helix = p_zero_helix * p_value_given_helix / max(1.e-10,
    p_zero_helix * p_value_given_helix + p_zero_strand *p_value_given_strand)
  if p_helix >= confidence_threshold:
    ss_type = 'H'
  elif p_helix <= 1-confidence_threshold:
    ss_type = 'S'
  else:
    ss_type = None
  return group_args(
    group_args_type = 'ss_type',
    p_helix = p_helix,
    p_strand = 1 - p_helix,
    ss_type = ss_type,
   )
def get_dist_of_closest_n_pairs(s1,s2, n_pairs):
  max_dist = 0
  for i in range(n_pairs):
    dist,id1, id2 = s1.min_distance_between_any_pair_with_id(s2)
    max_dist = max(max_dist,dist)
    s1a = s1[:id1]
    s1a.extend(s1[id1+1:])
    s1 = s1a
    s2a = s2[:id2]
    s2a.extend(s2[id2+1:])
    s2 = s2a
  return max_dist
def identify_structure_elements(params, sss_info,
    residues_to_skip_at_ends_of_elements = 2,
    minimum_remaining = 3,
    n_pairs = 2,
    log = sys.stdout):
  # Initial identification from original sites
  ss_dict = get_ss_dict(sss_info)
  # Now more detailed identification
  # Get inter-segment distances
  clusters = sss_info.cluster_info_list
  neighbor_dict = {}
  n = len(clusters)
  cluster_centers = flex.vec3_double()
  total_in_clusters = 0
  for i in range(n):
    c = clusters[i]
    total_in_clusters += c.original_end_res + 1 - c.original_start_res
    cluster_centers.append(c.split_sites.mean())
  print("\nTotal in clusters: %s " %(total_in_clusters), file = log)
  nearest_neighbor_dict = {}
  for i in range(n):
    nearest_neighbor_distances_dict = {}
    neighbor_dict[i] = nearest_neighbor_distances_dict
    c = clusters[i]
    assert c.cluster_id == i
    nearest_neighbor_dict[i] = None
    for j in range(n):
      other_c = clusters[j]
      if j <= i:
        nearest_neighbor_distances_dict[j] = 0
      else:
        maximum_skip_i = max(0,(c.split_sites.size() - minimum_remaining)//2)
        maximum_skip_j = max(0,(
           other_c.split_sites.size() - minimum_remaining)//2)
        skip_i = min(maximum_skip_i, residues_to_skip_at_ends_of_elements)
        skip_j = min(maximum_skip_j, residues_to_skip_at_ends_of_elements)
        remaining_i = c.split_sites.size() - 2 * skip_i
        remaining_j = other_c.split_sites.size() - 2 * skip_j
        if j == i + 1:  # do not include close parts in sequence
          n_skip_i = min(maximum_skip_i,max(skip_i, 3))
          n_skip_j = min(maximum_skip_j,max(skip_j, 3))
          split_sites_i = c.split_sites[skip_i:-max(1,n_skip_i)]
          split_sites_j = other_c.split_sites[n_skip_j:-max(1,skip_j)]
        else: # usual
          split_sites_i = c.split_sites[skip_i:-max(1,skip_i)]
          split_sites_j = other_c.split_sites[skip_j:-max(1,skip_j)]
        dist = get_dist_of_closest_n_pairs(split_sites_i,split_sites_j, n_pairs)
        nearest_neighbor_distances_dict[j] = dist
  all_neighbors_dict = {}
  for i in range(n):
    all_neighbors_dict[i] = []
    for j in range( n):
      if i < j:
        dist = neighbor_dict[i][j]
      elif i > j:
        dist = neighbor_dict[j][i]
      else:
        continue
      new_ga = group_args(
          group_args_type = 'nearest_neighbor',
            cluster_id = i,
            other_cluster_id = j,
            known_ss = None,
            dist = dist,)
      all_neighbors_dict[i].append(new_ga)
      ga = nearest_neighbor_dict[i]
      if ga is None or dist < ga.dist:
        ga = new_ga
        nearest_neighbor_dict[i] = ga
  for i in range(n):
    all_neighbors_dict[i] = sorted(all_neighbors_dict[i],
         key = lambda ga: ga.dist)
  if params.verbose:
    print("\nNeighbors ", file = log)
    for i in range(n):
      c = clusters[i]
      print("%10s" %("%s %s:%s " %(
        c.original_chain_id, c.original_start_res, c.original_end_res)),
           end = "", file = log)
      for j in range(n):
        other_c = clusters[j]
        dist = neighbor_dict[i][j]
        print(" %2s" %(min(99,int(dist))), end = "", file = log)
      print(file = log)
  # Classify segments based on original ss
  classify_segments(clusters, ss_dict, nearest_neighbor_dict)
  # Now get inter-segment info and reclassify
  inter_segment_distance_info = get_inter_segment_distance_info(
     clusters, ss_dict, nearest_neighbor_dict)
  if params.reclassify_based_on_distances:
    classify_segments(clusters, ss_dict, nearest_neighbor_dict,
      inter_segment_distance_info = inter_segment_distance_info)
  if params.create_strand_library:
    library_entries = []
  if params.verbose:
    print("\nNearest neighbors:", file = log)
  for i in range(n):
    c = clusters[i]
    if params.verbose:
      print("%4s %10s" %(c.cluster_id,
       " %s %s:%s " %(
         c.original_chain_id, c.original_start_res, c.original_end_res)),
         end = "", file = log)
      print(" Orig class:%7s (H %2s S %2s) " %(
        ss_dict[i].original_sites_ss,
        ss_dict[i].helix_n,
        ss_dict[i].sheet_n,
       ), end = "", file = log)
    ga = nearest_neighbor_dict[i]
    if ga:
      other_c = clusters[ga.other_cluster_id]
      if params.verbose:
        print("%10s" %("%s %s:%s " %(
          other_c.original_chain_id, other_c.original_start_res,
           other_c.original_end_res)),
            end = "", file = log)
        print("%.2f A " %(ga.dist), end = "", file = log)
        print(" CLASS: %s" %(ga.known_ss), file = log)
      c.add(key = 'known_ss', value = ga.known_ss)
      if params.create_strand_library and \
          c.original_helix_sheet_designation.original_sites_ss == 'S' and \
          c.known_ss == 'S':  # Make sure not helix
        # get pair of strands and save it
        s1 = sss_info.mmm.model().apply_selection_string(
          "(not water and not element ca) and (name ca or name c or name o or name n or name cb) and %s" %(
           get_selections([c])[0]))
        s2 = sss_info.mmm.model().apply_selection_string(
          "(not water and not element ca) and (name ca or name c or name o or name n or name cb) and %s" %(
          get_selections([other_c])[0]))
        s1_ca_sites = s1.apply_selection_string('name ca').get_sites_cart()
        s2_ca_sites = s2.apply_selection_string('name ca').get_sites_cart()
        library_entries.append(group_args(
            s1_ph = s1.get_hierarchy(),
            s2_ph = s2.get_hierarchy(),
            s1_ca_sites = s1_ca_sites,
            s2_ca_sites = s2_ca_sites,
            parallel = is_parallel(s1_ca_sites, s2_ca_sites),
            ))
    else:
      c.add(key = 'known_ss', value = None)
    selection_string = "name ca and %s" %(get_selections([c])[0])
    c.add(key = 'original_ca_sites_cart',
       value = sss_info.mmm.model().apply_selection_string(
         selection_string).get_sites_cart())
  if params.create_strand_library:
    from libtbx import easy_pickle
    easy_pickle.dump(params.create_strand_lib_entry,library_entries)
  return group_args(
    group_args_type = 'cluster info',
    clusters = clusters,
    all_neighbors_dict = all_neighbors_dict,
    cluster_centers = cluster_centers,
    )
def is_parallel(sites_1,sites_2):
  dist, id1,id2 = sites_1[:1].min_distance_between_any_pair_with_id(sites_2)
  dist_b, id2b,id1b = sites_2[:1].min_distance_between_any_pair_with_id(sites_1)
  # closest in sites_2 to n-term end of sites 1 should be n_term of sites 2
  if id2 < sites_2.size()//2 and id1b < sites_1.size()//2:
     return True
  elif id2 > sites_2.size()//2 and id1b > sites_1.size()//2:
     return False
  else:
    return None
def classify_segments(clusters, ss_dict, nearest_neighbor_dict,
   inter_segment_distance_info = None):
  n = len(clusters)
  for i in range(n):
    c = clusters[i]
    ss = ss_dict[i]
    ga = nearest_neighbor_dict[i]
    if ga and not inter_segment_distance_info: # generate it
      if(ss.helix_n, ss.sheet_n).count(0) ==1: # known helix or sheet
        if ss.helix_n:
          ga.add(key = 'known_ss', value = 'H')
        else:
          ga.add(key = 'known_ss', value = 'S')
      else:
        ga.add(key = 'known_ss', value = None)
    elif ga: # reclassify if strong evidence
      if not ga.known_ss: # reclassify for sure
        helix_strand_info = get_p_helix_strand_from_inter_segment_distance(
          ga.dist,
          inter_segment_distance_info = inter_segment_distance_info)
        ga.known_ss = helix_strand_info.ss_type
      else:
        if ga.known_ss == 'H':
          p_zero_helix = 0.95 # high confidence in helix assignments
        elif ga.known_ss == 'S':
          p_zero_helix = 0.2 # lower confidence in strand assignments
        before_known = ga.known_ss
        before_p = p_zero_helix
        helix_strand_info = get_p_helix_strand_from_inter_segment_distance(
          ga.dist,
          inter_segment_distance_info = inter_segment_distance_info,
           p_zero_helix = p_zero_helix, p_zero_strand = 1 - p_zero_helix)
        ga.known_ss = helix_strand_info.ss_type
def get_inter_segment_distance_info(
     clusters, ss_dict, nearest_neighbor_dict,
     expected_mean_sheets = 3.5,
     expected_sd_sheets = 1,
     expected_mean_helix = 8.5,
     expected_sd_helix = 2,
     minimum_residues_to_update = 100,
     ):
  # Get range of ga.dist for known helices or strands
  n = len(clusters)
  distances_helix = []
  distances_sheets = []
  for i in range(n):
    c = clusters[i]
    ss = ss_dict[i]
    ga = nearest_neighbor_dict[i]
    length = c.original_end_res - c.original_start_res + 1
    if ga and ga.known_ss == 'H':
      distances_helix += length * [ga.dist]
    elif ga and ga.known_ss == 'S':
      distances_sheets += length * [ga.dist]
  distances_helix = flex.double(distances_helix)
  distances_sheets = flex.double(distances_sheets)
  if distances_helix.size() >= 3:
    w = distances_helix.size()/(
      distances_helix.size() + max(1, minimum_residues_to_update))
    mean_helix = w * distances_helix.min_max_mean().mean +\
       (1-w) * expected_mean_helix
    sd_helix =  w * distances_helix.standard_deviation_of_the_sample() +\
       (1-w) * expected_sd_helix
  else:
    mean_helix = expected_mean_helix
    sd_helix = expected_sd_helix
  if distances_sheets.size() >= 3:
    w = distances_sheets.size()/(
      distances_sheets.size() + max(1, minimum_residues_to_update))
    mean_sheets = w * distances_sheets.min_max_mean().mean +\
       (1-w) * expected_mean_sheets
    sd_sheets =  w * distances_sheets.standard_deviation_of_the_sample() +\
       (1-w) * expected_sd_sheets
  else:
    mean_sheets = expected_mean_sheets
    sd_sheets = expected_sd_sheets
  return group_args(
    group_args_type = 'inter_segment distance info',
    mean_helix = mean_helix,
    sd_helix = sd_helix,
    mean_sheets = mean_sheets,
    sd_sheets = sd_sheets,
  )
def get_sss_info(params, mmm = None, ca_model = None, model = None,
     log = sys.stdout):
  # Get map and model
  if not mmm:
    mmm = get_map_and_model(params, model = model, log = log)
    if not mmm:
      return None
    write = True
  else:
    write = False
  #print("Finding secondary structure ...", file = log)
  # make duplicate reverse model if include_reverse
  if params.include_reverse:
    add_reverse_to_model(params, mmm)
  if not ca_model:
    ca_model = mmm.model().apply_selection_string('name ca')
  # Identify secondary structure in input model with fss
  ss_info = get_ss_info(ca_model.get_hierarchy())
  if not ss_info:
    #print("Failed to get secondary structure", file = log)
    return None
  cluster_info_list = []
  mmm_id = 0
  for m in split_model(model_info(hierarchy = ca_model.get_hierarchy())):
    mmm_id += 1
    sites_info = get_sites_info(m.hierarchy, ss_info)
    segment_cluster_info = run_one_chain(params, mmm,
        sites_info,
        mmm_id, log = log)
    if segment_cluster_info.cluster_info_list:
      cluster_info_list += segment_cluster_info.cluster_info_list
  # Number clusters
  cluster_id = -1
  for c in cluster_info_list:
    cluster_id += 1
    c.add(key = 'cluster_id', value = cluster_id)
  return group_args(
    group_args_type = 'secondary_structure segments',
    cluster_info_list = cluster_info_list,
    mmm = mmm,
    )
def get_ss_info(ph):
  args = ['tolerant=true']
  try:
    fss=find_secondary_structure(args = args,hierarchy=ph, out=null_out())
  except Exception as e:
    return None # failed
  ann = fss.get_annotation()
  helix_selections = []
  for helix in ann.helices:
    helix_selections.append(group_args(
      group_args_type = 'helix_selection',
      selection_string = helix.as_atom_selections(),
      start_resseq = helix.get_start_resseq_as_int(),
      end_resseq = helix.get_end_resseq_as_int(),)
      )
  sheet_selections = []
  for sheet in ann.sheets:
    for strand in sheet.strands:
      sheet_selections.append(group_args(
        group_args_type = 'sheet_selection',
        selection_string = strand.as_atom_selections(),
        start_resseq = strand.get_start_resseq_as_int(),
        end_resseq = strand.get_end_resseq_as_int(),)
        )
  # see if we can merge any strands
  new_sheet_selections = []
  for i in range(len(sheet_selections)):
    for j in range(i+1, len(sheet_selections)):
      if ranges_overlap(sheet_selections[i],sheet_selections[j]):
        sheet_selections[i].start_resseq = min(
          sheet_selections[i].start_resseq, sheet_selections[j].start_resseq)
        sheet_selections[i].end_resseq = max(
          sheet_selections[i].end_resseq, sheet_selections[j].end_resseq)
        sheet_selections[j] = None
    if sheet_selections[i]:
      new_sheet_selections.append(sheet_selections[i])
  sheet_selections = new_sheet_selections
  ss_info = group_args(
    group_args_type = 'helices and strands selections',
    helix_selections = helix_selections,
    sheet_selections = sheet_selections,
    helix_selection = ann.overall_helix_selection(),
    sheet_selection = ann.overall_sheet_selection(),
    )
  return ss_info
def ranges_overlap(s1,s2):
  if not s1 or not s2:
    return False
  elif s1.end_resseq >= s2.start_resseq and s1.start_resseq<= s2.end_resseq:
    return True
  elif s2.end_resseq >= s1.start_resseq and s2.start_resseq<= s1.end_resseq:
    return True
  else:
    return False
def write_models(params, mmm, cluster_info_list, log = sys.stdout):
  selection_list = get_selections(cluster_info_list)
  selection_string = " or ".join(selection_list)
  sss_model = mmm.model().apply_selection_string(selection_string)
  mmm.add_model_by_id(model_id = 'sss_model', model = sss_model)
  mmm.write_model(model_id = 'sss_model', file_name = 'sss_model.pdb')
  # Now put reduced sites in reduced_sites_model
  reduced_sites_model = mmm.model().apply_selection_string("name ca and (%s)" %(
       selection_string))
  sites_cart = reduced_sites_model.get_sites_cart()
  for cluster_info, selection in zip(cluster_info_list, selection_list):
    sel = reduced_sites_model.selection(selection)
    sites_cart.set_selected(sel,cluster_info.split_sites)
  reduced_sites_model.set_sites_cart(sites_cart)
  mmm.add_model_by_id(model_id = 'reduced_sites_model',
      model = reduced_sites_model)
  mmm.write_model(model_id = 'reduced_sites_model',
       file_name = 'reduced_sites_model.pdb')
def get_neighbor_selections(cluster_info_list, residues_on_end = None):
  selection_list = []
  n = residues_on_end if residues_on_end else 0
  for cluster_info,next_cluster_info in zip(
      cluster_info_list,cluster_info_list[1:]):
    if cluster_info.original_chain_id != next_cluster_info.original_chain_id:
      selection_list.append(None)
    elif next_cluster_info.original_end_res + n < \
         cluster_info.original_start_res - n:
      selection_list.append(None)  # didn't work
    else:
      selection_list.append("(chain %s and resseq %s:%s)" %(
        cluster_info.original_chain_id,
        cluster_info.original_start_res - n,
        next_cluster_info.original_end_res + n))
  return selection_list
def get_selections(cluster_info_list, residues_on_end = None):
  selection_list = []
  n = residues_on_end if residues_on_end else 0
  for cluster_info in cluster_info_list:
    selection_list.append("(chain %s and resseq %s:%s)" %(
      cluster_info.original_chain_id,
      cluster_info.original_start_res - n,
      cluster_info.original_end_res + n))
  return selection_list
def get_sites_info(ph, ss_info):
  # Get sites and known ss info for a segment
  sites_cart = ph.atoms().extract_xyz()
  first_res = get_first_resno(ph)
  last_res = get_last_resno(ph)
  chain_id = get_chain_id(ph)
  helix_selections = []
  sheet_selections = []
  for helix_selection in ss_info.helix_selections:
    if helix_selection.start_resseq>= first_res and \
        helix_selection.end_resseq<= last_res:
      hs = deepcopy(helix_selection)
      hs.add(key = 'first_i', value = hs.start_resseq - first_res)
      hs.add(key = 'last_i', value = hs.end_resseq - first_res)
      helix_selections.append(hs)
  for sheet_selection in ss_info.sheet_selections:
    if sheet_selection.start_resseq>= first_res and \
        sheet_selection.end_resseq<= last_res:
      hs = deepcopy(sheet_selection)
      hs.add(key = 'first_i', value = hs.start_resseq - first_res)
      hs.add(key = 'last_i', value = hs.end_resseq - first_res)
      sheet_selections.append(hs)
  if ss_info.helix_selection:
    asc1 = ph.atom_selection_cache()
    sel_helices = asc1.selection(string = ss_info.helix_selection)
  else:
    sel_helices = flex.bool(sites_cart.size(), False)
  if ss_info.sheet_selection:
    asc1 = ph.atom_selection_cache()
    sel_sheets = asc1.selection(string = ss_info.sheet_selection)
  else:
    sel_sheets = flex.bool(sites_cart.size(), False)
  sel_ss = (sel_sheets | sel_helices)
  original_id_dict = {}
  for i in range(sites_cart.size()):
    original_id_dict[i] = group_args(
     group_args_type = 'original_id dict',
     chain_id = chain_id,
     resno = first_res + i)
  return group_args(
   group_args_type = 'sites info',
   sites_cart = sites_cart,
   original_id_dict = original_id_dict,
   chain_id = chain_id,
   start_res = first_res,
   end_res = last_res,
   helix_selections = helix_selections,
   sheet_selections = sheet_selections,
   sel_sheets = sel_sheets,  # bool ID of sheet residues
   sel_helices = sel_helices,  # bool ID of helix residues
   sel_ss = sel_ss,  # bool id if helix or sheet resid
    )
def run_one_chain(params, mmm,  sites_info,
       mmm_id, log = sys.stdout):
  #print("Getting reduced sites for one chain", file = log)
  sites_cart = sites_info.sites_cart
  original_id_dict = sites_info.original_id_dict
  new_cluster_info_list = []
  result = group_args(
      group_args_type = 'split reduced sites for one chain',
      cluster_info_list = new_cluster_info_list,
      reduced_sites = None)
  # Set up reduced model (grouped CA only)
  reduced_sites_info = get_reduced_sites(params,
     sites_cart,
     original_id_dict,
     log = log)
  reduced_sites = reduced_sites_info.reduced_sites
  reduced_sites_id_dict = reduced_sites_info.original_id_dict
  result.reduced_sites = reduced_sites
  if not reduced_sites:
    #print("No reduced sites for set %s" %(mmm_id), file = log)
    return result
  #print("Getting split reduced sites for one chain", file = log)
  # Split up chain at all turns. Keep ss already identified
  split_reduced_sites_info = get_split_reduced_sites(params,
     reduced_sites,
     sites_cart,
     sites_info,
     log = log)
  if not split_reduced_sites_info:
    return result
  ii = 0
  last_end= None
  for one_split_info in split_reduced_sites_info.split_info_list:
    if last_end is not None:
      assert one_split_info.cluster_info.start > last_end
    last_end = one_split_info.cluster_info.end
    ii += 1
    split_sites = one_split_info.split_sites
    start_point = one_split_info.cluster_info.start
    end_point = one_split_info.cluster_info.end
    resseq_list = []
    for i in range(split_sites.size()):
      j = i + start_point
      resseq_list.append(reduced_sites_id_dict[j].resno)
    helix_sheet_designation = get_helix_sheet_designation(sites_info,
       start_point, end_point)
    new_cluster_info_list.append(group_args(
        group_args_type = 'cluster',
        mmm_id = mmm_id,
        split_id = ii,
        reduced_sites_start = start_point,
        reduced_sites_end = end_point,
        original_start_res = reduced_sites_id_dict[start_point].resno,
        original_end_res = reduced_sites_id_dict[end_point].resno,
        original_chain_id = reduced_sites_id_dict[start_point].chain_id,
        original_helix_sheet_designation = helix_sheet_designation,
        split_sites = split_sites,
        center = split_sites.mean(),
        split_sites_resseq_list = resseq_list,
        replacement_model = None))
  #print("Done getting split reduced sites for one chain", file = log)
  return result
def get_helix_sheet_designation(sites_info,start_point,end_point):
  # Note original helix/sheet designation
  # start_point and end_point are indices, not residue numbers
  original_sites_ss = None
  helix_n = 0
  for helix_selection in sites_info.helix_selections:
     if helix_selection.first_i <= end_point and \
        helix_selection.last_i >= start_point:
       i = max(helix_selection.first_i, start_point)
       j = min(helix_selection.last_i, end_point)
       helix_n += max(0,j + 1 - i)
  sheet_n = 0
  for sheet_selection in sites_info.sheet_selections:
     if sheet_selection.first_i <= end_point and \
        sheet_selection.last_i >= start_point:
       i = max(sheet_selection.first_i, start_point)
       j = min(sheet_selection.last_i, end_point)
       sheet_n += max(0,j + 1 - i)
  if helix_n and not sheet_n:
    original_sites_ss = 'H'
  elif sheet_n and not helix_n:
    original_sites_ss = 'S'
  else:
    original_sites_ss = None
  return group_args(
    group_args_type = 'helix_sheet designation',
    original_sites_ss = original_sites_ss,
    helix_n = helix_n,
    sheet_n = sheet_n)
def get_split_reduced_sites(params, reduced_sites, ca_sites, sites_info,
    log = sys.stdout):
  # Split up this set of reduced sites at places where it turns.
  # Try to keep sites marked as sel_ss in sites_info together
  # Get directions at each point in reduced model
  direction_vectors = get_direction_vectors(params, reduced_sites, log = log)
  # Find clusters of direction_vectors that are similar
  # Get highest local clusters
  if direction_vectors.size() < 2:
    return None
  smoothed_direction_vectors = smooth_vectors(direction_vectors, window = 10)
  smoothed_magnitudes = smoothed_direction_vectors.norms()
  smoothed_magnitudes.set_selected(smoothed_magnitudes <= 0, 1.e-10)
  smoothed_direction_vectors = smoothed_direction_vectors/smoothed_magnitudes
  #print("Finding clusters based on secondary structure", file = log)
  used_list = []
  # First get clusters based on existing identified secondary structure
  cluster_list_info = find_best_clusters(params, direction_vectors,
    used_list,
    sites_info,
    require_ss_at_center = True)
  cluster_list = cluster_list_info.cluster_list
  used_list = cluster_list_info.used_list
  # Sort clusters on start position
  cluster_list = sorted(cluster_list, key = lambda c: c.start)
  #print("Filling in clusters ...", file = log)
  # Then fill in with whatever we can find
  cluster_list_info = find_best_clusters(params, direction_vectors,
    used_list,
    sites_info,
    require_ss_at_center = False,
    log = log)
  cluster_list += cluster_list_info.cluster_list
  # Sort clusters on start position
  cluster_list = sorted(cluster_list, key = lambda c: c.start)
  #print("Removing dangling ends ...", file = log)
  # Get rid of dangling ends
  cluster_list = split_and_trim_cluster_ends(
     params, reduced_sites, direction_vectors, ca_sites,
     cluster_list, sites_info)
  # Sort clusters on start position
  cluster_list = sorted(cluster_list, key = lambda c: c.start)
  # Make a list of results
  split_info_list = []
  for c in cluster_list:
    if c.end - c.start + 1 >= params.reduce_min_length:
      split_info_list.append(group_args(
          group_args_type = 'one split reduced sites info',
          split_sites = reduced_sites[c.start:c.end+1],
          cluster_info = c))
      assert reduced_sites[c.start:c.end+1].size() == c.end - c.start + 1
  split_reduced_sites_info = group_args(
    group_args_type = 'split reduced sites info',
    split_info_list = split_info_list)
  return split_reduced_sites_info
def find_best_clusters(params, direction_vectors, used_list,
   sites_info,
   require_ss_at_center = None,
   keep_identified_helices_as_is = True,
   log = sys.stdout):
  have_helices_or_sheets = sites_info.helix_selections != [] or \
     sites_info.sheet_selections != []
  if require_ss_at_center and not have_helices_or_sheets: # nothing to do
    return group_args(
      group_arg_type = 'cluster_list',
      cluster_list = [],
      used_list = used_list)
  cluster_list = []
  for cy in range(direction_vectors.size()):
    end_number = -1
    n_tot = direction_vectors.size()
    nproc = max(1,
       min(params.nproc,n_tot//params.minimum_units_per_processor))
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
    kw_dict={
      'params':  params,
      'direction_vectors':  direction_vectors,
      'used_list':  used_list,
      'sites_info':  sites_info,
      'require_ss_at_center':  require_ss_at_center,
      'have_helices_or_sheets':  have_helices_or_sheets,
      'keep_identified_helices_as_is':  keep_identified_helices_as_is,
     }
    runs_carried_out = run_jobs_with_large_fixed_objects(
      nproc = params.nproc,
      verbose = params.verbose,
      kw_dict = kw_dict,
      run_info_list = runs_to_carry_out,
      job_to_run = group_of_get_one_local_cluster,
      log = log)
    one_cluster_list = []
    for run_info in runs_carried_out:
      result = run_info.result
      if result and result.cluster_list:
        for cluster in result.cluster_list:
           one_cluster_list.append(cluster)
    if not one_cluster_list:
      break
    one_cluster_list = sorted(one_cluster_list,
       key = lambda cluster: cluster.end+1 - cluster.start, reverse = True)
    last_cluster = None
    unique_cluster_list = []
    for c in one_cluster_list:
      if not last_cluster or \
           c.start != last_cluster.start or c.end != last_cluster.end:
        unique_cluster_list.append(c)
        last_cluster = c
    one_cluster_list = unique_cluster_list
    # Now take all clusters that do not overlap
    ok = True
    ii = 0
    for c in one_cluster_list:
      ii += 1
      for kk in range(c.start, c.end + 1):
        if kk in used_list:  # cannot use this range
          ok = False
          break
      if ok:
        cluster_list.append(c)
        used_list += list(range(c.start, c.end+1))
      else:
        break  # need to get a new list
  return group_args(
    group_arg_type = 'cluster_list',
    cluster_list = cluster_list,
    used_list = used_list)
def group_of_get_one_local_cluster(run_info,
    params, direction_vectors, used_list,
        sites_info,
        require_ss_at_center, have_helices_or_sheets,
        keep_identified_helices_as_is,
        log = sys.stdout):
  cluster_list = []
  for i in range(run_info.start_number, run_info.end_number + 1):
    if i in used_list: continue
    cluster = get_one_local_cluster(
       i, params, direction_vectors, used_list,
        sites_info,
        require_ss_at_center, have_helices_or_sheets,
        keep_identified_helices_as_is,
        log = log)
    if cluster:
      cluster_list.append(cluster)
  return group_args(
    group_args_type = 'one local cluster',
    cluster_list = cluster_list)
def get_one_local_cluster(i, params, direction_vectors, used_list,
        sites_info,
        require_ss_at_center, have_helices_or_sheets,
        keep_identified_helices_as_is,
        log = sys.stdout):
  minimum_range = None
  blocked_indices = []
  if have_helices_or_sheets:
    indices = flex.int(list(range(sites_info.sel_ss.size())))
    helix_range_covered = get_range_covered_info(
      i,sites_info.helix_selections)  # range in indices
    sheet_range_covered = get_range_covered_info(
      i,sites_info.sheet_selections)
    if helix_range_covered:# this is helix. keep only identified
      minimum_range = helix_range_covered
      if keep_identified_helices_as_is:
        range_covered_info = minimum_range
        blocked_indices = list(indices[:range_covered_info.first_i]) + \
         list(indices[range_covered_info.last_i+1:])
        assert len(blocked_indices) == indices.size() - \
          (range_covered_info.last_i+1 - range_covered_info.first_i)
      else:
        blocked_indices = list(indices.select(sites_info.sel_sheets))
    elif sheet_range_covered: # this is strand. block out helices
      minimum_range = sheet_range_covered
      blocked_indices = list(indices.select(sites_info.sel_helices))
  cluster = get_one_cluster(params, i, direction_vectors,
     used_list + blocked_indices,
     minimum_range = minimum_range,)
  return cluster
def contains_complete_region(cluster, i, minimum_range):
  ''' does this cluster contain all the contiguous sites marked'''
  if cluster.start <= minimum_range.first_i and \
       cluster.end >= minimum_range.last_i:
    return True
  else:
    return False
def get_range_covered_info( i, selections):
  first_i = None
  last_i = None
  for selection in selections:
    if i >= selection.first_i and i <= selection.last_i:
      return group_args(
        group_args_type = 'range covered',
        first_i = selection.first_i,
        last_i = selection.last_i,
        )
  return None
def get_one_cluster(params, i, direction_vectors, used_list,
    minimum_range = None):
  # Try to make a cluster around i:
  smoothed_direction_vectors = smooth_vectors(direction_vectors, window = 10)
  smoothed_magnitudes = smoothed_direction_vectors.norms()
  smoothed_direction_vectors = smoothed_direction_vectors/smoothed_magnitudes
  target_dir = col(direction_vectors[i])
  if minimum_range and minimum_range.last_i is not None and \
        minimum_range.first_i is not None:
    last_i = minimum_range.last_i
    first_i = minimum_range.first_i
    assert last_i >= i
    assert first_i <= i
    ok = True
    for kk in range(first_i, last_i + 1):
      if kk in used_list:  # cannot use this range
        ok = False
        break
    if not ok:
      last_i = i
      first_i = i
      if i in used_list:
        return None # did not work
  else:
    last_i = i
    first_i = i
  done_forward = False
  for k in range(1,direction_vectors.size()-i):
    ii = i + k
    if not done_forward:
      if not ii in used_list and \
        target_dir.dot(col(direction_vectors[ii])) > params.reduce_dot:
        last_i = max(last_i,ii)
      else:
        done_forward = True
    if done_forward:
      break
  if last_i - i < params.reduce_min_length/2:
    return None
  done_reverse = False
  for k in range(1,i+1):
    jj = i - k
    assert jj >=0
    if not done_reverse:
      if not jj in used_list and \
        target_dir.dot (col(direction_vectors[jj])) > params.reduce_dot:
        first_i = min(first_i,jj)
      else:
        done_reverse= True
    if done_reverse:
      break
  if i - first_i < params.reduce_min_length/2:
    return None
  if last_i - first_i + 1 >= params.reduce_min_length:
    return group_args(
          group_args_type = 'cluster',
          start = first_i,
          end = last_i,
          minimum_range = minimum_range)
  else:
    return None
def split_and_trim_cluster_ends(params, reduced_sites,
     direction_vectors, ca_sites, cluster_list, sites_info):
  # Trim ends that deviate more than reduce_tol from smoothed predicted location
  new_clusters = []
  for c in cluster_list:
    trim_one_cluster_end(params, reduced_sites, direction_vectors, c)
    if c.start is not None and c.end is not None:
      new_clusters.append(c)
  cluster_list = new_clusters
  # Now split any clusters that have a sharp corner
  new_clusters = []
  for c in cluster_list:
    split_clusters = split_one_cluster(params, reduced_sites,
       ca_sites,
       direction_vectors, c, sites_info,)
    new_clusters += split_clusters
  clusters = new_clusters
  return clusters
def split_one_cluster(params, reduced_sites, ca_sites,
       direction_vectors, c, sites_info,maximum_strand_length = None):
  new_clusters = [c]
  #  See if sites go in one direction and then suddenly change to a new one
  local_direction_vectors = direction_vectors[c.start: c.end + 1]
  if local_direction_vectors.size() < 4:
    return new_clusters # nothing to do
  n = local_direction_vectors.size()
  window = n//4
  low_i = None
  low_dot = None
  original_helix_sheet_designation = get_helix_sheet_designation(sites_info,
       c.start,c.end)
  if original_helix_sheet_designation.original_sites_ss == 'S':
    is_straight_info = is_straight(ca_sites[c.start: c.end + 1],
      maximum_strand_length =maximum_strand_length)
  else:
    is_straight_info = None
  for i in range(local_direction_vectors.size() - 2* window):
    if c.minimum_range and c.minimum_range.first_i is not None \
        and c.minimum_range.last_i is not None and \
        i >= c.minimum_range.first_i and i <= c.minimum_range.last_i:
      continue  # don't break inside identified ss
    start_dir = col(local_direction_vectors[i:min(n -1, i+window)].mean())
    start_dir = start_dir/max(1.e-10,start_dir.length())
    end_dir = col(local_direction_vectors[i+window:min(n-1,i+2*window)].mean())
    end_dir = end_dir/max(1.e-10,end_dir.length())
    dd = start_dir.dot(end_dir)
    if low_dot is None or dd < low_dot:
      low_dot = dd
      low_i = window + i
  if (is_straight_info is not None and not is_straight_info.is_straight) or (
     low_dot is not None and low_dot < params.similar_dot):
    # Break it here
    if is_straight_info is not None and not is_straight_info.is_straight:
      low_i = is_straight_info.worst_middle
    new_clusters = []
    new_c = group_args(
          group_args_type = 'cluster',
          start = c.start,
          end = c.start + low_i - 1,
         )
    if new_c.end + 1 - new_c.start >=  params.reduce_min_length:
      new_clusters.append(new_c)
    new_c = group_args(
          group_args_type = 'cluster',
          start = c.start + low_i + 1,
          end = c.end,
         )
    if new_c.end + 1 - new_c.start >=  params.reduce_min_length:
      new_clusters.append(new_c)
  return new_clusters
def trim_one_cluster_end(params, reduced_sites, direction_vectors, cluster,
    sd_ratio = 2.0):
  # Trim ends where directions start to vary a lot
  # Don't trim inside minimum_range
  local_direction_vectors = direction_vectors[cluster.start: cluster.end + 1]
  initial_start = cluster.start
  initial_end = cluster.end
  dot_values = flex.double()
  # get variation in middle part
  n_middle = min(local_direction_vectors.size(),
    max (params.reduce_min_length, local_direction_vectors.size()//2))
  n_start = (local_direction_vectors.size() - n_middle)//2
  n_end = min(local_direction_vectors.size()-1, n_start + n_middle)
  middle_part = local_direction_vectors[n_start:n_end + 1]
  middle_direction = col(
     middle_part.mean())/max(1.e-10,col(middle_part.mean()).length())
  for i in range(local_direction_vectors.size()):
    dot_values.append(middle_direction.dot(col(local_direction_vectors[i])))
  dot_values_middle = dot_values[n_start:n_end + 1]
  mean_value = dot_values_middle.min_max_mean().mean
  sd_value = dot_values_middle.standard_deviation_of_the_sample()
  # Now trim in where value > 3 * sd_value .
  i_start = cluster.start
  for i in range(n_start):
    if dot_values[i] < mean_value - sd_ratio * sd_value:
      i_start = cluster.start + i
  i_end = cluster.end
  for i in  range(dot_values.size()-1, n_end-1, -1):
    if dot_values[i] < mean_value - sd_ratio * sd_value:
      i_end = cluster.start + i
  cluster.start = i_start
  cluster.end = i_end
  if cluster.minimum_range and cluster.minimum_range.first_i is not None \
        and cluster.minimum_range.last_i is not None:
    cluster.start = max(initial_start,
      min(cluster.start,cluster.minimum_range.first_i))
    cluster.end = min(initial_end,
        max(cluster.end,cluster.minimum_range.last_i))
  assert cluster.start is not None
  assert cluster.end is not None
  if cluster.end + 1 - cluster.start < params.reduce_min_length:
    cluster.start = None
    cluster.end= None
def get_clusters(params, ok_list):
  cluster_list = []
  current_group = None
  for i in range(len(ok_list)):
    if ok_list[i]:
      if current_group is None:
        current_group = group_args(
          group_args_type = 'cluster',
          start = i,
          end = i,
         )
        cluster_list.append(current_group)
      else:
        current_group.end = i
    else:
      current_group = None
  new_cluster_list = []
  for c in cluster_list:
    if c.end - c.start + 1 >= params.reduce_min_length:
      new_cluster_list.append(c)
  return new_cluster_list
def get_direction_vectors(params, reduced_sites, log = sys.stdout):
  direction_vectors = flex.vec3_double()
  ca_sites = reduced_sites
  for i in range(ca_sites.size()):
    low_i = max(0, i - params.reduce_window)
    high_i = min(ca_sites.size() -1 , i + params.reduce_window)
    dd = col(ca_sites[high_i]) - col(ca_sites[low_i])
    dd = dd/max(dd.length(), 1.e-10)
    direction_vectors.append(dd)
  return direction_vectors
def smooth_vectors(direction_vectors, window = 10):
  new_dv = flex.vec3_double()
  for i in range(direction_vectors.size()):
    i_low = max(0, i - window//2 )
    i_high = min(direction_vectors.size() - 1, i + window//2)
    new_dv.append(direction_vectors[i_low:i_high + 1].mean())
  return new_dv
def smooth_values(dot_values, window = 10):
  new_dv = flex.double()
  for i in range(dot_values.size()):
    i_low = max(0, i - window//2 )
    i_high = min(dot_values.size() - 1, i + window//2)
    new_dv.append(dot_values[i_low:i_high + 1].min_max_mean().mean)
  return new_dv
def get_reduced_sites(params, ca_sites, original_id_dict, log = sys.stdout):
  reduced_sites = flex.vec3_double()
  for i in range(ca_sites.size()):
    reduced_site = get_reduced_site(params, ca_sites,i,)
    reduced_sites.append(reduced_site)
  # original_id_dict is still the same here
  return group_args(
    group_args_type = 'reduced sites',
    reduced_sites = reduced_sites,
    original_id_dict = original_id_dict)
def get_reduced_site(params, ca_sites,i):
  '''  Get one reduced site.  Average coords over +/- params.reduce_window '''
  averages = flex.vec3_double()
  for offset in [-1,0,1]:
    if i + offset - params.reduce_window < 0: continue
    if i + offset + params.reduce_window >= ca_sites.size(): continue
    for delta in [-params.reduce_window,params.reduce_window]:
      averages.append(ca_sites[i+offset+delta])
  if averages.size() > 0:
    return averages.mean()
  else:
    return ca_sites[i]
def get_map_and_model(params,
   model = None,
   log = sys.stdout):
  from iotbx.data_manager import DataManager
  dm = DataManager()
  dm.set_overwrite(True)
  if params.create_strand_library or params.ignore_symmetry_conflicts:
    ignore_symmetry_conflicts = True
  else:
    ignore_symmetry_conflicts = False
  if params.map_file_name: # read in map and model
    mmm = dm.get_map_model_manager(map_files = params.map_file_name,
      model_file = params.model_file_name,
      ignore_symmetry_conflicts = ignore_symmetry_conflicts)
    if model: # use supplied model
      model.set_crystal_symmetry(mmm.unit_cell_crystal_symmetry())
      model.set_shift_cart((0,0,0))
      mmm.add_model_by_id(model = model, model_id = 'model')
      if hasattr(model.info(),'pdb_id'):
        file_name = "%s_%s" %(model.info().pdb_id,model.info().chain_id)
      else:
        file_name = 'supplied model'
    else:
      model = mmm.model()
      file_name = params.model_file_name
  else:
    if model:
      file_name = 'supplied model'
    else:
      model = dm.get_model(params.model_file_name)
      file_name = params.model_file_name
    mmm = model.as_map_model_manager()
  if params.map_file_name:
    print("\nRead map and model from %s and %s" %(params.map_file_name,
     file_name),
     file = log)
  else:
    print("\nRead model from %s" %(file_name), file = log)
  if not model.get_hierarchy() or not get_chain_id(model.get_hierarchy()):
    return None
  if not get_chain_id(model.get_hierarchy()).replace(" ",""):
    set_chain_id(model.get_hierarchy(),'A')
  model = clean_model(model)
  mmm.set_model(model)
  if params.resolution:
    mmm.set_resolution(params.resolution)
  return mmm
def clean_model(model):
  model.remove_alternative_conformations(
      always_keep_one_conformer=True)
  model.convert_to_isotropic()
  model = model.apply_selection_string("protein and (not water) and not element ca")
  return model
def get_ssm_params(shift_field_distance = None,
    max_shift_field_ca_ca_deviation = None,
    shift_field_interval = None,
    starting_distance_cutoff = None,
    delta = None,
    max_models_to_fetch = None,
    local_pdb_dir = None,
    trim = None,
    superpose_brute_force = None,
    score_superpose_brute_force_with_cc = None,
    ok_brute_force_cc = None,
    quick = None):
  if delta is None: delta = 10 # overall distance cutoff
  params = group_args(
      group_args_type = 'parameters',
      nproc = 1,
      quick = quick,
      write_files = True,
      output_file_name = None,
      model_file_name = None,
      map_file_name = None,
      other_model_file_name =  None,
      reduce_window = 2, # range for getting reduced coordinates,
      reduce_dot = 0.5, # angle for breaking up reduced coordinates,
      very_similar_dot = 0.9, # angle for breaking up reduced coordinates,
      similar_dot = 0.7, # angle for breaking up reduced coordinates,
      reduce_min_length = 4, # minimum length of a reduced segment
      reduce_tol = 2, # max distance of reduced points from smoothed line
      reduce_min_separation = 5, # min residue separation of clusters
      create_strand_library = False, # create library
      create_strand_lib_entry = None, # create a strand library entry with this name
      strand_library_list = None,
      ignore_symmetry_conflicts = True,
      use_fragment_search_in_find_loops = True,
      min_overlap_fraction = 0.8,
      require_complete_model = True, # XXX check False
      max_loops_to_refine = 15,
      max_final_triples = 10,
      refine_initial_fragment_search = False,
      refine_fragment_search_at_end = False,
      quick_refine = False,
      require_sequential = True,
      maximum_passes = 10,
      cluster_window = 7,  # Vary this
      matching_window = 10,
      matching_min_residues = 6,
      matching_density_sigma= 1, # keep sites smoothed density_sigma of typical
      minimum_good_triples = 10, # if we have this many, go on Vary this
      good_triple_pairs = 10, # a good triple has 10 pairs
      minimum_hits_to_keep = 300,
      minimum_matches = 4,
      minimum_matching_fraction = 0.7,
      max_tries = 2,
      delta = delta,  # tolerance
      tolerance_ratio_min = 0.25 if starting_distance_cutoff is None else \
         starting_distance_cutoff/delta,
      tolerance_ratio_tries = 2,
      min_sequential_fraction = 0.1,
      target_sequential_fraction = 0.2,
      max_shift_field_ca_ca_deviation = max_shift_field_ca_ca_deviation if \
        max_shift_field_ca_ca_deviation is not None else 2.75,
      max_non_superposing_residues = 50, # how many residues away we include
      use_density_to_score = True, # use density if available
      max_gap = 100,
      dist_max = 60,
      make_database = False, # requires database_file
      make_database_from_pdb_include_list = True,
      data_dir = None,
      database_file = None,
      segment_database_file = None,
      use_original_ss_for_unknown_in_db = False,
      close_distance = 3 if starting_distance_cutoff is None else starting_distance_cutoff * 3/2.5,
      database_cluster_window = 7, # minimum of 3
      include_nearby_indices = True, # look nearby in target structure
      include_nearby_indices_n = 1, # vary 1 of 6 indices (choices are 1 or 3)
      match_distance_low = 2,
      match_distance_high = 5,
      score_distance_low = 1.0,
      score_distance_high = 0.5,
      models_to_keep_in_fragment_search = 1,
      optimization_iterations = 5,
      shift_field_distance = shift_field_distance if shift_field_distance is \
        not None else 10,
      shift_field_interval = shift_field_interval if shift_field_interval is \
        not None else 1,
      max_models_to_fetch = max_models_to_fetch,
      scale_offset = 0.1,
      target_loops_to_find = 2,
      target_loop_rmsd = 2,
      max_loops_to_try = 20,
      max_loop_delta = 3,
      min_abs_dot = 0.80,
      minimum_units_per_processor = 100, # don't use lots of processors for small jobs
      comparison_pdb_file_names = None, #"standard_model.pdb m1b_7179.pdb ".split(),
      verbose = False,
      pdb_exclude_list = None,
      pdb_include_list = None,
      reclassify_based_on_distances = None, # XXX vary this
      is_sequence_neighbors_search = False, # used internally
      refine_cycles = 3,
      include_reverse = False,  # XXX vary this one
      resolution = 3, # best to set working resolution directly
      max_change_in_residues = 5,
      max_triples_for_neighbors = 3,
      search_sequence_neighbors = 100, # search this many seq neighbors for each triple after replace with ssm
      morph = True, #  Apply distortion field after ssm
      find_ssm_matching_segments = True, # replace with ssm
      find_loops = True,  # replace loops
      replace_structure_elements = True, # replace segments
      replace_segments_in_model = True, # replace everything as specified above
      dump_replacement_segments = True,
      read_replacement_segments = False,
      segment_info_file = 'segments.pkl',
      return_best_triples = False, # Used internally
      splice_directly = True, # splice directly
      merge_loops_in_splice_directly = True, # merge loops in splice directly
      max_crossover_distance = 3, # crossover to template
      superpose_model_on_other = False, #usual is superpose other_model on model
      trim = trim if trim is not None else True, # trim model to match target
      superpose_brute_force = superpose_brute_force if \
         superpose_brute_force is not None else False, # use brute force method
      score_superpose_brute_force_with_cc = \
           score_superpose_brute_force_with_cc if \
         score_superpose_brute_force_with_cc \
             is not None else False, # use brute force method
      max_brute_force = 20,
      ok_brute_force_cc = ok_brute_force_cc if \
        ok_brute_force_cc is not None else 0.25,  # ok cc
      local_pdb_dir = local_pdb_dir, # local copy of PDB
      )
  return params
def get_segment_lib_dir():
  import libtbx.load_env
  return libtbx.env.find_in_repositories(
       relative_path=os.path.join("chem_data","segment_lib"),
       test=os.path.isdir)
def set_pdb_100(params):
  segment_lib_dir = get_segment_lib_dir()
  params.data_dir = os.path.join(segment_lib_dir, 'pdb100_fragment_db')
  params.database_file = 'fragment_db_pdb100.pkl.gz'
  params.segment_database_file = 'segment_db_pdb100.pkl.gz'
  return params
def set_top2018_filtered(params):
  segment_lib_dir = get_segment_lib_dir()
  if not segment_lib_dir:
     return params # don't set it
  params.data_dir = os.path.join(segment_lib_dir, 'pdb2018_filtered')
  params.database_file = 'fragment_db_top2018.pkl.gz'
  params.segment_database_file = 'segment_db.pkl.gz'
  return params
if __name__=="__main__":
  args = sys.argv[1:]
  params = get_ssm_params()
  # Set some params from args
  if "pdb100" in args:
    params = set_pdb_100(params)
    print("Using pdb 100 database")
    args.remove("pdb100")
  else:
    params = set_top2018_filtered(params)
    print("Using top 2018 filtered database")
  if "database" in args:
    params.make_database = True
    params.make_database_from_pdb_include_list = False
    for arg in args:
      if arg.startswith("nproc="):
        params.nproc = int(arg.replace("nproc=",""))
        print("Running with nproc=%s" %(params.nproc))
    run(params)
  elif args:
    params_sav = deepcopy(params)
    for file_name in args:
      params = deepcopy(params_sav)
      params.model_file_name = file_name
      print("\n========== RUNNING WITH %s ===============" %(file_name))
      run(params)
  elif params.strand_library_list and params.create_strand_library:
    for x in params.strand_library_list:
      params.model_file_name = x
      params.create_strand_lib_entry = x.replace(".pdb",".pkl")
      run(params)
  else:
    run(params)
