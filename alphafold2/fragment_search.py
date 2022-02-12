from __future__ import division, print_function
import sys, os
from copy import deepcopy
from libtbx import group_args
from libtbx.utils import null_out
from scitbx.matrix import col
from scitbx.array_family import flex
from phenix.autosol.get_pdb_inp import get_pdb_hierarchy
import random
from libtbx.easy_mp import run_jobs_with_large_fixed_objects
import numpy as np
from phenix.model_building.compact_hierarchy import compact_hierarchy
from libtbx import easy_pickle
from mmtbx.secondary_structure.find_ss_from_ca import \
     get_first_resno,get_last_resno, get_chain_id, set_chain_id, split_model, \
     model_info, find_secondary_structure, find_helix, find_beta_strand, \
     merge_hierarchies_from_models
def run_distort(params, log = sys.stdout):
  #  Find replacement segments from params.model_file to replace
  #    parts of model in params.match_model_file
  #  Output a new model that is composed of pieces from params.model_file
  #  Allow distortion of model_file pieces
  print("\nGet replacement segments from similar structures\n", file = log)
  # Get models
  model_info = get_models(params, log =log)
  # Find segments in model to match the match_model and distort them to match
  replacement_info = get_replacements(params, model_info, log = log)
  for replacement in replacement_info.replacement_info_list:
    segment = replacement.replacement_segment
    first_res = replacement.replaces_start_resno
    file_name = "replaces_%s-%s.pdb" %(
        first_res,
        first_res + segment.get_hierarchy().overall_counts().n_residues - 1)
    model_info.dm.write_model_file(segment, file_name)
    print("Wrote replacement model to %s" %(file_name), file = log)
def get_replacements(params, model_info, log = sys.stdout):
  # Find replacements for model_info.match_model from the pool in
  #    model_info.model
  #  Split model by segments and work with each separately
  model_list = split_one_model(params, model_info.model, log = log)
  replacement_info = group_args(
    group_args_type = 'replacement_info',
    replacement_info_list = [],
   )
  # Make sure match_model is just one chain
  if params.require_match_model_is_single_chain:
    match_model_list = split_one_model(
       params, model_info.match_model, log = log)
    assert len(match_model_list) == 1
  for model in model_list:
    group_match_info = morph_with_segment_matching(params = params,
          model = model,
        match_model = model_info.match_model,
        log = log)
    if group_match_info and group_match_info.replacement_info_list:
       replacement_info.replacement_info_list += \
          group_match_info.replacement_info_list
  return replacement_info
def get_model_from_hierarchy(ph):
    from mmtbx.model import manager as model_manager
    model = model_manager(
      model_input = ph.as_pdb_input(),
      crystal_symmetry=crystal_symmetry_from_sites(ph.atoms().extract_xyz()),
      log = null_out())
    return model
def get_model_ca_ranges(indices):
  indices.sort()
  range_info = group_args(
    group_args_type = 'range_info',
    range_list = [],
   )
  empty_range_ga = group_args(
    group_args_type = 'range_ga',
    range_start = None,
    range_end = None,)
  range_ga = None
  for i in indices:
    if not range_ga or i != range_ga.range_end + 1:
      range_ga = deepcopy(empty_range_ga)
      range_info.range_list.append(range_ga)
      range_ga.range_start = i
      range_ga.range_end = i
    else:
      range_ga.range_end = i
  return range_info
def get_group_match_info(
    params,
    group_match_info,
    match_sites,
    model_ca_sites,
    log = sys.stdout):
  used_model_sites = group_match_info.used_model_sites
  model_to_match_dict = group_match_info.model_to_match_dict
  # Find a stretch of resides that are pretty close in model and match
  # Method: Find a run of residues that are pretty close.  Then slide along
  #  in both directions from there and note the shifts required to adjust
  best_match_info = get_best_match_info(params, model_ca_sites, match_sites,
    working_offset = None, starting_index = None,
    used_model_sites = used_model_sites,
    )
  working_offset = best_match_info.working_offset
  if best_match_info.rmsd is None  or \
     best_match_info.rmsd > params.close_distance or (
     not best_match_info.ok):
    group_match_info.ok = False
    return group_match_info
  # Forward
  found_something = False
  starting_residue_delta = best_match_info.best_i+\
      group_match_info.model_first_resno- \
     best_match_info.matching_i-group_match_info.match_first_resno
  last_residue_delta = starting_residue_delta
  for k in range(best_match_info.best_i, model_ca_sites.size()):
    match_info = get_best_match_info(params, model_ca_sites, match_sites,
       working_offset = working_offset, starting_index = k,
       used_model_sites = used_model_sites)
    if not match_info.ok:  break
    working_offset = match_info.working_offset
    residue_delta = k+\
      group_match_info.model_first_resno-match_info.matching_i-\
        group_match_info.match_first_resno
    if  abs(residue_delta - last_residue_delta) > \
        params.max_local_register_shift:
      break # not ok
    elif match_info.rmsd > params.close_distance:
      break
    last_residue_delta = residue_delta
    used_model_sites.append(k)
    model_to_match_dict[k] = match_info
    found_something = True
  # Reverse
  working_offset = best_match_info.working_offset
  last_residue_delta = starting_residue_delta
  for k in range(best_match_info.best_i-1, -1, -1):
    match_info = get_best_match_info(params, model_ca_sites, match_sites,
       working_offset = working_offset, starting_index = k,
       used_model_sites = used_model_sites)
    if not match_info.ok:
      break
    working_offset = match_info.working_offset
    residue_delta = k+\
        group_match_info.model_first_resno-match_info.matching_i-\
        group_match_info.match_first_resno
    if  abs(residue_delta - last_residue_delta) > \
        params.max_local_register_shift:
      break # not ok
    elif match_info.rmsd > params.close_distance:
      break
    last_residue_delta = residue_delta
    used_model_sites.append(k)
    model_to_match_dict[k] = match_info
    found_something = True
  group_match_info.ok = found_something
  return group_match_info
def get_best_match_info(params, model_ca_sites, match_sites,
     working_offset = None,
     starting_index = None,
     used_model_sites = None,):
  if starting_index is not None:
    first_index = starting_index
    last_index = starting_index
  else:
    first_index = 0
    last_index = model_ca_sites.size() - 1
  if working_offset is None:
    working_offset = col((0,0,0))
  else:
    working_offset = col(working_offset)
  best_info = get_dummy_match_info()
  for i in range(first_index, last_index + 1):
    if i in used_model_sites: continue
    distances = flex.double()
    other_sites = flex.vec3_double()
    k_start = max(0,i - params.n_half_window)
    k_end = min(model_ca_sites.size() - 1,i+params.n_half_window)
    if k_end + 1 - k_start < params.minimum_match_length:
      continue  # too short
    matching_i = None
    matching_last_i = None
    # Expect that only sites near matching_i will be relevant..
    dist,id1,id2 = model_ca_sites[i:i+1].min_distance_between_any_pair_with_id(
         match_sites)
    matching_i = id2
    test_match_sites_start = max(0,matching_i - 2 * params.n_half_window)
    test_match_sites_end = min(match_sites.size() - 1,
       matching_i + 2 * params.n_half_window)
    test_match_sites = match_sites[test_match_sites_start:test_match_sites_end+1]
    weights = flex.double()
    for k in range(k_start, k_end + 1):
      target_one_site = flex.vec3_double()
      target_site = col(model_ca_sites[k]) + working_offset
      target_one_site.append(target_site)
      dist,id1,id2 = target_one_site.min_distance_between_any_pair_with_id(
         test_match_sites)
      id2 = id2 + test_match_sites_start
      distances.append(min(params.max_distance, dist))
      other_sites.append(match_sites[id2])
      weights.append(max(0, min(1,
         (1 - abs(k - i)/max(1,params.n_half_window + 1))) # 1 at i, 0 at N+1
        ))
      if k == k_end:
        matching_last_i = id2
    weights = weights /max(1.e-10, weights.min_max_mean().mean) # avg = 1 now
    diffs = other_sites - (model_ca_sites[k_start:k_end + 1] + working_offset)
    # delta = col(diffs.mean())
    delta =   col(( diffs * weights).mean()) # triangle function 2021-04-28
    diffs = diffs - delta
    rmsd = diffs.rms_length()
    if best_info.rmsd is None or rmsd < best_info.rmsd:
      best_info = group_args(
        group_args_type = 'best_match info',
        best_i = i,
        matching_i = matching_i,
        first_i = k_start,
        last_i = k_end,
        matching_last_i = matching_last_i,
        rmsd = rmsd,
        working_offset = working_offset + delta,
        ok = True)
  return best_info
def get_dummy_match_info(matching_i = None,
    working_offset = None):
  return group_args(
      group_args_type = 'best_match info',
      best_i = None,
      matching_i = matching_i,
      first_i = None,
      last_i = None,
      matching_last_i = None,
      rmsd = None,
      working_offset = working_offset,
      ok = False)
def get_bounds_of_sites(sites_cart, box_buffer = 6):
  lower_bounds = tuple(
     col(sites_cart.min()) - col((box_buffer,box_buffer,box_buffer)))
  upper_bounds = tuple(
     col(sites_cart.max()) + col((box_buffer,box_buffer,box_buffer)))
  return group_args(
    group_args_type = 'bounds of sites',
      lower_bounds = lower_bounds,
      upper_bounds = upper_bounds,)
def remove_distant_sites(params,
    sites_to_trim = None,
     sites_to_be_close_to = None,):
  # remove all match_model_ca_sites that are far from model_ca_sites
  info = get_bounds_of_sites(sites_to_be_close_to,
     box_buffer = params.max_distance)
  x,y,z = sites_to_trim.parts()
  s = (
         (x < info.lower_bounds[0]) |
         (y < info.lower_bounds[1]) |
         (z < info.lower_bounds[2]) |
         (x > info.upper_bounds[0]) |
         (y > info.upper_bounds[1]) |
         (z > info.upper_bounds[2])
         )
  return sites_to_trim.select(~s)
def get_ca_sites(model):
  return model.apply_selection_string(
   'name ca and (not water and not element ca)').get_sites_cart()
def split_one_model(params, model, log = sys.stdout):
  from mmtbx.model import manager as model_manager
  model_list = []
  for info in split_model(model_info(model.get_hierarchy())):
    if params.n_segments is not None and len(model_list) >= params.n_segments:
      break
    new_model = model_manager(
      model_input = info.hierarchy.as_pdb_input(),
      crystal_symmetry = model.crystal_symmetry(),
      log = null_out())
    if new_model.get_hierarchy().overall_counts().n_residues < \
        params.n_half_window*2:
       continue
    print("Split model with %s residues" %(
      new_model.get_hierarchy().overall_counts().n_residues), file = log)
    model_list.append(new_model)
  return model_list
def get_models(params, log = sys.stdout):
  from iotbx.data_manager import DataManager
  dm = DataManager()
  dm.set_overwrite(True)
  match_model = dm.get_model(params.match_model_file)
  model = dm.get_model(params.model_file)
  print("Read model to match from %s" %(params.match_model_file), file = log)
  print("Read model to distort and trim from %s" %(params.model_file),
      file = log)
  crystal_symmetry = None
  for m in match_model, model:
    if not crystal_symmetry and m.crystal_symmetry():
      crystal_symmetry = m.crystal_symmetry()
  if not crystal_symmetry:
    from cctbx.maptbx.box import shift_and_box_model
    model = shift_and_box_model(model = model,
       shift_model = False)
    crystal_symmetry = model.crystal_symmetry()
  for m in match_model, model:
    m.set_crystal_symmetry(crystal_symmetry)
    m.remove_alternative_conformations(always_keep_one_conformer=True)
    m.convert_to_isotropic()
  match_model = match_model.apply_selection_string(
      "not water and not element ca")
  model = model.apply_selection_string(
      "not water and not element ca")
  return group_args(
    group_args_type = 'models',
    match_model = match_model,
    model = model,
    dm = dm)
def get_distort_params(close_distance = None,
    max_segment_matching_ca_ca_deviation = None):
  return group_args(
      group_args_type = 'parameters',
      model_file = 'ensemble_merged.pdb',
      match_model_file = 'sequence_from_map_real_space_refined_000_sfm.pdb',
      n_cycles = 1,
      n_segments = None,
      max_distance = 6,
      close_distance = close_distance if close_distance is not None else 2.5,
      n_half_window = 5,
      max_local_register_shift = 2,
      max_gap = 20,
      require_match_model_is_single_chain = False,
      minimum_match_length = 5,
      max_segment_matching_ca_ca_deviation = \
         max_segment_matching_ca_ca_deviation if \
        max_segment_matching_ca_ca_deviation is not None else 0.8,
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
def make_database(
   output_database_file_name = 'fragment_db_top2018.pkl.gz',
   output_dir = 'db',
   source_dir = None,
   nproc = 64,
   file_name_list = None,
   minimum_length = 40,
   skip_indexing = True,
   log = sys.stdout):
  print("\nCreating database from files in %s" %(source_dir), file = log)
  print("Minimum length of segments: %s" %(minimum_length), file = log)
  assert source_dir is not None
  if not file_name_list:
    file_name_list = os.listdir(source_dir)
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  end_number = -1
  n_tot = len(file_name_list)
  n = n_tot//nproc
  if n * nproc < n_tot:
    n = n + 1
  assert n * nproc >= n_tot
  runs_to_carry_out = []
  for run_id in range(nproc):
    start_number = end_number + 1
    end_number = min(n_tot-1, start_number + n -1)
    if end_number < start_number: continue
    runs_to_carry_out.append(group_args(
      run_id = run_id,
      start_number = start_number,
      end_number = end_number,
      ))
  kw_dict = {
    'file_name_list':file_name_list,
    'source_dir':source_dir,
    'output_dir':output_dir,
    'minimum_length':minimum_length,
    'skip_indexing':skip_indexing,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = False,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_run_make_database,
    log = log)
  db = group_args(
    group_args_type = 'ca database from %s' %(source_dir),
    segment_info_list = [],
   )
  info_list = []
  for run_info in runs_carried_out:
    segment_info_list = easy_pickle.load(run_info.result.saved_file_name)
    db.segment_info_list += segment_info_list
    os.remove(run_info.result.saved_file_name)
  segment_id = -1
  for segment_info in db.segment_info_list:
    segment_id += 1
    segment_info.segment_id = segment_id
  # Compress values as int( 100* value )
  for segment_info in db.segment_info_list:
    acc= segment_info.ca_sites_cart.accessor()
    segment_info.ca_sites_cart = vec3_double_to_int_100(
       segment_info.ca_sites_cart)
  full_file_name = os.path.join(output_dir, output_database_file_name)
  easy_pickle.dump(full_file_name, db)
  print("Wrote database to %s" %(full_file_name), file = log)
def group_of_run_make_database(
  run_info = None,
  file_name_list = None,
  source_dir = None,
  output_dir = None,
  minimum_length = None,
  skip_indexing = None,
  log = sys.stdout):
  segment_info_list = []
  for file_id in range(run_info.start_number, run_info.end_number + 1):
    file_name = file_name_list[file_id]
    segment_info_list += run_make_database(
      file_name = file_name,
      source_dir = source_dir,
      output_dir = output_dir,
      skip_indexing = skip_indexing,
      minimum_length = minimum_length,
      log = log)
  # Save the database so we do not pass it back
  saved_file_name = 'temp_%s' %(file_name.replace(".pdb",".pkl"))
  from libtbx import easy_pickle
  easy_pickle.dump(saved_file_name, segment_info_list)
  result = group_args(
    group_args_type = 'one segment_info_list',
    saved_file_name = saved_file_name)
  return result
def run_make_database(
  file_name = None,
  source_dir = None,
  output_dir = None,
  minimum_length = None,
  skip_indexing = None,
  as_info = True,
  log = sys.stdout):
  segment_info_list = []
  full_file_name = os.path.join(source_dir,file_name)
  ph = easy_pickle.load(full_file_name).as_ph() # it is a compact_hierarchy
  if not ph:
    return segment_info_list
  remove_ph_alt_conformations(ph)
  convert_ph_to_isotropic(ph)
  remove_extra_models(ph)
  models = split_model(model_info(ph))
  for m in models:
    sc = get_ca_sites_from_hierarchy(m.hierarchy)
    if sc.size() <  minimum_length: continue
    if skip_indexing:
      segment_index_db = None
    else:
      segment_index_db = get_segment_index_db(sc)
    segment_info = group_args(
      group_args_type = 'segment info',
      pdb_file_name = file_name,
      start_resno = get_first_resno(m.hierarchy),
      end_resno = get_last_resno(m.hierarchy),
      n_residues = m.hierarchy.overall_counts().n_residues,
      ca_sites_cart = sc,
      segment_index_db = segment_index_db,
     )
    segment_info_list.append(segment_info)
  # Write trimmed hierarchy to file in database_dir
  output_file_name = os.path.join(output_dir,file_name)
  ch = compact_hierarchy(ph)
  easy_pickle.dump(output_file_name, ch)
  return segment_info_list
def remove_ph_alt_conformations(ph):
  ph.remove_alt_confs(always_keep_one_conformer=True)
  ph.sort_atoms_in_place()
  ph.atoms_reset_serial()
def remove_extra_models(ph):
  if len(ph.models()) > 1:
    for model in ph.models()[1:]:
      ph.remove_model(model)
    assert len(ph.models()) == 1
def convert_ph_to_isotropic(ph):
  from cctbx import adptbx
  atoms = ph.atoms()
  selection = flex.bool(atoms.size(), True).iselection()
  for i in selection:
      a = atoms[i]
      u_cart = a.uij
      if(u_cart != (-1,-1,-1,-1,-1,-1)):
        a.set_uij((-1,-1,-1,-1,-1,-1))
        a.set_b( adptbx.u_as_b(adptbx.u_cart_as_u_iso(u_cart)) )
def get_segment_index_db(sc,
   minimum_residue_separation = 15,
   maximum_residue_separation = 80,
   int_minimum_dist = 1,
   int_maximum_dist = 10,
   int_dist_tol_tolerance = 1):
  '''
   Create an index of residue pairs by number of residues between them
   and distance between the pairs
   We want to come in with a distance (5.5 A) and residue separation (40)
   and get a short list of residue pairs that qualify.
   dist_tol_tolerance is how far off our tolerance can be (int only)
  '''
  residue_separation_dict = {}
  segment_index_db = group_args(
    residue_separation_dict = residue_separation_dict,
    int_minimum_dist = int_minimum_dist,
    int_maximum_dist = int_maximum_dist,
    int_dist_tol_tolerance = int_dist_tol_tolerance,
    )
  for sep in range(minimum_residue_separation,
     min(sc.size(),maximum_residue_separation + 1)):
    residue_separation_dict[sep] = {}
    start_i = 0
    start_j = start_i + sep
    end_j = sc.size() - 1
    end_i = end_j - sep
    distances = (sc[start_i:end_i+1] - sc[start_j:end_j+1]).norms()
    found = flex.bool(distances.size(), False)
    for i_dist in range(int_minimum_dist,
      int_maximum_dist+int_dist_tol_tolerance, int_dist_tol_tolerance):
      found_at_i =  (distances < i_dist) # those shorter than i_dist
      new_at_i = found_at_i & ~found
      found = found_at_i | found
      residue_separation_dict[sep][i_dist] = new_at_i.iselection()
  # Verify
  for ratio in [1,3,5]:
    dist_tol = int_dist_tol_tolerance * ratio
    for sep in range(minimum_residue_separation,
       maximum_residue_separation + 1):
      for dist in range(int_minimum_dist, int_maximum_dist+1):
        indices = get_indices_with_target_distance_n_residues_away(
          segment_index_db, dist = dist, residue_separation = sep,
           dist_tol = dist_tol)
        if not indices: continue
        for index in indices:
          actual_dist = (col(sc[index]) - col(sc[index+sep])).length()
          if not abs(actual_dist - dist) <= dist_tol + int_dist_tol_tolerance:
            print(index,sep,dist,actual_dist,abs(actual_dist - dist))
            assert abs(actual_dist - dist) <= dist_tol + int_dist_tol_tolerance
  return segment_index_db
def get_indices_with_target_distance_n_residues_away(
    segment_index_db, dist = None, residue_separation = None,
    dist_tol = 5):
  '''
   Return list of indices where x(i) and x(i+residue_separation) are
    separated by dist +- dist_tol
  '''
  # Index by residue separation
  dist_dict = segment_index_db.residue_separation_dict.get(residue_separation)
  if not dist_dict:
    return None # don't have a dict for this separation
  # Index by distance between residues at this separation
  indices_list = []
  checked = False
  for i in get_distances_to_check(segment_index_db, dist, dist_tol):
    local_indices = dist_dict.get(i,None)
    if local_indices is not None:
      checked = True
      for jj in local_indices:
        if not jj in indices_list:
          indices_list.append(jj)
  if checked:
    return indices_list
  else:
    return None
def get_distances_to_check(segment_index_db, dist, dist_tol):
  t = segment_index_db.int_dist_tol_tolerance
  low_i = segment_index_db.int_minimum_dist
  high_i = segment_index_db.int_maximum_dist
  # we want all values of i where this distance could have ended up
  # If distance is exactly dist it will be indexed in bin n where
  #   low_i + n*t is equal to or just larger than dist: (< low_i = 0)
  #  suppose low_i = 2, t = 3, dist = 5.9, 6 or 6.1
  #    will be indexed at n = 1, 2, 2
  #   n = 1 + int( (dist - low_i)/t)  ;  1 +int(( 5.9 - 2)/3) = 1
  #      1 +int(( 6 - 2)/3) = 2
  # if distance is between dist-t and dist+t it will be indexed
  # in  1 + int( (dist - t - low_i)/t) through  1 + int( (dist + t - low_i)/t)
  #   or  n-1 through n+1  ; n = 1 + int( (dist - low_i)/t)
  n = 1 + int( (dist - low_i)/t)
  distances_to_check = []
  for nn in range(n-1, n+2):
    distances_to_check.append(low_i + n * t)
  return distances_to_check
def run_find_all_fragments(
     target_sites_cart = None,
     max_final_results = 1,
     maximum_insertions_or_deletions = 0,
     max_final_rmsd = 4,
     nproc = 1,
     target_file = None,   # can supply file name instead of ca_sites_cart
     database_file = None,
     data_dir = None,
     default_database_file = 'fragment_db_top2018.pkl.gz',
     residue_offset = 10,
     length = 40,
     length_min = 12,
     length_increments = 4,
     delta_ends = 5,
     trim = True,
     db_info = None,
     pdb_exclude_list = None,
     pdb_include_list = None,
     mmm = None,
     log = sys.stdout):
  print("Finding fragments in fragment library for all segments", file = log)
  if not db_info:
    db_info = get_db_info(
      data_dir = data_dir,
      database_file = database_file,
      default_database_file = default_database_file,
      log = log)
  if not db_info:
    return None
  if not target_sites_cart:
    target_ph = get_pdb_hierarchy(file_name=target_file)
    target_sites_cart = get_ca_sites_from_hierarchy(target_ph)
  target_list = []
  for j in range(length_increments):
    fraction = (length_increments - j- 1)/(length_increments)
    length_use = int(0.5 + length_min + fraction * (length - length_min))
    for i in range (0, target_sites_cart.size() - (length_use - 2 * delta_ends),
      residue_offset):
      start_pos = max(0,i + random.randint(-delta_ends,delta_ends))
      end_pos = min(target_sites_cart.size() - 1,
        i + length_use+random.randint(-delta_ends,delta_ends))
      target_list.append(target_sites_cart[start_pos:end_pos+1])
  info_list = []
  for target in target_list:
    result = run_find_fragment(
     target_sites_cart = target,
     max_final_results = max_final_results,
     maximum_insertions_or_deletions = maximum_insertions_or_deletions,
     max_final_rmsd = max_final_rmsd,
     nproc = nproc,
     db_info = db_info,
     trim = trim,
     pdb_exclude_list = pdb_exclude_list,
     pdb_include_list = pdb_include_list,
     mmm = mmm,
     log = log)
    if result:
      info_list += result.info_list
  return group_args(
    group_args_type = 'replacement segment info',
    info_list = info_list,
    )
def run_find_fragment(
     model = None,
     target_sites_cart = None,
     max_final_results = 10,
     maximum_insertions_or_deletions = 0,
     max_final_rmsd = 4,
     nproc = 1,
     target_file = None,   # can supply file name instead of ca_sites_cart
     database_file = None,
     data_dir = None,
     default_database_file = 'fragment_db_top2018.pkl.gz',
     db_info = None,
     trim = True,
     pdb_exclude_list = None,
     pdb_include_list = None,
     n_res = None,
     mmm = None,
     refine_model = None,
     log = sys.stdout):
  # If trim is False, also return full matching models
  # NOTE: finds fragment in current location of model (ignores any shift_cart)
  shift_cart = model.shift_cart()
  if not target_sites_cart:
    if model:
      target_ph = model.get_hierarchy()
    else:
      target_ph = get_pdb_hierarchy(file_name=target_file)
    target_sites_cart = get_ca_sites_from_hierarchy(target_ph)
  if not target_sites_cart:
    return None
  if n_res is None:
    n_res = target_sites_cart.size()
  print("Finding fragment of length %s in fragment library" %(
     n_res), file = log)
  if maximum_insertions_or_deletions:
     max_indel_tries = 3 + maximum_insertions_or_deletions**2
     print("Maximum insertions/deletions: %s" %(
        maximum_insertions_or_deletions), file = log)
  else:
     max_indel_tries = 0
  n_optimize = max(4,max_final_results * 4)
  if not db_info:
    db_info = get_db_info(
      data_dir = data_dir,
      database_file = database_file,
      default_database_file = default_database_file,
      log = log)
  if not db_info:
    return None
  exclude_segment_id_list = get_exclude_segment_id_list(
     db_info, pdb_exclude_list)
  required_segment_id_list = get_exclude_segment_id_list(
    db_info, pdb_include_list)
  if exclude_segment_id_list:
    print("Excluded PDB:",pdb_exclude_list,exclude_segment_id_list, file = log)
  if pdb_include_list:
    print("Required PDB:",pdb_include_list,required_segment_id_list, file = log)
  target_sites_cart_with_indel_list = get_target_sites_cart_with_indel(
    target_sites_cart, max_indel_tries = max_indel_tries,
    n_res = n_res)
  distance_group_info_list = []
  for target_sites_cart_with_indel in target_sites_cart_with_indel_list:
    distance_group_info = set_up_distances(target_sites_cart_with_indel)
    if distance_group_info:
      distance_group_info_list.append(distance_group_info)
  end_number = -1
  n_tot = len(db_info.db.segment_info_list)
  n = n_tot//nproc
  if n * nproc < n_tot:
    n = n + 1
  assert n * nproc >= n_tot
  runs_to_carry_out = []
  for run_id in range(nproc):
    start_number = end_number + 1
    end_number = min(n_tot-1, start_number + n -1)
    if end_number < start_number: continue
    runs_to_carry_out.append(group_args(
      run_id = run_id,
      start_number = start_number,
      end_number = end_number,
      ))
  kw_dict = {
    'db':db_info.db,
    'mmm':mmm,
    'refine_model':refine_model,
    'data_dir':db_info.data_dir,
    'distance_group_info_list':distance_group_info_list,
    'max_final_rmsd':max_final_rmsd,
    'n_optimize':1+max(0,n_optimize-1)//nproc,
    'exclude_segment_id_list':exclude_segment_id_list,
    'trim': trim,
    'required_segment_id_list':required_segment_id_list,
   }
  runs_carried_out = run_jobs_with_large_fixed_objects(
    nproc = nproc,
    verbose = False,
    kw_dict = kw_dict,
    run_info_list = runs_to_carry_out,
    job_to_run = group_of_find_matching_atoms,
    log = log)
  info_list = []
  for run_info in runs_carried_out:
    for info in run_info.result.info_list:
      if info.final_rmsd is not None and info.final_rmsd <= max_final_rmsd:
        info_list.append(info)
  info_list = sorted(info_list, key = lambda info: info.final_rmsd)
  for info in info_list[:max_final_results]:
    print(" %s (%s)  %.2f (length = %s)" %(
    info.pdb_text, info.segment_id, info.final_rmsd,
    info.model.get_hierarchy().overall_counts().n_residues),
        file = log)
    if not info.model.info().get('pdb_text'):
      info.model.info().pdb_text = info.pdb_text
    if shift_cart:  # put back shift_cart
      info.model.set_shift_cart(shift_cart)
      if info.full_model:
        info.full_model.set_shift_cart(shift_cart)
  return group_args(
    group_args_type = 'replacement segment info',
    info_list = info_list[:max_final_results],
    )
def get_exclude_segment_id_list(db_info, pdb_exclude_list):
  exclude_segment_id_list = []
  if not pdb_exclude_list:
    return exclude_segment_id_list
  db = db_info.db
  for segment_id in range(len(db.segment_info_list)):
    segment_info = db.segment_info_list[segment_id]
    file_name = segment_info.pdb_file_name
    for pdb in pdb_exclude_list:
      if file_name[:4].lower().find(pdb.lower()) > -1:
        exclude_segment_id_list.append(segment_id)
        break
  return exclude_segment_id_list
def get_db_info(
    data_dir = None,
    database_file = None,
    default_database_file = None,
    log = sys.stdout):
  if not data_dir:
    import libtbx.load_env
    data_dir = libtbx.env.find_in_repositories(
       relative_path=os.path.join("chem_data","segment_lib",'pdb2018_filtered'),
       test=os.path.isdir)
    if not data_dir:
      print("Data directory %s is missing..."  %(data_dir), file = log)
      return None
  if not database_file or not os.path.isfile(database_file):
    if not database_file:
      database_file = default_database_file
    database_file = os.path.join(data_dir,database_file)
    if not os.path.isfile(database_file):
      print("Database file %s is missing..."  %(database_file), file = log)
      return None
  db = get_database(database_file = database_file)
  print("Read %s entries from database" %(len(db.segment_info_list)),
      file = log)
  return group_args(
     group_args_info = 'db info',
     db = db,
     data_dir = data_dir)
def get_target_sites_cart_with_indel(target_sites_cart, max_indel_tries = None,
    n_res = None):
  target_sites_list = []
  if target_sites_cart.size() < 1:
    return target_sites_list
  if n_res == target_sites_cart.size():
    target_sites_list.append(target_sites_cart) # always
  # Find places where chain turns around and randomly add/subtrace a CA there
  indel_locations_list = get_extrema_list(target_sites_cart)
  if n_res == target_sites_cart.size() and (
      not indel_locations_list or max_indel_tries == 0):
    return target_sites_list
  if n_res > target_sites_cart.size():
    max_in_one = n_res - target_sites_cart.size()
    p_insert = 1
    p_delete =  -1
    max_indel_tries = 1
    n_dup = int(0.5+max_in_one/max(1,len(indel_locations_list)))
    indel_locations_list = n_dup * indel_locations_list
    prob_in_one = 1
  elif n_res < target_sites_cart.size():
    max_in_one = target_sites_cart.size() - n_res
    p_insert =  -1
    p_delete = 1
    max_indel_tries = 1
    prob_in_one = 1
    n_dup = int(0.5+max_in_one/max(1,len(indel_locations_list)))
    indel_locations_list = n_dup * indel_locations_list
  else: # usual
    p_insert = 0.5
    p_delete = 0.5
    max_in_one = max(1,int((max_indel_tries-3)**0.5))
    prob_in_one = max_in_one/max(1,len(indel_locations_list))
  for i in range(max_indel_tries): # make some random indel at these locations
    sc = target_sites_cart.deep_copy()
    changes = 0
    for k in indel_locations_list:
      if changes < max_in_one and random.uniform(0,1) <= prob_in_one:
        changes += 1
        if random.randint(0,1) <= p_insert: # insert
          new_sc = sc[:k]
          new_sc.append(sc[k])
          new_sc.extend(sc[k:])
          sc = new_sc
        elif random.randint(0,1) <= p_delete: # delete
          new_sc = sc[:k]
          new_sc.extend(sc[k+1:])
          sc = new_sc
    if changes:
      target_sites_list.append(sc)
  return target_sites_list
def get_extrema_list(target_sites_cart, from_start = True,
    window = 5,):
  if not from_start:
    target_sites_cart = list(target_sites_cart)
    target_sites_cart.reverse()
    target_sites_cart = flex.vec3_double(target_sites_cart)
  extrema_list = []
  if target_sites_cart.size()< 1:
    return  []
  distances = (target_sites_cart - target_sites_cart[0]).norms()
  for i in range(window, distances.size() - window):
    if distances[i] == distances[
        max(0,i-window):min(distances.size()-1,i+window)].min_max_mean().max:
      extrema_list.append(i)
    elif distances[i] == distances[
        max(0,i-window):min(distances.size()-1,i+window)].min_max_mean().min:
      extrema_list.append(i)
  if not from_start:
    new_list = []
    for i in extrema_list:
      new_list.append(target_sites_cart.size() - 1 - i)
    extrema_list = new_list
  return extrema_list
#  Start library routines
def crystal_symmetry_from_sites(sites_cart):
  '''
  Create crystal_symmetry big enough to hold sites_cart
  '''
  from cctbx import crystal
  box_end=tuple(col(sites_cart.max()) - col(sites_cart.min()))
  a,b,c=box_end
  if a==0 or b==0 or c==0:
    a = 10
    b = 10
    c = 10
  return crystal.symmetry((a,b,c, 90,90,90),1)
def model_from_sites(sites_cart):
  ''' Create model object from sites_cart'''
  from mmtbx.model import manager
  crystal_symmetry=crystal_symmetry_from_sites(sites_cart)
  return manager.from_sites_cart(
         sites_cart = sites_cart,
         crystal_symmetry = crystal_symmetry)
def get_ca_sites_from_hierarchy(ph):
  ''' extract CA sites from hierarchy'''
  asc1 = ph.atom_selection_cache()
  sel = asc1.selection(
     string = 'name ca and (not water and not element ca) and protein')
  return ph.select(sel).atoms().extract_xyz()
def get_rmsd_with_indel(x,y, close_distance = 3):
  '''
    Get rmsd between each value in x and closest in y (or reverse, take smaller)
    Also return number in x or y closer than close_distance
  '''
  rmsd_value= None
  forward = True
  close_n = None
  if x is None or y is None or x.size()<1 or y.size()<1:
    return group_args(
    group_args_type = 'rmsd with indel',
      rmsd_value = None,
      close_n = None,)
  for xx,yy in [[x,y],[y,x]]:
    distances = flex.double()
    for i in range(yy.size()): # smaller one
      dist,id1,id2 = yy[i:i+1].min_distance_between_any_pair_with_id(xx)
      distances.append(dist)
    rmsd = distances.rms()
    local_close_n = (distances <= close_distance).count(True)
    if close_n is None or local_close_n > close_n:
       close_n = local_close_n
    forward = False
    if rmsd_value is None or rmsd_value > rmsd:
      rmsd_value = rmsd
  return group_args(
    group_args_type = 'rmsd with indel',
      rmsd_value = rmsd_value,
      close_n = close_n,)
def superpose_sites_with_indel(sites_cart = None,
      target_sites = None,
      allow_indel = True,
      sites_to_apply_superposition_to = None,
      indel_cycles = 10):
  '''
     Superpose sites_cart onto target_sites, allowing insertions/deletions.
     Then apply result to sites_to_apply_superposition_to
  '''
  if not sites_to_apply_superposition_to:
    sites_to_apply_superposition_to = sites_cart
  if target_sites.size() < 3 or sites_cart.size() < 3: # return  what we have
    return sites_to_apply_superposition_to
  from scitbx.math import superpose
  if allow_indel and target_sites.size() != sites_cart.size():
    delta_n = target_sites.size() - sites_cart.size()
    best_lsq_fit_obj = None
    working_target_sites = target_sites
    working_sites_cart = sites_cart
    for cy in range(indel_cycles):
      if delta_n > 0:
        working_target_sites = delete_sites(target_sites, delta_n)
      else:
        working_sites_cart = delete_sites(sites_cart, delta_n)
      lsq_fit_obj = superpose.least_squares_fit(
        reference_sites = working_target_sites,
        other_sites     = working_sites_cart)
      new_working_sites_cart = lsq_fit_obj.r.elems * working_sites_cart + \
          lsq_fit_obj.t.elems
      rmsd = (working_target_sites - new_working_sites_cart).rms_length()
      if best_lsq_fit_obj is None or best_lsq_fit_obj.rmsd > rmsd:
        best_lsq_fit_obj = lsq_fit_obj
        best_lsq_fit_obj.rmsd = rmsd
    # Use best one now
    return best_lsq_fit_obj.r.elems * sites_to_apply_superposition_to \
        + best_lsq_fit_obj.t.elems
  else:
    working_sites_cart = sites_cart
    lsq_fit_obj = superpose.least_squares_fit(
      reference_sites = target_sites,
      other_sites     = sites_cart)
    new_working_sites_cart = lsq_fit_obj.r.elems * sites_cart+ \
          lsq_fit_obj.t.elems
    rmsd = (target_sites - new_working_sites_cart).rms_length()
    return lsq_fit_obj.r.elems * sites_to_apply_superposition_to \
       + lsq_fit_obj.t.elems
def delete_sites(sc, n):
  # randomly remove n sites
  n = abs(n)
  n_start = sc.size()
  if sc.size() < n+1 or n == 0:
    return sc.deep_copy()
  for i in range(n):
    k = random.randint(1,sc.size()-2)
    new_sc = sc[:k]
    new_sc.extend(sc[k+1:])
    sc = new_sc
  n_end= sc.size()
  assert n_end + n == n_start
  return sc
def morph_with_segment_matching(
      model = None,   # model to transform
      match_model = None, # model to match
      match_model_ca_sites = None, # alternative to match_model
      match_first_resno = None,    # alternative to match_model
      params = None,
      max_distance = 6,    # alternative to params
      max_local_register_shift = 2, # alternative to params
      close_distance = 2.5, # alternative to params
      n_half_window = 5, # alternative to params
      minimum_match_length = 5,
      max_segment_matching_ca_ca_deviation = 0.8,
      return_replacement_fragments_only = False, # return list of replacements
      trim = True,
      log = sys.stdout):
  ''' Find fragments in model that match CA in match_model
   Iteratively find CA in model close to CA in match_model, identify local
    shifts to morph model and apply.
   Requires that match_model already superimposes on model
   Returns nothing if models do not match within close_distance
     or distortions are greater than max_segment_matching_ca_ca_deviation
     or fraction matching is less than minimum_match_length
   Can supply either match_model or (match_model_ca_sites and match_first_resno)
   Can supply params or  close_distance, max_local_register_shift n_half_window,
      max_distance
   Returns replacement_info_list  (one per segment in match_model)
     Each replacement (entire model, only part is matched):
     segment = replacement.replacement_segment  # the morphed replacement model
     replacement.range_start = index where match to model starts
     replacement.range_end = index where match to model ends
     To get the part of each segment that matches model only, specify
       return_replacement_fragments_only
  '''
  if not params:
    params = group_args(
      close_distance = close_distance,
      max_local_register_shift = max_local_register_shift,
      n_half_window = n_half_window,
      max_distance = max_distance,
      minimum_match_length = minimum_match_length,
      max_segment_matching_ca_ca_deviation = max_segment_matching_ca_ca_deviation,
     )
  model_ca_sites = get_ca_sites(model).deep_copy()
  if match_model:
    match_model_ca_sites = get_ca_sites(match_model)
    match_first_resno = get_first_resno(match_model.get_hierarchy())
  else:
    assert match_model_ca_sites is not None and match_first_resno is not None
  dist,id1,id2 = model_ca_sites.min_distance_between_any_pair_with_id(
     match_model_ca_sites)
  if dist > params.close_distance:
    print("Models too far apart (%.2f A)" %(dist), file = log)
    return # nothing to do
  trimmed_new_model_info = distort_coords(params,
     model = model,
     model_ca_sites = model_ca_sites,
     match_model_ca_sites = match_model_ca_sites,
     model_first_resno = get_first_resno(model.get_hierarchy()),
     model_last_resno = get_last_resno(model.get_hierarchy()),
     match_first_resno = match_first_resno,
     trim = trim,
     log = log)
  if not return_replacement_fragments_only: # usual
    return trimmed_new_model_info
  if not trimmed_new_model_info:
    return None # nothing to do
  if not trimmed_new_model_info.replacement_info_list:
    print("No replacements...", file = log)
    return None # nothing to do
  # Trim the model info
  for replacement in trimmed_new_model_info.replacement_info_list:
     segment = replacement.replacement_segment  # the morphed replacement model
     range_start = replacement.range_start
     range_end = replacement.range_end
     first_resno = get_first_resno(model.get_hierarchy())
     last_resno = get_last_resno(model.get_hierarchy())
     residue_start = first_resno + range_start
     residue_end = first_resno + range_end
     replacement_fragment = segment.apply_selection_string("resseq %s:%s" %(
        residue_start,residue_end))
     replacement.replacement_segment = replacement_fragment
  return trimmed_new_model_info
def distort_coords(params,
   model = None,
   model_ca_sites = None,
   match_model_ca_sites = None,
   model_first_resno = None,
   model_last_resno = None,
   match_first_resno = None,
   trim =True,
   log = sys.stdout):
  match_sites = match_model_ca_sites
  ''' Find a long run of model_ca_sites and match_sites that match.
   Distort the coordinates of model sites so that the end sites match.
   Then cross those off and find another one, until none left
  '''
  used_model_sites = []
  model_to_match_dict = {}
  group_match_info = group_args(
      group_args_type = 'matched model sites',
      model_to_match_dict = model_to_match_dict,
      used_model_sites = used_model_sites,
      selection_string = None,
      replacement_info_list = None,
      model_first_resno = model_first_resno,
      model_last_resno = model_last_resno,
      match_first_resno = match_first_resno,
      ok = False)
  for iter in range(model_ca_sites.size()): # max we could run
    group_match_info = get_group_match_info(params,
      group_match_info,
      match_sites,
      model_ca_sites,
      log = log)
    if not group_match_info.ok:
      break # all done
  if len(list(model_to_match_dict.keys())) < params.minimum_match_length:
    return None
  # Now identify range(s) of model_ca_sites that have entries in our dict.
  # If values present on both sides of a gap and they are similar, interpolate
  #  through the gap.  Trim ends back if gap > max_gap.
  #  If ends present < max_gap, keep them with offset of first/last present
  model_ca_range_info = get_model_ca_ranges(list(list(model_to_match_dict.keys())))
  update_group_match_info(params,
     group_match_info = group_match_info,
     model_ca_range_info = model_ca_range_info,
     log = log)
  group_match_info = apply_shifts_to_selected_residues(
    params,
    model = model,
    trim = trim,
    group_match_info = group_match_info)
  return group_match_info
def apply_shifts_to_selected_residues(
    params,
    model = None,
    group_match_info = None,
    trim = True,
   ):
  # Now select just the residues we want and apply the shifts to them and return
  # Make sure that sequential shifts (residue to residue) are less than
  #  max_segment_matching_ca_ca_deviation
  keys = list(list(group_match_info.model_to_match_dict.keys()))
  keys.sort()
  for k1,k2 in zip(keys,keys[1:]):
    offset1 = group_match_info.model_to_match_dict[k1].working_offset
    offset2 = group_match_info.model_to_match_dict[k2].working_offset
    if k1 + 1 == k2 and (col(offset1) - col(offset2)).length() > \
         params.max_segment_matching_ca_ca_deviation:
      return None # too distorted
  if (trim):
    selected_model = model.apply_selection_string(
      group_match_info.selection_string)
  else:
    selected_model = model
  # And apply shifts for each residue
  sites_cart = selected_model.get_sites_cart()
  all_selected = flex.bool(sites_cart.size(), False)
  for index in list(list(group_match_info.model_to_match_dict.keys())):
    working_offset = group_match_info.model_to_match_dict[index].working_offset
    sel = selected_model.selection('resseq %s' %(
        index + group_match_info.model_first_resno))
    sc = sites_cart.select(sel)
    duplicate = (all_selected & sel)
    assert duplicate.count(True) == 0
    all_selected = (all_selected | sel)
    sc += working_offset
    sites_cart.set_selected(sel,sc)
  selected_model.set_sites_cart(sites_cart)
  # Now split up by segment
  for replacement_info in group_match_info.replacement_info_list:
    selection_string = "resseq %s:%s" %(
      replacement_info.start_resno, replacement_info.end_resno)
    replacement_info.replacement_segment = \
       selected_model.apply_selection_string(selection_string)
  group_match_info.full_selected_model = selected_model
  return group_match_info
def update_group_match_info(params,
     group_match_info = None,
     model_ca_range_info = None,
     log = sys.stdout):
  used_model_sites = group_match_info.used_model_sites
  model_to_match_dict = group_match_info.model_to_match_dict
  group_match_info.replacement_info_list = []
  selection_list = []
  replacement_info = None
  for ga, ga1 in zip(
      model_ca_range_info.range_list,
      model_ca_range_info.range_list[1:]+[None]):
    selection_list.append("(resseq %s:%s)" %(
         ga.range_start + group_match_info.model_first_resno,
        ga.range_end + group_match_info.model_first_resno))
    if not replacement_info:
      replacement_info = group_args(
        group_args_type = "one replacement segment. Note resno = indices +%s" %(
        group_match_info.model_first_resno),
        replacement_segment = None,
        range_start = ga.range_start,  # range of indices
        model_first_resno = group_match_info.model_first_resno,
        start_resno = ga.range_start + group_match_info.model_first_resno,
          # range of residues
        replaces_start_resno =  \
            model_to_match_dict[ga.range_start].matching_i +\
             group_match_info.match_first_resno,
        range_end = ga.range_end,
        end_resno = ga.range_end + group_match_info.model_first_resno,
        replaces_end_resno = \
           model_to_match_dict[ga.range_end].matching_i + \
            group_match_info.match_first_resno,
        model_to_match_dict = model_to_match_dict, # dict keyed on indices
        )
      group_match_info.replacement_info_list.append(replacement_info)
    else:  # extend it
      replacement_info.end_resno = ga.range_end + \
         group_match_info.model_first_resno
      replacement_info.range_end = ga.range_end
      replacement_info.replaces_end_resno = \
           model_to_match_dict[ga.range_end].matching_i + \
           group_match_info.match_first_resno
    shifts = flex.vec3_double()
    for k in range(ga.range_start, ga.range_end + 1):
      shifts.append(model_to_match_dict[k].working_offset)
    if ga1:  # fill in gap
      gap = ga1.range_start - ga.range_end
      if gap > 1 and gap <= params.max_gap:
        selection_list.append("(resseq %s:%s)" %(
          ga.range_end+1+ group_match_info.model_first_resno,
          ga1.range_start-1+ group_match_info.model_first_resno))
      for k in range(ga.range_end+1, ga1.range_start):
          w2 = (k - ga.range_end)/max(
              1, ga1.range_start - ga.range_end)
          w1 = 1 - w2
          working_offset = \
           w1 * model_to_match_dict[ga.range_end].working_offset + \
           w2 * model_to_match_dict[ga1.range_start].working_offset
          model_to_match_dict[k] = get_dummy_match_info(matching_i = k,
            working_offset = working_offset)
      else:
        replacement_info = None  # start new replacement_info
  replacement_info = None
  # anything before or after...
  ga1 = model_ca_range_info.range_list[0]
  gap = ga1.range_start
  if gap > 1 and gap <= params.max_gap:
      selection_list.append("(resseq %s:%s)" %(
        group_match_info.model_first_resno,
        ga1.range_start-1+ group_match_info.model_first_resno))
  for k in range(ga1.range_start):
        working_offset = model_to_match_dict[ga1.range_start].working_offset
        model_to_match_dict[k] = get_dummy_match_info(matching_i = k,
          working_offset = working_offset)
  ga = model_ca_range_info.range_list[-1]
  n_tot = group_match_info.model_last_resno - \
       group_match_info.model_first_resno + 1
  gap = n_tot - ga.range_end
  if gap > 1 and gap <= params.max_gap:
      selection_list.append("(resseq %s:%s)" %(
        ga.range_end+1+ group_match_info.model_first_resno,
        n_tot - 1 + group_match_info.model_first_resno))
  for k in range(ga.range_end+1, n_tot):
        working_offset =model_to_match_dict[ga.range_end].working_offset
        model_to_match_dict[k] = get_dummy_match_info(matching_i = k,
          working_offset = working_offset)
  group_match_info.selection_string = " or ".join(selection_list)
#  End library routines
def get_database(database_file = None):
  from libtbx import easy_pickle
  db = easy_pickle.load(database_file)
  for segment_info in db.segment_info_list:
    segment_info.ca_sites_cart = vec3_int_to_double_100(
      segment_info.ca_sites_cart)
  return db
def set_up_distances(sc):
  n_residues = sc.size()
  nn = n_residues - 1
  n_third = n_residues//3
  if sc.size()< 1:
    return
  distance_groups = group_args(
     group_args_type = 'distance groups',
     n_residues = n_residues,
     nn = nn,
     n_third = n_third,
     dist = (col(sc[0]) - col(sc[-1])).length(),
     diffs_target_list = [],
     dist_list = [],
     index_list = [],
     sites_cart = sc,
   )
  for j in [0, n_third, 2*n_third]:
    diffs_target = (sc- col(sc[j])).norms()
    distance_groups.diffs_target_list.append(diffs_target)
    distance_groups.index_list.append(j)
    distance_groups.dist_list.append((col(sc[nn]) - col(sc[j])).length()),
  return distance_groups
def group_of_find_matching_atoms(run_info = None,
     mmm = None,
     refine_model = None,
     db = None,
     data_dir = None,
     distance_group_info_list = None,
     dist_tol = 5,
     max_rms = 5,
     max_final_rmsd = 3,
     n_optimize = 1,
     trim = True,
     exclude_segment_id_list = None,
     required_segment_id_list = None,
     log = sys.stdout):
  result = group_args(
    group_args_type = 'result of best_fit',
    info_list = [])
  for distance_group_info in distance_group_info_list:
    for segment_id in range(run_info.start_number, run_info.end_number + 1):
      if exclude_segment_id_list and segment_id in exclude_segment_id_list:
        continue # skip this one
      if required_segment_id_list and not segment_id in required_segment_id_list:
        continue # skip this one
      info = find_matching_atoms(segment_id = segment_id,
       db = db,
       distance_group_info = distance_group_info,
       dist_tol = dist_tol, max_rms = max_rms, log = log)
      if info.rmsd is not None and \
            info.rmsd <= max_final_rmsd and info.sites_cart:
        #print("%s %.2f  (%s) " %( info.file_name, info.rmsd, info.index), file = log)
        result.info_list.append(info)
        info.distance_group_info = distance_group_info
  result.info_list = sorted(result.info_list, key = lambda info: info.rmsd)
  # Now optimize the top results in list
  for info in result.info_list[:n_optimize]:
    # Target we want to match
    match_model = model_from_sites(info.distance_group_info.sites_cart)
    # Read in model with source information now and use the whole thing
    # Pull original model from database
    original_model_file_name = "%s.gz" %(
        os.path.join(data_dir,info.file_name.replace(".gz","")))
    if not os.path.isfile(original_model_file_name):
      print("Missing database file %s" %(original_model_file_name), file = log)
      continue
    import iotbx.pdb
    original_ch = easy_pickle.load(original_model_file_name)
    if not original_ch:
      continue
    original_ph = original_ch.as_ph()
    if (not trim):
      original_ph_sav = original_ph.deep_copy()
    # Select residue range matching our final ca
    start_resno = info.first_resno_to_add_to_index + info.index
    end_resno = start_resno + info.sites_cart.size() - 1
    asc1 = original_ph.atom_selection_cache()
    sel = asc1.selection(string = 'resseq %s:%s' %(start_resno,end_resno))
    ph = original_ph.select(sel)
    full_sites_cart = ph.atoms().extract_xyz()
    ca_sites_from_new_full_sites_cart = get_ca_sites_from_hierarchy(ph)
    if (not trim):
      ca_sites_from_new_full_sites_cart_sav = \
        ca_sites_from_new_full_sites_cart.deep_copy()
    # Optimize coords for this fit
    new_full_sites_cart  = superpose_sites_with_indel(
        sites_cart = ca_sites_from_new_full_sites_cart,
          target_sites = distance_group_info.sites_cart,
        sites_to_apply_superposition_to = full_sites_cart)
    ph.atoms().set_xyz(new_full_sites_cart)
    ca_sites_from_new_full_sites_cart = get_ca_sites_from_hierarchy(ph)
    # Now get model from full ph
    model = get_model_from_hierarchy(ph)
    if (not trim):
      original_full_sites_cart = original_ph_sav.atoms().extract_xyz()
      new_original_full_sites_cart =superpose_sites_with_indel(
        sites_cart = ca_sites_from_new_full_sites_cart_sav,
          target_sites = distance_group_info.sites_cart,
        sites_to_apply_superposition_to = original_full_sites_cart)
      original_ph_sav.atoms().set_xyz(new_original_full_sites_cart)
      info.full_model = get_model_from_hierarchy(original_ph_sav)
    # Now just use this already-superimposed model as source of
    #   coordinates and match to match_model and distort as necessary to match
    params = get_distort_params(close_distance = 3)
    replacement_info = morph_with_segment_matching(  # Distortion routine
       params = params,
       model = model,
       match_model = match_model,
       log = null_out())
    # Should be just 1 replacement segment.  Put it in
    if not replacement_info:
      continue
    replacement = replacement_info.replacement_info_list[0]
    segment = replacement.replacement_segment
    # where optimized segment starts in match_model
    range_start = replacement.range_start
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
        target_sites = distance_group_info.sites_cart,
        sites_to_apply_superposition_to = all_sites_cart)
    model.set_sites_cart(new_all_sites_cart)
    if mmm: # set symmetry and refine the model now
      local_mmm = mmm.customized_copy(model_dict = {})
      model.set_crystal_symmetry(local_mmm.crystal_symmetry())
      model.set_unit_cell_crystal_symmetry(
        local_mmm.unit_cell_crystal_symmetry())
      model.set_shift_cart(mmm.shift_cart())
      match_model.set_crystal_symmetry(local_mmm.crystal_symmetry())
      match_model.set_unit_cell_crystal_symmetry(
          local_mmm.unit_cell_crystal_symmetry())
      match_model.set_shift_cart(mmm.shift_cart())
      local_mmm.add_model_by_id(model_id = 'model', model = model) # sets shift_cart
      if refine_model in [None, True]:
        box_mmm = local_mmm.extract_all_maps_around_model()
        build = box_mmm.model_building()
        build.refine(reference_model = model.deep_copy(), refine_cycles = 5)
        model = build.model()
        # put model back in original coordinate system
        model = mmm.get_model_from_other(box_mmm)
    new_ca = get_ca_sites(model)
    # Update rmsd:
    info.final_rmsd = get_rmsd_with_indel(
        new_ca,distance_group_info.sites_cart).rmsd_value
    if info.final_rmsd > max_final_rmsd:
       continue # skip it
    info.model = model
    info.sites_cart = new_ca
  return result
def intersection(list1, list2):
  list1 = deepcopy(list1) # deepcopy so that we don't sort original
  list2 = deepcopy(list2)
  in_common = []
  list1.sort()
  list2.sort()
  a = make_iselection_unique(flex.size_t(list1))
  b = make_iselection_unique(flex.size_t(list2))
  return list(a.intersection(b))
def make_iselection_unique(iselection):
  np_array = iselection.as_numpy_array()
  unique_np_array = np.unique(np_array)
  return flex.size_t(unique_np_array)
def find_matching_atoms(segment_id = None,
     db = None,
     distance_group_info = None,
     dist_tol = 5, max_rms = 5, max_final_rmsd = 3, log = sys.stdout):
  '''
    Find atoms in db matching distance_group_info.sites_cart
    Just look at one entry (segment_id) in database
    Return index of CA in segment_id matching atoms in sites_cart
  '''
  segment_info = db.segment_info_list[segment_id]
  file_name = segment_info.pdb_file_name
  sc = segment_info.ca_sites_cart
  best_i = None
  best_rmsd = None
  for i in range(sc.size() - distance_group_info.n_residues):
    ok = True
    for j,dist in zip(
        distance_group_info.index_list,
        distance_group_info.dist_list,):
      dd = (col(sc[i + distance_group_info.nn]) - col(sc[i + j])).length()
      if abs(dd - dist) > dist_tol:
        ok = False
        break
    if not ok: continue
    match = sc[i:i+distance_group_info.n_residues]
    ok = True
    sum_rmsd_sq = 0
    sum_n = 0
    for j,diffs_target in zip(distance_group_info.index_list,
      distance_group_info.diffs_target_list):
      diffs_obs = (match - col(match[j])).norms()
      rmsd = (diffs_target - diffs_obs).rms()
      sum_rmsd_sq += distance_group_info.n_residues * rmsd**2
      sum_n += distance_group_info.n_residues
      rmsd = (sum_rmsd_sq/sum_n)**0.5
      if rmsd > max_rms or (best_rmsd is not None and rmsd > best_rmsd):
        ok = False
        break
    if not ok: continue
    if best_rmsd is None or rmsd < best_rmsd:
      best_rmsd = rmsd
      best_i = i
  info = group_args(
    group_args_type = 'best_fit',
    rmsd = best_rmsd,
    index = best_i,
    first_resno_to_add_to_index = segment_info.start_resno,
    sites_cart = sc[best_i:best_i+distance_group_info.n_residues] \
      if best_i is not None else None,
    file_name = file_name,
    pdb_text = file_name.replace("_trimmed.pdb.pkl.gz","") \
       if file_name else None,
    segment_id = segment_info.segment_id,
    final_rmsd = None,
   )
  return info
if __name__=="__main__":
  args = sys.argv[1:]
  if 'distort' in args:
    run_distort(get_distort_params())
  elif "database" in args:
    import libtbx.load_env
    segment_lib_dir = libtbx.env.find_in_repositories(
       relative_path=os.path.join("chem_data","segment_lib"),
       test=os.path.isdir)
    args.remove("database")
    use_filtered_pdb = (not "pdb100" in args)
    skip_indexing = (not "include_indexing" in args)
    if not skip_indexing:
      print("Including indexing (not used later)")
    if use_filtered_pdb:
      print("Using filtered PDB (90% seq identity, 60% complete)")
      source_dir = 'db'
      output_dir = os.path.join(segment_lib_dir, 'pdb2018_filtered')
      if skip_indexing:
        output_database_file_name = 'fragment_db_top2018.pkl.gz'
      else:
        output_database_file_name = 'indexed_fragment_db_top2018.pkl.gz'
    else:
      print("Using 100% chain coverage PDB ")
      source_dir = 'pdb100'
      output_dir = os.path.join(segment_lib_dir, 'pdb100_fragment_db')
      output_database_file_name = 'fragment_db_pdb100.pkl.gz'
    print("Source dir will be %s" %(os.path.join(os.getcwd(),source_dir)))
    print("Output dir will be %s" %(output_dir))
    print("Output db file will be %s" %(output_database_file_name))
    make_database(
     source_dir = source_dir,
     output_dir = output_dir,
     output_database_file_name = output_database_file_name,
     skip_indexing = skip_indexing,
     log = sys.stdout)
  else:
    output_file = None
    maximum_insertions_or_deletions = 0
    run_all = False
    pdb_exclude_list = None
    pdb_include_list = None
    data_dir = None
    nproc = 4
    n_res = None
    orig_args = deepcopy(args)
    for arg in orig_args:
      print(arg)
      if arg.startswith("nproc="):
        nproc= int(arg.replace("nproc=",""))
        args.remove(arg)
      elif arg.startswith("indel="):
        maximum_insertions_or_deletions = int(arg.replace("indel=",""))
        args.remove(arg)
      elif arg.startswith("n_res="):
        n_res = int(arg.replace("n_res=",""))
        args.remove(arg)
      elif arg.startswith("output_file="):
        output_file = arg.replace("output_file=","")
        args.remove(arg)
        print ("output_file: %s" %(output_file))
      elif arg.startswith("data_dir="):
        data_dir = arg.replace("data_dir=","")
        args.remove(arg)
        print ("data_dir: %s" %(data_dir))
      elif arg == 'all':
        run_all = True
        args.remove(arg)
      elif arg.startswith("pdb_exclude_list="):
        pdb_exclude_list = arg.replace("pdb_exclude_list=","").split()
        args.remove(arg)
        print ("pdb_exclude_list: %s" %(pdb_exclude_list))
      elif arg.startswith("pdb_include_list="):
        pdb_include_list = arg.replace("pdb_include_list=","").split()
        args.remove(arg)
        print ("pdb_include_list: %s" %(pdb_include_list))
    target_file = args[0]
    if len(args) > 1:
      file_name_list_file = args[1]
      file_name_list = open(file_name_list_file).read().split()
      print("File names:",file_name_list)
    else:
      file_name_list = None
    print("Target file :",target_file)
    if run_all:
      result = run_find_all_fragments(target_file = target_file,
       maximum_insertions_or_deletions = maximum_insertions_or_deletions,
       pdb_exclude_list = pdb_exclude_list,
       pdb_include_list = pdb_include_list,
       nproc = nproc)
    else:
      result = run_find_fragment(target_file = target_file,
       maximum_insertions_or_deletions = maximum_insertions_or_deletions,
       n_res = n_res,
       pdb_exclude_list = pdb_exclude_list,
       pdb_include_list = pdb_include_list,
       data_dir = data_dir,
       nproc = nproc)
    if not result:
      print("No result for ",target_file)
      result = group_args(info_list = [])
    i = 0
    for info in result.info_list:
      i+=1
      if output_file and i == 1:
        f=open(output_file, 'w')
      else:
        f=open('result_%s.pdb' %(i), 'w')
      print(info.model.model_as_pdb(), file  = f)
      f.close()
