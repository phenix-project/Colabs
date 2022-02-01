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

from phenix_alphafold_utils import clear_directories


#####################  Imports for ColabFold and alphafold ###########

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace

if os.path.isfile('AF2_READY'):

  import numpy as np
  import pickle
  from alphafold.common import protein
  from alphafold.data import templates
  from alphafold.model import data
  from alphafold.model import config
  from alphafold.model import model
  from alphafold.data.tools import hhsearch

  import warnings
  from absl import logging

  # plotting libraries
  import py3Dmol
  import matplotlib.pyplot as plt
  import ipywidgets
  from ipywidgets import interact, fixed, GridspecLayout, Output


  from alphafold.data.templates import (_get_pdb_id_and_chain,
                                    _process_single_hit,
                                    _assess_hhsearch_hit,
                                    _build_query_to_hit_index_mapping,
                                    _extract_template_features,
                                    SingleHitResult,
                                    TEMPLATE_FEATURES)


