"""
    Reduce code redundancy
    Usage: Start script with 'from shared_imports import *'
"""

from copy import deepcopy
from functools import cache
from os import listdir
import json
import pickle
import gc
from datetime import datetime
from functools import partial

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, List, Tuple, Dict
import time

import torch
import dataclasses
from tqdm import tqdm


# # |Set root for GPUDrive import
# import os
# import sys
# from pathlib import Path

# # Set working directory to the base directory 'gpudrive'
# working_dir = Path.cwd()
# while working_dir.name != 'gpudrive-CoDec':
#     working_dir = working_dir.parent
#     if working_dir == Path.home():
#         raise FileNotFoundError("Base directory 'gpudrive' not found")
# os.chdir(working_dir)
# sys.path.append(str(working_dir))


# |GPUDrive imports
from gpudrive.utils.config import load_config

# |CoDec imports
from examples.CoDec_Research.code.gpuDrive_utils import *
from examples.CoDec_Research.code.shared_config import *

# # |Lower level imports
# from examples.CoDec_Research.code.simulation.construal_main import *
# from examples.CoDec_Research.code.analysis.evaluate_construal_actions import *

# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]



####################################################
################ SET EXP PARAMETERS ################
####################################################

curr_config = get_active_config()

construal_count_baseline = curr_config['construal_count_baseline']      # Number of construals to sample for baseline data generation
trajectory_count_baseline = curr_config['trajectory_count_baseline']    # Number of baseline trajectories to generate per construal


### Specify Environment Configuration ###

# |Location to store (and retrieve pre-computed) simulation results
simulation_results_path = curr_config["simulation_results_path"]
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")

# |Set scenario path
dataset_path = curr_config['dataset_path']
processID = dataset_path.split('/')[-2]                 # Used for storing and retrieving relevant data

# |Set simulator config
moving_veh_count = training_config.max_controlled_agents      # Get total vehicle count
num_parallel_envs = curr_config['num_parallel_envs']
total_envs = curr_config['total_envs']
device = eval(curr_config['device'])

# |Set construal config
construal_size = curr_config['construal_size']
observed_agents_count = moving_veh_count - 1                              # Agents observed except self (used for vector sizes)
sample_size_utility = curr_config['sample_size_utility']            # Number of samples to compute expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1                           # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))