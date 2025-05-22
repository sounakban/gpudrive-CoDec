"""
    This part of pipeline 1 is dedicated to synthetic data generation. The code generates synthetic 
    data based on sampled construals (previous stage of pipeline).
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

from typing import Any, List, Tuple
import time

import torch
import dataclasses


# |Set root for GPUDrive import
import os
import sys
from pathlib import Path

from traitlets import default

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))


# |GPUDrive imports
from gpudrive.utils.config import load_config
from examples.CoDec_Research.code.simulation.construal_main import generate_baseline_data, generate_selected_construal_traj, \
                                                                    get_constral_heurisrtic_values, generate_all_construal_trajnval
from examples.CoDec_Research.code.gpuDrive_utils import get_gpuDrive_vars, save_pickle
from examples.CoDec_Research.code.config import active_config


# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]


# |START TIMER
start_time = time.perf_counter()

####################################################
################ SET EXP PARAMETERS ################
####################################################

curr_config = active_config

# Parameters for Inference
heuristic_params = {"ego_distance": 0.5, "cardinality": 1}              # Hueristics and their weight parameters (to be inferred)

construal_count_baseline = curr_config['construal_count_baseline']      # Number of construals to sample for baseline data generation
trajectory_count_baseline = curr_config['trajectory_count_baseline']    # Number of baseline trajectories to generate per construal


### Specify Environment Configuration ###

# |Location to store (and retrieve pre-computed) simulation results
simulation_results_path = "examples/CoDec_Research/results/simulation_results/"
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")

# |Set scenario path
dataset_path = 'data/processed/construal/Set3.1/'
processID = dataset_path.split('/')[-2]                 # Used for storing and retrieving relevant data

# |Set simulator config
max_agents = training_config.max_controlled_agents      # Get total vehicle count
num_parallel_envs = 1
total_envs = 1
device = eval(curr_config['device'])

# |Set construal config
construal_size = 1
observed_agents_count = max_agents - 1                              # Agents observed except self (used for vector sizes)
sample_size_utility = curr_config['sample_size_utility']            # Number of samples to compute expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1                           # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))


env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config=training_config,
                                                                                device=device,
                                                                                num_parallel_envs=num_parallel_envs,
                                                                                dataset_path=dataset_path,
                                                                                max_agents=max_agents,
                                                                                total_envs=total_envs,
                                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                        )

env.close(); env_multi_agent.close()
del env_config, train_loader, env, env_multi_agent, sim_agent
print("+++++++++ 1 ++++++++")
time.sleep(10)


env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config=training_config,
                                                                                device=device,
                                                                                num_parallel_envs=num_parallel_envs,
                                                                                dataset_path=dataset_path,
                                                                                max_agents=max_agents,
                                                                                total_envs=total_envs,
                                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                        )

env.close(); env_multi_agent.close()
del env_config, train_loader, env, env_multi_agent, sim_agent
print("+++++++++ 2 +++++++++")
time.sleep(10)

env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config=training_config,
                                                                                device=device,
                                                                                num_parallel_envs=num_parallel_envs,
                                                                                dataset_path=dataset_path,
                                                                                max_agents=max_agents,
                                                                                total_envs=total_envs,
                                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                        )

env.close(); env_multi_agent.close()
del env_config, train_loader, env, env_multi_agent, sim_agent
print("+++++++++ 3 +++++++++")
time.sleep(10)
