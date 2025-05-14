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
from tqdm import tqdm


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
from examples.CoDec_Research.code.gpuDrive_utils import get_gpuDrive_vars, get_mov_veh_masks, save_pickle
from examples.CoDec_Research.code.config import get_active_config


# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]


# |START TIMER
start_time = time.perf_counter()

####################################################
################ SET EXP PARAMETERS ################
####################################################

curr_config = get_active_config()

# Parameters for Inference
heuristic_params = {"ego_distance": 0.5, "cardinality": 1}              # Hueristics and their weight parameters (to be inferred)

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
num_parallel_envs = curr_config['num_parallel_envs_light']
total_envs = curr_config['total_envs']
device = eval(curr_config['device'])

# |Set construal config
construal_size = curr_config['construal_size']
observed_agents_count = moving_veh_count - 1                              # Agents observed except self (used for vector sizes)
sample_size_utility = curr_config['sample_size_utility']            # Number of samples to compute expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1                           # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))

moving_veh_masks = get_mov_veh_masks(
                                    training_config=training_config, 
                                    device=device, 
                                    dataset_path=dataset_path,
                                    max_agents=moving_veh_count,
                                    result_file_loc=simulation_results_path,
                                    processID=processID
                                    )

env_config, train_loader, env, sim_agent = get_gpuDrive_vars(
                                                            training_config=training_config,
                                                            device=device,
                                                            num_parallel_envs=num_parallel_envs,
                                                            dataset_path=dataset_path,
                                                            total_envs=total_envs,
                                                            sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                            )







#############################################
################ SIMULATIONS ################
#############################################

### Retrieve Saved Construal Sampling Resuts ###
scene_constr_dict = None

#2# |Check if saved construal sampling data is available
for srFile in simulation_results_files:
    if "sampled_construals" in srFile:
        with open(srFile, 'rb') as opn_file:
            scene_constr_dict = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):
            print(f"Using sampled construal information from file: {srFile}")
            file_params = scene_constr_dict.pop('params')
            break
        else:
            scene_constr_dict = None
if scene_constr_dict is None:
    raise FileNotFoundError("Could not find saved file for sampled construals for current scenes")

### Generate Synthetic Ground Truth for Selected Construals (Baseline Data on Which to Perform Inference) ###

# |Loop through all files in batches
for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                    total=len(train_loader), colour="blue"):
    # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
    # |Update simulator with the new batch of data
    env.swap_data_batch(batch)

    state_action_pairs = None

    # |Check if saved data is available
    curr_dataset_scenes = set(env_path2name(scene_path_) for scene_path_ in env.data_batch)
    for srFile in simulation_results_files:
        if "baseline_state_action_pairs" in srFile:
            with open(srFile, 'rb') as opn_file:
                state_action_pairs = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            fileScenes = set(state_action_pairs.keys()); fileScenes.remove('params'); fileScenes.remove('dict_structure')
            # print(fileScenes)
            # print(curr_dataset_scenes)
            if fileScenes == curr_dataset_scenes and state_action_pairs["params"] == heuristic_params:
                print(f"Synthetic baseline data for current batch already exists in file: {srFile}")
                break
            else:
                state_action_pairs = None

    if state_action_pairs is None and \
        all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in env.data_batch):

        print(f"Could not find baseline data for current batch. Now computing.")
                
        lambdaPath = simulation_results_path + f"lambda{heuristic_params['ego_distance']}_"
        state_action_pairs = generate_baseline_data(sim_agent=sim_agent,
                                                    num_parallel_envs=num_parallel_envs,
                                                    max_agents=moving_veh_count,
                                                    sample_size=trajectory_count_baseline,
                                                    device=device,
                                                    env=env,
                                                    moving_veh_masks=moving_veh_masks,
                                                    observed_agents_count=observed_agents_count,
                                                    construal_size=construal_size,
                                                    selected_construals=scene_constr_dict,
                                                    generate_animations=False)
                
        #2# |Save data
        savefl_path = simulation_results_path+processID+"_"+"baseline_state_action_pairs_"+str(datetime.now())+".pickle"
        state_action_pairs["params"] = heuristic_params # Save parameters for data generation
        save_pickle(savefl_path, state_action_pairs, "Baseline")
        # |Clear memory for large variable
        del state_action_pairs
        gc.collect()




env.close()

# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")