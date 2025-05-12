"""This part of pipeline 1 is dedicated to synthetic data generation. The code (1) Computes values of all possible
    construals (given heuristic parameters), samples construals based on their values, and generates synthetic 
    data based on sampled construals. The second part performs bayesian inference to retrieve the original value of 
    the parameters."""

from copy import deepcopy
from functools import cache
from os import listdir
import json
import pickle
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


# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]


# |START TIMER
start_time = time.perf_counter()

####################################################
################ SET EXP PARAMETERS ################
####################################################


# Parameters for Inference
heuristic_params = {"ego_distance": 0.5, "cardinality": 1} # Hueristics and their weight parameters (to be inferred)

construal_count_baseline = 2 # Number of construals to sample for baseline data generation
trajectory_count_baseline = 3 # Number of baseline trajectories to generate per construal


### Specify Environment Configuration ###

# |Location to store (and retrieve pre-computed) simulation results
simulation_results_path = "examples/CoDec_Research/results/simulation_results/"
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")

# |Set scenario path
dataset_path = 'data/processed/construal/Set2/'
processID = dataset_path.split('/')[-2]        # Used for storing and retrieving relevant data

# |Set simulator config
max_agents = training_config.max_controlled_agents   # Get total vehicle count
num_parallel_envs = 25
total_envs = 25
# device = "cpu" # cpu just because we're in a notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# |Set construal config
construal_size = 1
observed_agents_count = max_agents - 1      # Agents observed except self (used for vector sizes)
sample_size_utility = 40                    # Number of samples to compute expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1    # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))


# TODO (Post NeurIPS): Optimize code
# 1. Reduce redunduncy in baseline data (use data-class to save data)
# 2. Convert for loops to list comprihension in env_torch.py: function get_structured_observation
# 3. We might have to reevaluate our measure of construal utilities or use other data
#   --- This is great for inferring discrete values of one parameter 
#   --- We might need more expressive utility values as our problem becomes more complex


### Instantiate Variables ###

env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config = training_config,
                                                                                device = device,
                                                                                num_parallel_envs = num_parallel_envs,
                                                                                dataset_path = dataset_path,
                                                                                max_agents = max_agents,
                                                                                total_envs = total_envs,
                                                                                sim_agent_path= "daphne-cornelisse/policy_S10_000_02_27",
                                                                            )






#############################################
################ SIMULATIONS ################
#############################################



### Select Construals for Baseline Data ###
scene_constr_dict = None

# |Compute construal  utilities through simulator sampling
default_values = None

#2# |Check if saved data is available
for srFile in simulation_results_files:
    if "construal_vals" in srFile:
        with open(srFile, 'rb') as opn_file:
            default_values = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in default_values.keys() for scene_path_ in train_loader.dataset):
            print(f"Using construal values from file: {srFile}")
            break
        else:
            default_values = None

if default_values is None:
    default_values, traj_obs, ground_truth, _ = generate_all_construal_trajnval(sim_agent=sim_agent,
                                                                                observed_agents_count=observed_agents_count,
                                                                                construal_size=construal_size,
                                                                                num_parallel_envs=num_parallel_envs,
                                                                                max_agents=max_agents,
                                                                                sample_size=sample_size_utility,
                                                                                device=device,
                                                                                train_loader=train_loader,
                                                                                env=env,
                                                                                env_multi_agent=env_multi_agent,
                                                                                generate_animations=False)
    #3# |Save data
    savefl_path = simulation_results_path+processID+"_"+"construal_vals_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, default_values, "Construal value")
    savefl_path = simulation_results_path+processID+"_"+"constr_traj_obs_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, traj_obs, "Trajectory observation")
    savefl_path = simulation_results_path+processID+"_"+"ground_truth_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, ground_truth, "Ground truth")
    #3# Free up memory
    del traj_obs, ground_truth



# |Generate Construal Heuristic Values
heuristic_values = get_constral_heurisrtic_values(env, train_loader, default_values, heuristic_params=heuristic_params)

# |Sample construals for generating baseline data
def sample_construals(heuristic_values: dict, sample_count: int) -> dict:
    """
    Sample construals based on heuristic values.
    """
    sampled_construals = {}
    for scene_name, construal_info in heuristic_values.items():
        constr_indices, constr_values = zip(*construal_info.items())
        sampled_indices = torch.multinomial(torch.tensor(constr_values), num_samples=sample_count, \
                                                replacement=False).tolist()
        sampled_construals[scene_name] = {constr_indices[i]: constr_values[i] for i in sampled_indices}
        print(f"Sampled construals for scene {scene_name}: {sampled_construals[scene_name].keys()}")

    return sampled_construals

scene_constr_dict = sample_construals(heuristic_values, sample_count=construal_count_baseline)






### Generate Synthetic Ground Truth for Selected Construals (Baseline Data on Which to Perform Inference) ###
data_subset_paths = [dataset_path[:-1]+'.'+str(i)+'/' for i in range(1,6)]

for curr_dataset_path in data_subset_paths:
    if not os.path.isdir(curr_dataset_path):
        # |Skip if directory does not exist
        continue

    num_parallel_envs = total_envs = len(listdir(curr_dataset_path))
    state_action_pairs = None

    env.close(); env_multi_agent.close()
    del env, env_multi_agent
    time.sleep(5)       # Let madrona clear memory to avoid multiple parallel instances of GPUDrive
    
    env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                    training_config = training_config,
                                                                                    device = device,
                                                                                    num_parallel_envs = num_parallel_envs,
                                                                                    dataset_path = curr_dataset_path,
                                                                                    max_agents = max_agents,
                                                                                    total_envs = total_envs,
                                                                                    sim_agent_path= "daphne-cornelisse/policy_S10_000_02_27",
                                                                                )

    # |Check if saved data is available
    for srFile in simulation_results_files:
        if "baseline_state_action_pairs_" in srFile:
            with open(srFile, 'rb') as opn_file:
                state_action_pairs = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in state_action_pairs.keys() for scene_path_ in train_loader.dataset) and \
                    state_action_pairs["params"] == heuristic_params:
                print(f"Synthetic baseline data for current batch already exists in file: {srFile}")
                break
            else:
                state_action_pairs = None

    if state_action_pairs is None and \
        all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):

        lambdaPath = simulation_results_path + f"lambda{heuristic_params['ego_distance']}_"
        state_action_pairs = generate_baseline_data(sim_agent=sim_agent,
                                                    num_parallel_envs=num_parallel_envs,
                                                    max_agents=max_agents,
                                                    sample_size=trajectory_count_baseline,
                                                    device=device,
                                                    train_loader=train_loader,
                                                    env=env,
                                                    env_multi_agent=env_multi_agent,
                                                    observed_agents_count=observed_agents_count,
                                                    construal_size=construal_size,
                                                    selected_construals=scene_constr_dict,
                                                    generate_animations=False)
        #2# |Save data
        savefl_path = simulation_results_path+processID+"_"+"baseline_state_action_pairs_"+str(datetime.now())+".pickle"
        state_action_pairs["params"] = heuristic_params # Save parameters for data generation
        save_pickle(savefl_path, state_action_pairs, "Baseline")



# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")