"""
    This part of pipeline 1 is dedicated to construal value computation, which is later used for both sampling 
    for synthetic data generation and inference logic. The code computes values of all possible construals 
    (given some parameters), and samples construals based on computed values, 
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
from examples.CoDec_Research.code.config import local_config, server_config


# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]


# |START TIMER
start_time = time.perf_counter()

####################################################
################ SET EXP PARAMETERS ################
####################################################

curr_config = server_config

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
dataset_path = curr_config['dataset_path']
processID = dataset_path.split('/')[-2]                 # Used for storing and retrieving relevant data

# |Set simulator config
max_agents = training_config.max_controlled_agents      # Get total vehicle count
num_parallel_envs = curr_config['num_parallel_envs']
total_envs = curr_config['total_envs']
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






# TODO (Post NeurIPS): Optimize code
# 1. Reduce redunduncy in baseline data (use data-class to save data)
# 2. Convert for loops to list comprihension in env_torch.py: function get_structured_observation
# 3. We might have to reevaluate our measure of construal utilities or use other data
#   --- This is great for inferring discrete values of one parameter 
#   --- We might need more expressive utility values as our problem becomes more complex



#############################################
################ SIMULATIONS ################
#############################################




### Compute construal  utilities through simulator sampling ###
default_values = None

#2# |Check if saved construal utility data is available
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




### Select Construals for Baseline Data ###
scene_constr_dict = None

#2# |Check if saved construal sampling data is available
for srFile in simulation_results_files:
    if "sampled_construals" in srFile:
        with open(srFile, 'rb') as opn_file:
            scene_constr_dict = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):
            print(f"Using sampled construal information from file: {srFile}")
            break
        else:
            scene_constr_dict = None

if scene_constr_dict is None:
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

    scene_constrFile = simulation_results_path + processID + "_" + "sampled_construals_"+str(datetime.now())+".pickle"
    scene_constr_dict["params"] = heuristic_params
    save_pickle(scene_constrFile, scene_constr_dict, "Sampled construals")

    env.close(); env_multi_agent.close()




# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")