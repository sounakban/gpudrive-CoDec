"""Script containing all functions calls necessary for testing different parts of the construal pipeline."""


from copy import deepcopy
from functools import cache
from os import listdir
import json
import pickle
from datetime import datetime

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
from examples.CoDec_Research.code.gpuDrive_utils import get_gpuDrive_vars
from examples.CoDec_Research.code.analysis.evaluate_construal_actions import evaluate_construals, get_best_construals_likelihood




####################################################
################ SET EXP PARAMETERS ################
####################################################

### Parameters for Inference ###
construal_selection_logic = "heuristic_1" # Set Construal Selection Logic. 
                                          # Options: "log_likelihood", "heuristic_1", "heuristic_2", "heuristic_3"
heuristic_params = {"dist_ego": 0.5} # Parameters for weighing the heuristics

### Specify Environment Configuration ###

# |Location to store (and find pre-computed) simulation results
simulation_results_path = "examples/CoDec_Research/results/simulation_results/"

# |Location to store simulation results
out_dir = "examples/CoDec_Research/results/simulation_results/"

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")

# |Set scenario path
dataset_path = 'data/processed/construal'

# |Set simulator config
max_agents = training_config.max_controlled_agents   # Get total vehicle count
num_parallel_envs = 3
total_envs = 12
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# |Set construal config
construal_size = 1
observed_agents_count = max_agents - 1      # Agents observed except self (used for vector sizes)
sample_size = 1                             # Number of samples to calculate expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1    # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))



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





#################################################
################ LOG LIKELIHOODS ################
#################################################


### Generate Baseline Data ###
state_action_pairs = None

# |Check if saved data is available
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]
for scdFile in simulation_results_files:
    if "baseline_state_action_pairs" not in scdFile:
        continue
    with open(scdFile, 'rb') as opn_file:
        state_action_pairs = pickle.load(opn_file)

if state_action_pairs is None:
    state_action_pairs = generate_baseline_data(out_dir=out_dir,
                                                sim_agent=sim_agent,
                                                num_parallel_envs=num_parallel_envs,
                                                max_agents=max_agents,
                                                sample_size=3,
                                                device=device,
                                                train_loader=train_loader,
                                                env=env,
                                                env_multi_agent=env_multi_agent,
                                                generate_animations=False)





### Compute Construal Log Likelihoods ###
construal_action_likelihoods = None

# |Check if saved data is available
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]
for scdFile in simulation_results_files:
    if "log_likelihood_measures" not in scdFile:
        continue
    with open(scdFile, 'rb') as opn_file:
        construal_action_likelihoods = pickle.load(opn_file)

construal_action_likelihoods = evaluate_construals(state_action_pairs, construal_size, sim_agent, out_dir, device=device)

# |Clear memory for large variable, once it has served its purpose
del state_action_pairs


### Select Construals for Inference ###
scene_constr_dict = None
if construal_selection_logic == "log_likelihood":
    # |Select construals based on Minimum Log Likelihood values
    scene_constr_diff_dict = None

    # |Check if saved data is available
    simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]
    for scdFile in simulation_results_files:
        if "highest_construal_dict_log_likelihood_diff" in scdFile:
            with open(scdFile, 'rb') as opn_file:
                scene_constr_diff_dict = pickle.load(opn_file)
        elif "highest_construal_dict_log_likelihood" in scdFile:
            with open(scdFile, 'rb') as opn_file:
                scene_constr_dict = pickle.load(opn_file)
        else:
            continue       

    if scene_constr_dict is None:
        scene_constr_dict = get_best_construals_likelihood(construal_action_likelihoods, out_dir)
    if scene_constr_diff_dict is None:
        scene_constr_diff_dict = get_best_construals_likelihood(construal_action_likelihoods, out_dir, likelihood_key="log_likelihood_diff")

elif construal_selection_logic == "heuristic_1":
    # |Select construals based on heuristic 1 (distance from ego)
    #2# |Generate Construal Execution Values through simulator sampling
    default_values = None

    #3# |Check if saved data is available
    simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]
    for scdFile in simulation_results_files:
        if "construal_vals" in scdFile:
            with open(scdFile, 'rb') as opn_file:
                default_values = pickle.load(opn_file)
        else:
            continue

    if scene_constr_dict is None:
        default_values, _, _, _ = generate_all_construal_trajnval(out_dir=out_dir,
                                                        sim_agent=sim_agent,
                                                        observed_agents_count=observed_agents_count,
                                                        construal_size=construal_size,
                                                        num_parallel_envs=num_parallel_envs,
                                                        max_agents=max_agents,
                                                        sample_size=sample_size,
                                                        device=device,
                                                        train_loader=train_loader,
                                                        env=env,
                                                        env_multi_agent=env_multi_agent,
                                                        generate_animations=False)

    #2# |Generate Construal Heuristic Values (Heuristic 1: Distance from ego)
    scene_constr_dict = get_constral_heurisrtic_values(env, default_values, heuristic_params=heuristic_params)

else:
    raise ValueError("Construal selection logic not supported yet. Please use either 'log_likelihood' or 'heuristic_1'.")





### Generate Trajectories For Selected Construals ###

# |Reload environment
env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config = training_config,
                                                                                device = device,
                                                                                num_parallel_envs = num_parallel_envs,
                                                                                dataset_path = dataset_path,
                                                                                max_agents = max_agents,
                                                                                total_envs = total_envs,
                                                                                sim_agent_path= "daphne-cornelisse/policy_S10_000_02_27",
                                                                            )

generate_selected_construal_traj(out_dir=out_dir,
                                    sim_agent=sim_agent,
                                    observed_agents_count=observed_agents_count,
                                    construal_size=construal_size,
                                    num_parallel_envs=num_parallel_envs,
                                    max_agents=max_agents,
                                    sample_size=sample_size,
                                    device=device,
                                    train_loader=train_loader,
                                    env=env,
                                    env_multi_agent=env_multi_agent,
                                    selected_construals = scene_constr_dict,
                                    generate_animations=False,)






##################################################
################ CONSTRUAL VALUES ################
##################################################
