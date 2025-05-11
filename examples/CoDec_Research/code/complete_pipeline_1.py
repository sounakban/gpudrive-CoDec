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
from examples.CoDec_Research.code.analysis.evaluate_construal_actions import evaluate_construals, get_best_construals_likelihood


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
state_action_pairs = None

# |Check if saved data is available
for srFile in simulation_results_files:
    if "baseline_state_action_pairs_" in srFile:
        with open(srFile, 'rb') as opn_file:
            state_action_pairs = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in state_action_pairs.keys() for scene_path_ in train_loader.dataset) and \
                state_action_pairs["params"] == heuristic_params:
            print(f"Using synthetic baseline data from file: {srFile}")
            break
        else:
            state_action_pairs = None

if state_action_pairs is None:
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

# |Use parameters only for matching data identity (not needed beyond this point)
state_action_pairs.pop("params")




#####################################################
################ PARAMETER INFERENCE ################
#####################################################


### Compute Construal Log Likelihoods ###
construal_action_likelihoods = None

# |Check if saved data is available
for srFile in simulation_results_files:
    if "log_likelihood_measures" in srFile:
        with open(srFile, 'rb') as opn_file:
            construal_action_likelihoods = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in construal_action_likelihoods.keys() for scene_path_ in train_loader.dataset) and \
                construal_action_likelihoods["params"] == heuristic_params:
            print(f"Using log-likelihodd measures from file: {srFile}")
            break
        else:
            construal_action_likelihoods = None

if construal_action_likelihoods is None:
    # lambdaPath = simulation_results_path + f"lambda{heuristic_params['ego_distance']}_"
    construal_action_likelihoods = evaluate_construals(state_action_pairs, construal_size, sim_agent, device=device)

    # |Save data
    savefl_path = simulation_results_path+processID+"_"+"log_likelihood_measures_"+str(datetime.now())+".pickle"
    construal_action_likelihoods["params"] = heuristic_params # Save parameters for data generation
    save_pickle(savefl_path, construal_action_likelihoods, "Log Likelihood")

# |Use parameters only for matching data identity (not needed beyond this point)
construal_action_likelihoods.pop("params")

# |Clear memory for large variable, once it has served its purpose
del state_action_pairs



### DEBUG: Sanity Check ###
# scene_constr_dict = None
# scene_constr_diff_dict = None

# # |Check if saved data is available
# for srFile in simulation_results_files:
#     if "highest_construal_dict_log_likelihood_diff" in srFile:
#         with open(srFile, 'rb') as opn_file:
#             scene_constr_diff_dict = pickle.load(opn_file)
#     if "highest_construal_dict_log_likelihood" in srFile:
#         with open(srFile, 'rb') as opn_file:
#             scene_constr_dict = pickle.load(opn_file)

# if scene_constr_dict is None:
#     scene_constr_dict = get_best_construals_likelihood(construal_action_likelihoods, simulation_results_path)
# if scene_constr_diff_dict is None:
#     scene_constr_diff_dict = get_best_construals_likelihood(construal_action_likelihoods, simulation_results_path, likelihood_key="log_likelihood_diff")

# for scene_name_, scene_info_ in construal_action_likelihoods.items():
#     print(f"Scene: {scene_name_}")
#     for base_construal_name_, base_construal_info_ in scene_info_.items():
#         print_dict = {}
#         for test_construal_name_, test_construal_info_ in base_construal_info_.items():
#             for sample_num_, sample_info_ in test_construal_info_.items():
#                 # print_dict.update({(base_construal_name_, test_construal_name_, sample_num_): abs(sample_info_['log_likelihood_diff'])})
#                 print_dict.update({(base_construal_name_, test_construal_name_, sample_num_): sample_info_['log_likelihood']})
#         print(dict(sorted(print_dict.items(), key=lambda item: item[1])))





### Inference Logic ###

# |Get probability of lambda values
get_constral_heurisrtic_values_partial = partial(get_constral_heurisrtic_values, env=env, 
                                                 train_loader=train_loader, default_values=default_values)
p_lambda = {}
curr_heuristic_params = deepcopy(heuristic_params)
for curr_lambda in np.linspace(0,1,11):
    curr_lambda = curr_lambda.item()
    curr_heuristic_params["ego_distance"] = curr_lambda
    curr_heuristic_values = get_constral_heurisrtic_values_partial(heuristic_params=curr_heuristic_params)
    p_lambda[curr_lambda] = {}

    for scene_name, sampled_construals in construal_action_likelihoods.items():
        p_lambda[curr_lambda][scene_name] = {}
        for base_construal, base_construal_info in sampled_construals.items():

            p_lambda[curr_lambda][scene_name][base_construal] = []
            for traj_sample, traj_sample_info in base_construal_info.items():
                curr_p_lambda = []
                for test_construal, test_construal_info in traj_sample_info.items():
                    construal_heur_value = curr_heuristic_values[scene_name][test_construal]
                    # p_a = torch.exp( -1*test_construal_info['log_likelihood'].type(torch.float64) ).item()
                    p_a = test_construal_info['likelihood'].item()
                    curr_p_lambda.append(p_a*construal_heur_value)
                p_lambda[curr_lambda][scene_name][base_construal].append( torch.log(torch.sum(torch.tensor(curr_p_lambda, dtype=torch.float64))) )
            p_lambda[curr_lambda][scene_name][base_construal] = -1*torch.sum(torch.tensor(p_lambda[curr_lambda][scene_name][base_construal])).item()

# |Get product over lambda probability across sampled construals
lamda_inference = {}
for curr_lambda, scene_info in p_lambda.items():
    lamda_inference[curr_lambda] = np.sum([val for scene_name, construal_info in scene_info.items() for val in construal_info.values() if val < np.inf])
print(lamda_inference)





### Convert Results to Pandas Table and Save ###
import pandas as pd

construal_action_likelihoods_df = {(scene,baseC,testC,sample): construal_action_likelihoods[scene][baseC][sample][testC]['log_likelihood'].item()
                                        for scene in construal_action_likelihoods.keys() 
                                        for baseC in construal_action_likelihoods[scene].keys() 
                                        for sample in construal_action_likelihoods[scene][baseC].keys()
                                        for testC in construal_action_likelihoods[scene][baseC][sample].keys()
                                        }


multi_index = pd.MultiIndex.from_tuples(construal_action_likelihoods_df.keys(), names=['scene', 'base_construal', 'test_construal', 'sample'])
construal_action_likelihoods_df = pd.DataFrame(construal_action_likelihoods_df.values(), index=multi_index)
construal_action_likelihoods_df.columns = ['-log_likelihood']

construal_action_likelihoods_summarydf = construal_action_likelihoods_df.groupby(level=(0,1,2)).mean().sort_values(by='-log_likelihood', ascending=True).\
                                            groupby(level=(0,1)).head(5).sort_index(level=(0,1), sort_remaining=False)


heuristic_params_str = '_'.join([heur_+str(param_) for heur_, param_ in heuristic_params.items()])
# construal_action_likelihoods_df.to_csv(simulation_results_path + f"lambda{heuristic_params['ego_distance']}_construal_action_likelihoods.tsv", sep="\t", index=True, header=True)
construal_action_likelihoods_df.to_csv(simulation_results_path + processID + "_" + f"construal_action_likelihoods_{heuristic_params_str}.tsv", sep="\t", index=True, header=True)
# construal_action_likelihoods_summarydf.to_csv(simulation_results_path + f"lambda{heuristic_params['ego_distance']}_construal_action_likelihoods_summary.tsv", sep="\t", index=True, header=True)
construal_action_likelihoods_summarydf.to_csv(simulation_results_path + processID + "_" + f"construal_action_likelihoods_summary_{heuristic_params_str}.tsv", sep="\t", index=True, header=True)



# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")