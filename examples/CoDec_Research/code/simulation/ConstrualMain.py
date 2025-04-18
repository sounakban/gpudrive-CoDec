# |Local import
from construal_functions import simulate_construal_policies


# |Default library imports
from copy import deepcopy
from functools import cache
from os import listdir
import json
from datetime import datetime

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, List, Tuple
import time

start_time = time.perf_counter()


# |Set root for GPUDrive import
import os
from pathlib import Path

## |Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)


# |GPUDrive imports
import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet
from tqdm import tqdm

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv, GPUDriveConstrualEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config


##############################################
################### CONFIG ###################
##############################################

# |Location to store simulation results
out_dir = "examples/CoDec_Research/results/simulation_results/"

# |Model Config (on which model was trained)
config = load_config("examples/experimental/config/reliable_agents_params")
# print(config)

# |Set scenario path
# dataset_path='data/processed/examples'
# dataset_path='data/processed/training'
dataset_path = 'data/processed/construal'

# |Set simulator config
max_agents = config.max_controlled_agents   # Get total vehicle count
num_parallel_envs = 7
total_envs = 25
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# |Set construal config
construal_size = 1
observed_agents_count = max_agents - 1      # Agents observed except self (used for vector sizes)
sample_size = 10                            # Number of samples to calculate expected utility of a construal

# |Other changes to variables
config.max_controlled_agents = 1    # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))

# Set params
env_config = dataclasses.replace(
    EnvConfig(),
    ego_state=config.ego_state,
    road_map_obs=config.road_map_obs,
    partner_obs=config.partner_obs,
    reward_type=config.reward_type,
    norm_obs=config.norm_obs,
    dynamics_model=config.dynamics_model,
    collision_behavior=config.collision_behavior,
    dist_to_goal_threshold=config.dist_to_goal_threshold,
    polyline_reduction_threshold=config.polyline_reduction_threshold,
    remove_non_vehicles=config.remove_non_vehicles,
    lidar_obs=config.lidar_obs,
    disable_classic_obs=config.lidar_obs,
    obs_radius=config.obs_radius,
    steer_actions = torch.round(
        torch.linspace(-torch.pi, torch.pi, config.action_space_steer_disc), decimals=3  
    ),
    accel_actions = torch.round(
        torch.linspace(-4.0, 4.0, config.action_space_accel_disc), decimals=3
    ),
)




#############################################################
################### INSTANTIATE VARIABLES ###################
#############################################################

# |Create data loader
train_loader = SceneDataLoader(
    root=dataset_path,
    batch_size=num_parallel_envs,
    dataset_size=max(total_envs,num_parallel_envs),
    sample_with_replacement=False,
)

# |Make env [Original]
# env = GPUDriveTorchEnv(
#     config=env_config,
#     data_loader=train_loader,
#     max_cont_agents=config.max_controlled_agents,
#     device=device,
# )

# |Make env [Construal]
env = GPUDriveConstrualEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=config.max_controlled_agents,
    device=device,
)

# |Import Pre-trained Model
sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")





##################################################
################### MAIN LOGIC ###################
##################################################

# |Create multi-agent environment to get information about moving vehicles
env_multi_agent = GPUDriveConstrualEnv(
                    config=env_config,
                    data_loader=train_loader,
                    max_cont_agents=max_agents,
                    device=device,
                    )


if __name__ == "__main__":

    construal_values = {}
    all_obs = {}

    # |Loop through all batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
        total=len(train_loader), colour="blue"):
        # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        # |Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # |Get moving vehicle information
        env_multi_agent.swap_data_batch(batch)
        moving_veh_mask = env_multi_agent.cont_agent_mask
        moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in moving_veh_mask]
        print("Indices of all moving vehicles (by scene): ", moving_veh_indices)
        control_mask = env.cont_agent_mask

        # |Simulate on Construals
        construal_values_, all_obs_ = simulate_construal_policies(env = env, 
                                                            observed_agents_count = observed_agents_count,
                                                            construal_size= construal_size,
                                                            total_envs = num_parallel_envs,
                                                            max_agents = max_agents,
                                                            moving_veh_indices = moving_veh_indices,
                                                            sample_size = sample_size,
                                                            sim_agent = sim_agent,
                                                            control_mask = control_mask,
                                                            device = device)
        construal_values.update(construal_values_)
        all_obs.update(all_obs_)

    with open(out_dir+"construal_vals_"+str(datetime.now())+".txt", 'w') as file:
        file.write(str(construal_values))
    with open(out_dir+"all_obs_"+str(datetime.now())+".txt", 'w') as file:
        file.write(str(all_obs))

    
    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    env.close()