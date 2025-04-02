import os
from pathlib import Path

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

###########################################################################################

import torch
import dataclasses
import mediapy
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import ModelCard
from gpudrive.networks.late_fusion import NeuralNet

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv, GPUDriveConstrualEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

from os import listdir
import json

import numpy as np
import itertools

###########################################################################################

# Configs model has been trained with
config = load_config("examples/experimental/config/reliable_agents_params")
print(config)

# datase_path='data/processed/examples'
dataset_path='data/processed/training'
# datase_path = 'data/processed/construal'

max_agents = config.max_controlled_agents
num_envs = 150
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Other changes to variables
config.max_controlled_agents = 1    # Control only the first vehicle in the environment
num_envs = min(num_envs, len(os.listdir(dataset_path)))

###########################################################################################

# Create data loader
train_loader = SceneDataLoader(
    root=dataset_path,
    batch_size=num_envs,
    dataset_size=max(100,num_envs),
    sample_with_replacement=False,
)

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

# |Make env
# env = GPUDriveTorchEnv(
#     config=env_config,
#     data_loader=train_loader,
#     max_cont_agents=config.max_controlled_agents,
#     device=device,
# )

# |Make env
env = GPUDriveConstrualEnv(
    config=env_config,
    data_loader=train_loader,
    max_cont_agents=config.max_controlled_agents,
    device=device,
)

control_mask = env.cont_agent_mask


###########################################################################################

def straightness_ratio(curve):
    """Calculates the straightness of a curve using the path length ratio method.

    Args:
        curve: A list or numpy array of 2D points representing the curve.

    Returns:
        The straightness ratio.
    """
    curve = np.array(curve)
    total_length = np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
    straight_distance = np.sqrt(np.sum((curve[0] - curve[-1])**2))
    return total_length / straight_distance if straight_distance > 0 else 0

###########################################################################################

# |Get objects with most complex trajectories

def fileterScenes_1(dirpath, sceneCount):
    """Select cars based on first non-zero velocity"""
    scenario_files = [dirpath+fl_name for fl_name in listdir(dirpath)]
    traj_ratios = {}
    for scFile in scenario_files:
        with open(scFile, 'r') as opn_file:
            data = json.load(opn_file)
        veh_num = 0
        for obj in data['objects']:
            # Logic to select first vehicle with non-zero velocity
            all_velocities = [[veldict['x'],veldict['y']] for veldict in obj['velocity'] if veldict['x']!=-10000] # -10000 velocities mean object no longer exists
            all_velocities = [item for sublist in all_velocities for item in sublist] # Flatten list
            total_velocity = sum(all_velocities)/len(all_velocities) # Get avg velocity accross dimensions
            # TODO: Find a better way to identify controlled vehicles
            if total_velocity > 0:
                #2# |Weed out parked cars
                all_positions = [[posdict['x'],posdict['y'],posdict['z']] for posdict in obj['position'] if posdict['x']!=-10000]
                traj_ratios[scFile] = {'veh_num': veh_num, 'traj_ratio' : straightness_ratio(all_positions)}
                break # move on to next file after getting ratio for first moving vehicle
            veh_num += 1

    # traj_ratios

    sorted_dict = dict(sorted(traj_ratios.items(), key=lambda item: item[1]['traj_ratio'], reverse = True))
    # sorted_dict
    traj_ratios_topN = {entry[0]:entry[1] for n,entry in enumerate(sorted_dict.items()) if n < sceneCount}

    return traj_ratios_topN



def fileterScenes_2(env, sceneCount):
    """Select cars based on simulator logic velocity"""
    # |Get file and controlled vehicle lists
    data_files = env.data_batch
    control_mask = env.cont_agent_mask

    # |Logic to select controlled vehicle using environment control  mask
    traj_ratios = {}
    for scFile, veh_num in zip(data_files, control_mask.nonzero()):
        veh_num = veh_num[1]
        with open(scFile, 'r') as opn_file:
            data = json.load(opn_file)
        contVeh_pos = data['objects'][veh_num]['position']
        all_positions = [[posdict['x'],posdict['y'],posdict['z']] for posdict in contVeh_pos if posdict['x']!=-10000]
        traj_ratios[scFile] = {'veh_num': veh_num, 'traj_ratio' : straightness_ratio(all_positions)}

    # traj_ratios

    sorted_dict = dict(sorted(traj_ratios.items(), key=lambda item: item[1]['traj_ratio'], reverse = True))
    # sorted_dict
    traj_ratios_topN = {entry[0]:entry[1] for n,entry in enumerate(sorted_dict.items()) if n < sceneCount}

    return traj_ratios_topN

###########################################################################################


    
dirpath = 'data/processed/training/'
sceneCount = 20
# traj_ratios_topN = fileterScenes_1(dirpath, sceneCount)
traj_ratios_topN = fileterScenes_2(env, sceneCount)
print(traj_ratios_topN)