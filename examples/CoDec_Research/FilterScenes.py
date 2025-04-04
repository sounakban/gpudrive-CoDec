""" The code selects scenes which have the most complex primary vehicle trajectories """

import math
import os
from pathlib import Path

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
print("Reached root driectory")

###########################################################################################

import torch
import dataclasses

from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv, GPUDriveConstrualEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config
from tqdm import tqdm

from os import listdir
import json

from shapely.geometry import LineString
import numpy as np
import math
import itertools

print ("Imported libraries")

###########################################################################################

# Configs model has been trained with
config = load_config("examples/experimental/config/reliable_agents_params")
print(config)

# dataset_path='data/processed/examples/'
dataset_path='data/processed/training/'
# dataset_path = 'data/processed/construal/'

max_agents = config.max_controlled_agents
num_parallel_envs = 2
total_envs = 8
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Other changes to variables
config.max_controlled_agents = 1    # Control only the first vehicle in the environment
num_parallel_envs = min(num_parallel_envs, len(os.listdir(dataset_path)))

print("Set up configs")

###########################################################################################

def get_simulation_environment(dataset_path):
    # Create data loader
    train_loader = SceneDataLoader(
        root=dataset_path,
        batch_size=num_parallel_envs,
        dataset_size=max(total_envs,num_parallel_envs),
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

    # control_mask = env.cont_agent_mask

    print("Set up environment complete")

    return (train_loader, env)



###########################################################################################

def get_moving_vehicles(all_veh_objs, first_only = False):
    '''Create list of all moving vehicles in an environment

    Args:
        all_veh_objs: A list of vehicle objects in the environment
        first_only: breaks the loop after finding the first moving vehicle and only returns details for that object.

    Returns:
        A list or a single vehicle objects.
    '''
    moving_veh = []
    for obj in all_veh_objs:
        all_velocities = [[veldict['x'],veldict['y']] for veldict in obj['velocity'] if veldict['x']!=-10000] # -10000 velocities mean object no longer exists
        all_velocities = [item for sublist in all_velocities for item in sublist] # Flatten list
        total_velocity = sum(all_velocities)/len(all_velocities) # Get avg velocity accross dimensions
        if abs(total_velocity) > 0:
            moving_veh.append(obj)
            if first_only: break
    return moving_veh



def get_topN_traj(traj_ratios, sceneCount):
    """Given trajectory ratios for multiple scenes, get top sceneCount scenes with highest complexity"""
    sorted_dict = dict(sorted(traj_ratios.items(), key=lambda item: item[1]['traj_complexity'], reverse = True))
    # sorted_dict
    traj_ratios_topN = {entry[0]:entry[1] for n,entry in enumerate(sorted_dict.items()) if n < sceneCount}

    return traj_ratios_topN




def straightness_ratio(curve):
    """
    Calculates the straightness of a curve using the path length ratio method.

    Args:
        curve: A list or numpy array of 2D points representing the curve.

    Returns:
        The straightness ratio.
    """
    curve = np.array(curve)
    total_length = np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
    straight_distance = np.sqrt(np.sum((curve[0] - curve[-1])**2))
    traj_ratio = total_length / straight_distance if straight_distance > 0 else -1
    return (total_length, straight_distance, traj_ratio)




def get_trajectory_intersection(trajectory1, trajectory2):
    """
    Calculates the intersection point(s) of two trajectories.

    Args:
        trajectory1: List of coordinate tuples representing the first trajectory.
        trajectory2: List of coordinate tuples representing the second trajectory.

    Returns:
        A list of coordinate tuples representing the intersection points,
        or an empty list if no intersection occurs.
    """
    line1 = LineString(trajectory1)
    line2 = LineString(trajectory2)
    intersection = line1.intersection(line2, grid_size=3)

    if intersection.is_empty:
        return []
    elif intersection.geom_type == 'Point':
        return [intersection.coords[0]]
    elif intersection.geom_type == 'MultiPoint':
        return [point.coords[0] for point in intersection.geoms]
    elif intersection.geom_type == 'LineString':
      return list(intersection.coords)
    elif intersection.geom_type in ['MultiLineString', "GeometryCollection"]:
      return [intersect_point for intersect_line in intersection.geoms for intersect_point in intersect_line.coords]
    else:
        print("Intersection type ", intersection.geom_type, " did not match any predefined types")
        return []
    # Example usage:
    trajectory_a = [(0, 0), (1, 1), (2, 2)]
    trajectory_b = [(0, 2), (1, 1), (2, 0)]
    intersection_points = get_trajectory_intersection(trajectory_a, trajectory_b)
    print(f"Intersection points: {intersection_points}")

    trajectory_c = [(0, 0), (1, 1)]
    trajectory_d = [(2, 2), (3, 3)]
    intersection_points_2 = get_trajectory_intersection(trajectory_c, trajectory_d)
    print(f"Intersection points: {intersection_points_2}")

    trajectory_e = [(0,0), (1,0)]
    trajectory_f = [(0.5,0),(1.5,0)]
    intersection_points_3 = get_trajectory_intersection(trajectory_e, trajectory_f)
    print(f"Intersection points: {intersection_points_3}")




def get_traj_complexity(veh_obj, all_veh_objs):
    """
    Function that computes all trajectory complexity using different measures.

    Args:
        veh_obj: a dictionary object containing all information for a specific variable.
        all_veh_objs: list of all vehicle objects

    Returns:
        A dictionary with all complexity-related measures.
    """
    # |Complexity based on trajectory ratio
    # all_positions = [[posdict['x'],posdict['y'],posdict['z']] for posdict in veh_obj['position'] if posdict['x']!=-10000]
    # total_length, straight_distance, traj_ratio = straightness_ratio(all_positions)
    # traj_complexity = (traj_ratio-1)*total_length
    # traj_complexity_dict = {'veh_ID': veh_obj['id'], 'traj_ratio' : traj_ratio, 'straight_distance' : straight_distance, \
    #                         'path_len' : total_length, 'traj_complexity' : traj_complexity}
    # |Complexity based path intersections
    moving_vehs = get_moving_vehicles(all_veh_objs)
    # veh1_pos = [[posdict['x'],posdict['y'],posdict['z']] for posdict in veh_obj['position'] if posdict['x']!=-10000]
    veh1_pos = [[posdict['x'],posdict['y']] for posdict in veh_obj['position'] if posdict['x']!=-10000]
    if len(veh1_pos) < 2: 
        intersection_list.append(0)
        return  {'veh_ID': veh_obj['id'], 'traj_complexity' : -1}
    intersection_list = []
    for veh_obj2 in moving_vehs:
        if veh_obj2 == veh_obj:
            continue
        # veh2_pos = [[posdict['x'],posdict['y'],posdict['z']] for posdict in veh_obj2['position'] if posdict['x']!=-10000]
        veh2_pos = [[posdict['x'],posdict['y']] for posdict in veh_obj2['position'] if posdict['x']!=-10000]
        if len(veh2_pos) < 2: 
            intersection_list.append(0)
            continue
        intersection_list.append(len(get_trajectory_intersection(veh1_pos, veh2_pos)))
    intersection_list = list(filter(lambda num: num != 0, intersection_list))   # Remove zero interactions
    traj_complexity = sum(intersection_list) * ( len(intersection_list)/(math.log2(len(intersection_list)+1)+1) )
    traj_complexity_dict = {'veh_ID': veh_obj['id'], 'traj_complexity' : traj_complexity, 'intersection_list': intersection_list}

    return traj_complexity_dict




###########################################################################################

def fileterScenes_1(dirpath):
    """Select primary vehicle based on first non-zero velocity"""
    scenario_files = [dirpath+fl_name for fl_name in listdir(dirpath)]
    traj_complexity = {}
    for scFile in scenario_files:
        with open(scFile, 'r') as opn_file:
            data = json.load(opn_file)
        moving_veh = get_moving_vehicles(data['objects'], first_only = True)    # Get first moving vehicle
        traj_complexity[scFile] = get_traj_complexity(moving_veh[0], data['objects'])
    return traj_complexity



def fileterScenes_2(dirpath):
    """Select primary vehicle based on sdc vehicle index"""
    scenario_files = [dirpath+fl_name for fl_name in listdir(dirpath)]
    traj_complexity = {}
    for scFile in scenario_files:
        print("Processing file ", scFile, "\r")
        with open(scFile, 'r') as opn_file:
            data = json.load(opn_file)
        sdc_veh_indx = data['metadata']['sdc_track_index']
        sdc_veh_data = data['objects'][sdc_veh_indx]
        traj_complexity[scFile] = get_traj_complexity(sdc_veh_data, data['objects'])
    return traj_complexity



def fileterScenes_3(env):
    """Select primary vehicle based on simulator logic velocity"""
    # |Get file and controlled vehicle lists
    data_files = env.data_batch
    control_mask = env.cont_agent_mask

    # |Logic to select controlled vehicle using environment control  mask
    traj_complexity = {}
    for scFile, veh_num in zip(data_files, control_mask.nonzero()):
        veh_num = veh_num[1]
        with open(scFile, 'r') as opn_file:
            data = json.load(opn_file)
        contVeh = data['objects'][veh_num]
        traj_complexity[scFile] = get_traj_complexity(contVeh, data['objects'])
    return traj_complexity





###########################################################################################


vehicle_selection_logic = 2
final_sceneCount = 20

# |Get Trajectory ratios
if vehicle_selection_logic == 1:
    #2# |Get trajectory rations based on custom logic (for primary-vehicle selection)
    all_traRatios = fileterScenes_1(dataset_path)
elif vehicle_selection_logic == 2:
    #2# |Get trajectory rations based on custom logic (for primary-vehicle selection)
    all_traRatios = fileterScenes_2(dataset_path)
elif vehicle_selection_logic == 3:
    #2# |Get trajectory rations based on environment logic (for primary-vehicle selection)
    train_loader, env = get_simulation_environment(dataset_path)
    all_traRatios = {}
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
        total=len(train_loader), colour="blue"):
        #3# |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        #3# |Update simulator with the new batch of data
        env.swap_data_batch(batch)
        #3# |Get Trajectory ratios
        all_traRatios.update(fileterScenes_3(env))




# |Get top-N trajectories
traj_complexity_topN = get_topN_traj(all_traRatios,final_sceneCount)


# Formatted print
# import pprint
# pprint.pprint(traj_complexity_topN)


# Copy to N files to custom directory
import shutil

for fl in traj_complexity_topN.keys():
    shutil.copyfile(fl, 'data/processed/construal/'+fl.split('/')[-1])