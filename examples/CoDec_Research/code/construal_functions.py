# |Default library imports
from copy import deepcopy
from curses import raw
from functools import cache
import json
from unittest import result

import torch
from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, List, Tuple


# |Set root for GPUDrive import
import os
from pathlib import Path

from zmq import device

from gpudrive.env.env_torch import GPUDriveConstrualEnv



# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]



def get_moving_vehicles(all_veh_objs, first_only = False):
    '''
    Create list of all moving vehicles in an environment

    Args:
        all_veh_objs: A list of vehicle objects in the environment
        first_only: breaks the loop after finding the first moving vehicle and only returns details for that object.

    Returns:
        A list or a single vehicle objects.
    '''
    moving_veh_list = []
    for i, obj in enumerate(all_veh_objs):
        all_velocities = [[veldict['x'],veldict['y']] for veldict in obj['velocity'] if veldict['x']!=-10000] # -10000 velocities mean object no longer exists
        all_velocities = [item for sublist in all_velocities for item in sublist] # Flatten list
        total_velocity = sum(all_velocities)/len(all_velocities) # Get avg velocity accross dimensions
        if abs(total_velocity) > 0:
            obj['index'] = i
            moving_veh_list.append(obj)
            if first_only: break
    return moving_veh_list



def expand_construal_mask(constr_mask: list):
    '''
    Expand construal mask from [objects] to [objects, observations] to (where observations = objects - 1). 
        That is, specify the observation mask for each object in the environment.

    Args:
        construal_mask: A boolean list with construals
        construal_indices: A tuple of indices containing all objects of interest in the bollean list
        total_obj_count: Total number of objects (used to determine length of mask)

    Returns:
        A boolean list of lists with all objects in the environment
    '''
    expanded_mask = [list(constr_mask) for _ in range(len(constr_mask))]     # Create multiple copies of the mask, one for each vehicle
    [msk_.pop(i) for i, msk_ in enumerate(expanded_mask)]                    # Remove ego-vehicle entry from each mask
    return expanded_mask





@cache
def get_construals( total_obj_count: int, 
                    target_obj_indices: tuple, 
                    construal_size: int, 
                    expanded_mask: bool = False,
                    device: str = 'cpu',):
    '''
    Create construed masks based on complete mask and objects of interest

    Args:
        total_obj_count: Total number of objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual
        expanded_mask: If True, return the expanded mask of shape [objects, observations]
                        If False, return the construal indices and mask of shape [objects]

    Returns:
        Dictionary with construal indices as keys and coorresponding masks (boolean lists) as values.
            The dictionary also containts a default entry for '[]', with no onbservable object
    '''
    construal_size = construal_size if construal_size < len(target_obj_indices) else len(target_obj_indices)
    construal_indices_list = combinations(target_obj_indices, construal_size)
    construal_info = dict()
    for construal_num, construal_indices in enumerate(construal_indices_list):
        # |Mask all non-contrual target objects
        curr_mask = [True if i in target_obj_indices and i not in construal_indices else False for i in range(total_obj_count)]
        if expanded_mask:
            curr_mask = expand_construal_mask(curr_mask)
        construal_info[construal_num] = (construal_indices, curr_mask)
    # |Default construal where all vehicles are observed
    if expanded_mask:
        construal_info['default'] = ((), expand_construal_mask([False,]*total_obj_count))
    else:
        construal_info['default'] = ((), [False,]*total_obj_count)
    return construal_info




def get_construal_byIndex(total_obj_count: int, 
                          target_obj_indices: List, 
                          construal_size: int, 
                          indx: int, 
                          expanded_mask: bool = False,
                          device: str = 'cpu',):
    '''
    Create construed masks based on complete mask and objects of interest

    Args:
        total_obj_count: Total number of objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual
        indx: The construal number
        expanded_mask: If True, return the expanded mask of shape [objects, observations]
                        If False, return the construal indices and mask of shape [objects]

    Returns:
        Tuple of construal object indices and coorresponding mask (boolean list).
        If index is greater than number of constrauls it returns a default value, with no observable objects
    '''
    all_construals = get_construals(total_obj_count, target_obj_indices, construal_size, expanded_mask, device)
    if indx in all_construals.keys():
        return all_construals[indx]
    else:
        # If index out of bounds, return default construal
        return all_construals['default']
    



def get_selected_construal_byIndex(total_obj_count: int,
                                    target_obj_indices: List,
                                    construal_size: int,
                                    indx: int, 
                                    selected_construals: dict,
                                    expanded_mask: bool = False,
                                    device: str = 'cpu'):
    '''
    Create construed masks based on complete mask and objects of interest

    Args:
        total_obj_count: Total number of objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual
        indx: The construal number
        selected_construals: Dictionary containing the selected construals for each scene
        expanded_mask: If True, return the expanded mask of shape [objects, observations]
                        If False, return the construal indices and mask of shape [objects]

    Returns:
        Tuple of construal object indices and coorresponding mask (boolean list).
        If index is greater than number of constrauls it returns a default value, with no observable objects
    '''
    all_construals = get_construals(total_obj_count, target_obj_indices, construal_size, expanded_mask, device)
    selected_construal_indices = list(selected_construals.keys())
    selected_construal_info = [curr_constr_info_ for curr_constr_info_ in all_construals.values() if curr_constr_info_[0] in selected_construal_indices]
    return selected_construal_info[indx]



def get_construal_count(total_obs_count, target_obj_indices, construal_size):
    '''
    Get the number of construals given number of objects of interest and construal size

    Args:
        total_obs_count: Total number of observed objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual

    Returns:
        The number of construals
    '''
    return len(get_construals(total_obs_count, target_obj_indices, construal_size))



#######################################################
################# HEURISTICS FUNCTIONS ################
#######################################################

### Support Functions ###
euclidean_distance = lambda point1, point2: math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))



### Generate Construal Heuristic Values (Heuristic 1: Distance from ego) ###
def get_construal_veh_distance(env: GPUDriveConstrualEnv, construal_indices: tuple, average: bool = True,
                               normalize: bool = False):
    '''
    Get the (average or) distance of each vehicle in the construal to the ego vehicle

    Args:
        env: The environment object
        construal_indices: A list of indices containing all objects of interest in the bollean list
        average: If true, return the average distance of all vehicles in the construal to the ego vehicle

    Returns:
        The average distance or a list of distances from the ego vehicle to each vehicle in the construal
    '''
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    # |Populate dictionary will all relevant information
    info_dict = dict()
    for env_num, env_name in enumerate(curr_data_batch):
        info_dict[env_name] = dict()
        info_dict[env_name]['ego_index'] = torch.where(env.cont_agent_mask[env_num])[0].item()
        info_dict[env_name]['construal_indices'] = construal_indices[env_name]
    
    # |Get all vehicle distances
    all_pos = env.get_data_log_obj().pos_xy
    distance_dict = dict()

    for env_num, env_name in enumerate(curr_data_batch):
        distance_dict[env_name] = dict()
        all_distances = [euclidean_distance(all_pos[env_num][info_dict[env_name]['ego_index']][0].cpu().numpy(),
                                            all_pos[env_num][i][0].cpu().numpy()) 
                            for i in range(len(all_pos[env_num]))]  
              
        if normalize:
            #2# |Normalize distances to [0,1] using min-max scaling 
            all_distances = (np.array(all_distances) - np.min(all_distances)) / (np.max(all_distances) - np.min(all_distances))

        for curr_indices in info_dict[env_name]['construal_indices']:
            distance_dict[env_name][curr_indices] = [all_distances[i] for i in curr_indices]
            if average:
                if len(distance_dict[env_name][curr_indices]) > 0:
                    distance_dict[env_name][curr_indices] = sum(distance_dict[env_name][curr_indices])/len(distance_dict[env_name][curr_indices])
                else:
                    distance_dict[env_name][curr_indices] = 0

    return distance_dict