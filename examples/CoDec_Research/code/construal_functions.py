# |Default library imports
from copy import deepcopy
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
def get_construals(total_obj_count: int, 
                   target_obj_indices: tuple, 
                   construal_size: int, 
                   expanded_mask: bool = False):
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
        construal_info['default'] = ((), torch.tensor(expand_construal_mask([False,]*total_obj_count)))
    else:
        construal_info['default'] = ((), torch.tensor([False,]*total_obj_count))
    return construal_info




def get_construal_byIndex(total_obj_count: int, 
                          target_obj_indices: List, 
                          construal_size: int, 
                          indx: int, 
                          expanded_mask: bool = False):
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
    all_construals = get_construals(total_obj_count, target_obj_indices, construal_size, expanded_mask)
    if indx in all_construals.keys():
        return all_construals[indx]
    else:
        # If index out of bounds, return default construal
        return all_construals['default']
    



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

