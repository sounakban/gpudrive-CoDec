# |Default library imports
from copy import deepcopy
from functools import cache
import json

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, List, Tuple


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
import mediapy
from gpudrive.networks.late_fusion import NeuralNet

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig




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






@cache
def get_construals(total_obs_count: int, target_obj_indices: tuple, construal_size: int):
    '''
    Create construed masks based on complete mask and objects of interest

    Args:
        total_obs_count: Total number of observed objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual

    Returns:
        Dictionary with construal indices as keys and coorresponding masks (boolean lists) as values.
            The dictionary also containts a default entry for '[]', with no onbservable object
    '''
    construal_size = construal_size if construal_size < len(target_obj_indices) else len(target_obj_indices)
    construal_indices_list = combinations(target_obj_indices, construal_size)
    construal_info = dict()
    for construal_num, construal_indices in enumerate(construal_indices_list):
        # |Mask all non-contrual target objects
        # curr_mask = [True if i in target_obj_indices else False for i in range(total_obs_count)]  # Mask all target objects
        # curr_mask = [False if i in construal_indices else val for i, val in enumerate(curr_mask)]  # Unmask objects in construal
        # |OR
        curr_mask = [True if i in target_obj_indices and i not in construal_indices else False for i in range(total_obs_count)]
        construal_info[construal_num] = (construal_indices, curr_mask)
    construal_info['default'] = ((), [False,]*total_obs_count)    # Default construal where all vehicles are observed
    return construal_info


def get_construal_byIndex(total_obs_count, target_obj_indices, construal_size, indx):
    '''
    Create construed masks based on complete mask and objects of interest

    Args:
        total_obs_count: Total number of observed objects (used to determine length of mask)
        target_obj_indices: A list of indices containing all objects of interest in the bollean list
        construal_size: Size of each contrual
        indx: The construal number

    Returns:
        Tuple of construal object indices and coorresponding mask (boolean list).
        If index is greater than number of constrauls it returns a default value, with no observable objects
    '''
    if indx in get_construals(total_obs_count, target_obj_indices, construal_size).keys():
        return get_construals(total_obs_count, target_obj_indices, construal_size)[indx]
    else:
        # If index out of bounds, return default construal
        return get_construals(total_obs_count, target_obj_indices, construal_size)['default']
    




def run_policy(env: GPUDriveTorchEnv,
               sim_agent: NeuralNet,
               next_obs: torch.Tensor,
               control_mask: List,
               construal_masks: List,
               time_step: int,
               total_envs: int,
               max_agents: int,
               device: str,
               frames: dict,
               const_num: int,
               sample_num: int,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    # |Predict actions
    action, _, _, _, _ = sim_agent(
        next_obs[control_mask], deterministic=False
    )
    action_template = torch.zeros(
        (total_envs, max_agents), dtype=torch.int64, device=device
    )
    action_template[control_mask] = action.to(device)

    # |Step
    env.step_dynamics(action_template)

    # |Render
    sim_states = env.vis.plot_simulator_state(
        env_indices=list(range(total_envs)),
        time_steps=[time_step]*total_envs,
        zoom_radius=70,
    )
    
    if construal_masks:
        for env_num in range(total_envs):
            frames[f"env_{env_num}-constr_{const_num}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num])) 
    else:
        for env_num in range(total_envs):
            frames[f"env_{env_num}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num])) 

    next_obs = env.get_obs(partner_mask=construal_masks)
    reward = env.get_rewards()
    done = env.get_dones()
    info = env.get_infos()

    #3# Populate variables for trajectory illustrations
    plottable_obs = env.get_structured_obs()

    return next_obs, reward, done, info, plottable_obs





def simulate_construal_policies(env: GPUDriveTorchEnv, 
                        observed_agents_count: int,
                        construal_size: int,
                        total_envs: int,
                        max_agents: int,
                        moving_veh_indices: List,
                        sample_size: int,
                        sim_agent: NeuralNet,
                        control_mask: List,
                        device: str):
    """
    Simulate environment under construed observation spaces
    
    Args:
        env: GPUDrive simulation environment
        observed_agents_count: Number of agents being observed by model agent
        construal_size: The number of agents beings observed in each construal
        total_envs: Number of scenarios being simulated
        max_agents: Maximum number of (tracked) agents in the environment
        moving_veh_indices: Indices of (moving) vehicles of interest in the environment
        sample_size: The number of samples to draw to estimate value of construals
        sim_agent: The (pre-trained) agent model
        control_mask: The mask specifying which agent(s) is controlled by the model
        device: Whether use CPU or GPU for computation
    
    Returns (two dictionaries):
        construal_values: Dictionary that contains the expected utility of each construal
                            Structure: {scene_name: {construal_mask: expected_utility}}
        all_obs: Dictionary that contains the observations (agent trajectories) for each construal
                    Structure: {scene_name: {construal_mask: {sample_num: [vehicles,timestep,coord]}}}

    """
    construal_values = {env_name: {} for env_name in env.data_batch}               # Dictionary that contains the expected utility per construal
    all_obs = {env_name: {} for env_name in env.data_batch}                        # Dictionary that contains the observations (agent trajectories) for each construal
    for const_num in range(math.comb(observed_agents_count,construal_size)):
        # |Repeat rollout for each construal

        # next_obs = env.reset()
        # print("Observation shape: ", next_obs.shape)

        #2# |Define observation mask for construal
        construal_masks = [] 
        for scene_num, _ in enumerate(env.data_batch):
            construal_masks.append( get_construal_byIndex(max_agents, moving_veh_indices[scene_num], construal_size, const_num) )
        #3# |get_construal_byIndex produces masks of shape [scenes,objs], reshape to [scenes,objs,obs]
        tmp = []
        mask_indices = ()
        for mask_info in construal_masks:
            curr_indices, mask = mask_info
            curr_masks = [list(mask) for _ in range(len(mask))]     # Create multiple copies of the mask, one for each vehicle
            [msk.pop(i) for i, msk in enumerate(curr_masks)]        # Remove ego-vehicle entry from the mask
            tmp.append(curr_masks)
            mask_indices += (tuple(curr_indices),)
        construal_masks = tmp
        
        frames = {f"env_{env_num}-constr_{const_num}-sample_{sample_num}": [] for sample_num in range(sample_size) for env_num in range(total_envs)}
        curr_samples = []   # Keep track of reards
        for sample_num in range(sample_size):
            print("\tsample ", sample_num)
            
            _ = env.reset()
            next_obs = env.get_obs(partner_mask=construal_masks)
            for time_step in range(env.episode_len):
                #2# |Roll out policy for a specific construal
                print(f"\r\t\tStep: {time_step}", end="", flush=True)

                #3# |Execute policy
                next_obs, reward, done, info, plottable_obs = run_policy(
                    env=env,
                    sim_agent=sim_agent,
                    next_obs=next_obs,
                    control_mask=control_mask,
                    construal_masks=construal_masks,
                    time_step=time_step,
                    total_envs=total_envs,
                    max_agents=max_agents,
                    device=device,
                    frames=frames,
                    const_num=const_num,
                    sample_num=sample_num,
                )

                #3# |Record observations for each construal
                for env_num, all_pos in enumerate(plottable_obs['pos_ego']):
                    all_pos = torch.stack(all_pos, dim=1).unsqueeze(1)      # Reshape from [vehicles,coord] to [vehicles,1,coord] for timesteps
                    env_name = env.data_batch[env_num]
                    if mask_indices[env_num] in all_obs[env_name] and sample_num in all_obs[env_name][mask_indices[env_num]]:
                        all_obs[env_name][mask_indices[env_num]][sample_num] = torch.cat([all_obs[env_name][mask_indices[env_num]][sample_num], all_pos], dim=1)
                    elif mask_indices[env_num] not in all_obs[env_name]:
                        all_obs[env_name][mask_indices[env_num]] = {sample_num : all_pos}                        
                    elif sample_num not in all_obs[env_name][mask_indices[env_num]]:
                        all_obs[env_name][mask_indices[env_num]][sample_num] = all_pos
                    else:
                        raise ValueError(f"Unknown Situation: {env_name}, {mask_indices[env_num]}, {sample_num}")
                
                if done.all():
                    break
            print() # Change to new line after step prints
                
            curr_samples.append(reward[control_mask].tolist())

        for env_num in range(total_envs):
            for sample_num in range(sample_size):
                #2# Convert from tensor to list for storage
                all_obs[env_name][mask_indices[env_num]][sample_num] = all_obs[env_name][mask_indices[env_num]][sample_num].tolist()

        #3# |Calculate value (average reward) for each construal
        curr_vals = [sum(x)/sample_size for x in zip(*curr_samples)]
        for env_num, val in enumerate(curr_vals):
            construal_values[env.data_batch[env_num]][mask_indices[env_num]] = val
        print("Processed masks: ", mask_indices, ", with values:", curr_vals)

        if all([mask == () for mask in mask_indices]):
            #3# |Break loop once list of construals for all scenarios have been exhausted
            break

        #2# |Save animations
        # mediapy.set_show_save_dir('./sim_vids')
        # mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

    # print("\nExpected utility by contrual: ", construal_values)
    
    # TODO: Extract valid flags, and ground truth trajectories
    return (construal_values, all_obs)








def simulate_full_policies(env: GPUDriveTorchEnv, 
                            total_envs: int,
                            max_agents: int,
                            sample_size: int,
                            sim_agent: NeuralNet,
                            control_mask: List,
                            device: str):
    """
    Simulate environment under construed observation spaces
    
    Args:
        env: GPUDrive simulation environment
        total_envs: Number of scenarios being simulated
        max_agents: Maximum number of (tracked) agents in the environment
        sample_size: The number of samples to draw to estimate value of construals
        sim_agent: The (pre-trained) agent model
        control_mask: The mask specifying which agent(s) is controlled by the model
        device: Whether use CPU or GPU for computation
    
    Returns (two dictionaries):
        construal_values: Dictionary that contains the expected utility of each construal
                            Structure: {scene_name: {construal_mask: expected_utility}}
        all_obs: Dictionary that contains the observations (agent trajectories) for each construal
                    Structure: {scene_name: {construal_mask: {sample_num: [vehicles,timestep,coord]}}}

    """
    model_values = {env_name: {} for env_name in env.data_batch}               # Dictionary that contains the expected utility
    all_obs = {env_name: {} for env_name in env.data_batch}                        # Dictionary that contains the observations (agent trajectories) 
    frames = {f"env_{i}": [] for i in range(total_envs)}
    curr_samples = []   # Keep track of reards
    for sample_num in range(sample_size):
        print("\tsample ", sample_num)
        _ = env.reset()
        next_obs = env.get_obs()
        for time_step in range(env.episode_len):
            #2# |Roll out policy for a specific construal
            print(f"\r\t\tStep: {time_step}", end="", flush=True)

            #3# |Execute policy
            next_obs, reward, done, info, plottable_obs = run_policy(
                env=env,
                sim_agent=sim_agent,
                next_obs=next_obs,
                control_mask=control_mask,
                construal_masks=None,
                time_step=time_step,
                total_envs=total_envs,
                max_agents=max_agents,
                device=device,
                frames=frames,
                const_num=-1,
                sample_num=sample_num,
            )

            #3# |Record observations for each construal
            for env_num, all_pos in enumerate(plottable_obs['pos_ego']):
                all_pos = torch.stack(all_pos, dim=1).unsqueeze(1)      # Reshape from [vehicles,coord] to [vehicles,1,coord] for timesteps
                env_name = env.data_batch[env_num]
                if sample_num in all_obs[env_name]:
                    all_obs[env_name][sample_num] = torch.cat([all_obs[env_name][sample_num], all_pos], dim=1)
                elif sample_num not in all_obs[env_name]:
                    all_obs[env_name][sample_num] = all_pos
                else:
                    raise ValueError(f"Unknown Situation: {env_name}, {sample_num}")
            
            if done.all():
                break
        print() # Change to new line after step prints
            
        curr_samples.append(reward[control_mask].tolist())

    for env_num in range(total_envs):
        for sample_num in range(sample_size):
            #2# Convert from tensor to list for storage
            all_obs[env_name][sample_num] = all_obs[env_name][sample_num].tolist()

    #3# |Calculate value (average reward) for each construal
    curr_vals = [sum(x)/sample_size for x in zip(*curr_samples)]
    for env_num, val in enumerate(curr_vals):
        model_values[env.data_batch[env_num]] = val

    #2# |Save animations
    # mediapy.set_show_save_dir('./sim_vids')
    # mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

    # print("\nExpected utility by contrual: ", construal_values)
    return (model_values, all_obs)