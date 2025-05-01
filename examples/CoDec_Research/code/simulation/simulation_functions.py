# |Default library imports
from copy import deepcopy
from functools import cache
import json

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, Dict, List, Tuple

from torch.distributions.utils import logits_to_probs


# |Set root for GPUDrive import
import os
import sys
from pathlib import Path

## |Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))

# |Local imports
from examples.CoDec_Research.code.construal_functions import get_construal_byIndex, get_construal_count, get_construals
from examples.CoDec_Research.code.analysis.evaluate_construal_actions import process_state

# |GPUDrive imports
import torch
import mediapy
from gpudrive.networks.late_fusion import NeuralNet

from gpudrive.env.env_torch import GPUDriveConstrualEnv, GPUDriveTorchEnv
from gpudrive.visualize.utils import img_from_fig





# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]




def print_gpu_usage(device: str):
    """
    Print GPU usage
    """
    if torch.cuda.is_available() and torch.device("cuda") == device:
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / (1024 ** 2)
        utilization = torch.cuda.utilization()
        print(f"GPU usage: {utilization:3.2f}% and {mem_used_MB:3.2f}MB", end="\r")




def save_animations(sim_state_frames: dict, save_dir: str = './sim_vids'):
    """
    Save animations to a specified directory
    Args:
        sim_state_frames: Dictionary containing the frames for each environment
        save_dir: Directory to save the animations
    """
    print("Saving animations to: ", save_dir)
    sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}

    # |Display and save videos locally
    # mediapy.set_show_save_dir(save_dir)
    # mediapy.show_videos(sim_state_arrays, fps=15, width=500, height=500, columns=2, codec='gif')
    
    # |Save videos locally
    for env_id, frames in sim_state_arrays.items():        
        mediapy.write_video(
            str(save_dir+'/'+env_id+'.gif'),
            frames,
            fps=15,
            codec='gif',
        )






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
               generate_animations: bool = False,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    # |Predict actions
    action, _, _, _, action_probs = sim_agent(next_obs[control_mask], deterministic=False)

    action_template = torch.zeros(
        (total_envs, max_agents), dtype=torch.int64, device=device
    )
    action_template[control_mask] = action.to(device)

    # |Garb raw observations before stepping through environment for debug logic later
    # tmp1_ = env.get_obs(raw_obs=True)

    # |Step
    env.step_dynamics(action_template)

    #2# |Print GPU usage
    print_gpu_usage(device)

    # |Render
    if generate_animations:
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(total_envs)),
            time_steps=[time_step]*total_envs,
            zoom_radius=70,
        )
    
    if construal_masks:
        for env_num_, env_path_ in enumerate(env.data_batch):
            frames[f"env_{env_path2name(env_path_)}-constr_{const_num}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num_])) 
    else:
        for env_num_, env_path_ in enumerate(env.data_batch):
            frames[f"env_{env_path2name(env_path_)}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num_])) 

    next_obs = env.get_obs(partner_mask=construal_masks)
    reward = env.get_rewards()
    done = env.get_dones()
    info = env.get_infos()

    # |Variable for trajectory illustrations
    plottable_obs = env.get_structured_obs()

    # |DEBUG LOGIC: Check if state processing logic is operating correctly
    # tmp2_ind_, tmpt2_mask_ = get_construals(64, (1,), 1, expanded_mask = True)['default']
    # tmpt2_mask_ = torch.tensor(tmpt2_mask_)  # Get generalist mask
    # tmp1_ = [process_state(tmp1_tmp_, tmpt2_mask_, 10) for tmp1_tmp_ in tmp1_]
    # tmp1_ = torch.stack(tmp1_)
    # print("\nProcess obsrvation logic working:", (tmp1_==next_obs).all())
    # tmp_action_, _, _, _, tmp_action_probs_ = sim_agent(tmp1_[control_mask], deterministic=False)
    # print("Action probabilities match?: ", (tmp_action_probs_== action_probs).all())
    # print("Action match?: ", torch.max(torch.abs(logits_to_probs(tmp_action_probs_) - logits_to_probs(action_probs))))

    return next_obs, reward, done, info, plottable_obs, action_probs







def simulate_construal_policies(env: GPUDriveConstrualEnv, 
                        observed_agents_count: int,
                        construal_size: int,
                        total_envs: int,
                        max_agents: int,
                        moving_veh_indices: List,
                        sample_size: int,
                        sim_agent: NeuralNet,
                        control_mask: List,
                        device: str,
                        generate_animations: bool = False,
                        ) -> Tuple[dict, dict, dict]:
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
    loop_count = math.comb(observed_agents_count, construal_size)                  # Calculate the number of loops
    for const_num in range(loop_count):
        # |Repeat rollout for each construal

        # next_obs = env.reset()
        # print("Observation shape: ", next_obs.shape)

        #2# |Define observation mask for construal
        construal_info = [get_construal_byIndex(max_agents, moving_veh_indices[scene_num], construal_size, const_num, expanded_mask=True, device=device) 
                            for scene_num in range(len(env.data_batch))]
        mask_indices, construal_masks = zip(*construal_info)   # Unzip construal masks
                
        frames = {f"env_{env_path2name(env_path_)}-constr_{const_num}-sample_{sample_num_}": [] for sample_num_ in range(sample_size) for env_path_ in env.data_batch}
        curr_samples = []   # Keep track of rewards
        for sample_num in range(sample_size):
            print("\tsample ", sample_num)
            
            _ = env.reset()
            next_obs = env.get_obs(partner_mask=construal_masks)
            for time_step in range(env.episode_len):
                #2# |Roll out policy for a specific construal
                print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

                #3# |Execute policy
                next_obs, reward, done, info, plottable_obs, action_probs = run_policy( env=env,
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
                                                                                        generate_animations=generate_animations,
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

        #2# Convert from tensor to list for storage
        # for env_num, env_name in enumerate(env.data_batch):
        #     for sample_num in range(sample_size):
        #         all_obs[env_name][mask_indices[env_num]][sample_num] = all_obs[env_name][mask_indices[env_num]][sample_num].tolist()

        #2# |Calculate value (average reward) for each construal
        curr_vals = [sum(x)/sample_size for x in zip(*curr_samples)]
        for env_num, val in enumerate(curr_vals):
            construal_values[env.data_batch[env_num]][mask_indices[env_num]] = val
        print("Processed masks: ", mask_indices, ", with values:", curr_vals)

        if all([mask == () for mask in mask_indices]):
            #2# |Break loop once list of construals for all scenarios have been exhausted
            break

        #2# |Save animations
        if generate_animations:
            save_animations(frames)

    #2# Extract ground-truth data
    ground_truth = {'traj': {}, 'traj_valids': {}, 'contr_veh_indices': {}}
    for env_num, env_name in enumerate(env.data_batch):
        ground_truth['traj'][env_name] = env.get_data_log_obj().pos_xy[env_num].tolist()
        ground_truth['traj_valids'][env_name] = env.get_data_log_obj().valids[env_num].tolist()
        ground_truth['contr_veh_indices'][env_name] = torch.where(control_mask[env_num])[0].tolist()


    # print("\nExpected utility by contrual: ", construal_values)
    
    return (construal_values, all_obs, ground_truth)






def simulate_selected_construal_policies(env: GPUDriveConstrualEnv, 
                                        observed_agents_count: int,
                                        construal_size: int,
                                        total_envs: int,
                                        max_agents: int,
                                        moving_veh_indices: List[List],
                                        sample_size: int,
                                        sim_agent: NeuralNet,
                                        control_mask: List,
                                        device: str,
                                        selected_construals: Dict[str, List[Tuple[int]]],
                                        generate_animations: bool = False,
                                        ) -> Tuple[dict, dict, dict]:
    """
    Simulate environment under hand-picked construed observation spaces
    
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
        selected_construals: Dictionary containing construals to be simulated for each environment.
    
    Returns:

    """
    construal_values = {env_name: {} for env_name in env.data_batch}               # Dictionary that contains the expected utility per construal
    all_obs = {env_name: {} for env_name in env.data_batch}                        # Dictionary that contains the observations (agent trajectories) for each construal
    
    # |Get all construal info then filter for selected construals
    selected_construal_info = {scene_name_: [] for scene_name_ in env.data_batch}
    for scene_num, scene_name in enumerate(env.data_batch):
        curr_env_construals = get_construals(max_agents, moving_veh_indices[scene_num], construal_size, expanded_mask = True)
        curr_env_selected_construals = selected_construals[scene_name]
        selected_construal_info[scene_name]= []
        for constr_ind, _ in curr_env_selected_construals:
            selected_construal_info[scene_name].extend([constr_info_ for _, constr_info_ in 
                                                        curr_env_construals.items() if constr_info_[0] == constr_ind])
    
    loop_count = set([len(env_constr_) for env_constr_ in selected_construals.values()])    # Get the number of loops = number of selected construals per scene
    if len(loop_count) > 1:
        raise ValueError("Number of construals per scene must be the same")
    
    loop_count = loop_count.pop()
    print("Construals per scene: ", loop_count)
    for const_num in range(loop_count):
        # |Repeat rollout for each construal

        # next_obs = env.reset()
        # print("Observation shape: ", next_obs.shape)

        curr_mask_indices = [constr_info_[const_num][0] for constr_info_ in selected_construal_info.values()]
        curr_construal_masks = [constr_info_[const_num][1] for constr_info_ in selected_construal_info.values()]
        frames = {f"env_{env_path2name(env_path_)}-constr_{const_num}-sample_{sample_num_}": [] for sample_num_ in range(sample_size) for env_path_ in env.data_batch}
        curr_samples = []   # Keep track of rewards
        print("Processing construals: ", curr_mask_indices)
        for sample_num in range(sample_size):
            print("\tsample ", sample_num)
            
            _ = env.reset()
            next_obs = env.get_obs(partner_mask=curr_construal_masks)
            for time_step in range(env.episode_len):
                #2# |Roll out policy for a specific construal
                print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

                #3# |Execute policy
                next_obs, reward, done, info, plottable_obs, action_probs = run_policy( env=env,
                                                                                        sim_agent=sim_agent,
                                                                                        next_obs=next_obs,
                                                                                        control_mask=control_mask,
                                                                                        construal_masks=curr_construal_masks,
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
                    if curr_mask_indices[env_num] in all_obs[env_name] and sample_num in all_obs[env_name][curr_mask_indices[env_num]]:
                        all_obs[env_name][curr_mask_indices[env_num]][sample_num] = torch.cat([all_obs[env_name][curr_mask_indices[env_num]][sample_num], all_pos], dim=1)
                    elif curr_mask_indices[env_num] not in all_obs[env_name]:
                        all_obs[env_name][curr_mask_indices[env_num]] = {sample_num : all_pos}                        
                    elif sample_num not in all_obs[env_name][curr_mask_indices[env_num]]:
                        all_obs[env_name][curr_mask_indices[env_num]][sample_num] = all_pos
                    else:
                        raise ValueError(f"Unknown Situation: {env_name}, {curr_mask_indices[env_num]}, {sample_num}")
                
                if done.all():
                    break
            print() # Change to new line after step prints
                
            curr_samples.append(reward[control_mask].tolist())

        # #2# Convert from tensor to list for storage
        # for env_num, env_name in enumerate(env.data_batch):
        #     for sample_num in range(sample_size):
        #         all_obs[env_name][curr_mask_indices[env_num]][sample_num] = all_obs[env_name][curr_mask_indices[env_num]][sample_num].tolist()

        if all([mask == () for mask in curr_mask_indices]):
            #2# |Break loop once list of construals for all scenarios have been exhausted
            break


        #2# |Save animations
        if generate_animations:
            save_animations(frames)
            # mediapy.set_show_save_dir('./sim_vids')
            # mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

    #2# Extract ground-truth data
    # ground_truth = {'traj': {}, 'traj_valids': {}, 'contr_veh_indices': {}}
    # for env_num, env_name in enumerate(env.data_batch):
    #     ground_truth['traj'][env_name] = env.get_data_log_obj().pos_xy[env_num].tolist()
    #     ground_truth['traj_valids'][env_name] = env.get_data_log_obj().valids[env_num].tolist()
    #     ground_truth['contr_veh_indices'][env_name] = torch.where(control_mask[env_num])[0].tolist()


    # print("\nExpected utility by contrual: ", construal_values)
    
    # TODO: Extract valid flags, and ground truth trajectories
    return all_obs







def simulate_generalist_policies1(env: GPUDriveConstrualEnv, 
                                total_envs: int,
                                max_agents: int,
                                sample_size: int,
                                sim_agent: NeuralNet,
                                control_mask: List,
                                device: str,
                                generate_animations: bool = False,
                                ):
    """
    Run normal simulation and record model trajectories and values

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
    all_obs = {env_name: {} for env_name in env.data_batch}                    # Dictionary that contains the observations (agent trajectories) 
    frames = {f"env_{env_path2name(env_path_)}-sample_{sample_num_}": [] for sample_num_ in range(sample_size) for env_path_ in env.data_batch}
    curr_samples = []   # Keep track of reards
    for sample_num in range(sample_size):
        print("\tsample ", sample_num)
        _ = env.reset()
        next_obs = env.get_obs()
        for time_step in range(env.episode_len):
            # |Roll out policy for a specific construal
            print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

            #2# |Execute policy
            next_obs, reward, done, info, plottable_obs, action_probs = run_policy( env=env,
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

            #2# |Record observations
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

    #2# Convert observations from tensor to list for storage
    for env_num in range(total_envs):
        env_name = env.data_batch[env_num]
        for sample_num in range(sample_size):            
            all_obs[env_name][sample_num] = all_obs[env_name][sample_num].tolist()

    #2# Extract ground-truth data
    ground_truth = {'traj': {}, 'traj_valids': {}, 'contr_veh_indices': {}}
    for env_num in range(total_envs):
        ground_truth['traj'][env.data_batch[env_num]] = env.get_data_log_obj().pos_xy[env_num].tolist()
        ground_truth['traj_valids'][env.data_batch[env_num]] = env.get_data_log_obj().valids[env_num].tolist()
        ground_truth['contr_veh_indices'][env.data_batch[env_num]] = torch.where(control_mask[env_num])[0].tolist()

    #2# |Calculate value (average reward) for each construal
    curr_vals = [sum(x)/sample_size for x in zip(*curr_samples)]
    for env_num, val in enumerate(curr_vals):
        model_values[env.data_batch[env_num]] = val

    #2# |Save animations
    if generate_animations:
        save_animations(frames)
        # mediapy.set_show_save_dir('./sim_vids')
        # mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

    # print("\nExpected utility by contrual: ", construal_values)
    return (model_values, all_obs, ground_truth)








def simulate_generalist_policies2(env: GPUDriveConstrualEnv, 
                                total_envs: int,
                                max_agents: int,
                                sample_size: int,
                                sim_agent: NeuralNet,
                                control_mask: List,
                                device: str,
                                generate_animations: bool = False,
                                ):
    """
    Run normal simulation and record states and action prob. dist. of model agent

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
    state_action_pairs = {env_name: {} for env_name in env.data_batch}               # Dictionary that contains the expected utility
    frames = {f"env_{env_path2name(env_path_)}-sample_{sample_num_}": [] for sample_num_ in range(sample_size) for env_path_ in env.data_batch}
    for sample_num in range(sample_size):
        print("\tsample ", sample_num)
        _ = env.reset()
        next_obs = env.get_obs()
        for time_step in range(env.episode_len):
            # |Roll out policy
            print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

            #2# |Get raw observations before stepping through environment
            raw_obs = env.get_obs(raw_obs=True)

            #2# |Execute policy
            next_obs, reward, done, info, plottable_obs, action_probs = run_policy( env=env,
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

            #2# |Record observations
            for env_num, action_dist in enumerate(action_probs):
                env_name = env.data_batch[env_num]
                if sample_num in state_action_pairs[env_name]:
                    state_action_pairs[env_name][sample_num].append((raw_obs[env_num], action_dist))
                elif sample_num not in state_action_pairs[env_name]:
                    state_action_pairs[env_name][sample_num] = [(raw_obs[env_num], action_dist)]
                else:
                    raise ValueError(f"Unknown Situation: {env_name}, {sample_num}")
            
            if done.all():
                break
        print() # Change to new line after step prints

    #2# |Save animations
    if generate_animations:
        save_animations(frames)
        # mediapy.set_show_save_dir('./sim_vids')
        # mediapy.show_videos(frames, fps=15, width=500, height=500, columns=2, codec='gif')

    # print("\nExpected utility by contrual: ", construal_values)
    return state_action_pairs