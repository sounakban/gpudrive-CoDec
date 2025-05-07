# |Default library imports
from copy import deepcopy
from functools import cache
from locale import normalize
from os import listdir
import json
import pickle
from datetime import datetime

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, Dict, List, Tuple
import time

import torch
import dataclasses
from tqdm import tqdm


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


# |GPUDrive imports
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv, GPUDriveConstrualEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config

# |CoDec imports
from examples.CoDec_Research.code.simulation.simulation_functions import simulate_policies, simulate_selected_construal_policies
from examples.CoDec_Research.code.construal_functions import get_construal_veh_distance_ego, get_construal_cardinality
from examples.CoDec_Research.code.gpuDrive_utils import get_gpuDrive_vars


##############################################
################### CONFIG ###################
##############################################

# |Location to store simulation results
out_dir = "examples/CoDec_Research/results/simulation_results/"

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")
# print(config)

# |Set scenario path
# dataset_path='data/processed/examples'
# dataset_path='data/processed/training'
dataset_path = 'data/processed/construal'

# |Set simulator config
max_agents = training_config.max_controlled_agents   # Get total vehicle count
num_parallel_envs = 2
total_envs = 2
device = "cpu" # cpu just because we're in a notebook
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# |Set construal config
construal_size = 1
observed_agents_count = max_agents - 1      # Agents observed except self (used for vector sizes)
sample_size = 2                            # Number of samples to calculate expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1    # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))






##################################################
################### MAIN LOGIC ###################
##################################################



# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]



def generate_all_construal_trajnval(out_dir: str,
                                sim_agent: NeuralNet,
                                observed_agents_count: int,
                                construal_size: int,
                                num_parallel_envs: int,
                                max_agents: int,
                                sample_size: int,
                                device: str,
                                train_loader: SceneDataLoader,
                                env: GPUDriveConstrualEnv,
                                env_multi_agent: GPUDriveConstrualEnv,
                                generate_animations: bool = False,
                                ) -> None:
    """
    Generate values and trajectory observations for construed agent states
    """

    construal_values = {"dict_structure": '{scene_name: {construal_index: value}}'}
    traj_obs = {"dict_structure": '{scene_name: {construal_index: {sample_num: 3Dmatrix[vehicles,timestep,coord]}}}'}
    ground_truth = {"dict_structure": '{"traj": {scene_name: 3Dmatrix[vehicles,timestep,coord]}, "traj_valids": {scene_name: 3Dmatrix[vehicles,timestep,bool]}, "contr_veh_indices": {scene_name: list[controlled_vehicles]} }'}

    # |Loop through all batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        # |Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # |Get moving vehicle information
        env_multi_agent.swap_data_batch(batch)
        moving_veh_mask = env_multi_agent.cont_agent_mask
        # moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in moving_veh_mask]
        moving_veh_indices = [tuple(torch.where(mask)[0].cpu().tolist()) for mask in moving_veh_mask]
        print("Indices of all moving vehicles (by scene): ", moving_veh_indices)
        control_mask = env.cont_agent_mask

        # |Simulate on Construals
        construal_values_, traj_obs_, ground_truth_, _ = simulate_policies(env = env,
                                                                        observed_agents_count = observed_agents_count,
                                                                        construal_size= construal_size,
                                                                        total_envs = num_parallel_envs,
                                                                        max_agents = max_agents,
                                                                        moving_veh_indices = moving_veh_indices,
                                                                        sample_size = sample_size,
                                                                        sim_agent = sim_agent,
                                                                        control_mask = control_mask,
                                                                        device = device,
                                                                        generate_animations = generate_animations,
                                                                        save_trajectory_obs=True)
        construal_values.update(construal_values_)
        traj_obs.update(traj_obs_)
        ground_truth.update(ground_truth_)

    # |Save the construal value information to a file
    savefl_path = out_dir+"construal_vals_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(construal_values, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Baseline data saved to: ", savefl_path)
    # |Save the construal trajectory information to a file
    savefl_path = out_dir+"all_constr_obs_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(traj_obs, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Baseline data saved to: ", savefl_path)
    # |Save the ground truth information to a file
    savefl_path = out_dir+"ground_truth_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(ground_truth, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Baseline data saved to: ", savefl_path)

    return construal_values, traj_obs, ground_truth





def generate_selected_construal_traj(out_dir: str,
                                        sim_agent: NeuralNet,
                                        observed_agents_count: int,
                                        construal_size: int,
                                        num_parallel_envs: int,
                                        max_agents: int,
                                        sample_size: int,
                                        device: str,
                                        train_loader: SceneDataLoader,
                                        env: GPUDriveConstrualEnv,
                                        env_multi_agent: GPUDriveConstrualEnv,
                                        selected_construals: Dict[str, List[Tuple[int]]],
                                        generate_animations: bool = False,
                                        ) -> None:
    """
    Generate values and trajectory observations for construed agent states
    """

    all_obs = {"dict_structure": '{scene_name: {construal_index: {sample_num: 3Dmatrix[vehicles,timestep,coord]}}}'}
    # ground_truth = {"dict_structure": '{"traj": {scene_name: 3Dmatrix[vehicles,timestep,coord]}, "traj_valids": {scene_name: 3Dmatrix[vehicles,timestep,bool]}, "contr_veh_indices": {scene_name: list[controlled_vehicles]} }'}

    # |Loop through all batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        # |Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # |Get moving vehicle information
        env_multi_agent.swap_data_batch(batch)
        moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in env_multi_agent.cont_agent_mask]
        # print("Indices of all moving vehicles (by scene): ", moving_veh_indices)
        control_mask = env.cont_agent_mask

        # |Simulate on Construals
        all_obs_ = simulate_selected_construal_policies(env = env, 
                                                        observed_agents_count = observed_agents_count,
                                                        construal_size= construal_size,
                                                        total_envs = num_parallel_envs,
                                                        max_agents = max_agents,
                                                        moving_veh_indices = moving_veh_indices,
                                                        sample_size = sample_size,
                                                        sim_agent = sim_agent,
                                                        control_mask = control_mask,
                                                        device = device,
                                                        selected_construals = selected_construals,
                                                        generate_animations = generate_animations)
        all_obs.update(all_obs_)
    #     ground_truth.update(ground_truth_)

    with open(out_dir+"selected_constr_obs_"+str(datetime.now())+".txt", 'w') as file:
        file.write(str(all_obs))






def generate_baseline_data( out_dir: str,
                            sim_agent: NeuralNet,
                            num_parallel_envs: int,
                            max_agents: int,
                            sample_size: int,
                            device: str,
                            train_loader: SceneDataLoader,
                            env: GPUDriveConstrualEnv,
                            env_multi_agent: GPUDriveConstrualEnv,
                            observed_agents_count: int = 0,
                            construal_size: int = 0,
                            selected_construals: Dict[str, List[Tuple[int]]] = None,
                            generate_animations: bool = False,
                            ) -> None:
    """
    Generate baseline state representation and action probability pairs
    """
    state_action_pairs = {"dict_structure": '{scene_name: {\"control_mask\": mask, \"max_agents\": int, \"moving_veh_ind\": list, sample_num: ((raw_states, action_probs),...timesteps)}}'}

    # |Loop through all batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        #2# |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        #2# |Update simulator with the new batch of data
        env.swap_data_batch(batch)
        env_multi_agent.swap_data_batch(batch)
        control_mask = env.cont_agent_mask
        curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
        
        #2# |Get moving vehicle information
        moving_veh_mask = env_multi_agent.cont_agent_mask
        moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in moving_veh_mask]
        # print("Indices of all moving vehicles (by scene): ", moving_veh_indices)

        #2# |Simulate on Construals
        _, _, _, state_action_pairs_ = simulate_policies(env = env, 
                                                        total_envs = num_parallel_envs,
                                                        max_agents = max_agents,
                                                        sample_size = sample_size,
                                                        sim_agent = sim_agent,
                                                        control_mask = control_mask,
                                                        device = device,
                                                        observed_agents_count = observed_agents_count,
                                                        moving_veh_indices = moving_veh_indices,
                                                        construal_size = construal_size,
                                                        selected_construals = selected_construals,
                                                        generate_animations = generate_animations,
                                                        save_state_action_pairs=True,
                                                        )

        for scene_num, scene_name in enumerate(curr_data_batch):
            #2# |Add environment config metadata
            state_action_pairs_[scene_name]["control_mask"] = env.cont_agent_mask[scene_num]
            state_action_pairs_[scene_name]["max_agents"] = env.config.max_controlled_agents
            state_action_pairs_[scene_name]["moving_veh_ind"] = moving_veh_indices[scene_num]
        state_action_pairs.update(state_action_pairs_)

    # |Save the state-action pairs to a file
    savefl_path = out_dir+"baseline_state_action_pairs_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(state_action_pairs, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Baseline data saved to: ", savefl_path)

    return state_action_pairs





def get_constral_heurisrtic_values(env: GPUDriveConstrualEnv, train_loader: SceneDataLoader,
                                   default_values: dict, heuristic_params: dict = None, 
                                   average: bool = True,  normalize:bool = True) -> dict:
    '''
    Get the construal values based on some heuristic on average or for each vehicle in the construal

    Args:
        env: The environment object
        default_values: Dictionary containing the default values for all construals in  each scene
        average: If true, return the average construal value for all vehicles in the construal
        heuristic: The heuristic to use for the construal value
            0: Default construal values
            1: Distance from ego vehicle
            2: -
            3: -
            4: -
            5: -
        heuristic_params: Dictionary containing the parameters for the heuristic. Keys:
            "ego_distance": parameter for (ego) distance heuristic
        normalize: If true, return the normalized [0,1] heuristics values for all construals
                    using min-max scaling

    Returns:
        The average distance or a list of distances from the ego vehicle to each vehicle in the construal
    '''
    heuristics_to_func = {"ego_distance": get_construal_veh_distance_ego,
                          "cardinality": get_construal_cardinality,}
    active_heuristics = {heuristics_to_func[curr_heuristic_]: curr_heuristic_val_
                            for curr_heuristic_, curr_heuristic_val_ in heuristic_params.items()}
    
    construal_indices = {scene_name: construal_info.keys() for scene_name, construal_info in default_values.items()
                                                                                if scene_name != "dict_structure"}

    result_dict = dict()
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        #2# |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        #2# |Update simulator with the new batch of data
        env.swap_data_batch(batch)
        heuristics_vars = [(curr_heuristic_func_(env, construal_indices, average=average, normalize=normalize), 
                            curr_heuristic_val_)
                                for curr_heuristic_func_, curr_heuristic_val_ in active_heuristics.items()]
        curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
        for scene_num, scene_name in enumerate(curr_data_batch):
            result_dict[scene_name] = dict()
            construal_info = default_values[scene_name]            
            for construal_index, construal_val in construal_info.items():
                curr_heuristic_penalty = sum(curr_heuristic_val_*curr_heuristic_dict_[scene_name][construal_index]
                                                for curr_heuristic_dict_, curr_heuristic_val_ in heuristics_vars)
                result_dict[scene_name][construal_index] = construal_val + curr_heuristic_penalty  
            # |Softmax construal values          
            constr_indices, constr_values = zip(*result_dict[scene_name].items())
            constr_values_softmax = torch.nn.functional.softmax(torch.tensor(constr_values), dim=0)
            result_dict[scene_name] = {curr_index_: curr_values_softmax_.item() for 
                                        curr_index_, curr_values_softmax_ in zip(constr_indices, constr_values_softmax)}
    return result_dict











#################################################
################### MAIN TEST ###################
#################################################
    
if __name__ == "__main__":
    start_time = time.perf_counter()

    env_config, train_loader, env, env_multi_agent, sim_agent = get_gpuDrive_vars(
                                                                                training_config = training_config,
                                                                                device = device,
                                                                                num_parallel_envs = num_parallel_envs,
                                                                                dataset_path = dataset_path,
                                                                                max_agents = max_agents,
                                                                                total_envs = total_envs,
                                                                                sim_agent_path= "daphne-cornelisse/policy_S10_000_02_27",
                                                                            )

    if torch.cuda.is_available():
        print("Using GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    generate_all_construal_trajnval(out_dir=out_dir,
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
    
    # results = generate_baseline_data(out_dir=out_dir,
    #                         sim_agent=sim_agent,
    #                         num_parallel_envs=num_parallel_envs,
    #                         max_agents=max_agents,
    #                         sample_size=sample_size,
    #                         device=device,
    #                         train_loader=train_loader,
    #                         env=env,
    #                         env_multi_agent=env_multi_agent)

    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")