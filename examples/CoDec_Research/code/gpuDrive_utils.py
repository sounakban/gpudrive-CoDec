# |Python imports
from sqlite3 import DataError
import torch

from datetime import datetime
import pickle
from tqdm import tqdm

# |Set root for GPUDrive import
import os
import sys
from pathlib import Path

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))


import dataclasses
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveConstrualEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config



def get_gpuDrive_vars(training_config, 
                      device: str, 
                      num_parallel_envs: int, 
                      dataset_path: str,
                      total_envs: int,
                      sim_agent_path: str = "daphne-cornelisse/policy_S10_000_02_27",
                      env: GPUDriveConstrualEnv = None,
                      ):
    
    env_config = get_env_config(training_config, device)

    # |Create data loader
    train_loader = SceneDataLoader(
        root=dataset_path,
        batch_size=num_parallel_envs,
        dataset_size=max(total_envs,num_parallel_envs),
        sample_with_replacement=False,
    )

    if env is None:
        # Only initialize environment if one does not exist
        #   (Multiple initializations may lead to segmentation fault)
        # |Make env [Construal]
        env = GPUDriveConstrualEnv(
            config=env_config,
            data_loader=train_loader,
            max_cont_agents=training_config.max_controlled_agents,
            device=device,
        )

        # # |DEBUG: Make env [Original]
        # env = GPUDriveTorchEnv(
        #     config=env_config,
        #     data_loader=train_loader,
        #     max_cont_agents=training_config.max_controlled_agents,
        #     device=device,
        # )
        print("Initialized default environment")

    # |Import Pre-trained Model
    sim_agent = NeuralNet.from_pretrained(sim_agent_path).to(device)

    return (env_config, train_loader, env, sim_agent)




def get_mov_veh_masks(
                      training_config, 
                      device: str, 
                      dataset_path: str,
                      max_agents: int,
                      result_file_loc: str,
                      processID: str,
                      save_data = True,
                    ) -> dict:
    
    # |Set total_envs and num_parallel_envs
    total_envs = len([entry_ for entry_ in os.listdir(dataset_path) if entry_.endswith('.json')])   # Load all scenes
    if total_envs > 100:
        #2# |If dataset is too big: divide dataset into equal fragments
        #2# |Get the maximum possible devisor that is less than 100
        for num_parallel_envs in reversed(range(1, 101)):
            if total_envs%num_parallel_envs == 0:
                break
    else:
        num_parallel_envs = total_envs
    if num_parallel_envs == 1 and total_envs > num_parallel_envs:
        #2# |Number of scenes is prime
        raise DataError("Could not divide dataset into equal fragments, try changing number of scenes")
    
    env_config = get_env_config(training_config, device)

    # |Create data loader
    train_loader = SceneDataLoader(
        root=dataset_path,
        batch_size=num_parallel_envs,
        dataset_size=max(total_envs,num_parallel_envs),
        sample_with_replacement=False,
    )

    # |Check if data is available before initializing environment
    moving_veh_masks = None

    simulation_results_files = [result_file_loc+fl_name for fl_name in os.listdir(result_file_loc)]
    for srFile in simulation_results_files:
        if "movVeh_masks" in srFile:
            with open(srFile, 'rb') as opn_file:
                moving_veh_masks = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(scene_path_ in moving_veh_masks.keys() for scene_path_ in train_loader.dataset):
                print(f"Using moving vehicle masks from file: {srFile}")
                break
            else:
                moving_veh_masks = None

    # |Otherwise generate data
    if moving_veh_masks is None:
        print("Saved masks were not found for the current data, generatinig new maks.")
        moving_veh_masks = {}
        env_multi_agent = GPUDriveConstrualEnv(
                            config=env_config,
                            data_loader=train_loader,
                            max_cont_agents=max_agents,
                            device=device,
                            )
        
        # |Loop through all batches
        for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                            total=len(train_loader), colour="blue"):
            # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
            # |Update simulator with the new batch of data
            env_multi_agent.swap_data_batch(batch)
            print("Initialized multi-agent environment")

            moving_veh_masks.update({scene_name_: scene_mask_ for scene_name_, scene_mask_ in zip(env_multi_agent.data_batch, \
                                                                                            env_multi_agent.cont_agent_mask)})
        
        # |Prevent multiple active instances of the environment
        env_multi_agent.close()
        del env_multi_agent

        # |Save resuts to file
        if save_data:
            savefl_path = result_file_loc+processID+'_'+"movVeh_masks_"+str(datetime.now())+".pickle"
            with open(savefl_path, 'wb') as file:
                pickle.dump(moving_veh_masks, file, protocol=pickle.HIGHEST_PROTOCOL)

    return moving_veh_masks



def save_pickle(fliePath, fileData, dataTag: str = "Unspecified"):
    with open(fliePath, 'wb') as file:
        pickle.dump(fileData, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{dataTag} data saved to: ", fliePath)



def get_env_config(training_config,
                   device: str,):
    # |Create environment config (with default device)
    env_config = dataclasses.replace(
        EnvConfig(),
        ego_state=training_config.ego_state,
        road_map_obs=training_config.road_map_obs,
        partner_obs=training_config.partner_obs,
        reward_type=training_config.reward_type,
        norm_obs=training_config.norm_obs,
        dynamics_model=training_config.dynamics_model,
        collision_behavior=training_config.collision_behavior,
        dist_to_goal_threshold=training_config.dist_to_goal_threshold,
        polyline_reduction_threshold=0.2 if device == "cpu" else training_config.polyline_reduction_threshold,
        remove_non_vehicles=training_config.remove_non_vehicles,
        lidar_obs=training_config.lidar_obs,
        disable_classic_obs=training_config.lidar_obs,
        obs_radius=training_config.obs_radius,
        steer_actions = torch.round(
            torch.linspace(-torch.pi, torch.pi, training_config.action_space_steer_disc), decimals=3  
        ),
        accel_actions = torch.round(
            torch.linspace(-4.0, 4.0, training_config.action_space_accel_disc), decimals=3
        ),
    )

    return env_config