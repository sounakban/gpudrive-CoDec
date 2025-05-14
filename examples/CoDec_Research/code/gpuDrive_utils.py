# |Python imports
import torch

# |Set root for GPUDrive import
import os
import sys
from pathlib import Path
import pickle

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
                      max_agents: int, 
                      total_envs: int,
                      sim_agent_path: str = "daphne-cornelisse/policy_S10_000_02_27",
                      env: GPUDriveConstrualEnv = None,
                      env_multi_agent: GPUDriveConstrualEnv = None,
                      ):
    # |Create environment config (for default device)
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

    # |Create environment config (for the CPU version of the environment)
    env_config_cpu = dataclasses.replace(
        EnvConfig(),
        ego_state=training_config.ego_state,
        road_map_obs=training_config.road_map_obs,
        partner_obs=training_config.partner_obs,
        reward_type=training_config.reward_type,
        norm_obs=training_config.norm_obs,
        dynamics_model=training_config.dynamics_model,
        collision_behavior=training_config.collision_behavior,
        dist_to_goal_threshold=training_config.dist_to_goal_threshold,
        polyline_reduction_threshold=0.3,
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

    if env_multi_agent is None:
        # Only initialize environment if one does not exist
        #   (Multiple initializations may lead to segmentation fault)
        # |Create multi-agent environment to get information about moving vehicles
        env_multi_agent = GPUDriveConstrualEnv(
                            config=env_config_cpu,
                            data_loader=train_loader,
                            max_cont_agents=max_agents,
                            device="cpu",
                            )

    # |Import Pre-trained Model
    sim_agent = NeuralNet.from_pretrained(sim_agent_path).to(device)

    return (env_config, train_loader, env, env_multi_agent, sim_agent)



def save_pickle(fliePath, fileData, dataTag: str = "Unspecified"):
    with open(fliePath, 'wb') as file:
        pickle.dump(fileData, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{dataTag} data saved to: ", fliePath)