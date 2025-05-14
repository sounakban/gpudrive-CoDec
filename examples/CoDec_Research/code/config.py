import numpy as np

def get_active_config():
    return server_config


ego_dis_param_values = np.linspace(0,10,11)


local_config = {
                'dataset_path': 'data/processed/construal/Set3/',         # Path to scenario files
                'simulation_results_path': 'examples/CoDec_Research/results/simulation_results/',
                'num_parallel_envs': 2,
                'num_parallel_envs_light': 2,                             # NUmber of parallel environments for memory intensive operations
                'total_envs': 6,
                'device': "'cpu'",
                'sample_size_utility': 1,                                 # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 4,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                }

local_config_2= {# |run inference on baseline data already generated on the server
                'dataset_path': 'data/processed/construal/Set2/',         # Path to scenario files
                'simulation_results_path': '/mnt/d/Data/',
                'num_parallel_envs': 25,
                'num_parallel_envs_light': 3,                             # NUmber of parallel environments for memory intensive operations
                'total_envs': 25,
                'device': "'cpu'",
                'sample_size_utility': 1,                                 # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 6,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                }

server_config = {
                'dataset_path': 'data/processed/construal/Set2/',         # Path to scenario files
                'simulation_results_path': 'examples/CoDec_Research/results/simulation_results/',
                'num_parallel_envs': 25,
                'num_parallel_envs_light': 4,                             # Number of parallel environments for memory intensive operations
                'total_envs': 25,
                'device': "'cuda' if torch.cuda.is_available() else 'cpu'",
                'sample_size_utility': 40,                                # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 6,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                }



