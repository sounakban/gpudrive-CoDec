

# |Shared Imports
from examples.CoDec_Research.code.shared_imports import *



def get_active_config():
    return local_config



# |Hueristics and their weight parameters (to be inferred)
heuristic_params_vals = {  
                        "ego_distance": np.linspace(0,10,11),                                             
                        "rel_heading": np.linspace(-15,15,31),
                        "cardinality": np.linspace(0,10,11)
                        }

# Preset parameters for Inference
# heuristic_params = {"ego_distance": ego_dis_param_values[5],            # Hueristics and their weight parameters (to be inferred)
#                     "rel_heading": ego_head_param_values[1],
#                     "cardinality": 1}

# |Hueristics and their weight parameters (to be inferred)
heuristic_params = {                                                    
                    "rel_heading": heuristic_params_vals['rel_heading'][15],
                    "cardinality": 2
                    }



local_config = {
                'dataset_path': 'data/processed/construal/Set3/',         # Path to scenario files
                'simulation_results_path': 'examples/CoDec_Research/results/simulation_results/',
                'construal_size': 1,
                'num_parallel_envs': 3,
                'num_parallel_envs_light': 3,                             # NUmber of parallel environments for memory intensive operations
                'total_envs': 3,
                'device': "'cpu'",
                'sample_size_utility': 1,                                 # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 6,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                'ego_in_construal': False                                 # Boolean flag indicating whether to keep ego in construals. 
                                                                          #     Ego is observed anyway
                }

local_config_2= {# |run inference on baseline data already generated on the server
                'dataset_path': 'data/processed/construal/Set1V2/',         # Path to scenario files
                'simulation_results_path': 'examples/CoDec_Research/results/simulation_results/',
                'construal_size': 1,
                'num_parallel_envs': 10,
                'num_parallel_envs_light': 2,                             # NUmber of parallel environments for memory intensive operations
                'total_envs': 10,
                'device': "'cpu'",
                'sample_size_utility': 1,                                 # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 6,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                'ego_in_construal': False                                 # Boolean flag indicating whether to keep ego in construals. 
                                                                          #     Ego is observed anyway
                }

server_config = {
                'dataset_path': 'data/processed/construal/Set1V2/',         # Path to scenario files
                'simulation_results_path': 'examples/CoDec_Research/results/simulation_results/',
                'construal_size': 1,
                'num_parallel_envs': 10,
                'num_parallel_envs_light': 5,                             # Number of parallel environments for memory intensive operations
                'total_envs': 10,
                'device': "'cuda' if torch.cuda.is_available() else 'cpu'",
                'sample_size_utility': 10,                                # Number of samples to compute expected utility of a construal
                'construal_count_baseline': 8,                            # Number of construals to sample for baseline data generation
                'trajectory_count_baseline': 1,                           # Number of baseline trajectories to generate per construal
                'ego_in_construal': False                                 # Boolean flag indicating whether to keep ego in construals. 
                                                                          #     Ego is observed anyway
                }


# shared_config = {
#                     save_intermediate_files = False,     # If set to False, files apart from construal values and final results are not 
#                                                         # saved. If True, everything is saved
#                     save_

# }
