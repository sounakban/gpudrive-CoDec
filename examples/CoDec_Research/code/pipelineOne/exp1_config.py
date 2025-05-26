from examples.CoDec_Research.code.shared_config import *

target_param = "rel_heading"    # ego_distance or rel_heading



####################################################
################ SET EXP PARAMETERS ################
####################################################

curr_config = get_active_config()

construal_count_baseline = curr_config['construal_count_baseline']      # Number of construals to sample for baseline data generation
trajectory_count_baseline = curr_config['trajectory_count_baseline']    # Number of baseline trajectories to generate per construal


### Specify Environment Configuration ###

# |Location to store (and retrieve pre-computed) simulation results
simulation_results_path = curr_config["simulation_results_path"]
simulation_results_files = [simulation_results_path+fl_name for fl_name in listdir(simulation_results_path)]

# |Model Config (on which model was trained)
training_config = load_config("examples/experimental/config/reliable_agents_params")

# |Set scenario path
dataset_path = curr_config['dataset_path']
processID = dataset_path.split('/')[-2]                 # Used for storing and retrieving relevant data

# |Set simulator config
moving_veh_count = training_config.max_controlled_agents      # Get total vehicle count
num_parallel_envs = curr_config['num_parallel_envs']
total_envs = curr_config['total_envs']
device = eval(curr_config['device'])

# |Set construal config
construal_size = curr_config['construal_size']
observed_agents_count = moving_veh_count - 1                              # Agents observed except self (used for vector sizes)
sample_size_utility = curr_config['sample_size_utility']            # Number of samples to compute expected utility of a construal

# |Other changes to variables
training_config.max_controlled_agents = 1                           # Control only the first vehicle in the environment
total_envs = min(total_envs, len(listdir(dataset_path)))