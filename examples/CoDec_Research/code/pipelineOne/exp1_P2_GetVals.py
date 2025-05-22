"""
    This part of pipeline 1 is dedicated to construal value computation, which is later used for both sampling 
    for synthetic data generation and inference logic. The code computes values of all possible construals 
    (given some parameters), and samples construals based on computed values, 
"""

# |Set parent to current working directory for imports
import os
import sys
from pathlib import Path

working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))

# |Import everything
from examples.CoDec_Research.code.pipelineOne.pipe1_imports import *

# |START TIMER
start_time = time.perf_counter()




#####################################################
################ SET UP ENVIRONMENTS ################
#####################################################

env_config, train_loader, env, sim_agent = get_gpuDrive_vars(
                                                            training_config=training_config,
                                                            device=device,
                                                            num_parallel_envs=num_parallel_envs,
                                                            dataset_path=dataset_path,
                                                            total_envs=total_envs,
                                                            sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                            )


#############################################
################ SIMULATIONS ################
#############################################


### Compute construal  utilities through simulator sampling ###
default_values = None
traj_obs = None

#2# |Check if saved construal utility data is available
for srFile in simulation_results_files:
    if "construal_vals" in srFile:
        with open(srFile, 'rb') as opn_file:
            default_values = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in default_values.keys() for scene_path_ in train_loader.dataset):
            print(f"Using construal values from file: {srFile}")
            break
        else:
            default_values = None

if default_values is None:
    default_values, traj_obs, ground_truth, _ = generate_all_construal_trajnval(sim_agent=sim_agent,
                                                                                observed_agents_count=observed_agents_count,
                                                                                construal_size=construal_size,
                                                                                num_parallel_envs=num_parallel_envs,
                                                                                max_agents=moving_veh_count,
                                                                                sample_size=sample_size_utility,
                                                                                device=device,
                                                                                train_loader=train_loader,
                                                                                env=env,
                                                                                moving_veh_masks=moving_veh_masks,
                                                                                generate_animations=False,
                                                                                expConfig=curr_config)
    #3# |Save data
    savefl_path = simulation_results_path+processID+"_"+"construal_vals_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, default_values, "Construal value")
    savefl_path = simulation_results_path+processID+"_"+"constr_traj_obs_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, traj_obs, "Trajectory observation")
    savefl_path = simulation_results_path+processID+"_"+"ground_truth_"+str(datetime.now())+".pickle"
    save_pickle(savefl_path, ground_truth, "Ground truth")
    #3# Free up memory
    del traj_obs, ground_truth



discounted_rewards = False
if discounted_rewards:
    if traj_obs is None:
        for srFile in simulation_results_files:
            if "constr_traj_obs_" in srFile:
                with open(srFile, 'rb') as opn_file:
                    traj_obs = pickle.load(opn_file)
                #2# |Ensure the correct file is being loaded
                if all(env_path2name(scene_path_) in traj_obs.keys() for scene_path_ in train_loader.dataset):
                    print(f"Using construal values from file: {srFile}")
                    break
                else:
                    traj_obs = None
    # default_values = default_to_discounted(default_values, traj_obs)



### Select Construals for Baseline Data ###
scene_constr_dict = None

#2# |Check if saved construal sampling data is available
for srFile in simulation_results_files:
    if "sampled_construals" in srFile:
        with open(srFile, 'rb') as opn_file:
            scene_constr_dict = pickle.load(opn_file)
        #2# |Ensure the correct file is being loaded
        if all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):
            print(f"Using sampled construal information from file: {srFile}")
            break
        else:
            scene_constr_dict = None

if scene_constr_dict is None:
    # |Generate Construal Heuristic Values
    heuristic_values = get_constral_heurisrtic_values(env, train_loader, default_values, heuristic_params=heuristic_params)

    # |Sample construals for generating baseline data
    def sample_construals(heuristic_values: dict, sample_count: int) -> dict:
        """
        Sample construals based on heuristic values.
        """
        sampled_construals = {}
        for scene_name, construal_info in heuristic_values.items():
            if (0,) in construal_info: construal_info.pop((0,))                # Do not sample empty construal
            full_construal_ = max(construal_info.keys(), key=len)
            construal_info.pop(full_construal_)     # Do not sample full state space
            constr_indices, constr_values = zip(*construal_info.items())
            sampled_indices = torch.multinomial(torch.tensor(constr_values), num_samples=sample_count, \
                                                    replacement=True).tolist()
            # sampled_construals[scene_name] = {constr_indices[i]: constr_values[i] for i in sampled_indices}
            # print(f"Sampled construals for scene {scene_name}: {sampled_construals[scene_name].keys()}")
            sampled_construals[scene_name] = tuple(constr_indices[i] for i in sampled_indices)
            print(f"Sampled construals for scene {scene_name}: {sampled_construals[scene_name]}")

        return sampled_construals

    scene_constr_dict = sample_construals(heuristic_values, sample_count=construal_count_baseline)

    scene_constrFile = simulation_results_path + processID + "_" + "sampled_construals_"+str(datetime.now())+".pickle"
    scene_constr_dict["params"] = heuristic_params
    save_pickle(scene_constrFile, scene_constr_dict, "Sampled construals")




env.close()

# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")