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

# # |Import everything
from examples.CoDec_Research.code.Experiment_1.exp1_imports import *








##################################################
################ DEFINE FUNCTIONS ################
##################################################


def get_construal_utility_value(
                                sim_agent: NeuralNet,
                                observed_agents_count: int,
                                construal_size: int,
                                num_parallel_envs: int,
                                max_agents: int,
                                sample_size: int,
                                device: str,
                                train_loader: SceneDataLoader,
                                env: GPUDriveConstrualEnv,
                                moving_veh_masks: dict,
                                generate_animations: bool = False,
                                saveResults: bool = False,
                                out_dir: str = None,
                                processID: str = None,
                                expConfig: dict = None
                                ):
    """
      Compute construal  utilities through simulator sampling.
      Look for already saved values. If not available, call function to compute.
    """
    default_values = None
    traj_obs = None

    # |Check if saved construal utility data is available
    intermediate_results_files = get_files_at_loc(out_dir)
    for srFile in intermediate_results_files:
        if "construal_vals" in srFile:
            with open(srFile, 'rb') as opn_file:
                default_values = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in default_values.keys() for scene_path_ in train_loader.dataset):
                print(f"Using construal values from file: {srFile}")
                break
            else:
                default_values = None

    # |If not, simluate and compute
    if default_values is None:
        default_values, traj_obs, ground_truth, _ = generate_all_construal_trajnval(sim_agent=sim_agent,
                                                                                    observed_agents_count=observed_agents_count,
                                                                                    construal_size=construal_size,
                                                                                    num_parallel_envs=num_parallel_envs,
                                                                                    max_agents=max_agents,
                                                                                    sample_size=sample_size,
                                                                                    device=device,
                                                                                    train_loader=train_loader,
                                                                                    env=env,
                                                                                    moving_veh_masks=moving_veh_masks,
                                                                                    generate_animations=generate_animations,
                                                                                    saveResults = False,
                                                                                    out_dir = None,
                                                                                    expConfig=expConfig)

        #2# |If using discounted rewards for construal values, convert [NOT IMPLEMENTED]
        discounted_rewards = False
        if discounted_rewards:
            # NOTE: Not implemented
            # default_values = default_to_discounted(default_values, traj_obs)
            pass

        if saveResults:
            #2# |Save data
            savefl_path = out_dir+processID+"_"+"construal_vals_"+str(datetime.now())+".pickle"
            save_pickle(savefl_path, default_values, "Construal value")
            savefl_path = out_dir+processID+"_"+"constr_traj_obs_"+str(datetime.now())+".pickle"
            save_pickle(savefl_path, traj_obs, "Trajectory observation")
            savefl_path = out_dir+processID+"_"+"ground_truth_"+str(datetime.now())+".pickle"
            save_pickle(savefl_path, ground_truth, "Ground truth")
        #2# Free up memory (for unused variables)
        del traj_obs, ground_truth

    return default_values






def sample_construals(
                        construal_values: dict, 
                        sample_count: int,
                        train_loader: SceneDataLoader,
                        out_dir: str,
                        processID: str,
                        heuristic_params: Dict = None,
                      ) -> dict:
    """
    Sample construals for baseline data, given their values.
    """
    scene_constr_dict = None

    # |Check if saved construal sampling data is available
    intermediate_results_files = get_files_at_loc(out_dir)
    for srFile in intermediate_results_files:
        if "sampled_construals" in srFile:
            with open(srFile, 'rb') as opn_file:
                scene_constr_dict = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):
                print(f"Using sampled construal information from file: {srFile}")
                break
            else:
                scene_constr_dict = None

    # |If not, sample construals
    if scene_constr_dict is None:
        scene_constr_dict = {}
        for scene_name, construal_info in construal_values.items():
            if (0,) in construal_info: construal_info.pop((0,))                # Do not sample empty construal
            full_construal_ = max(construal_info.keys(), key=len)
            construal_info.pop(full_construal_)     # Do not sample full state space
            constr_indices, constr_values = zip(*construal_info.items())
            sampled_indices = torch.multinomial(torch.tensor(constr_values), num_samples=sample_count, \
                                                    replacement=True).tolist()
            # sampled_construals[scene_name] = {constr_indices[i]: constr_values[i] for i in sampled_indices}
            # print(f"Sampled construals for scene {scene_name}: {sampled_construals[scene_name].keys()}")
            scene_constr_dict[scene_name] = tuple(constr_indices[i] for i in sampled_indices)
            print(f"Sampled construals for scene {scene_name}: {scene_constr_dict[scene_name]}")

        #2# Save sampling data
        scene_constrFile = out_dir + processID + "_" + "sampled_construals_"+str(datetime.now())+".pickle"
        scene_constr_dict["params"] = heuristic_params
        save_pickle(scene_constrFile, scene_constr_dict, "Sampled construals")

    return scene_constr_dict





















if __name__ == "__main__":

    ##### Set Up Environments #####
    
    from examples.CoDec_Research.code.Experiment_1.exp1_config import *

    if len(sys.argv) > 1:
        # |If calling code with arguments
        # |Uses values from configuration file if values are not passed as argument
        try:
            target_param, target_heur_val = sys.argv[1], sys.argv[2]     # Changing global variable
            target_param = target_param.split(',')
            target_heur_val = [float(i) for i in target_heur_val.split(',')]
            for param_, val_ in zip(target_param, target_heur_val):
                active_heuristic_params[param_] = val_
        except Exception as e:
            raise ValueError("could not run python file: invalid arguments."
                                " Please ensure first argument is a comma-separated (no space)"
                                " list of heuristic names and the second is a comma-separated"
                                " (no space) list of values (Ex: 2,3,4)")

    moving_veh_masks = get_mov_veh_masks(
                                        training_config=training_config, 
                                        device=device, 
                                        dataset_path=dataset_path,
                                        max_agents=moving_veh_count,
                                        result_file_loc=intermediate_results_path,
                                        processID=processID
                                        )

    env_config, train_loader, env, sim_agent = get_gpuDrive_vars(
                                                                training_config=training_config,
                                                                device=device,
                                                                num_parallel_envs=num_parallel_envs,
                                                                dataset_path=dataset_path,
                                                                total_envs=total_envs,
                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                )


    ##### Run Main Code #####

    start_time = time.perf_counter()

    default_values = get_construal_utility_value(
                                                sim_agent = sim_agent,
                                                observed_agents_count = observed_agents_count,
                                                construal_size = construal_size,
                                                num_parallel_envs = num_parallel_envs,
                                                max_agents = moving_veh_count,
                                                sample_size = sample_size_utility,
                                                device = device,
                                                train_loader = train_loader,
                                                env = env,
                                                moving_veh_masks = moving_veh_masks,
                                                generate_animations = False,
                                                saveResults = True,
                                                out_dir = intermediate_results_path,
                                                processID = processID,
                                                expConfig = curr_config
                                                )

    heuristic_values = get_constral_heurisrtic_values(env, train_loader, default_values, heuristic_params=active_heuristic_params)    

    scene_constr_dict = sample_construals(
                                            construal_values = heuristic_values, 
                                            sample_count = construal_count_baseline,
                                            train_loader = train_loader,
                                            out_dir = intermediate_results_path,
                                            processID = processID,
                                            heuristic_params = active_heuristic_params,
                                        )

    env.close()

    # |Print the execution time
    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")