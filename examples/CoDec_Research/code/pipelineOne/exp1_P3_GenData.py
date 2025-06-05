"""
    This part of pipeline 1 is dedicated to synthetic data generation. The code generates synthetic 
    data based on sampled construals (previous stage of pipeline).
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
from examples.CoDec_Research.code.pipelineOne.exp1_imports import *








##################################################
################ DEFINE FUNCTIONS ################
##################################################


def generate_synthData(
                        out_dir: str,
                        processID: str,
                        expConfig: Dict,
                        sim_agent: NeuralNet,
                        num_parallel_envs: int,
                        max_agents: int,
                        sample_size: int,
                        device: str,
                        env: GPUDriveConstrualEnv,
                        moving_veh_masks: Dict,
                        observed_agents_count: int,
                        construal_size: int,
                        generate_animations: bool = False,
                        heuristic_params: Dict = None,
                    ):
    ### Retrieve Saved Construal Sampling Resuts ###
    scene_constr_dict = None

    #2# |Check if saved construal sampling data is available
    intermediate_results_files = get_files_at_loc(out_dir)
    for srFile in intermediate_results_files:
        if "sampled_construals" in srFile:
            with open(srFile, 'rb') as opn_file:
                scene_constr_dict = pickle.load(opn_file)
            #2# |Ensure the correct file is being loaded
            if all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in train_loader.dataset):
                print(f"Using sampled construal information from file: {srFile}")
                file_params = scene_constr_dict.pop('params')
                break
            else:
                scene_constr_dict = None
    if scene_constr_dict is None:
        raise FileNotFoundError("Could not find saved file for sampled construals for current scenes "
                                "in out_dir. Please generate construal samples for this code to work.")

    ### Generate Synthetic Ground Truth for Selected Construals (Baseline Data on Which to Perform Inference) ###

    # |Loop through all files in batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        # |Update simulator with the new batch of data
        env.swap_data_batch(batch)

        state_action_pairs = None

        # |Check if saved data is available
        curr_dataset_scenes = set(env_path2name(scene_path_) for scene_path_ in env.data_batch)
        for srFile in intermediate_results_files:
            if "baseline_state_action_pairs" in srFile:
                state_action_pairs = ObservationDataManager.load_data(srFile, decompress_data=expConfig['compress_synthetic_data'])
                #2# |Ensure the correct file is being loaded
                fileScenes = set(state_action_pairs.keys()); fileScenes.remove('params'); fileScenes.remove('dict_structure')
                # print(fileScenes)
                # print(curr_dataset_scenes)
                if fileScenes == curr_dataset_scenes and state_action_pairs["params"] == heuristic_params:
                    print(f"Synthetic baseline data for current batch already exists in file: {srFile}")
                    break
                else:
                    state_action_pairs = None

        if state_action_pairs is None and \
            all(env_path2name(scene_path_) in scene_constr_dict.keys() for scene_path_ in env.data_batch):

            print(f"Could not find baseline data for current batch. Now computing.")
                    
            state_action_pairs = generate_baseline_data(sim_agent=sim_agent,
                                                        num_parallel_envs=num_parallel_envs,
                                                        max_agents=max_agents,
                                                        sample_size=sample_size,
                                                        device=device,
                                                        env=env,
                                                        moving_veh_masks=moving_veh_masks,
                                                        observed_agents_count=observed_agents_count,
                                                        construal_size=construal_size,
                                                        selected_construals=scene_constr_dict,
                                                        generate_animations=generate_animations,
                                                        expConfig=expConfig)
                    
            #2# |Save data
            savefl_path = out_dir+processID+"_"+"baseline_state_action_pairs_"+str(datetime.now())+".pickle"
            state_action_pairs["params"] = heuristic_params # Save parameters for data generation
            ObservationDataManager.save_data(state_action_pairs, savefl_path, compress_data=expConfig['compress_synthetic_data'])
            # |Clear memory for large variable
            del state_action_pairs
            gc.collect()






















if __name__ == "__main__":

    ##### Set Up Environments #####
    
    from examples.CoDec_Research.code.pipelineOne.exp1_config import *

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
                                                                # num_parallel_envs=num_parallel_envs_dataGen,
                                                                dataset_path=dataset_path,
                                                                total_envs=total_envs,
                                                                sim_agent_path="daphne-cornelisse/policy_S10_000_02_27",
                                                                )

    

    
    ##### Run Main Code #####

    start_time = time.perf_counter()

    generate_synthData(
                        out_dir = intermediate_results_path,
                        processID = processID,
                        expConfig = curr_config,
                        sim_agent = sim_agent,
                        num_parallel_envs = num_parallel_envs,
                        max_agents = moving_veh_count,
                        sample_size = trajectory_count_baseline,
                        device = device,
                        env = env,
                        moving_veh_masks = moving_veh_masks,
                        observed_agents_count = observed_agents_count,
                        construal_size = construal_size,
                        heuristic_params = active_heuristic_params,
                    )

    env.close()

    # |Print the execution time
    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")