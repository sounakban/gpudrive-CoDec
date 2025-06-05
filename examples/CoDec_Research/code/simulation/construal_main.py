"""

"""

# |Higher-level imports
from examples.CoDec_Research.code.simulation.simulation_imports import *

# |Same-level imports
from examples.CoDec_Research.code.construals.heuristic_functions import *

# |Local imports
from examples.CoDec_Research.code.simulation.simulation_functions import *
from examples.CoDec_Research.code.simulation.data_manager import *




##################################################
################### MAIN LOGIC ###################
##################################################



# # Function to extract filename from path
# env_path2name = lambda path: path.split("/")[-1].split(".")[0]



def generate_all_construal_trajnval(sim_agent: NeuralNet,
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
                                    expConfig: dict = None
                                    ) -> None:
    """
    Generate values and trajectory observations for construed agent states
    """
    if saveResults:
        assert out_dir, "Provide save location for data"

    construal_values = {"dict_structure": '{scene_name: {construal_index: value}}'}
    traj_obs = {"dict_structure": '{scene_name: {construal_index: {sample_num: 3Dmatrix[vehicles,timestep,coord]}}}'}
    ground_truth = {"dict_structure": '{"traj": {scene_name: 3Dmatrix[vehicles,timestep,coord]}, "traj_valids": {scene_name: 3Dmatrix[vehicles,timestep,bool]}, "contr_veh_indices": {scene_name: list[controlled_vehicles]} }'}
    veh_indx2ID = {}

    # |Loop through all batches
    for batch in tqdm(train_loader, desc=f"Processing Waymo batches",
                        total=len(train_loader), colour="blue"):
        # |BATCHING LOGIC: https://github.com/Emerge-Lab/gpudrive/blob/bd618895acde90d8c7d880b32d87942efe42e21d/examples/experimental/eval_utils.py#L316
        # |Update simulator with the new batch of data
        env.swap_data_batch(batch)

        # |Get moving vehicle information
        curr_moving_veh_mask = torch.stack([scene_mask_ for scene_name_, scene_mask_ in moving_veh_masks.items() if scene_name_ in env.data_batch], dim=0)
        moving_veh_indices = [torch.where(mask)[0].cpu().tolist() for mask in curr_moving_veh_mask]
        if not expConfig['ego_in_construal']:
            [constr_indcs_.pop(0) for constr_indcs_ in moving_veh_indices]  # Remove ego from construals
        moving_veh_indices = [tuple(constr_indcs_) for constr_indcs_ in moving_veh_indices] # Convert to immutable objects
        print("Indices of all moving vehicles (by scene): ", moving_veh_indices)
        control_mask = env.cont_agent_mask

        # |Get IDs of construal vehicles
        curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
        for scene_num, scene_name in enumerate(curr_data_batch):
            veh_indx2ID[scene_name] = env.get_veh_ids(veh_indices=moving_veh_indices)[scene_num]

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

    if saveResults:
        print("Saving simulation results")
        # |Save the construal value information to a file
        savefl_path = out_dir+"construal_vals_"+str(datetime.now())+".pickle"
        with open(savefl_path, 'wb') as file:
            pickle.dump(construal_values, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Construal value information saved to: ", savefl_path)
        # |Save the construal trajectory information to a file
        savefl_path = out_dir+"constr_traj_obs_"+str(datetime.now())+".pickle"
        with open(savefl_path, 'wb') as file:
            pickle.dump(traj_obs, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Trajectory observation data saved to: ", savefl_path)
        # |Save the ground truth information to a file
        savefl_path = out_dir+"ground_truth_"+str(datetime.now())+".pickle"
        with open(savefl_path, 'wb') as file:
            pickle.dump(ground_truth, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Ground truth data saved to: ", savefl_path)

    return construal_values, traj_obs, ground_truth, veh_indx2ID





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
                                        moving_veh_masks: dict,
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
        curr_moving_veh_mask = torch.stack([scene_mask_ for scene_name_, scene_mask_ in moving_veh_masks.items() if scene_name_ in env.data_batch], dim=0)
        moving_veh_indices = [torch.where(mask)[0].cpu().tolist() for mask in curr_moving_veh_mask]
        # moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in curr_moving_veh_mask]
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






def generate_baseline_data( sim_agent: NeuralNet,
                            num_parallel_envs: int,
                            max_agents: int,
                            sample_size: int,
                            device: str,
                            env: GPUDriveConstrualEnv,
                            moving_veh_masks: dict,
                            observed_agents_count: int = 0,
                            construal_size: int = 0,
                            selected_construals: Dict[str, List[Tuple[int]]] = None,
                            generate_animations: bool = False,
                            saveResults: bool = False,
                            out_dir: str = None,
                            expConfig: dict = None,
                            ) -> None:
    """
    Generate baseline state representation and action probability pairs
    """
    if saveResults:
        assert out_dir, "Provide save location for data"

    state_action_pairs = {"dict_structure": '{scene_name: {\"control_mask\": mask, \"max_agents\": int, \"moving_veh_ind\": list, sample_num: ((raw_states, action_probs),...timesteps)}}'}

    control_mask = env.cont_agent_mask
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    
    #2# |Get moving vehicle information
    curr_moving_veh_mask = torch.stack([scene_mask_ for scene_name_, scene_mask_ in moving_veh_masks.items() if scene_name_ in env.data_batch], dim=0)
    moving_veh_indices = [torch.where(mask)[0].cpu().tolist() for mask in curr_moving_veh_mask]
    if not expConfig['ego_in_construal']:
        [constr_indcs_.pop(0) for constr_indcs_ in moving_veh_indices]  # Remove ego from construals
    moving_veh_indices = [tuple(constr_indcs_) for constr_indcs_ in moving_veh_indices] # Convert to immutable objects
    # moving_veh_indices = [tuple([i for i, val in enumerate(mask) if val]) for mask in moving_veh_mask]
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

    if saveResults:
        print("Saving baseline data")
        # |Save the state-action pairs to a file
        savefl_path = out_dir+"baseline_state_action_pairs_"+str(datetime.now())+".pickle"
        ObservationDataManager.save_data(state_action_pairs, savefl_path)
        # with open(savefl_path, 'wb') as file:
        #     pickle.dump(state_action_pairs, file, protocol=pickle.HIGHEST_PROTOCOL)
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
        heuristic_params: Dictionary containing the parameters for the heuristic. Keys:
            "ego_distance": parameter for (ego) distance heuristic
        normalize: If true, return the normalized [0,1] heuristics values for all construals
                    using min-max scaling

    Returns:
        The average distance or a list of distances from the ego vehicle to each vehicle in the construal
    '''
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

        #2# |Ensure the correct combination of environments and values are being used
        assert all(env_path2name(scene_path_) in construal_indices.keys() for scene_path_ in env.data_batch), \
            "Mismatch between environment data batch and default values"

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

    env_config, train_loader, env, sim_agent = get_gpuDrive_vars(
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
                                moving_veh_masks={},
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