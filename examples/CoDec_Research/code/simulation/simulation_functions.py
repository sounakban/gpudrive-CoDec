
# |Higher-level imports
from examples.CoDec_Research.code.simulation.simulation_imports import *

# |Local imports
from examples.CoDec_Research.code.construals.construal_functions import *
from examples.CoDec_Research.code.analysis.evaluate_construal_actions import process_state




def print_gpu_usage(device: str):
    """
    Print GPU usage
    """
    if torch.cuda.is_available() and torch.device("cuda") == device:
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / (1024 ** 2)
        utilization = torch.cuda.utilization()
        print(f"GPU usage: {utilization:3.2f}% and {mem_used_MB:3.2f}MB", end="\r")




def save_animations(sim_state_frames: dict, save_dir: str = './sim_vids'):
    """
    Save animations to a specified directory
    Args:
        sim_state_frames: Dictionary containing the frames for each environment
        save_dir: Directory to save the animations
    """
    print("Saving animations to: ", save_dir)
    sim_state_arrays = {k: np.array(v) for k, v in sim_state_frames.items()}

    # |Display and save videos locally
    # mediapy.set_show_save_dir(save_dir)
    # mediapy.show_videos(sim_state_arrays, fps=15, width=500, height=500, columns=2, codec='gif')
    
    # |Save videos locally
    for env_id, frames in sim_state_arrays.items():        
        mediapy.write_video(
            str(save_dir+'/'+env_id+'.gif'),
            frames,
            fps=15,
            codec='gif',
        )


@cache
def get_action_template(total_envs: int, max_agents: int, device: str) -> torch.Tensor:
    """
    Create an action template for the environment based on the control mask.
    This function is created for code optimization through caching.
    
    Args:
        total_envs: Total number of environments
        max_agents: Maximum number of agents in the environment
        device: Device to place the tensor on (CPU or GPU)
    
    Returns:
        action_template: A tensor with shape (total_envs, max_agents) initialized to zeros
    """
    action_template = torch.zeros(
        (total_envs, max_agents), dtype=torch.int64, device=device
    )
    return action_template



def run_policy(env: GPUDriveTorchEnv,
               sim_agent: NeuralNet,
               next_obs: torch.Tensor,
               control_mask: List,
               construal_masks: List,
               time_step: int,
               total_envs: int,
               max_agents: int,
               device: str,
               frames: dict,
               const_num: int,
               sample_num: int,
               generate_animations: bool = False,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    # |Predict actions
    action, _, _, _, action_logits = sim_agent(next_obs[control_mask], deterministic=False)

    # action_template = torch.zeros(
    #     (total_envs, max_agents), dtype=torch.int64, device=device
    # )
    action_template = get_action_template(total_envs, max_agents, device)
    action_template[control_mask] = action.to(device)

    # |Garb raw observations before stepping through environment for debug logic later
    # tmp1_ = env.get_obs(raw_obs=True)

    # |Step
    env.step_dynamics(action_template)

    #2# |DEBUG: Print GPU usage
    # print_gpu_usage(device)

    # |Render
    if generate_animations:
        sim_states = env.vis.plot_simulator_state(
            env_indices=list(range(total_envs)),
            time_steps=[time_step]*total_envs,
            zoom_radius=70,
        )
    
        if construal_masks:
            for env_num_, env_path_ in enumerate(env.data_batch):
                frames[f"env_{env_path2name(env_path_)}-constr_{const_num}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num_])) 
        else:
            for env_num_, env_path_ in enumerate(env.data_batch):
                frames[f"env_{env_path2name(env_path_)}-sample_{sample_num}"].append(img_from_fig(sim_states[env_num_])) 

    next_obs = env.get_obs(partner_mask=construal_masks)
    reward = env.get_rewards()
    done = env.get_dones()
    info = env.get_infos()

    # |Variable for trajectory illustrations
    plottable_obs = env.get_structured_obs(partner_mask=construal_masks)

    # |DEBUG LOGIC: Check if state processing logic is operating correctly
    # tmp2_ind_, tmpt2_mask_ = get_construals(64, (1,), 1, expanded_mask = True)['default']
    # tmpt2_mask_ = torch.tensor(tmpt2_mask_)  # Get generalist mask
    # tmp1_ = [process_state(tmp1_tmp_, tmpt2_mask_, 10) for tmp1_tmp_ in tmp1_]
    # tmp1_ = torch.stack(tmp1_)
    # print("\nProcess obsrvation logic working:", (tmp1_==next_obs).all())
    # tmp_action_, _, _, _, tmp_action_probs_ = sim_agent(tmp1_[control_mask], deterministic=False)
    # print("Action probabilities match?: ", (tmp_action_probs_== action_probs).all())
    # print("Action match?: ", torch.max(torch.abs(logits_to_probs(tmp_action_probs_) - logits_to_probs(action_probs))))

    return next_obs, reward, done, info, plottable_obs, action_logits







def simulate_policies(env: GPUDriveConstrualEnv, 
                        total_envs: int,
                        max_agents: int,
                        sample_size: int,
                        sim_agent: NeuralNet,
                        control_mask: List,
                        device: str,
                        observed_agents_count: int = 0,
                        moving_veh_indices: List = [],
                        construal_size: int = 0,
                        selected_construals: Dict[str, List[Tuple[int]]] = None,
                        generate_animations: bool = False,
                        save_state_action_pairs: bool = False,
                        save_trajectory_obs: bool = False,
                        ) -> Tuple[dict, dict, dict, dict]:
    """
    Simulate environment under construed (or non-construed) observation spaces, and
        save trajectories, observations, and values of model agents (when for each construal)
    
    Args:
        env: GPUDrive simulation environment
        total_envs: Number of scenarios being simulated
        max_agents: Maximum number of (tracked) agents in the environment
        sample_size: The number of samples to draw to estimate value of construals
        sim_agent: The (pre-trained) agent model
        control_mask: The mask specifying which agent(s) is controlled by the model
        device: Whether use CPU or GPU for computation
        observed_agents_count: Number of agents being observed by model agent
        moving_veh_indices: Indices of (moving) vehicles of interest in the environment
        construal_size: The number of agents beings observed in each construal,
                            set to 0, when only genersalist policies are to be simulated
        selected_construals: Dictionary containing construals to be simulated for each environment.
        generate_animations: Boolean values indicates whether to save simulation animations
        save_state_action_pairs: Boolean values indicates whether to save state-action pairs
        save_trajectory_obs: Boolean values indicates whether to save vehicle trajectory data
    
    Returns (two dictionaries):
        construal_values: Dictionary that contains the expected utility of each construal
                            Structure: {scene_name: {construal_mask: expected_utility}}
        all_obs: Dictionary that contains the observations (agent trajectories) for each construal
                    Structure: {scene_name: {construal_mask: {sample_num: [vehicles,timestep,coord]}}}

    """
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    all_obs = {env_name: {} for env_name in curr_data_batch}                        # Dictionary that contains the observations (agent trajectories) for each construal
    state_action_pairs = {env_name: {} for env_name in curr_data_batch}             # Dictionary that contains all states and corresponding action distributions
    construal_values = {env_name: {} for env_name in curr_data_batch}               # Dictionary that contains the expected utility per construal
    frames = {}

    # Compute the number of loops
    if selected_construals is None:
        loop_count = math.comb(observed_agents_count, construal_size)
    else:
        loop_count = set([len(sel_constr_) for sel_constr_ in selected_construals.values()])
        if len(loop_count) > 1:
            raise ValueError("Number of construals per scene must be the same")
        loop_count = loop_count.pop()
 
    for const_num in range(loop_count):
        # |LOOP rollout for each construal

        # next_obs = env.reset()
        # print("Observation shape: ", next_obs.shape)

        if construal_size > 0:
            #2# |IF simulating on construals
            #3# |Define observation mask for construal
            if selected_construals is None:
                construal_info = [get_construal_byIndex(max_agents, moving_veh_indices[scene_num], construal_size, const_num, \
                                                        expanded_mask=True, device=device) 
                                    for scene_num in range(len(curr_data_batch))]
            else:                
                construal_info = [get_selected_construal_byIndex(max_agents, moving_veh_indices[scene_num_], construal_size, \
                                                                 const_num, selected_construals[sene_name_], expanded_mask=True, \
                                                                 device=device) 
                                    for scene_num_, sene_name_ in enumerate(curr_data_batch)]
            
            construal_info, construal_done = zip(*construal_info)
            mask_indices, construal_masks = zip(*construal_info)   # Unzip construal masks
            
            for scene_num, scene_name in enumerate(curr_data_batch):
                if mask_indices[scene_num] in all_obs[scene_name]:
                    #3# | If these construals were encountered before, when sampling with replacement.
                    #       Shift previous data by sample size
                    curr_tot_samples = max(all_obs[scene_name][mask_indices[scene_num]].keys()) + 1
                    for sample_num_ in reversed(range(curr_tot_samples)):
                        all_obs[scene_name][mask_indices[scene_num]][sample_num_+sample_size] = \
                            all_obs[scene_name][mask_indices[scene_num]][sample_num_]
                        all_obs[scene_name][mask_indices[scene_num]][sample_num_] = None
                        state_action_pairs[scene_name][mask_indices[scene_num]][sample_num_+sample_size] = \
                            state_action_pairs[scene_name][mask_indices[scene_num]][sample_num_]
                        state_action_pairs[scene_name][mask_indices[scene_num]][sample_num_] = []
                        frames[f"env_{scene_name}-constr_{const_num}-sample_{sample_num_}"] = []
                else:
                    #3# |Otherwise, create empty dictionaries to store information
                    all_obs[scene_name][mask_indices[scene_num]] = {sample_num_: None for sample_num_ in range(sample_size)}
                    state_action_pairs[scene_name][mask_indices[scene_num]] = {sample_num_: [] for sample_num_ in range(sample_size)}                
                    frames.update({f"env_{scene_name}-constr_{const_num}-sample_{sample_num_}": [] for sample_num_ in range(sample_size)})
        else:
            #2# |IF only simulating generalist policy
            construal_masks = None
            const_num = -1
            #3# |Create empty dictionaries to store information
            for scene_num, scene_name in enumerate(curr_data_batch):
                all_obs[scene_name][moving_veh_indices[scene_num]] = {sample_num_: None for sample_num_ in range(sample_size)}
                state_action_pairs[scene_name][moving_veh_indices[scene_num]] = {sample_num_: [] for sample_num_ in range(sample_size)}                
                frames.update({f"env_{scene_name}-sample_{sample_num_}": [] for sample_num_ in range(sample_size)})
        
        curr_sample_rewards = []   # Keep track of rewards
        for sample_num in range(sample_size):
            print("\tsample ", sample_num+1)
            
            _ = env.reset()
            next_obs = env.get_obs(partner_mask=construal_masks)
            for time_step in range(env.episode_len):
                #2# |Roll out policy for a specific construal
                print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

                #3# |Get raw observations before stepping through environment
                raw_obs = env.get_obs(raw_obs=True)

                #3# |Execute policy
                next_obs, reward, done, info, plottable_obs, action_logits = run_policy( env=env,
                                                                                        sim_agent=sim_agent,
                                                                                        next_obs=next_obs,
                                                                                        control_mask=control_mask,
                                                                                        construal_masks=construal_masks,
                                                                                        time_step=time_step,
                                                                                        total_envs=total_envs,
                                                                                        max_agents=max_agents,
                                                                                        device=device,
                                                                                        frames=frames,
                                                                                        const_num=const_num,
                                                                                        sample_num=sample_num,
                                                                                        generate_animations=generate_animations,
                                                                                    )

                #3# |Record state-action pairs
                if save_state_action_pairs:
                    for scene_num, action_dist in enumerate(action_logits):
                        scene_name = curr_data_batch[scene_num]
                        if const_num == -1:
                            # |Generalist policy
                            state_action_pairs[scene_name][moving_veh_indices[scene_num]][sample_num].append((raw_obs[scene_num], action_dist))
                        else:
                            # |Construal policy
                            state_action_pairs[scene_name][mask_indices[scene_num]][sample_num].append((raw_obs[scene_num], action_dist))

                #3# |Record observations
                if save_trajectory_obs:
                    for scene_num, all_pos in enumerate(plottable_obs['pos_ego']):
                        all_pos = torch.stack(all_pos, dim=1).unsqueeze(1)      # Reshape from [vehicles,coord] to [vehicles,1,coord] for timesteps
                        scene_name = curr_data_batch[scene_num]
                        if const_num == -1:
                            # |Generalist policy
                            if all_obs[scene_name][moving_veh_indices[scene_num]][sample_num] is None:                                
                                all_obs[scene_name][moving_veh_indices[scene_num]][sample_num] = all_pos
                            else:
                                all_obs[scene_name][moving_veh_indices[scene_num]][sample_num] = torch.cat([all_obs[scene_name][moving_veh_indices[scene_num]][sample_num], all_pos], dim=1)
                        else:
                            # |Construal policy
                            if all_obs[scene_name][mask_indices[scene_num]][sample_num] is None:
                                all_obs[scene_name][mask_indices[scene_num]][sample_num] = all_pos
                            else:
                                all_obs[scene_name][mask_indices[scene_num]][sample_num] = torch.cat([all_obs[scene_name][mask_indices[scene_num]][sample_num], all_pos], dim=1)

                if done.all():
                    break
            print() # Change to new line after step prints
                
            curr_sample_rewards.append(reward[control_mask].tolist())

        #2# |Save animations
        if generate_animations:
            save_animations(frames)

        #2# |Calculate value (average reward) for each construal
        if construal_size > 0:
            curr_vals = [sum(x)/sample_size for x in zip(*curr_sample_rewards)]
            for scene_num, val in enumerate(curr_vals):
                construal_values[curr_data_batch[scene_num]][mask_indices[scene_num]] = val
            print("Processed mask(s): ", mask_indices, ", with value(s):", curr_vals)

            # if all([mask == () for mask in mask_indices]):
            if all(construal_done):
                #2# |Break loop once list of construals for all scenarios have been exhausted
                break
    
    #2# Extract ground-truth data
    ground_truth = {'traj': {}, 'traj_valids': {}, 'contr_veh_indices': {}}
    for scene_num, scene_name in enumerate(curr_data_batch):
        ground_truth['traj'][scene_name] = env.get_data_log_obj().pos_xy[scene_num].tolist()
        ground_truth['traj_valids'][scene_name] = env.get_data_log_obj().valids[scene_num].tolist()
        ground_truth['contr_veh_indices'][scene_name] = torch.where(control_mask[scene_num])[0].tolist()


    # print("\nExpected utility by contrual: ", construal_values)
    
    return (construal_values, all_obs, ground_truth, state_action_pairs)






def simulate_selected_construal_policies(env: GPUDriveConstrualEnv, 
                                        observed_agents_count: int,
                                        construal_size: int,
                                        total_envs: int,
                                        max_agents: int,
                                        moving_veh_indices: List[List],
                                        sample_size: int,
                                        sim_agent: NeuralNet,
                                        control_mask: List,
                                        device: str,
                                        selected_construals: Dict[str, List[Tuple[int]]],
                                        generate_animations: bool = False,
                                        ) -> Tuple[dict, dict, dict]:
    """
    Simulate environment under hand-picked construed observation spaces
    
    Args:
        env: GPUDrive simulation environment
        observed_agents_count: Number of agents being observed by model agent
        construal_size: The number of agents beings observed in each construal
        total_envs: Number of scenarios being simulated
        max_agents: Maximum number of (tracked) agents in the environment
        moving_veh_indices: Indices of (moving) vehicles of interest in the environment
        sample_size: The number of samples to draw to estimate value of construals
        sim_agent: The (pre-trained) agent model
        control_mask: The mask specifying which agent(s) is controlled by the model
        device: Whether use CPU or GPU for computation
        selected_construals: Dictionary containing construals to be simulated for each environment.
    
    Returns:

    """
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    all_obs = {env_name: {} for env_name in curr_data_batch}                        # Dictionary that contains the observations (agent trajectories) for each construal
    state_action_pairs = {env_name: {} for env_name in curr_data_batch}             # Dictionary that contains all states and corresponding action distributions
    construal_values = {env_name: {} for env_name in curr_data_batch}               # Dictionary that contains the expected utility per construal
    
    # |Get all construal info then filter for selected construals
    selected_construal_info = {scene_name_: [] for scene_name_ in curr_data_batch}
    for scene_num, scene_name in enumerate(curr_data_batch):
        curr_env_construals = get_construals(max_agents, moving_veh_indices[scene_num], construal_size, expanded_mask = True)
        curr_env_selected_construals = selected_construals[scene_name]
        selected_construal_info[scene_name]= []
        for constr_ind, _ in curr_env_selected_construals:
            selected_construal_info[scene_name].extend([constr_info_ for _, constr_info_ in 
                                                        curr_env_construals.items() if constr_info_[0] == constr_ind])
    
    loop_count = set([len(env_constr_) for env_constr_ in selected_construals.values()])    # Get the number of loops = number of selected construals per scene
    if len(loop_count) > 1:
        raise ValueError("Number of construals per scene must be the same")
    
    loop_count = loop_count.pop()
    print("Construals per scene: ", loop_count)
    for const_num in range(loop_count):
        # |Repeat rollout for each construal

        # next_obs = env.reset()
        # print("Observation shape: ", next_obs.shape)

        curr_mask_indices = [constr_info_[const_num][0] for constr_info_ in selected_construal_info.values()]
        curr_construal_masks = [constr_info_[const_num][1] for constr_info_ in selected_construal_info.values()]
        frames = {f"env_{env_name_}-constr_{const_num}-sample_{sample_num_}": [] for sample_num_ in range(sample_size) for env_name_ in curr_data_batch}
        curr_samples = []   # Keep track of rewards
        print("Processing construals: ", curr_mask_indices)
        for sample_num in range(sample_size):
            print("\tsample ", sample_num)
            
            _ = env.reset()
            next_obs = env.get_obs(partner_mask=curr_construal_masks)
            for time_step in range(env.episode_len):
                #2# |Roll out policy for a specific construal
                print(f"\r\t\tStep: {time_step+1}", end="", flush=True)

                #3# |Execute policy
                next_obs, reward, done, info, plottable_obs, action_logits = run_policy( env=env,
                                                                                        sim_agent=sim_agent,
                                                                                        next_obs=next_obs,
                                                                                        control_mask=control_mask,
                                                                                        construal_masks=curr_construal_masks,
                                                                                        time_step=time_step,
                                                                                        total_envs=total_envs,
                                                                                        max_agents=max_agents,
                                                                                        device=device,
                                                                                        frames=frames,
                                                                                        const_num=const_num,
                                                                                        sample_num=sample_num,
                                                                                        generate_animations=generate_animations,
                                                                                    )

                #3# |Record observations for each construal
                for env_num, all_pos in enumerate(plottable_obs['pos_ego']):
                    all_pos = torch.stack(all_pos, dim=1).unsqueeze(1)      # Reshape from [vehicles,coord] to [vehicles,1,coord] for timesteps
                    env_name = curr_data_batch[env_num]
                    if curr_mask_indices[env_num] in all_obs[env_name] and sample_num in all_obs[env_name][curr_mask_indices[env_num]]:
                        all_obs[env_name][curr_mask_indices[env_num]][sample_num] = torch.cat([all_obs[env_name][curr_mask_indices[env_num]][sample_num], all_pos], dim=1)
                    elif curr_mask_indices[env_num] not in all_obs[env_name]:
                        all_obs[env_name][curr_mask_indices[env_num]] = {sample_num : all_pos}                        
                    elif sample_num not in all_obs[env_name][curr_mask_indices[env_num]]:
                        all_obs[env_name][curr_mask_indices[env_num]][sample_num] = all_pos
                    else:
                        raise ValueError(f"Unknown Situation: {env_name}, {curr_mask_indices[env_num]}, {sample_num}")
                
                if done.all():
                    break
            print() # Change to new line after step prints
                
            curr_samples.append(reward[control_mask].tolist())

        #2# |Save animations
        if generate_animations:
            save_animations(frames)

        if all([mask == () for mask in curr_mask_indices]):
            #2# |Break loop once list of construals for all scenarios have been exhausted
            break

    # print("\nExpected utility by contrual: ", construal_values)
    
    # TODO: Extract valid flags, and ground truth trajectories
    return all_obs






