# |Import Python Libraries
import time
from typing import Dict, List, Tuple
import pickle
from functools import cache
from datetime import datetime
import numpy as np

import torch
from torch.distributions.utils import logits_to_probs

import os
import sys
from pathlib import Path

# |Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))


# |Import CoDec Libraries
from examples.CoDec_Research.code.construal_functions import get_construal_byIndex, get_construal_count
from examples.CoDec_Research.code.analysis.metrics import log_likelihood


# |GPUDrive imports
from gpudrive.networks.late_fusion import NeuralNet


@cache
def get_masked_vals(size: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a tensor of masked values.

    Args:
        size (int): Size of the tensor to be created.

    Returns:
        torch.Tensor: Tensor of masked values.
    """
    return torch.tensor([0,]*size, dtype=dtype)



def process_state(raw_state: List, construal_mask: torch.tensor, timestep: int) -> torch.Tensor:
    """
    Process the raw state to make it model-compatible

    Args:
        raw_state tuple[lists]: The raw state information .

    Returns:
        torch.Tensor: Processed state tensor.
    """
    ego_states = raw_state[0].clone()
    partner_observations = raw_state[1].clone()
    road_map_observations = raw_state[2].clone()
    
    masked_values = get_masked_vals(partner_observations.shape[-1], partner_observations.dtype)  # Create masked values tensor
    partner_observations[construal_mask] = masked_values         # Mask non-construal objects
    # if timestep == 0:
    #     #|DEBUG LOGIC
    #     print(torch.where((partner_observations[0] == masked_values).all(dim=1)))
    partner_observations = partner_observations.flatten(start_dim=1)      # Flatten over feature dimension [max_veh, obs_veh, feature_size]

    compatible_obs = torch.cat(
                    (
                        ego_states,
                        partner_observations,
                        road_map_observations,
                    ),
                    dim=-1,
                )

    return compatible_obs



def get_actions(raw_state: Tuple[List], sim_agent: NeuralNet, control_mask: torch.tensor) -> torch.Tensor:
    
    next_obs = process_state(raw_state)

    action, _, _, _, action_probs = sim_agent(
        next_obs[control_mask], deterministic=False
    )

    return action_probs



def evaluate_construals(baseline_data: Dict,
                        construal_size: int, 
                        sim_agent: NeuralNet, 
                        out_dir: str,
                        ) -> Dict:
    """
    Evaluate the construals using the simulation agent.

    Args:
        construal_info (Dict): Dictionary with construal indices and masks.
        sim_agent (NeuralNet): The simulation agent.
        all_veh_objs (List[Dict]): List of all vehicle objects.
        construal_size (int): Size of each construal.

    Returns:
        Dict: Dictionary with construal indices and their corresponding actions.
    """    
    construal_action_likelihoods = {}
    for scene_name, scene_info in baseline_data.items():
        if scene_name == 'dict_structure':
            continue
        construal_action_likelihoods[scene_name] = {}
        print("Processing Scene: ", scene_name)
        control_mask = scene_info['control_mask']
        max_agents = scene_info['max_agents']
        moving_veh_indices = scene_info['moving_veh_ind']
        construal_count = get_construal_count(max_agents, moving_veh_indices, construal_size)
        prev_obs = None     # Used in code debugging below
        for construal_num in range(construal_count):
            construal_indices, construal_mask = get_construal_byIndex(max_agents, moving_veh_indices, construal_size, construal_num)
            print("Processing Construal: ", construal_indices)
            #3# |get_construal_byIndex produces masks of shape [scenes,objs], reshape to [scenes,objs,obs]
            curr_masks = [list(construal_mask) for _ in range(len(construal_mask))]     # Create multiple copies of the mask, one for each vehicle
            [msk.pop(i) for i, msk in enumerate(curr_masks)]                            # Remove ego-vehicle entry from the mask
            construal_mask = torch.tensor(curr_masks)
            true_action_dist = None
            pred_action_dist = None
            construal_action_likelihoods[scene_name][construal_indices] = dict()
            for sample_num, sample in scene_info.items():
                if sample_num == 'control_mask' or sample_num == 'max_agents' or sample_num == 'moving_veh_ind':
                    continue
                print("Processing Sample: ", sample_num)
                for timestep, (raw_state, true_action_logits) in enumerate(sample):
                    next_obs = process_state(raw_state, construal_mask, timestep)
                    # if timestep == 0:
                    #     #|DEBUG LOGIC
                    #     if prev_obs is not None:
                    #         print((next_obs==prev_obs).all())
                    #     prev_obs = next_obs
                    action, _, _, _, pred_action_logits = sim_agent(next_obs[control_mask], deterministic=False)
                    curr_true_action_dist = logits_to_probs(true_action_logits).reshape(1,-1)    # Covert to prob distribution
                    true_action_dist =  curr_true_action_dist if true_action_dist is None else torch.cat((true_action_dist, curr_true_action_dist), dim=0)
                    curr_pred_action_dist = logits_to_probs(pred_action_logits).reshape(1,-1)    # Covert to prob distribution
                    pred_action_dist = curr_pred_action_dist if pred_action_dist is None else torch.cat((pred_action_dist, curr_pred_action_dist), dim=0)
                log_likelihood = [torch.log(pred_dist_[torch.argmax(tru_dist_).item()]) for tru_dist_, pred_dist_ in 
                                                                                            zip(true_action_dist, pred_action_dist)]
                log_likelihood_diff = [torch.log( pred_dist_[torch.argmax(tru_dist_)]/torch.max(tru_dist_) ) for tru_dist_, pred_dist_ in 
                                                                                                                        zip(true_action_dist, pred_action_dist)]
                # print([(torch.argmax(pred_dist_), pred_dist_[torch.argmax(pred_dist_)]) for tru_dist_, pred_dist_ in zip(true_action_dist, pred_action_dist)
                #             if torch.argmax(tru_dist_).item() != torch.argmax(pred_dist_).item()][0])  
                construal_action_likelihoods[scene_name][construal_indices][sample_num] = {"true_likelihoods": true_action_dist,
                                                                               "pred_likelihoods": pred_action_dist,
                                                                               "log_likelihood": -1*sum(log_likelihood),
                                                                               "log_likelihood_diff": sum(log_likelihood_diff),}

    # |Save log likelihood moeasures to file
    savefl_path = out_dir+"log_likelihood_measures_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(construal_action_likelihoods, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Log likelihood measures saved to: ", savefl_path)
        
    return construal_action_likelihoods



def get_best_construals_likelihood( construal_action_likelihoods: Dict,
                                    out_dir: str,
                                    likelihood_key: str = "log_likelihood",
                                    ) -> Dict:
    """
    Get the construals with the highest likelihood to baseline model.

    Args:
        construal_action_likelihoods (Dict): Dictionary with construal indices and their corresponding actions.
        out_dir (str): Directory to save the results.
        likelihood_key (str): Key to access the likelihood values in the dictionary. Default is "log_likelihood".

    Returns:
        Dict: Dictionary with construal indices and their corresponding actions.
    """
    # |Extract construals with minimum log likelihood
    scene_constr_dict = {}
    for scene_, scene_lik_ in construal_action_likelihoods.items():
        curr_min_likelihood = np.inf
        curr_min_constr = None
        curr_generalist_likelihood = scene_lik_[()][0][likelihood_key].item()
        for const_ind_, const_lik_ in scene_lik_.items():
            for sample_num, sample_lik_ in const_lik_.items():
                if curr_min_likelihood > sample_lik_[likelihood_key] and const_ind_ != ():
                    curr_min_likelihood = sample_lik_[likelihood_key].item()
                    curr_min_constr = const_ind_
                # print(scene_, const_ind_, sample_num, sample_lik_[likelihood_key])
                # print(scene_, const_ind_, sample_num, sample_lik_["true_likelihoods"])
                # print(scene_, const_ind_, sample_num, sample_lik_["pred_likelihoods"])
        scene_constr_dict[scene_] = [(curr_min_constr, curr_min_likelihood), ((), curr_generalist_likelihood)]        
        
    savefl_path = out_dir+f"highest_construal_dict_{likelihood_key}_"+str(datetime.now())+".pickle"
    with open(savefl_path, 'wb') as file:
        pickle.dump(scene_constr_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Highest scene construals dict saved to: ", savefl_path)

    return scene_constr_dict




# |Import Pre-trained Model
# sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")

if __name__ == "__main__":    
    # |Location to store simulation results
    out_dir = "examples/CoDec_Research/results/simulation_results/"

    construal_size = 1
    
    # |Import Pre-trained Model
    sim_agent = NeuralNet.from_pretrained("daphne-cornelisse/policy_S10_000_02_27")

    dirpath = "examples/CoDec_Research/results/simulation_results/"
    # outputDir = "data/processed/Viz2/"
    baseline_data_files = [dirpath+fl_name for fl_name in os.listdir(dirpath)]
    for bdFile in baseline_data_files:
        if "baseline_state_action_pairs" not in bdFile:
            continue
        with open(bdFile, 'rb') as opn_file:
            data = pickle.load(opn_file)
        construal_action_likelihoods = evaluate_construals(data, construal_size, sim_agent)

    scene_constr_dict = get_best_construals_likelihood(construal_action_likelihoods)

