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

from zmq import device

# |Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive-CoDec':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)
sys.path.append(str(working_dir))


# |Import CoDec Libraries
from examples.CoDec_Research.code.construals.construal_functions import get_construal_byIndex, get_construal_count
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
                        saveResults: bool = False,
                        out_dir: str = None,
                        device: str = 'cpu',
                        ) -> Dict:
    """
    Evaluate the construals using the simulation agent.

    Args:
        baseline_data (Dict): the trajectory data on which the construals are evaluated.
                                Also contains information about the control masks, agent count,
                                and indices of moving vehicles.
        construal_size (int): Size of each construal.
        sim_agent (NeuralNet): The simulation agent.
        saveResults (bool): whether to save the results of the code.
        out_dir (str): Location to save the results.
        device (str): "cpu"/"cuda", device on which to run computations.

    Returns:
        Dict: Dictionary with construal indices and their corresponding actions.
    """    
    if saveResults:
        assert out_dir, "Provide save location for data"

    construal_action_likelihoods = {}
    for scene_name, scene_info in baseline_data.items():
        if scene_name == 'dict_structure':
            continue
        construal_action_likelihoods[scene_name] = {}
        print("Processing Scene: ", scene_name)
        if set(('control_mask','max_agents','moving_veh_ind')).issubset(set(scene_info.keys())):
            control_mask = scene_info['control_mask']
            max_agents = scene_info['max_agents']
            moving_veh_indices = scene_info['moving_veh_ind']
        else:
            raise ValueError("Please ensure baseline data contains control mask, max num of agents, "
                                "and indices of all vehicles of interest")
        construal_count = get_construal_count(max_agents, moving_veh_indices, construal_size)
        prev_obs = None     # Used in code debugging below
        for baseline_constr_indxs, baseline_constr_info in scene_info.items():
            if baseline_constr_indxs == 'control_mask' or baseline_constr_indxs == 'max_agents' or \
                baseline_constr_indxs == 'moving_veh_ind':
                continue
            construal_action_likelihoods[scene_name][baseline_constr_indxs] = dict()
            for sample_num, sample in baseline_constr_info.items():
                construal_action_likelihoods[scene_name][baseline_constr_indxs][sample_num] = {}
                for construal_num in range(construal_count):
                    (test_construal_indices, test_construal_mask), _ = get_construal_byIndex(max_agents, moving_veh_indices, 
                                                                                        construal_size, construal_num, 
                                                                                        expanded_mask=True, device=device)
                    true_action_dist = None
                    pred_action_dist = None
                    print(f"Processing baseline construal {baseline_constr_indxs} against construal {test_construal_indices}, sample {sample_num}")
                    for timestep, (raw_state, true_action_logits) in enumerate(sample):
                        next_obs = process_state(raw_state, test_construal_mask, timestep)
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
                    likelihood = torch.tensor([pred_dist_[torch.argmax(tru_dist_).item()] for tru_dist_, pred_dist_ in 
                                                                                            zip(true_action_dist, pred_action_dist)]).type(torch.float64)
                    likelihood_diff = torch.tensor([pred_dist_[torch.argmax(tru_dist_)]/torch.max(tru_dist_)  for tru_dist_, pred_dist_ in 
                                                                                                                zip(true_action_dist, pred_action_dist)]).type(torch.float64)
                    # print([(torch.argmax(pred_dist_), pred_dist_[torch.argmax(pred_dist_)]) for tru_dist_, pred_dist_ in zip(true_action_dist, pred_action_dist)
                    #             if torch.argmax(tru_dist_).item() != torch.argmax(pred_dist_).item()][0])  
                    construal_action_likelihoods[scene_name][baseline_constr_indxs][sample_num][test_construal_indices] = \
                                                                                {"true_likelihoods": true_action_dist,
                                                                                "pred_likelihoods": pred_action_dist,
                                                                                "likelihood": torch.prod(likelihood),
                                                                                "log_likelihood": -1*sum(torch.log(likelihood)),
                                                                                "log_likelihood_diff": sum(torch.log(likelihood_diff)),}

    if saveResults:
        # |Save log likelihood moeasures to file
        print("Saving construal likelihood computation results")
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
    # Get average likelihood values for each construal in each scene
    avg_dict = {}
    for scn_name, scn_info in construal_action_likelihoods.items():
        avg_dict[scn_name] = {}
        for base_constr, base_constr_info in scn_info.items():
            avg_dict[scn_name][base_constr] = {}
            # print(base_constr_info)
            for test_constr, test_constr_info in base_constr_info.items():
                likelihoods = [curr_info_[likelihood_key].item() for curr_info_ in test_constr_info.values()]
                avg_dict[scn_name][base_constr][test_constr] = sum(likelihoods) / len(likelihoods)
                # print(f"scn_name: {scn_name}, constr: {constr}, likelihoods: {avg_dict[scn_name][constr]}")

    # |Get the best construals
    scene_constr_dict = {}
    for scn_name, scn_info in avg_dict.items():
        scene_constr_dict[scn_name] = {}
        for base_constr, base_constr_info in scn_info.items():
            scene_constr_dict[scn_name][base_constr] = (min(base_constr_info, key=base_constr_info.get), min(base_constr_info.values()))
            # print(f"Best construal for {scn_name}: {best_construals[scn_name]} with likelihood {avg_dict[scn_name][best_construals[scn_name]]}")
            # if scene_constr_dict[scn_name][base_constr][0] != ():
            #     # The empty construal is always the second element in the list
            #     scene_constr_dict[scn_name][base_constr] = [scene_constr_dict[scn_name][base_constr], ((), base_constr_info[()])]
            # else:
            #     # Get second highest values (best-fit construal)
            #     curr_constr = sorted(base_constr_info, key=base_constr_info.get)[1]
            #     scene_constr_dict[scn_name][base_constr] = [(curr_constr, base_constr_info[curr_constr]), scene_constr_dict[scn_name][base_constr]]   

        
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

