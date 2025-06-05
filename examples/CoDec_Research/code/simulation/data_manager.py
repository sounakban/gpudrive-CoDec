# |Higher-level imports
from curses import raw

from sympy import comp
from examples.CoDec_Research.code.simulation.simulation_imports import *



# # |Specify observation type constants
# EGO_OBS = 'ego_observations'  # Observation data for ego vehicle
# PARTNER_OBS = 'partner_observations'  # Observation data for partner vehicles
# ROAD_OBS = 'road_observations'  # Observation data for road features



class ObservationDataManager:
    """
    Class containing functions to load and save observation data while performing compression (during save) and decompression (during load).


    The data is stored in a dictionary with the following structure:
        {scenario_id: {
                        control_mask: mask for controlled vehicles <torch.tensor>,
                        max_agents: maximum number of agents in the env <int>,
                        moving_veh_ind: indices of moving vehicles <tuple>,
                        construal1: {sample1: observation data <structure described below>
                                     sample2: observation data <structure described below>
                                     ...},
                        construal2: {sample1: observation data <structure described below>
                                     sample2: observation data <structure described below>
                                     ...},
                        ...
                        construalN: {sample1: observation data <structure described below>
                                     sample2: observation data <structure described below>
                                     ...}
                    },
        }
                    
    observation data is a list of size timestep, with each element being a tuple: 
        (
        ego_observation <torch.tensor>,
        partner_observation <torch.tensor>,
        road_observation <torch.tensor>
        )
    """

    @classmethod
    def compress_data(cls, raw_data):
        """
        Compresses the observation data by applying a compression algorithm.
        
        Args:
            data (dict): The observation data to compress.
        
        Returns:
            dict: Compressed observation data.
        """
        print("Compressing observation data...")

        for scene_name, scene_info in raw_data.items():
            if scene_name in ('dict_structure', 'params'):
                continue  # Skip the non-construal entry
            if set(('control_mask','max_agents','moving_veh_ind')).issubset(set(scene_info.keys())):
                for baseline_constr_indxs, baseline_constr_info in scene_info.items():
                    if baseline_constr_indxs == 'control_mask' or baseline_constr_indxs == 'max_agents' or \
                        baseline_constr_indxs == 'moving_veh_ind':
                        continue
                    for sample_num, sample in baseline_constr_info.items():
                        # |Since there is redunduncy in the observations state, we can compress it by storing only the first element and consecutive differences

                        raw_states, true_action_logits = list(zip(*sample))  # Unzip the sample across timesteps

                        ego_states, partner_observations, road_map_observations = list(zip(*raw_states))

                        temp = [(y - x).to_sparse() for x, y in zip(partner_observations[:-1], partner_observations[1:])] # Compute difference between consecutive values
                        partner_observations = [partner_observations[0]] + temp # Keep first element so all subsequent elements can be retrieved

                        temp = [(y - x).to_sparse() for x, y in zip(road_map_observations[:-1], road_map_observations[1:])] # Compute difference between consecutive values
                        road_map_observations = [road_map_observations[0]] + temp # Keep first element so all subsequent elements can be retrieved

                        raw_states = list(zip(ego_states, partner_observations, road_map_observations))

                        sample = list(zip(raw_states, true_action_logits))  # Reconstruct the sample with compressed states
                        baseline_constr_info[sample_num] = sample
            else:
                raise ValueError("Please ensure baseline data contains control mask, max num of agents, "
                                    "and indices of all vehicles of interest")

        print("Compression complete")
        compressed_data = raw_data
        return compressed_data
    

    @classmethod
    def decompress_data(cls, compressed_data):
        """
        Decompresses the observation data by applying a decompression algorithm.
        
        Args:
            compressed_data (dict): The compressed observation data to decompress.
        
        Returns:
            dict: Decompressed observation data.
        """
        print("Decompressing observation data...")

        for scene_name, scene_info in compressed_data.items():
            if scene_name in ('dict_structure', 'params'):
                continue  # Skip the non-construal entry
            if set(('control_mask','max_agents','moving_veh_ind')).issubset(set(scene_info.keys())):
                for baseline_constr_indxs, baseline_constr_info in scene_info.items():
                    if baseline_constr_indxs == 'control_mask' or baseline_constr_indxs == 'max_agents' or \
                        baseline_constr_indxs == 'moving_veh_ind':
                        continue
                    for sample_num, sample in baseline_constr_info.items():
                        # |Since there is redunduncy in the observations state, we can compress it by storing only the first element and consecutive differences

                        raw_states, true_action_logits = list(zip(*sample))  # Unzip the sample across timesteps

                        ego_states, partner_observations, road_map_observations = list(zip(*raw_states))

                        partner_observations = [x.to_dense() for x in partner_observations]  # Convert sparse tensors to dense
                        partner_observations = list(accumulate(partner_observations))  # Perform cumulative sum to reconstruct original values

                        road_map_observations = [x.to_dense() for x in road_map_observations]  # Convert sparse tensors to dense
                        road_map_observations = list(accumulate(road_map_observations))  # Perform cumulative sum to reconstruct original values

                        raw_states = list(zip(ego_states, partner_observations, road_map_observations))

                        sample = list(zip(raw_states, true_action_logits))  # Reconstruct the sample with compressed states
                        baseline_constr_info[sample_num] = sample
            else:
                raise ValueError("Please ensure baseline data contains control mask, max num of agents, "
                                    "and indices of all vehicles of interest")

        print("Decompression complete")
        raw_data = compressed_data
        return raw_data
    
    
    @classmethod
    def save_data(cls, raw_data, file_path, compress_data=True):
        if compress_data:
            compressed_data = cls.compress_data(raw_data)
        else:
            compressed_data = raw_data

        with open(file_path, 'wb') as file:
            pickle.dump(compressed_data, file, protocol=pickle.HIGHEST_PROTOCOL)

        print("Observation data saved to: ", file_path)


    @classmethod
    def load_data(cls, file_path, decompress_data=True):
        with open(file_path, 'rb') as opn_file:
            compressed_data = pickle.load(opn_file)
        print("Observation data loaded from: ", file_path)

        if decompress_data:
            raw_data = cls.decompress_data(compressed_data)
        else:
            raw_data = compressed_data

        return raw_data