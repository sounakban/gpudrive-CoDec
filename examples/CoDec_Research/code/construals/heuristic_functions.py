# |Higher-level imports
from examples.CoDec_Research.code.construals.construal_imports import *

from sklearn import preprocessing

### Support Functions ###
# |Compute eucledian distance between two points
@cache
def euclidean_distance(point1, point2):
    return norm(np.array(point1) - np.array(point2))
    # return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

# |Compute positional angle (in radians) of (2D) point 2 relative to point 1
@cache
def direction_rad(point1, point2):
    return math.atan2(point2[1]-point1[1], point2[0]-point1[0])

# |Compute angle between two vectors [+1 when heading in the same direction, -1 when heading in opposite directions]
@cache
def vector_angle(vec1, vec2):
    return dot(np.array(vec1),np.array(vec2))/( (norm(vec1)*norm(vec2))+1e-10 )

# |Compute relative vector (vector difference) given two vectors
@cache
def relative_vector(vec1, vec2):
    return tuple(np.array(vec2) - np.array(vec1))

# |Compute the unitary vecotor of any vector
@cache
def unit_vector(vec):
    return tuple( np.array(vec)/( norm(np.array(vec))+1e-10 ) ) # Adding a small value to avoid division by zero



#######################################################
################# HEURISTICS FUNCTIONS ################
#######################################################

### Construal Heuristic: Distance from ego ###
def get_construal_veh_distance_ego(env: GPUDriveConstrualEnv, construal_indices: dict, average: bool = True,
                               normalize: bool = False):
    '''
    Get the distance of each vehicle (or average) in the construal to the ego vehicle

    Args:
        env: The environment object
        construal_indices: A dictionary whose values are lists of indices corresponding to each construal in a scene
        average: If true, return the average distance of all vehicles in the construal to the ego vehicle
        normalize: If true, normalize distances of all vehicles for each scene to [0,1], using min-max scaling

    Returns:
        dict: The average distance or a list of distances from the ego vehicle to each vehicle in the construal
    '''
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    # |Populate dictionary with all relevant information
    info_dict = dict()
    for env_num, env_name in enumerate(curr_data_batch):
        info_dict[env_name] = dict()
        try:
            info_dict[env_name]['ego_index'] = torch.where(env.cont_agent_mask[env_num])[0].item()
        except RuntimeError:
            raise RuntimeError("Environment has more than one ego vehicle.")
        info_dict[env_name]['construal_indices'] = construal_indices[env_name]
    
    # |Get all vehicle distances
    all_pos = env.get_data_log_obj().pos_xy
    distance_dict = dict()

    for env_num, env_name in enumerate(curr_data_batch):
        distance_dict[env_name] = dict()
        all_distances = [euclidean_distance(
                                                tuple( all_pos[env_num][info_dict[env_name]['ego_index']][0].cpu().tolist() ),
                                                tuple( all_pos[env_num][i][0].cpu().tolist() )
                                            ) 
                                            for i in range(len(all_pos[env_num]))]
        all_distances = np.array(all_distances)
              
        if normalize:
            #2# |Normalize distances to [0,1] using min-max scaling
            all_distances = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(all_distances.reshape(-1,1)).reshape(1,-1)[0]

        #2# |Multiplied by -1 as distance is a penalty term, greater values are associated with higher penalty
        all_distances = -1*all_distances 

        for curr_indices in info_dict[env_name]['construal_indices']:
            distance_dict[env_name][curr_indices] = [all_distances[i] for i in curr_indices]
            if average:
                if len(distance_dict[env_name][curr_indices]) > 0:
                    distance_dict[env_name][curr_indices] = sum(distance_dict[env_name][curr_indices])/len(distance_dict[env_name][curr_indices])
                else:
                    #3# | If empty construal
                    distance_dict[env_name][curr_indices] = 0
                    
    return distance_dict



### Construal Heuristic: Deviation from ego heading ###
def get_construal_dev_ego_heading(env: GPUDriveConstrualEnv, construal_indices: dict, average: bool = True,
                               normalize: bool = False):
    '''
    Get the angle (in radians) between the heading of the ego vehicle and the positional direction of vehicles in the construal relative to the ego.

    Args:
        env: The environment object
        construal_indices: A dictionary whose values are lists of indices corresponding to each construal in a scene
        average: If true, return the average angle for all vehicles in the construal
        normalize: If true, normalize angles across all vehicles for each scene to [0,1], using min-max scaling

    Returns:
        dict: The average relative or a list of relavive headings from the ego vehicle to each vehicle in the construal
    '''
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    # |Populate dictionary with all relevant information
    info_dict = dict()
    for env_num, env_name in enumerate(curr_data_batch):
        info_dict[env_name] = dict()
        info_dict[env_name]['ego_index'] = torch.where(env.cont_agent_mask[env_num])[0].item()
        info_dict[env_name]['construal_indices'] = construal_indices[env_name]
    
    # |Get relative directional location for all vehicles
    all_pos = env.get_data_log_obj().pos_xy
    heading_dev_dict = dict()

    for env_num, env_name in enumerate(curr_data_batch):
        heading_dev_dict[env_name] = dict()
        heading_dev = [direction_rad(
                                        tuple( all_pos[env_num][info_dict[env_name]['ego_index']][0].cpu().tolist() ),
                                        tuple( all_pos[env_num][i][0].cpu().tolist())
                                    )
                                    for i in range(len(all_pos[env_num]))]  
        ego_vel_x, ego_vel_y = env.get_data_log_obj().vel_xy[env_num][info_dict[env_name]['ego_index']][0]
        ego_heading = math.atan2(ego_vel_y, ego_vel_x)
        heading_dev = np.abs(np.array(heading_dev) - ego_heading)
              
        if normalize:
            #2# |Normalize heading radians to [0,1] using min-max scaling 
            heading_dev = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(heading_dev.reshape(-1,1)).reshape(1,-1)[0]
            
        #2# |Multiplied by -1 as higher angles mean further away from the ego heading and are less likely to be considered
        heading_dev = -1*heading_dev

        for curr_indices in info_dict[env_name]['construal_indices']:
            heading_dev_dict[env_name][curr_indices] = [heading_dev[i] for i in curr_indices]
            if average:
                if len(heading_dev_dict[env_name][curr_indices]) > 0:
                    heading_dev_dict[env_name][curr_indices] = sum(heading_dev_dict[env_name][curr_indices])/len(heading_dev_dict[env_name][curr_indices])
                else:
                    #3# | If empty construal
                    heading_dev_dict[env_name][curr_indices] = 0
                    
    return heading_dev_dict



### Construal Heuristic: heading relative to ego ###
def get_construal_rel_heading_ego(env: GPUDriveConstrualEnv, construal_indices: dict, average: bool = True,
                               normalize: bool = False):
    '''
    Get the angle (in radians) between the heading of vehicles (or average) in the construal and the heading of the ego vehicle

    Args:
        env: The environment object
        construal_indices: A dictionary whose values are lists of indices corresponding to each construal in a scene
        average: If true, return the average angle for all vehicles in the construal
        normalize: If true, normalize angles across all vehicles for each scene to [0,1], using min-max scaling

    Returns:
        dict: The average relative or a list of relavive headings from the ego vehicle to each vehicle in the construal
    '''
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    # |Populate dictionary with all relevant information
    info_dict = dict()
    for env_num, env_name in enumerate(curr_data_batch):
        info_dict[env_name] = dict()
        info_dict[env_name]['ego_index'] = torch.where(env.cont_agent_mask[env_num])[0].item()
        info_dict[env_name]['construal_indices'] = construal_indices[env_name]
    
    # |Get relative heading for all vehicles
    all_vel = env.get_data_log_obj().vel_xy
    relative_heading_dict = dict()

    for env_num, env_name in enumerate(curr_data_batch):
        relative_heading_dict[env_name] = dict()
        relative_headings = [vector_angle(
                                            tuple( all_vel[env_num][info_dict[env_name]['ego_index']][0].cpu().tolist() ),
                                            tuple( all_vel[env_num][i][0].cpu().tolist() )
                                        ) 
                                        for i in range(len(all_vel[env_num]))]
        relative_headings = np.array(relative_headings)
              
        #2# |Normalization not necessary, as values are computed are already in range -1,1

        for curr_indices in info_dict[env_name]['construal_indices']:
            relative_heading_dict[env_name][curr_indices] = [relative_headings[i] for i in curr_indices]
            if average:
                if len(relative_heading_dict[env_name][curr_indices]) > 0:
                    relative_heading_dict[env_name][curr_indices] = sum(relative_heading_dict[env_name][curr_indices])/len(relative_heading_dict[env_name][curr_indices])
                else:
                    #3# | If empty construal
                    relative_heading_dict[env_name][curr_indices] = 0
                    
    return relative_heading_dict



### Construal Heuristic: Deviation from collision course ###
def get_construal_dev_collision_ego(env: GPUDriveConstrualEnv, construal_indices: dict, average: bool = True,
                               normalize: bool = False):
    '''
    Get the 'deviation from ego collision' $(vel_x - vel_ego).(pos_x - pos_ego)$, dot product of 
        relative velocity and relative displacement for vehicles in the construal, relative to the ego.
        This is a measure of how much the straight-line paths of the ego and construal vehicles deviate
        from the collission trajectory.

    Args:
        env: The environment object
        construal_indices: A dictionary whose values are lists of indices corresponding to each construal in a scene
        average: If true, return the average angle for all vehicles in the construal
        normalize: If true, normalize angles across all vehicles for each scene to [0,1], using min-max scaling

    Returns:
        dict: The average relative or a list of relavive headings from the ego vehicle to each vehicle in the construal
    '''
    curr_data_batch = [env_path2name(env_path_) for env_path_ in env.data_batch]
    # |Populate dictionary with all relevant information
    info_dict = dict()
    for env_num, env_name in enumerate(curr_data_batch):
        info_dict[env_name] = dict()
        info_dict[env_name]['ego_index'] = torch.where(env.cont_agent_mask[env_num])[0].item()
        info_dict[env_name]['construal_indices'] = construal_indices[env_name]
    
    # |Get deviation from collission course (with ego) for all vehicles
    all_pos = env.get_data_log_obj().pos_xy
    all_vel = env.get_data_log_obj().vel_xy
    deviation_collision_dict = dict()

    for env_num, env_name in enumerate(curr_data_batch):
        deviation_collision_dict[env_name] = dict()
        relative_dis = [relative_vector(
                                            tuple( all_pos[env_num][info_dict[env_name]['ego_index']][0].cpu().tolist() ),
                                            tuple( all_pos[env_num][i][0].cpu().tolist() )
                                        ) 
                                        for i in range(len(all_pos[env_num]))]
        relative_vel = [relative_vector(
                                            tuple( all_vel[env_num][info_dict[env_name]['ego_index']][0].cpu().tolist() ),
                                            tuple( all_vel[env_num][i][0].cpu().tolist() )
                                        ) 
                                        for i in range(len(all_vel[env_num]))]
        #2# |Compute deviation from collision course as dot product of relative velocity and relative displacement
        #2# |Value 0 indicates vehicles moving away from each other, 1 indicates vehicles moving directly towards each other
        dev_collision = [abs( dot(unit_vector(vel_vec), unit_vector(dis_vec)) ) for 
                                vel_vec, dis_vec in zip(relative_vel, relative_dis)]
        dev_collision = np.array(dev_collision)

        if normalize:
            #2# |Normalize values between [-1,1] using min-max scaling 
            dev_collision = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit_transform(dev_collision.reshape(-1,1)).reshape(1,-1)[0]

        for curr_indices in info_dict[env_name]['construal_indices']:
            deviation_collision_dict[env_name][curr_indices] = [dev_collision[i] for i in curr_indices]
            if average:
                if len(deviation_collision_dict[env_name][curr_indices]) > 0:
                    deviation_collision_dict[env_name][curr_indices] = sum(deviation_collision_dict[env_name][curr_indices])/len(deviation_collision_dict[env_name][curr_indices])
                else:
                    #3# | If empty construal
                    deviation_collision_dict[env_name][curr_indices] = 0
                    
    return deviation_collision_dict



### Construal Heuristic: Cardinality ###
def get_construal_cardinality(env: GPUDriveConstrualEnv, construal_indices: dict, average: bool = True,
                               normalize: bool = False): 
    '''
    Get the cardinality of each construal

    Args:
        env: The environment object
        construal_indices: A dictionary whose values are lists of indices corresponding to each construal in a scene
        average: Unused for this logic, but included for consistency
        normalize: If true, normalize cardinality values of all construals in each scene to [0,1], using min-max scaling

    Returns:
        dict: Cardinality values for each construal
    '''
    cardinality_dict = dict()
    for scene_name, scene_construal_indices in construal_indices.items():
        curr_cardinalities = {indices: len(indices) for indices in scene_construal_indices}
        min_cardinality = min(curr_cardinalities.values())
        max_cardinality = max(curr_cardinalities.values())  
        if normalize:
            #2# |Normalize cardinalities to [0,1] using min-max scaling 
            #2# |Multiplied by -1 as cardinality is a penalty term, greater values are associated with higher penalty
            curr_cardinalities = {indices: -1*( (cardinality_ - min_cardinality) / (max_cardinality - min_cardinality) ) for indices, cardinality_ in curr_cardinalities.items()}
        cardinality_dict[scene_name] = curr_cardinalities
    return cardinality_dict


heuristics_to_func = {
                        "cardinality": get_construal_cardinality,
                        "ego_distance": get_construal_veh_distance_ego,
                        "dev_ego_heading": get_construal_dev_ego_heading,
                        "rel_heading": get_construal_rel_heading_ego,
                        "dev_collission": get_construal_dev_collision_ego,
                        }