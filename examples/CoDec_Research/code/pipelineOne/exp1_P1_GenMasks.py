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

### Save Moving Veh Info ###
moving_veh_masks = get_mov_veh_masks(
                                    training_config=training_config, 
                                    device=device, 
                                    dataset_path=dataset_path,
                                    max_agents=moving_veh_count,
                                    result_file_loc=simulation_results_path,
                                    processID=processID
                                    )



# TODO (Post NeurIPS): Optimize code
# X> Clean up code
# X> See if the inference algorithm can be optimized (Bayesian Optimization)
# > Edit exp1_P1_GenMasks.py and exp1_P2_....py
# > Move Bayesian optimization code to complete_pipeline_1_P2.py
# > Optimize 'run_policy' function
# > Convert for loops to list comprihension in env_torch.py: function get_structured_observation
# > Reduce redunduncy in baseline data (use data-class to save data)
# > We might have to re-evaluate our measure of construal utilities or use other data
#   --- This is great for inferring discrete values of one parameter 
#   --- We might need more expressive utility values as our problem becomes more complex
#   --- I was thinking if we could train an attentional network alongside the PPO agent, which could 
#       provide behavioral utilities of various objects in the environment, and could be used to 
#       estimate the behavioral utility of various construals.
#   --- Or any construal, containing vehicles whith whom the trajectories of the ego vehicle
#       intersect in the next n-seconds (can be justified by the fact that experts often 
#       make decisions based on forward simulations)


# More thoughts (On expertise): Experts move away from computationally expensive simulations to heuristics

