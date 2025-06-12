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
from examples.CoDec_Research.code.Experiment_1.exp1_imports import *
from examples.CoDec_Research.code.Experiment_1.exp1_config import *








if __name__ == "__main__":
    # |Run the main function
    """
    Save Moving Veh Masks for a particular batch of data
    This is part of a separate code because, the process requires instansiation of a separate GPUDrive 
        envrironment, which fails when multiple environments are active simultaneously.
    """
    
    # |START TIMER
    start_time = time.perf_counter()
    
    moving_veh_masks = get_mov_veh_masks(
                                        training_config=training_config, 
                                        device=device, 
                                        dataset_path=dataset_path,
                                        max_agents=moving_veh_count,
                                        result_file_loc=intermediate_results_path,
                                        processID=processID
                                        )

    # |Print the execution time
    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {math.floor(execution_time/60)} minutes and {execution_time%60:.1f} seconds")







    

# TODO (Post NeurIPS): Optimize code
# X> Email Daphne and NYU-IT, Print and sign I-539
# X> Clean up code
# X> See if the inference algorithm can be optimized (Bayesian Optimization)
# X> Edit exp1_P1_GenMasks.py and exp1_P2_....py
# X> Move Bayesian optimization code to complete_pipeline_1_P2.py
# X> Optimize 'run_policy' function
#   ---> Improvement achieved: None (for CPU), already heavily optimized code
# > Implement new heuristics
#   --- Check current implementations
# X> Reduce redunduncy in baseline data (use data-class to save data)
# > Implement logic for looping through multiple values of lambda
#   --- Encapsulate operations into functions
#   --- Implement argument-based script execution
# > Implement tool for visually editing scenarios
# > We might have to re-evaluate our measure of construal utilities or use other data
#   --- This is great for inferring discrete values of one parameter 
#   --- We might need more expressive utility values as our problem becomes more complex
#   --- I was thinking if we could train an attentional network alongside the PPO agent, which could 
#       provide behavioral utilities of various objects in the environment, and could be used to 
#       estimate the behavioral utility of various construals.
#   --- Or any construal, containing vehicles with whom the trajectories of the ego vehicle
#       intersect in the next n-seconds (can be justified by the fact that experts often 
#       make decisions based on forward simulations)


# More thoughts (On expertise): Experts move away from computationally expensive simulations to heuristics 
# Two routes for implementation: 1. weight for the value guided part and (1 - weight) for the heuristic part. 
#                                   weight decreases with expertise. 
#                                2. The complete state-space is limited by heuristics, reducing the size (=complexity)
#                                   of both the planning and evaluation models.
