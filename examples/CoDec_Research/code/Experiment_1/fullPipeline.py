import os
import time

import numpy as np
import math

import pickle
import re



start_time = time.perf_counter()


# |Custom sampling function, ensuring more uniform sampling for smaller sample-sizes (<50)
sample_range = (5,11)       # Number of samples to consider for each iteration, generates sum(range(sample_range)) samples
custom_sampling = lambda range_min, range_max: [i for indx, num_samples in enumerate(range(*sample_range)) 
                                                    for i in np.linspace(range_min+(indx*0.2),range_max-(indx*0.2),num_samples)]


# |Loop through lambdas
lambda_heur = ["dev_ego_heading","rel_heading","dev_collission"]
# lambda_val = [[2,],[-2,],[-3,]]     #DEBUG: Test
#2# |Custom sampling
# lambda_val = [custom_sampling(-15,15), custom_sampling(-20,20), custom_sampling(-30,15)]
#2# |OR uniform sampling
# sample_count = 100
# lambda_val = [np.random.uniform(-15,15,sample_count), 
#                 np.random.uniform(-15,15,sample_count), 
#                 np.random.uniform(-20,20,sample_count), ]
#2# |OR get sampled lambda values from file
with open("examples/CoDec_Research/code/Experiment_1/sampleLambdas.pkl", 'rb') as file:
    lambda_val = pickle.load(file)


### LOGIC 1: Using a single sample size (for trajectories) ###
# lambda_val = list(zip(*lambda_val))                 # Change list structure for code compatibility
# lambda_heur = [lambda_heur]*len(lambda_val)         # Change list structure for code compatibility
# for type_, val_ in zip(lambda_heur,lambda_val):
#     val_ = ','.join([str(i) for i in val_])     # Pass argument as string of comma-spearated values
#     type_ = ','.join(type_)                     # Pass argument as string of comma-spearated values
#     os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P1_GenMasks.py {} {}".format(type_, val_))
#     os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P2_GetVals.py {} {}".format(type_, val_))
#     os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P3_GenData.py {} {}".format(type_, val_))
#     os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P4_Inference.py {} {}".format(type_, val_))


### LOGIC 2: Iterating over different sample sizes (for trajectories). Useful for sample-size efficiency test ###
lambda_val = list(zip(*lambda_val))                 # Change list structure for code compatibility
lambda_heur = [lambda_heur]*len(lambda_val)         # Change list structure for code compatibility
for traj_samples_ in list(range(1, 11))+list(range(15,51,5)):
    # Update sample size in config file
    #2# Read in the file
    with open('examples/CoDec_Research/code/shared_config.py', 'r') as config_file:
        filedata = config_file.readlines()
    #2# Replace the target string
    assert "construal_count_baseline" in filedata[71], "Configuration file changed, ensure line number in the next command is correct"
    filedata[71] = re.sub(r': [0-9]+,', f': {traj_samples_},', filedata[71])
    #2# Write the file out again
    with open('examples/CoDec_Research/code/shared_config.py', 'w') as config_file:
        config_file.writelines(filedata)

    # Run iterations
    for type_, val_ in zip(lambda_heur,lambda_val):
        val_ = ','.join([str(i) for i in val_])     # Pass argument as string of comma-spearated values
        type_ = ','.join(type_)                     # Pass argument as string of comma-spearated values
        os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P1_GenMasks.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P2_GetVals.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P3_GenData.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/Experiment_1/exp1_P4_Inference.py {} {}".format(type_, val_))


# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Pipeline execution time: {math.floor(execution_time/3600)} hours "
        f"{math.floor((execution_time%3600)/60)} minutes"
        f" and {execution_time%60:.1f} seconds"
    )