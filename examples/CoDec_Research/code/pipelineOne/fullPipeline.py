import os
import time

import numpy as np
import math



start_time = time.perf_counter()


# |Loop through lambdas
sample_count = 1
lambda_heur = ["rel_heading"]
lambda_val = [np.random.uniform(-5,5,sample_count)]
# lambda_val = [[6,]]     #DEBUG: Test


lambda_val = list(zip(*lambda_val))                 # Change list structure for code compatibility
lambda_heur = [lambda_heur]*len(lambda_val)         # Change list structure for code compatibility
for type_, val_ in zip(lambda_heur,lambda_val):
        val_ = ','.join([str(i) for i in val_])   # Pass argument as string of comma-spearated values
        type_ = ','.join(type_)                     # Pass argument as string of comma-spearated values
        os.system("python examples/CoDec_Research/code/pipelineOne/exp1_P1_GenMasks.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/pipelineOne/exp1_P2_GetVals.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/pipelineOne/exp1_P3_GenData.py {} {}".format(type_, val_))
        os.system("python examples/CoDec_Research/code/pipelineOne/exp1_P4_Inference.py {} {}".format(type_, val_))


# |Print the execution time
execution_time = time.perf_counter() - start_time
print(f"Pipeline execution time: {math.floor(execution_time/3600)} hours "
        f"{math.floor((execution_time%3600)/60)} minutes"
        f" and {execution_time%60:.1f} seconds"
    )