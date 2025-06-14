CoDec Research
==============

This folder contains research-related code for the NYU-TRI collaboration project on machine-teaching in human driving.


## Directory Structure

The root contains two direcoties:
1. **code**: Contains all the research code
2. **results**: Contains the results of various analyses. This directory also stores temporary files generated by the code (but are ignored during git push).



### code

This derectory cotains all research-related code, organized into directories by the functions they perform. It also contains [shared_imports](./code/shared_imports.py) and [shared_config](./code/shared_config.py) files which contain all library imports and configurations respectively, commonly used by all scripts in the research codebase.

Directories begining with 'Experiment_' contain the highest level of code which calls functions from other scripts inside the directory.

For example, [Experiment_1](./code/Experiment_1/) containtains the code for the research described in the 2025 NerIPS submission, which include the following files:
1. exp1_config: Experiment 1 specific configurations (in addition to shared_config).
2. exp1_imports: Experiment 1 specific imports (in addition to shared_imports).
3. exp1_P1_GenMasks: Experiment 1 pipeline part 1. Python script responsible for generating masks over vehicles of interest, used to define construed perceptial spaces in the simulator. 
4. exp1_P2_GetVals: Experiment 1 pipeline part 2. Python script responsible for computing the behavioral utilities of different construals through simulations.
5. exp1_P3_GenData: Experiment 1 pipeline part 3. Python script responsible for generating synthetic data treated as ground truth by the inference algorithm.
6. exp1_P4_Inference: Experiment 1 pipeline part 4. Python script responsible for performing inference over generated synthetic data.
7. fullPipeline: Python script that calls the four other python scripts (listed above) to execute the whole pipeline, iterating over multiple values of lambda.


### results

This directory contains final analysis resuts which are submitted for publication ([analysis_results](./results/analysis_results/)). The [simulation_results](./results/simulation_results/) folder is used for storing teporary files generated by the code during runtime, and is included in the gitignore list.