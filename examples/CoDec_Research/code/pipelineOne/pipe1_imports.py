"""
    Reduce code redundancy
    Usage: Start script with 'from shared_imports import *'
"""

# # |Set root for GPUDrive import
# import os
# import sys
# from pathlib import Path

# # Set working directory to the base directory 'gpudrive'
# working_dir = Path.cwd()
# while working_dir.name != 'gpudrive-CoDec':
#     working_dir = working_dir.parent
#     if working_dir == Path.home():
#         raise FileNotFoundError("Base directory 'gpudrive' not found")
# os.chdir(working_dir)
# sys.path.append(str(working_dir))


# |Shared Imports
from examples.CoDec_Research.code.shared_imports import *


# |Lower level imports
from examples.CoDec_Research.code.analysis.evaluate_construal_actions import *
from examples.CoDec_Research.code.simulation.construal_main import *




####################################################
################ SET EXP PARAMETERS ################
####################################################

##### Done in Shared Import #####