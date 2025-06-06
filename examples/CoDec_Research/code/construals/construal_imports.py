
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

# |GPUDrive imports
from gpudrive.env.env_torch import GPUDriveConstrualEnv

# |Library imports
from numpy import dot
from numpy.linalg import norm