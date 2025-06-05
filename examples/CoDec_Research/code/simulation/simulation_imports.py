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


# |Standard library imports
import mediapy
from sympy import parallel_poly_from_expr
from torch.distributions.utils import logits_to_probs

from itertools import accumulate


# |GPUDrive imports
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv, GPUDriveConstrualEnv
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.utils.config import load_config



# |Same-level imports
from examples.CoDec_Research.code.construals.construal_functions import *


