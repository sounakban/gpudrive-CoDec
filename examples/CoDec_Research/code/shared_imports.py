"""
    Reduce code redundancy
    Usage: Start script with 'from shared_imports import *'
"""

from copy import deepcopy
from functools import cache
from os import listdir
import json
import pickle
import gc
from datetime import datetime
from functools import partial

from scipy.special import softmax
import numpy as np
import math
from itertools import combinations

from typing import Any, List, Tuple, Dict
import time

import torch
import dataclasses
from tqdm import tqdm
import pandas as pd


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


# |GPUDrive imports
from gpudrive.utils.config import load_config

# |CoDec imports
from examples.CoDec_Research.code.gpuDrive_utils import *

# # |Lower level imports
# from examples.CoDec_Research.code.simulation.construal_main import *
# from examples.CoDec_Research.code.analysis.evaluate_construal_actions import *

# Function to extract filename from path
env_path2name = lambda path: path.split("/")[-1].split(".")[0]