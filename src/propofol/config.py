"""Configuration parameters for propofol dosing recommendation."""

import numpy as np

# Simulation parameters
SIM_MIN = 15
TIME = np.linspace(0.0, SIM_MIN, SIM_MIN * 60 + 1)  # 1-second steps, time in minutes
N_INTERVALS = SIM_MIN  # one maintenance decision per minute

# BIS targets
BIS_LOW = 40.0
BIS_HIGH = 60.0
BIS_TARGET = 50.0

# MAP thresholds
MAP_ABS_MIN = 65.0
MAP_REL_FRAC = 0.70

# Drug bounds
BOLUS_MGKG_BOUNDS = (0.2, 3.5)  # To-do: change to ranges in dataset
INFUSION_MGKGH_BOUNDS = (0.0, 20.0)  # To-do: change to ranges in dataset

# Optional clinical rounding for maintenance rates
# e.g. 0.5 or 1.0. Leave None for continuous rates.
MAINTENANCE_RATE_STEP = None
