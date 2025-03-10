import warnings
import numpy as np
from EM_K_tools import *
warnings.filterwarnings("ignore")
np.random.seed(42)
from Analysis import tvGMM

# Initilize estimated parameters:
m1k, m2k, m3k, m4k, m5k = 0.1, 0.3, 0.5, 0.7, 0.9
std1k, std2k, std3k, std4k, std5k = 0.05, 0.05, 0.05, 0.05, 0.05
InitialMeans = np.array([[m1k], [m2k], [m3k], [m4k]])
InitialSTDs = np.array([[std1k], [std2k], [std3k], [std4k]])
dimt = 5
K = 4
method = 'Splines'
PathToSave = 'TestBIC/'


# We generate data (t_data, e_data) for a given set of parameters (matrix W_true, Means_Stds_true).
# The goal is given this data, to find W_est, means_est and stds_est that these data points were generated.

W_true, MeansStds_true, K_true, e_data, t_data, z_data, p, samples, c = load_synthetic_exeperiment()
SaveGroundTruthParameters(PathToSave, t_data, e_data, z_data, c, W_true)
SaveSyntheticProbabilities(PathToSave, t_data, p, c)

# Initilize prior parameters:
Strength = 'Very Strong'
if Strength == 'Very Strong':
    power = 1.5
elif Strength == 'Strong':
    power = 1
elif Strength == 'Weak':
    power = 0.5
elif Strength == 'Very Weak':
    power = 0.0

PriorMu1 = [2, 0.25, samples**(power)]
PriorStd1 = [2, 0.02, samples**(power)]

# PriorMu2 = [3, 0.4, samples**(power)]
# PriorStd2 = [3, 0.03, samples**(power)]
PriorKnowledge = [[PriorMu1], [PriorStd1]]

# PriorKnowledge = [[PriorMu1, PriorMu2], [PriorStd1, PriorStd2]]

# Run our framework
# Initilize time-varying Gaussian Mixture Model:
model = tvGMM(e_data, t_data, K, dimt, method, InitialMeans, InitialSTDs)
model.setup_synthetic(MeansStds_true, K_true)
# Use Prior Knowledge (if you don't want to use, comment the below command):
model.enable_prior(PriorKnowledge)
# Run the EM Alogirthm:
W_est, Means_est, STDs_est  = model.EM_algorithm(max_iters=300, path=PathToSave)