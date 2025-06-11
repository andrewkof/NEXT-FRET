from code.Analysis import tvGMM
from code.EM_K_tools import *

# ───────────────────────────────────────────────────────────────────
# 1.  Basic settings
# ───────────────────────────────────────────────────────────────────
RESULTS_DIR         = "./results_MBP/"
DATA_FILE           = "MBP"

# Read real data and save ground truth
t_data, e_data = read_real_data_xlsx("data/folding_data.xlsx",
                                     sheet_name = DATA_FILE)
samples = len(t_data)

# initial parameter guesses
initial_means       = [0.63, 0.81]
initial_stds        = [0.06, 0.07]

# model / EM hyper-parameters
dimt                = 5
approx_method       = "Splines"

PriorMu1  = [1, 0.63, samples**(1.5)]
PriorStd1 = [1, 0.06, samples**(1.5)]          # prior knowledge on 1st state

PriorMu2  = [2, 0.81, 2+1e-5]
PriorStd2 = [2, 0.07, samples**(1.0)]          # prior knowledge on 2nd state

PriorKnowledge = [[PriorMu1, PriorMu2], [PriorStd1, PriorStd2]]
max_iters      = 300
# ───────────────────────────────────────────────────────────────────

SaveGroundTruthParametersREAL(RESULTS_DIR, t_data, e_data)

# Initialise tvGMM parameters
InitialMeans, InitialSTDs, K = initialize_means_stds(initial_means, initial_stds)

# Initialise time-varying Gaussian Mixture Model
model = tvGMM(e_data, t_data, K, dimt, approx_method,
              InitialMeans, InitialSTDs)

model.enable_prior(PriorKnowledge)      # uncomment to use the priors

# Run EM algorithm
W_est, Means_est, STDs_est = model.EM_algorithm(max_iters = max_iters,
                                                path      = RESULTS_DIR)

print("Material is saved in {}.".format(RESULTS_DIR))