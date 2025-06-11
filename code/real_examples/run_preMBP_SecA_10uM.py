from Analysis import tvGMM
from EM_K_tools import *

# ───────────────────────────────────────────────────────────────────
# 1.  Basic settings
# ───────────────────────────────────────────────────────────────────
RESULTS_DIR         = "./results_preMBP_SecA_10uM/"
DATA_FILE           = "preMBP_secA_10uM"

# Read real data and save ground truth
t_data, e_data = read_real_data_xlsx("data/folding_data.xlsx",
                                     sheet_name = DATA_FILE)
samples = len(t_data)

# initial parameter guesses
initial_means       = [0.64, 0.82, 0.42, 0.27]
initial_stds        = [0.06, 0.08, 0.06, 0.06]

# model / EM hyper-parameters
dimt                = 5
approx_method       = "Splines"


PriorMu1 = [1, 0.64, samples**(1.5)]
PriorStd1 = [1, 0.06, samples**(1.5)]

PriorMu2 = [2, 0.82, 2+1e-5]
PriorStd2 = [2, 0.08, samples**(1)]

PriorMu3 = [3, 0.42, 2+1e-5]
PriorStd3 = [3, 0.06, samples**(1)]

PriorMu4 = [4, 0.27, 2+1e-5]
PriorStd4 = [4, 0.06, samples**(1)]

PriorKnowledge = [[PriorMu1, PriorMu2, PriorMu3, PriorMu4], [PriorStd1, PriorStd2, PriorStd3, PriorStd4]]
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