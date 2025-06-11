from Analysis import tvGMM
from EM_K_tools import *

# ───────────────────────────────────────────────────────────────────
# 1.  Basic settings
# ───────────────────────────────────────────────────────────────────
RESULTS_DIR         = "./results_MBP_TF_v2/"
DATA_FILE           = "MBP_Tf_v2"

# Read real data and save ground truth
t_data, e_data = read_real_data_xlsx("data/folding_data.xlsx",
                                     sheet_name = DATA_FILE)
samples = len(t_data)

# initial parameter guesses
initial_means       = [0.65, 0.80, 0.5]
initial_stds        = [0.06, 0.08, 0.06]

# model / EM hyper-parameters
dimt                = 5
approx_method       = "Splines"

PriorMu1 = [1, 0.65, samples**(1.5)]         # prior knowledge on 1st state
PriorStd1 = [1, 0.06, samples**(1.5)]

PriorMu2 = [2, 0.8, 2+1e-5]                  # prior knowledge on 2nd state
PriorStd2 = [2, 0.08, samples**(1)]            

PriorMu3 = [3, 0.5, samples**(1)]            # prior knowledge on 3rd state
PriorStd3 = [3, 0.06, samples**(1)]

PriorKnowledge = [[PriorMu1, PriorMu2, PriorMu3], [PriorStd1, PriorStd2, PriorStd3]]
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