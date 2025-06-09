from ..Analysis import tvGMM
from ..EM_K_tools import *
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ───────────────────────────────────────────────────────────────────
# 1.  Basic settings
# ───────────────────────────────────────────────────────────────────
SEED                = 1
RESULTS_DIR         = os.path.join(ROOT_DIR, "results")

# synthetic-data generator
SAMPLES             = 1000
CONTAMINATED        = True
CONT_PERCENTAGE     = 0.20

# initial parameter guesses
initial_means       = [0.10, 0.30, 0.50, 0.70]
initial_stds        = [0.05] * 4

# prior guesses
prior_means         = [0.15, 0.25, 0.40, 0.60]
prior_stds          = [0.01, 0.02, 0.03, 0.04]

# model / EM hyper-parameters
dimt                = 5
approx_method       = "Splines"
p_mu, p_std         = 0.5, 0.5          # prior-strength exponents
max_iters           = 300
# ───────────────────────────────────────────────────────────────────

np.random.seed(SEED)

# Generate synthetic data
W_true, MeansStds_true, K_true, e_data, t_data, z_data, p, samples, c = (
    load_synthetic_exeperiment(
        approx_method,
        contamenated = CONTAMINATED,
        cont_percentage = CONT_PERCENTAGE,
        samples = SAMPLES
    )
)

SaveGroundTruthParameters(
    RESULTS_DIR, t_data, e_data, z_data,
    W_true, c, contaminated = CONTAMINATED
)

# Initialise tvGMM parameters
InitialMeans, InitialSTDs, K = initialize_means_stds(initial_means, initial_stds)

# Prior-strength coefficients
l_mu  = get_prior_strength(p_mu,  samples)
l_std = get_prior_strength(p_std, samples)

# Build prior lists
PriorMu  = [[k + 1, mu,  l_mu ] for k, mu  in enumerate(prior_means)]
PriorStd = [[k + 1, std, l_std] for k, std in enumerate(prior_stds)]
PriorKnowledge = [PriorMu, PriorStd]

# Initialise time-varying Gaussian Mixture Model
model = tvGMM(e_data, t_data, K, dimt, approx_method,
              InitialMeans, InitialSTDs)

model.setup_synthetic(MeansStds_true, K_true)

# model.enable_prior(PriorKnowledge)      # uncomment to use the priors

# Run EM algorithm
W_est, Means_est, STDs_est = model.EM_algorithm(max_iters = max_iters,
                                                path      = RESULTS_DIR)

print('Material is saved in {}.'.format(RESULTS_DIR))
