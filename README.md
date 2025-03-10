# Time-Varying Gaussian Mixture Model (tvGMM)

This repository provides an implementation of a Time-Varying Gaussian Mixture Model (tvGMM) for analyzing single-molecule FRET (smFRET) data. The model is built using Expectation-Maximization (EM) for parameter estimation and supports prior knowledge integration.

## Requirements

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```


## Usage Example

### 1. Import Required Modules

```python
import numpy as np
from tvGMM2 import tvGMM
from EM_K_tools import load_synthetic_exeperiment
```

### 2. Generate Synthetic Data

```python
# Load synthetic experiment data
W_true, MeansStds_true, K_true, e_data, t_data, z_data, p, samples, c = load_synthetic_exeperiment()
```

### 3. Initialize Model Parameters

```python
# Define initial means and standard deviations
initial_means = np.array([[0.1], [0.3], [0.5], [0.7]])
initial_stds = np.array([[0.05], [0.05], [0.05], [0.05]])

# Define model settings
dimt = 5  # Dimensionality of basis time vector
K = 4  # Number of states (mixture components)
method = 'Splines'  # Basis time vector method
path_to_save = 'TestBIC/'  # Directory for saving results
```

### 4. Define Prior Knowledge (Optional)

```python
# Set the strength of prior knowledge
strength = 'Very Strong'
if strength == 'Very Strong':
    power = 1.5
elif strength == 'Strong':
    power = 1
elif strength == 'Weak':
    power = 0.5
elif strength == 'Very Weak':
    power = 0.0

# Define priors for means and standard deviations
prior_mu1 = [2, 0.25, samples**(power)]
prior_std1 = [2, 0.02, samples**(power)]
prior_knowledge = [[prior_mu1], [prior_std1]]
```

### 5. Initialize and Run the Model

```python
# Initialize the tvGMM model
model = tvGMM(e_data, t_data, K, dimt, method, initial_means, initial_stds)

# Set up synthetic data parameters
model.setup_synthetic(MeansStds_true, K_true)

# Enable prior knowledge (optional)
model.enable_prior(prior_knowledge)

# Run the Expectation-Maximization (EM) algorithm
W_est, means_est, stds_est = model.EM_algorithm(max_iters=300, path=path_to_save)
```

### 6. Output Results

After running the EM algorithm, the estimated parameters (weights `W_est`, means `means_est`, and standard deviations `stds_est`) will be stored in `path_to_save` for further analysis.

## Directory Structure

```
├── tvGMM2.py                 # tvGMM implementation
├── EM_K_tools.py             # Helper functions for data processing
├── main2.py                  # Example script to run the model
├── requirements.txt           # Required dependencies
├── TestBIC/                   # Output directory for results
│   ├── ParametricEstimations/  # Estimated parameters
│   ├── Norms/                  # Convergence metrics
│   ├── GroundTruth/            # Ground truth data visualization
```

## References

If you use this model in your research, please cite the relevant literature on time-varying Gaussian mixture models for smFRET data analysis.

## License

This project is open-source and licensed under the MIT License.
