<!-- README badges -->
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-0C55A5?logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)



# Time-Varying Gaussian Mixture Model (tvGMM)

tvGMM is a Python implementation of a time-varying Gaussian mixture model for analysing single-molecule FRET (smFRET) trajectories. It relies on an Expectation–Maximisation (EM) procedure and optionally incorporates prior knowledge about expected states.

## Features

- Synthetic and real data support
- EM algorithm with optional priors
- Utility functions to generate ground truth data and save results
- Built-in visualisation of estimated parameters

## Installation

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the examples

Two example scripts are provided. Results are written to the `results/` directory.

### Synthetic data

```bash
python -m code.examples.run_synthetic_example
```

This script generates a synthetic experiment, runs the EM algorithm and stores the plots and estimated parameters.

### Real data

```bash
python -m code.examples.run_real_example
```

`run_real_example.py` expects a CSV file containing `Time` and `FRET` columns (default `data/MB.csv`). Provide your own file or adjust `DATA_FILE` inside the script.

## Using the model in your code

```python
from code.Analysis import tvGMM
from code.EM_K_tools import load_synthetic_exeperiment

# Load example data
W_true, MeansStds_true, K_true, e_data, t_data, *_ = load_synthetic_exeperiment()

# Initial guesses
init_means = [0.10, 0.30, 0.50, 0.70]
init_stds = [0.05] * 4

# Build and run the model
model = tvGMM(e_data, t_data, K_true, dimt=5, method='Splines',
              initial_means=init_means, initial_stds=init_stds)
model.setup_synthetic(MeansStds_true, K_true)
W_est, means_est, stds_est = model.EM_algorithm(max_iters=300, path='results/')
```

## Repository layout

```
├── code/
│   ├── Analysis.py               # tvGMM implementation
│   ├── EM_K_tools.py             # Helper functions
│   └── examples/
│       ├── run_real_example.py       # Example using real data
│       └── run_synthetic_example.py  # Example using synthetic data
├── requirements.txt          # Required Python packages
├── data/                     # Place your CSV file here
└── results/                  # Output files will appear here
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## References

If you use tvGMM in academic work, please cite relevant papers on time-varying GMMs for smFRET analysis.
