<!-- README badges -->
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?logo=matplotlib&logoColor=black)
![SciPy](https://img.shields.io/badge/SciPy-0C55A5?logo=scipy&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)



# Time-Varying Gaussian Mixture Model (tvGMM)

tvGMM is a Python implementation of a time-varying Gaussian mixture model for analysing single-molecule FRET (smFRET) trajectories. It relies on an Expectation–Maximisation (EM) procedure and optionally incorporates prior knowledge about expected states.

## Installation

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running examples

Two example scripts are provided. Results are written to the `results/` directory.

### Synthetic data

```bash
python -m code.real_examples.run_MBP
```

This script runs one of the example datasets and stores the results in a `results_*` directory.  Other scripts in `code/real_examples` can be run in the same manner.

### Real data

The real examples expect the Excel workbook `data/folding_data.xlsx` with worksheets named after the different datasets.  Adjust the `DATA_FILE` variable inside each script if needed.

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
│   └── real_examples/            # Example scripts using the model
├── requirements.txt          # Required Python packages
├── data/                     # `folding_data.xlsx` workbook
└── results/                  # Output files will appear here
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## References

If you use tvGMM in academic work, please cite relevant papers on time-varying GMMs for smFRET analysis.
