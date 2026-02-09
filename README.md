# BayesianBenchmark

A comprehensive framework for benchmarking and comparing Bayesian inference methods and optimization algorithms for parameter estimation in systems biology models.

## Overview

**BayesianBenchmark** provides tools to systematically evaluate and compare different statistical inference approaches on realistic parameter estimation problems defined in PEtab format. The framework measures both solution quality (log-likelihood achieved) and computational efficiency (function calls required).

## Features

### Supported Methods

**Bayesian Inference:**
- **PT-MCMC** (Parallel Tempering MCMC) via pypesto
- **SMC** (Sequential Monte Carlo) via pocomc
- **PMC** (Preconditioned Monte Carlo) via pocomc

**Multistart Local Optimization:**
- L-BFGS-B
- Nelder-Mead
- Powell
- Trust-Region Constrained

**Global Optimization:**
- Differential Evolution
- Dual Annealing
- Basin Hopping

### Key Capabilities

- **Unified Result Framework**: Universal container for results from any method with automatic type detection
- **Fair Comparison**: Consistent function call tracking and controlled threading
- **Statistical Analysis**: KS tests, pairwise posterior comparisons, convergence metrics
- **Visualization Tools**: Method comparison plots, convergence analysis, parameter recovery visualization
- **HPC Support**: SLURM job submission scripts for batch processing

## Installation

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate bayes
```

### Core Dependencies

- Python 3.12.4
- numpy, scipy, pandas, matplotlib, seaborn
- pypesto (PT-MCMC), pocomc (SMC/PMC)
- petab, python-libsbml, libroadrunner (PEtab/SBML support)

## Usage

### Basic Execution

The main script to generate results is [src/run_model_calibration.py](src/run_model_calibration.py).

```bash
# Bayesian inference
python src/run_model_calibration.py -m ptmcmc -p Hopf -n 1000 -s 42 -o results/ -c 4

# Multistart optimization
python src/run_model_calibration.py -m multistart_lbfgsb -p Hopf -n 100 -s 42 -o results/

# Global optimization
python src/run_model_calibration.py -m differential_evolution -p Michaelis_Menten -s 1 -o results/
```

**Arguments:**
- `-m, --method`: Method name (ptmcmc, smc, pmc, multistart_lbfgsb, etc.)
- `-p, --problem`: Model name (see Available Models below)
- `-n, --n_ensemble`: Ensemble size (typically 1000 for Bayesian, 50-100 for optimization)
- `-s, --seed`: Random seed for reproducibility
- `-o, --output_dir`: Results output directory
- `-c, --n_cpus`: Number of CPUs to use (default: 1)

### Standalone Demo

Run [src/demo_multistart.py](src/demo_multistart.py) for a self-contained demonstration that doesn't require PEtab/pypesto:

```bash
python src/demo_multistart.py
```

This script demonstrates multistart optimization on simple test problems (Michaelis-Menten, Rosenbrock, Rastrigin) with synthetic data.

### Batch Job Submission

For running multiple seeds and methods on an HPC cluster:

```bash
bash scripts/submit_jobs.sh
```

This submits 100 seeds across multiple methods to SLURM. See [scripts/](scripts/) for other job submission options.

## Available Models

The project includes 8 biological/biochemical models with varying complexity:

| Model | Parameters | Type |
|-------|-----------|------|
| **Hopf** | 2 | Dynamical System |
| **Michaelis_Menten** | 2 | Enzyme Kinetics |
| **Calcium_Oscillate** | - | Calcium Signaling |
| **long_Calcium_Oscillate** | - | Calcium Signaling |
| **Neg_Feed_Oscillate** | - | Gene Regulation |
| **EGFR** | - | Signal Transduction |
| **Shp2** | - | Signal Transduction |
| **linear_mRNA_self_reg** | - | Gene Regulation |

All models are located in the [models/](models/) directory. Each model subdirectory contains:
- PEtab YAML configuration
- SBML model definition
- Parameter definitions
- Experimental/simulated data
- Observable definitions
- Jupyter notebooks for data generation and visualization

## Analysis

### Result Analysis

Results are stored as pickled dictionaries in the output directory. Use the provided analysis tools:

```python
from result_classes import Result, MethodResults, BenchmarkComparison

# Load single result
result = Result.from_file('results/Hopf_multistart_lbfgsb_42.pkl')

# Load and compare methods
method_results = MethodResults.from_directory('results/Hopf/')
comparison = BenchmarkComparison([method_results1, method_results2])
```

### Visualization

Interactive analysis notebooks are in [notebooks/](notebooks/):
- [notebooks/compare_methods.ipynb](notebooks/compare_methods.ipynb) - Method comparison analysis
- [notebooks/vis_KS_stats.ipynb](notebooks/vis_KS_stats.ipynb) - KS statistics visualization

Use the visualization module (from project root):

```python
import sys
sys.path.insert(0, 'src')
from visualization import BenchmarkVisualizer

visualizer = BenchmarkVisualizer(comparison)
visualizer.plot_comparison()
visualizer.plot_convergence()
```

## Project Structure

```
BayesianBenchmark/
├── src/                         # Core source code
│   ├── Core Framework
│   │   ├── modelproblem.py          # Model loading and interface
│   │   ├── bayesianinference.py     # Abstract sampler interface
│   │   ├── result_classes.py        # Universal result containers
│   │   └── visualization.py         # Plotting utilities
│   │
│   ├── Samplers/Optimizers
│   │   ├── pestosampler.py          # PT-MCMC implementation
│   │   ├── pocosampler.py           # SMC/PMC implementation
│   │   └── multistart_sampler.py    # Multistart + global optimizers
│   │
│   ├── Execution Scripts
│   │   ├── run_model_calibration.py # Main CLI entry point
│   │   └── demo_multistart.py       # Standalone demo
│   │
│   └── Analysis Utilities
│       ├── pairwise_ks.py          # Statistical comparisons
│       ├── compress_results.py     # Result compression
│       └── weighted_quantile.py    # Quantile utilities
│
├── models/                      # Benchmark models (PEtab format)
│   ├── Hopf/
│   ├── Michaelis_Menten/
│   ├── Calcium_Oscillate/
│   ├── long_Calcium_Oscillate/
│   ├── Neg_Feed_Oscillate/
│   ├── EGFR/
│   ├── Shp2/
│   └── linear_mRNA_self_reg/
│
├── notebooks/                   # Analysis notebooks
│   ├── compare_methods.ipynb    # Method comparison
│   └── vis_KS_stats.ipynb       # KS statistics
│
├── scripts/                     # HPC job submission scripts
│   ├── submit_jobs.sh           # Standard job submission
│   ├── submit_bigmem_jobs.sh    # Big memory jobs
│   ├── submit_ptmcmc_jobs.sh    # PT-MCMC specific
│   ├── single_slurm_job.sh      # Single job template
│   ├── bigmem_slurm_job.sh      # Big memory template
│   └── ptmcmc_slurm_job.sh      # PT-MCMC template
│
├── environment.yml              # Conda environment specification
└── README.md                    # This file
```

## Workflow

1. **Setup**: Choose or add a model in PEtab format (in [models/](models/))
2. **Run**: Execute inference/optimization using [src/run_model_calibration.py](src/run_model_calibration.py)
3. **Collect**: Results saved as pickled dictionaries in `results/`
4. **Analyze**: Load results using classes in [src/result_classes.py](src/result_classes.py)
5. **Visualize**: Generate plots using [notebooks/](notebooks/) or [src/visualization.py](src/visualization.py)

## Recent Updates

- Fixed Hopf model with PEtab file edits and SBML export improvements
- Added multistart optimization support with multiple local/global algorithms
- Enhanced visualization with max log-likelihood plots
- Improved method comparison capabilities

## Contributing

To add a new model:
1. Create a directory under [models/](models/)
2. Add PEtab specification files (YAML, SBML, TSV files)
3. The model will be automatically available to [src/run_model_calibration.py](src/run_model_calibration.py)

To add a new inference method:
1. Implement the `BayesianInference` interface in [src/bayesianinference.py](src/bayesianinference.py)
2. Add method registration in [src/run_model_calibration.py](src/run_model_calibration.py)
3. Ensure results follow the standard format for compatibility

## Credits

### Main Contributors

**Caroline Larkin** - Original developer and primary contributor
- GitHub: [LarkinIt/BayesianBenchmark](https://github.com/LarkinIt/BayesianBenchmark)
- Developed the core framework and model implementations

**Achyudhan Kutuva** - Contributor
- GitHub: [akutuva21/BayesianBenchmark](https://github.com/akutuva21/BayesianBenchmark)
- Fork and extensions to the original framework

**Current Repository**
- Forked from Achyudhan Kutuva's repository
- Ongoing development and refinements

## License

This project is part of ongoing research in computational systems biology and parameter estimation methods.
