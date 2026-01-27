"""
Result Classes for Optimization Benchmarking.

Extended from the original BayesianBenchmark result_classes.py to support
multistart optimization results alongside Bayesian inference results.

Provides unified interface for:
- Loading and processing results from any method
- Comparing different optimization/inference methods
- Statistical analysis of convergence
- Visualization utilities
"""

import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from dataclasses import dataclass


class Result:
    """
    Universal result container for both Bayesian and optimization methods.
    
    Automatically detects the method type and provides appropriate
    accessor methods for analysis.
    """
    
    def __init__(self, result_dict: Dict) -> None:
        # Initialize sensible defaults to avoid shared mutable state and
        # make attributes explicit for static type checkers.
        self.method: str = ""
        self.algo_specific_info: Dict[str, Any] = {}
        self.posterior_llhs: Optional[np.ndarray] = None
        self.posterior_weights: Optional[np.ndarray] = None
        self.all_llhs: Optional[np.ndarray] = None
        self.all_samples: Optional[np.ndarray] = None
        self.n_chains: int = 1
        self.n_fun_calls: int = 0
        self.posterior_samples: Optional[np.ndarray] = None
        self.n_ensemble: int = 0
        self.converged: bool = False
        self.n_iter: int = 0

        # Load provided attributes from dictionary (overwrite defaults)
        for key, value in result_dict.items():
            setattr(self, key, value)

        # Determine method type
        self._is_bayesian = self.method in ["ptmcmc", "smc", "pmc"]
        self._is_multistart = self.method.startswith("multistart_")
        self._is_global = self.method in ["differential_evolution", "dual_annealing", "basinhopping"]

        # Set convergence flag if not explicitly provided
        if not any(k == 'converged' for k in result_dict.keys()):
            if self._is_bayesian and self.method != "ptmcmc":
                self.converged = True
            elif not self._is_bayesian:
                self.converged = True
        
        # Handle PT-MCMC specific convergence tracking
        if self.method == "ptmcmc" and self.converged:
            burn_in_idx = self.algo_specific_info.get("burn_in_idx", 0)
            n_chains = getattr(self, 'n_chains', 1)
            self.n_fun_calls = (burn_in_idx + 1) * n_chains
        
        # Ensure weights exist
        if self.posterior_weights is None:
            n = len(self.posterior_llhs) if self.posterior_llhs is not None else 1
            self.posterior_weights = np.array([1.0 / n] * n)
    
    @property
    def method_type(self) -> str:
        """Return the type of method: 'bayesian', 'multistart', or 'global'."""
        if self._is_bayesian:
            return "bayesian"
        elif self._is_multistart:
            return "multistart"
        else:
            return "global"
    
    def get_sampling_ratio(self, par_bounds: List[Tuple], par_idx: int = 0) -> float:
        """
        Measure the ratio of sampling space explored for a given parameter.
        
        Only meaningful for Bayesian methods with full trace.
        """
        if self.all_samples is None:
            return np.nan
        
        bound_diff = par_bounds[par_idx][1] - par_bounds[par_idx][0]
        par_samples = self.all_samples[:, :, par_idx]
        max_val = np.max(par_samples)
        min_val = np.min(par_samples)
        sample_diff = max_val - min_val
        return sample_diff / bound_diff
    
    def get_convergence(self, llh_threshold: float) -> float:
        """
        Get the number of function calls to reach likelihood threshold.
        
        Returns np.nan if threshold was never reached.
        """
        if not self.converged:
            return np.nan
        
        try:
            if self.all_llhs is not None:
                idxs = np.where(self.all_llhs > llh_threshold)
                if len(idxs[0]) == 0:
                    return np.nan
                first_iter = np.min(idxs[0])

                if self.method == "ptmcmc":
                    return float((int(first_iter) + 1) * int(self.n_chains))
                elif 'calls_by_iter' in self.algo_specific_info:
                    return float(self.algo_specific_info['calls_by_iter'][int(first_iter)])
                else:
                    # For multistart, estimate from first iteration reaching threshold
                    denom = float(len(self.all_llhs.flatten()))
                    return float(first_iter) * float(self.n_fun_calls) / denom if denom > 0.0 else float(np.nan)
        except (ValueError, IndexError):
            pass
        
        return np.nan
    
    def get_init_best_llh(self) -> float:
        """Get the best log-likelihood from initial samples."""
        if self.all_llhs is None:
            if self.posterior_llhs is not None:
                return float(np.min(self.posterior_llhs))  # Conservative for optimization
            return -np.inf
        
        if self.method != "ptmcmc":
            iter0 = self.all_llhs[0, :]
        else:
            # For PT-MCMC, look at first ~250 iterations across chains
            assert self.all_llhs is not None
            iter0 = self.all_llhs[:min(250, len(self.all_llhs)), :]
        
        return float(np.amax(iter0))
    
    def get_max_llh(self) -> float:
        """Get the maximum log-likelihood found."""
        if self.converged and self.posterior_llhs is not None:
            return float(np.max(self.posterior_llhs))
        elif self.all_llhs is not None:
            return float(np.amax(self.all_llhs))
        elif 'best_fval' in self.algo_specific_info:
            return float(-self.algo_specific_info['best_fval'])
        return -np.inf
    
    def get_best_parameters(self) -> Optional[np.ndarray]:
        """Get the best parameter set found."""
        if 'best_x' in self.algo_specific_info:
            return np.array(self.algo_specific_info['best_x'])
        if self.posterior_samples is not None and self.posterior_llhs is not None:
            best_idx = int(np.argmax(self.posterior_llhs))
            return self.posterior_samples[best_idx]
        return None
    
    # Multistart-specific methods
    def get_success_rate(self) -> float:
        """Get optimization success rate (multistart only)."""
        if 'success_rate' in self.algo_specific_info:
            return self.algo_specific_info['success_rate']
        return 1.0 if self.converged else 0.0
    
    def get_convergence_stats(self) -> Dict:
        """Get convergence statistics (multistart only)."""
        if 'convergence_distribution' in self.algo_specific_info:
            return self.algo_specific_info['convergence_distribution']
        return {}
    
    def get_all_optima(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all optima found (multistart only)."""
        if self.posterior_samples is not None:
            return self.posterior_samples, (self.posterior_llhs if self.posterior_llhs is not None else np.array([]))
        return np.array([]), np.array([])


class MethodResults:
    """
    Container for results from multiple runs of the same method.
    
    Provides aggregation and comparison functionality across runs.
    """
    
    METHOD_INFO = {
        "pmc": ("PMC", "Preconditioned Monte Carlo"),
        "smc": ("SMC", "Sequential Monte Carlo"),
        "ptmcmc": ("PT-MCMC", "Parallel Tempering MCMC"),
        "multistart_lbfgsb": ("MS-LBFGSB", "Multistart L-BFGS-B"),
        "multistart_nm": ("MS-NM", "Multistart Nelder-Mead"),
        "multistart_powell": ("MS-Powell", "Multistart Powell"),
        "multistart_trust": ("MS-Trust", "Multistart Trust-Region"),
        "differential_evolution": ("DE", "Differential Evolution"),
        "dual_annealing": ("DA", "Dual Annealing"),
        "basinhopping": ("BH", "Basin Hopping"),
    }
    
    def __init__(self, method: str) -> None:
        self.all_runs: List[Result] = []
        self.method = method
        
        # Get display names
        if method in self.METHOD_INFO:
            self.abbr, self.label = self.METHOD_INFO[method]
        elif method.startswith("multistart_"):
            base = method.replace("multistart_", "")
            self.abbr = f"MS-{base.upper()}"
            self.label = f"Multistart {base.upper()}"
        else:
            self.abbr = method.upper()
            self.label = method
    
    def add_result(self, result_obj: Result) -> None:
        """Add a result to the collection."""
        self.all_runs.append(result_obj)
    
    @property
    def n_runs(self) -> int:
        return len(self.all_runs)
    
    def get_fun_calls(self) -> np.ndarray:
        """Get array of function call counts for all runs."""
        return np.array([
            x.n_fun_calls if x.converged else np.nan 
            for x in self.all_runs
        ])
    
    def get_llhs(self) -> np.ndarray:
        """Get all posterior log-likelihoods from all runs."""
        all_llhs = []
        for x in self.all_runs:
            if x.converged and x.posterior_llhs is not None:
                all_llhs.append(x.posterior_llhs)
            else:
                # Return placeholder for non-converged
                n = getattr(x, 'n_ensemble', 100)
                all_llhs.append(np.full(n, -np.inf))
        return np.array(all_llhs, dtype=object)
    
    def get_avg_llhs(self) -> np.ndarray:
        """Get weighted average log-likelihood for each run."""
        avgs = []
        for x in self.all_runs:
            if x.converged and x.posterior_llhs is not None:
                avg = np.average(x.posterior_llhs, weights=x.posterior_weights)
            else:
                avg = -np.inf
            avgs.append(avg)
        return np.array(avgs)
    
    def get_max_llhs(self) -> np.ndarray:
        """Get maximum log-likelihood for each run."""
        return np.array([x.get_max_llh() for x in self.all_runs])
    
    def get_sampling_efficiency(self, bounds: List[Tuple], par_idx: int) -> np.ndarray:
        """Get sampling efficiency for each run."""
        return np.array([
            x.get_sampling_ratio(bounds, par_idx) if x.converged else np.nan
            for x in self.all_runs
        ])
    
    def get_convergence_times(self, llh_threshold: float) -> np.ndarray:
        """Get function calls to reach threshold for each run."""
        return np.array([x.get_convergence(llh_threshold) for x in self.all_runs])
    
    def get_best_inits(self) -> np.ndarray:
        """Get best initial log-likelihood for each run."""
        return np.array([x.get_init_best_llh() for x in self.all_runs])
    
    def get_success_rates(self) -> np.ndarray:
        """Get success rates for each run (multistart methods)."""
        return np.array([x.get_success_rate() for x in self.all_runs])
    
    def get_best_run(self) -> Result:
        """Get the run with the best log-likelihood."""
        max_llhs = self.get_max_llhs()
        best_idx = np.argmax(max_llhs)
        return self.all_runs[best_idx]
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all runs."""
        max_llhs = self.get_max_llhs()
        fun_calls = self.get_fun_calls()
        
        valid_llhs = max_llhs[np.isfinite(max_llhs)]
        valid_calls = fun_calls[np.isfinite(fun_calls)]
        
        return {
            'method': self.method,
            'n_runs': self.n_runs,
            'n_converged': np.sum([r.converged for r in self.all_runs]),
            'best_llh': np.max(valid_llhs) if len(valid_llhs) > 0 else -np.inf,
            'mean_llh': np.mean(valid_llhs) if len(valid_llhs) > 0 else -np.inf,
            'std_llh': np.std(valid_llhs) if len(valid_llhs) > 0 else np.nan,
            'median_llh': np.median(valid_llhs) if len(valid_llhs) > 0 else -np.inf,
            'mean_fun_calls': np.mean(valid_calls) if len(valid_calls) > 0 else np.nan,
            'std_fun_calls': np.std(valid_calls) if len(valid_calls) > 0 else np.nan,
        }
    
    # KS test for comparing posterior distributions
    def ks_weighted(self, data1, data2, wei1, wei2, alternative='two-sided'):
        """
        Weighted Kolmogorov-Smirnov test.
        
        Source: https://stackoverflow.com/questions/40044375/
        """
        ix1 = np.argsort(data1)
        ix2 = np.argsort(data2)
        data1 = data1[ix1]
        data2 = data2[ix2]
        wei1 = wei1[ix1]
        wei2 = wei2[ix2]
        data = np.concatenate([data1, data2])
        cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
        cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
        cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
        cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
        d = np.max(np.abs(cdf1we - cdf2we))
        
        # Calculate p-value
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = stats.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            expt = -2 * z**2 - 2 * z * (m + 2*n) / np.sqrt(m*n*(m+n)) / 3.0
            prob = np.exp(expt)
        return d, prob
    
    def calc_pairwise_matrix(self, par_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pairwise KS statistics for a parameter."""
        n_runs = len(self.all_runs)
        combos = itertools.combinations(range(n_runs), 2)
        ks_matrix = np.zeros(shape=(n_runs, n_runs))
        pval_matrix = np.zeros(shape=(n_runs, n_runs))
        
        for i, j in combos:
            runA = self.all_runs[i]
            runB = self.all_runs[j]
            
            if not (runA.converged and runB.converged):
                ks_matrix[i, j] = np.nan
                ks_matrix[j, i] = np.nan
                continue

            # Ensure posterior samples/weights exist
            if runA.posterior_samples is None or runB.posterior_samples is None:
                ks_matrix[i, j] = np.nan
                ks_matrix[j, i] = np.nan
                continue
            if runA.posterior_weights is None or runB.posterior_weights is None:
                ks_matrix[i, j] = np.nan
                ks_matrix[j, i] = np.nan
                continue

            # Narrow types for static checkers
            assert runA.posterior_samples is not None and runB.posterior_samples is not None
            assert runA.posterior_weights is not None and runB.posterior_weights is not None

            param_samplesA = runA.posterior_samples[:, par_index]
            param_samplesB = runB.posterior_samples[:, par_index]

            ks_stat, pval = self.ks_weighted(
                param_samplesA, param_samplesB,
                runA.posterior_weights, runB.posterior_weights
            )
            ks_matrix[j, i] = ks_stat
            ks_matrix[i, j] = ks_stat
            pval_matrix[j, i] = pval
            pval_matrix[i, j] = pval
        
        return ks_matrix, pval_matrix


class BenchmarkComparison:
    """
    Compare multiple methods on the same problem.
    
    Provides tools for analyzing and visualizing benchmark results.
    """
    
    def __init__(self, problem_name: str):
        self.problem_name = problem_name
        self.method_results: Dict[str, MethodResults] = {}
    
    def add_method(self, method_name: str, results: MethodResults) -> None:
        """Add results for a method."""
        self.method_results[method_name] = results
    
    def get_comparison_table(self) -> Dict[str, Dict]:
        """Get comparison table of all methods."""
        table = {}
        for method, results in self.method_results.items():
            table[method] = results.get_summary_stats()
        return table
    
    def rank_by_metric(self, metric: str = 'best_llh', ascending: bool = False) -> List[Tuple[str, float]]:
        """Rank methods by a metric."""
        table = self.get_comparison_table()
        ranked = [(m, table[m].get(metric, -np.inf)) for m in table]
        return sorted(ranked, key=lambda x: x[1], reverse=not ascending)
    
    def get_efficiency_comparison(self) -> Dict[str, Dict]:
        """
        Compare efficiency: likelihood achieved per function call.
        """
        efficiency = {}
        for method, results in self.method_results.items():
            stats = results.get_summary_stats()
            if stats['mean_fun_calls'] > 0:
                # Higher is better: likelihood improvement per call
                eff = stats['best_llh'] / np.log10(stats['mean_fun_calls'])
            else:
                eff = np.nan
            efficiency[method] = {
                'best_llh': stats['best_llh'],
                'mean_calls': stats['mean_fun_calls'],
                'efficiency': eff
            }
        return efficiency


def load_results(filepath: str) -> Result:
    """Load results from a pickle file (supports gzip)."""
    import gzip
    import pickle
    
    try:
        with gzip.open(filepath, "rb") as f:
            result_dict = pickle.load(f)
    except gzip.BadGzipFile:
        with open(filepath, "rb") as f:
            result_dict = pickle.load(f)
    
    return Result(result_dict)


def load_method_results(result_dir: str, method: str) -> MethodResults:
    """Load all results for a method from a directory."""
    import glob
    
    method_results = MethodResults(method)
    pattern = f"{result_dir}/*_{method}_*.pkl"
    fnames = glob.glob(pattern)
    
    for fname in fnames:
        try:
            result = load_results(fname)
            method_results.add_result(result)
        except Exception as e:
            print(f"Warning: Could not load {fname}: {e}")
    
    return method_results