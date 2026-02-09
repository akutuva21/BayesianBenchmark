"""
Multistart Optimization Sampler for BayesianBenchmark.

This module implements multistart local optimization as an alternative to
Bayesian inference methods (PT-MCMC, SMC, PMC) for parameter estimation.

Follows the same interface as pestoSampler and pocoSampler for easy integration.
"""

import os
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import time
import warnings


@dataclass
class OptimizationRun:
    """Result from a single optimization run."""
    x: np.ndarray  # Optimal parameters
    fun: float     # Objective value (negative log-likelihood)
    success: bool
    n_feval: int
    n_iter: int
    time: float
    message: str = ""
    start_point: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x.tolist() if isinstance(self.x, np.ndarray) else self.x,
            'fun': float(self.fun),
            'success': self.success,
            'n_feval': self.n_feval,
            'n_iter': self.n_iter,
            'time': self.time,
            'message': self.message,
            'start_point': self.start_point.tolist() if self.start_point is not None else None
        }


class MultistartSampler:
    """
    Multistart local optimization for parameter estimation.
    
    Follows the BayesianInference interface pattern from the BayesianBenchmark
    codebase for compatibility with existing infrastructure.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n_starts : int
        Number of starting points for multistart optimization
    model_problem : ModelProblem
        The model problem instance from modelproblem.py
    n_cpus : int
        Number of CPUs (for future parallel implementation)
    method : str
        Local optimizer: 'l-bfgs-b', 'nelder-mead', 'powell', 'trust-constr',
        or global: 'differential_evolution', 'dual_annealing', 'basinhopping'
    sampler : str
        Starting point sampling: 'lhs' (Latin Hypercube), 'sobol', 'uniform'
    maxiter : int
        Maximum iterations per local optimization
    """
    
    def __init__(
        self,
        seed: int,
        n_starts: int,
        model_problem,  # ModelProblem type
        n_cpus: int,
        method: str = 'l-bfgs-b',
        sampler: str = 'lhs',
        maxiter: int = 1000
    ):
        self.seed = seed
        self.n_starts = n_starts
        self.model_problem = model_problem
        self.n_cpus = n_cpus
        self.method = method
        self.sampler_type = sampler
        self.maxiter = maxiter
        
        # Will be set during initialization
        self.x0_list: List[np.ndarray] = []
        self.bounds: List[Tuple[float, float]] = []
        self.n_dim: int = 0
        
        # Results storage
        self.all_runs: List[OptimizationRun] = []
        self.best_run: Optional[OptimizationRun] = None
        
    def initialize(self):
        """Initialize the optimizer with starting points."""
        mod_prob = self.model_problem
        
        # Get bounds from model problem
        self.bounds = mod_prob.bounds
        self.n_dim = mod_prob.n_dim
        
        lbs = np.array([b[0] for b in self.bounds])
        ubs = np.array([b[1] for b in self.bounds])
        
        # Generate starting points based on sampler type
        np.random.seed(self.seed)
        
        if self.sampler_type == 'lhs':
            # Try common argument names for SciPy qmc (seed vs random_state)
            try:
                sampler = qmc.LatinHypercube(d=self.n_dim, seed=self.seed)  # type: ignore[arg-type]
            except TypeError:
                sampler = qmc.LatinHypercube(d=self.n_dim, random_state=self.seed)  # type: ignore[arg-type]
            scale_x0 = sampler.random(n=self.n_starts)
            x0_array = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)
        elif self.sampler_type == 'sobol':
            try:
                sampler = qmc.Sobol(d=self.n_dim, scramble=True, seed=self.seed)  # type: ignore[arg-type]
            except TypeError:
                sampler = qmc.Sobol(d=self.n_dim, scramble=True, random_state=self.seed)  # type: ignore[arg-type]
            scale_x0 = sampler.random(n=self.n_starts)
            x0_array = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)
        elif self.sampler_type == 'uniform':
            x0_array = np.random.uniform(lbs, ubs, size=(self.n_starts, self.n_dim))
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_type}")
        
        # Validate starting points (filter out invalid ones)
        valid_x0 = []
        for x0 in x0_array:
            fval = self.model_problem.log_likelihood_wrapper(x0)
            if fval < 1e9:  # Valid evaluation
                valid_x0.append(x0)
        
        # If we lost too many starting points, resample
        if len(valid_x0) < self.n_starts // 2:
            print(f"Warning: Only {len(valid_x0)} valid starting points found, resampling...")
            max_tries = 10 * self.n_starts
            n_tries = 0
            fval = 1e10
            while len(valid_x0) < self.n_starts and n_tries < max_tries:
                x_new = np.random.uniform(lbs, ubs, size=(self.n_dim,))
                fval = self.model_problem.log_likelihood_wrapper(x_new)
                if fval < 1e9:
                    valid_x0.append(x_new)
                n_tries += 1
        
        self.x0_list = valid_x0 if valid_x0 else [x0_array[0]]  # At least one start
        # Ensure x0_list is a list of numpy arrays
        self.x0_list = [np.array(x) for x in self.x0_list]
        
        # Reset function call counter
        self.model_problem.n_fun_calls = 0
        
        print(f"Initialized {len(self.x0_list)} starting points using {self.sampler_type}")
    
    def _objective(self, x: np.ndarray) -> float:
        """Objective function wrapper (minimization form)."""
        return self.model_problem.log_likelihood_wrapper(x, mode="pos")
    
    def _run_local_optimization(self, x0: np.ndarray) -> OptimizationRun:
        """Run a single local optimization from starting point x0."""
        start_time = time.time()
        n_feval = [0]
        
        def wrapped_objective(x):
            n_feval[0] += 1
            return self._objective(x)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.method == 'l-bfgs-b':
                    result = minimize(
                        wrapped_objective, x0,
                        method='L-BFGS-B',
                        bounds=self.bounds,
                        options={'maxiter': self.maxiter, 'ftol': 1e-9, 'gtol': 1e-6}
                    )
                elif self.method == 'nelder-mead':
                    result = minimize(
                        wrapped_objective, x0,
                        method='Nelder-Mead',
                        options={'maxiter': self.maxiter, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                elif self.method == 'powell':
                    result = minimize(
                        wrapped_objective, x0,
                        method='Powell',
                        bounds=self.bounds,
                        options={'maxiter': self.maxiter, 'ftol': 1e-6}
                    )
                elif self.method == 'trust-constr':
                    # Pass bounds as sequence of (lb, ub) pairs; avoid using Bounds constructor for typing
                    result = minimize(
                        wrapped_objective, x0,
                        method='trust-constr',
                        bounds=self.bounds,
                        options={'maxiter': self.maxiter, 'gtol': 1e-6}
                    )
                elif self.method == 'bfgs':
                    result = minimize(
                        wrapped_objective, x0,
                        method='BFGS',
                        options={'maxiter': self.maxiter, 'gtol': 1e-6}
                    )
                else:
                    raise ValueError(f"Unknown local method: {self.method}")
            
            elapsed = time.time() - start_time
            
            return OptimizationRun(
                x=result.x,
                fun=result.fun,
                success=result.success,
                n_feval=n_feval[0],
                n_iter=result.nit if hasattr(result, 'nit') else 0,
                time=elapsed,
                message=str(result.message) if hasattr(result, 'message') else "",
                start_point=x0.copy()
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return OptimizationRun(
                x=x0,
                fun=1e10,
                success=False,
                n_feval=n_feval[0],
                n_iter=0,
                time=elapsed,
                message=str(e),
                start_point=x0.copy()
            )
    
    def _run_global_optimization(self) -> OptimizationRun:
        """Run global optimization (DE, dual annealing, or basin hopping)."""
        start_time = time.time()
        n_feval = [0]
        
        def wrapped_objective(x):
            n_feval[0] += 1
            return self._objective(x)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if self.method == 'differential_evolution':
                    result = differential_evolution(
                        wrapped_objective,
                        self.bounds,
                        maxiter=self.maxiter,
                        workers=1,  # Single-threaded for fair comparison
                        polish=True
                    )
                elif self.method == 'dual_annealing':
                    result = dual_annealing(
                        wrapped_objective,
                        self.bounds,
                        maxiter=self.maxiter
                    )
                elif self.method == 'basinhopping':
                    x0 = self.x0_list[0] if self.x0_list else np.mean(np.array(self.bounds), axis=1)
                    minimizer_kwargs = {
                        'method': 'L-BFGS-B',
                        'bounds': self.bounds
                    }
                    result = basinhopping(
                        wrapped_objective, x0,
                        niter=self.maxiter,
                        minimizer_kwargs=minimizer_kwargs
                    )
                else:
                    raise ValueError(f"Unknown global method: {self.method}")
            
            elapsed = time.time() - start_time
            
            return OptimizationRun(
                x=result.x,
                fun=result.fun,
                success=result.success if hasattr(result, 'success') else True,
                n_feval=n_feval[0],
                n_iter=result.nit if hasattr(result, 'nit') else self.maxiter,
                time=elapsed,
                message=str(result.message) if hasattr(result, 'message') else "",
                start_point=None
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return OptimizationRun(
                x=np.zeros(self.n_dim),
                fun=1e10,
                success=False,
                n_feval=n_feval[0],
                n_iter=0,
                time=elapsed,
                message=str(e),
                start_point=None
            )
    
    def run(self) -> Dict:
        """
        Run multistart optimization and return results.
        
        Returns dict compatible with the Result class format.
        """
        self.all_runs = []
        
        # Check if this is a global or local method
        global_methods = ['differential_evolution', 'dual_annealing', 'basinhopping']
        
        if self.method in global_methods:
            print(f"Running {self.method} global optimization...")
            result = self._run_global_optimization()
            self.all_runs.append(result)
        else:
            # Run multistart local optimization
            print(f"Running multistart {self.method} with {len(self.x0_list)} starts...")
            for i, x0 in enumerate(self.x0_list):
                print(f"  Start {i+1}/{len(self.x0_list)}", end='\r')
                result = self._run_local_optimization(x0)
                self.all_runs.append(result)
            print()  # Newline after progress
        
        # Find best result (filter None to satisfy type checkers)
        valid_runs = [r for r in self.all_runs if r is not None]
        if len(valid_runs) == 0:
            raise RuntimeError("No optimization runs produced results")
        best_idx = int(np.argmin([r.fun for r in valid_runs]))
        self.best_run = valid_runs[best_idx]
        
        # Process results into standard format
        return self.process_results()
    
    def process_results(self) -> Dict:
        """
        Process optimization results into format compatible with Result class.
        
        The format mimics the structure from pocoSampler/pestoSampler:
        - posterior_samples: Best parameters replicated n_ensemble times
        - posterior_llhs: Best likelihood replicated
        - posterior_weights: Uniform weights
        - all_samples: All optimization trajectories
        - all_llhs: All final likelihoods
        """
        all_results = {}
        
        # Basic info
        all_results["seed"] = self.seed
        all_results["n_ensemble"] = len(self.all_runs)
        all_results["method"] = f"multistart_{self.method}"
        all_results["problem"] = self.model_problem.model_name
        # Guard against mypy thinking best_run may be None
        best_run = self.best_run
        if best_run is None:
            # No runs - produce safe defaults
            best_run = OptimizationRun(
                x=np.zeros(self.n_dim),
                fun=1e10,
                success=False,
                n_feval=0,
                n_iter=0,
                time=0.0,
                message="no result",
                start_point=None
            )
        all_results["converged"] = best_run.success
        
        # Optimization-specific info
        all_results["n_starts"] = len(self.x0_list) if self.x0_list else 1
        all_results["n_iter"] = sum(getattr(r, 'n_iter', 0) for r in self.all_runs)
        all_results["n_chains"] = 1  # For compatibility
        
        # Posterior samples: Use best result
        # For multistart, the "posterior" is essentially a point estimate
        # We replicate it to match expected format
        n_ensemble = len([r for r in self.all_runs if r is not None])
        best_run = best_run  # already guaranteed above
        best_x = best_run.x
        best_llh = -best_run.fun  # Convert back to log-likelihood
        
        # Create pseudo-posterior from all converged runs
        converged_runs = [r for r in self.all_runs if r is not None and getattr(r, 'success', False) and getattr(r, 'fun', 1e10) < 1e9]
        if converged_runs:
            posterior_samples = np.array([r.x for r in converged_runs])
            posterior_llhs = np.array([-r.fun for r in converged_runs])  # Convert to llh
            # Weight by likelihood (softmax-like weighting)
            llh_shifted = posterior_llhs - np.max(posterior_llhs)
            weights = np.exp(llh_shifted)
            posterior_weights = weights / np.sum(weights)
        else:
            posterior_samples = np.array([best_x])
            posterior_llhs = np.array([best_llh])
            posterior_weights = np.array([1.0])
        
        all_results["posterior_samples"] = posterior_samples
        all_results["posterior_llhs"] = posterior_llhs
        all_results["posterior_priors"] = np.zeros(len(posterior_llhs))  # Not used in optimization
        all_results["posterior_weights"] = posterior_weights
        
        # All samples/llhs for analysis
        # Shape: (n_iter, n_chains, n_dim) - we adapt this for optimization
        valid_runs = [r for r in self.all_runs if r is not None]
        all_x = np.array([r.x for r in valid_runs]) if valid_runs else np.array([])
        all_llhs = np.array([-r.fun for r in valid_runs]) if valid_runs else np.array([])
        
        # Reshape to match expected format (n_iter, n_chains, n_dim)
        # For multistart, treat each start as a "chain" with 1 iteration
        try:
            all_results["all_samples"] = all_x.reshape(1, -1, self.n_dim)
        except Exception:
            all_results["all_samples"] = np.array([]).reshape(1, 0, self.n_dim)
        try:
            all_results["all_llhs"] = all_llhs.reshape(1, -1)
        except Exception:
            all_results["all_llhs"] = np.array([]).reshape(1, 0)
        all_results["all_priors"] = np.zeros_like(all_llhs).reshape(1, -1) if all_llhs.size else np.zeros((1, 0))
        
        # Function calls
        all_results["n_fun_calls"] = sum(getattr(r, 'n_feval', 0) for r in self.all_runs)
        
        # Algorithm-specific info
        valid_runs = [r for r in self.all_runs if r is not None]
        n_runs = len(valid_runs)
        algo_info = {
            "optimizer": self.method,
            "sampler": self.sampler_type,
            "n_starts": len(self.x0_list) if self.x0_list else 1,
            "best_fval": float(best_run.fun),
            "best_x": (best_run.x.tolist() if hasattr(best_run.x, 'tolist') else list(best_run.x)),
            "success_rate": (sum(1 for r in valid_runs if getattr(r, 'success', False)) / n_runs) if n_runs > 0 else 0.0,
            "total_time": float(sum(getattr(r, 'time', 0.0) for r in valid_runs)),
            "all_fvals": [float(getattr(r, 'fun', 1e10)) for r in valid_runs],
            "all_success": [bool(getattr(r, 'success', False)) for r in valid_runs],
            "all_n_feval": [int(getattr(r, 'n_feval', 0)) for r in valid_runs],
            "convergence_distribution": self._compute_convergence_distribution()
        }
        all_results["algo_specific_info"] = algo_info
        
        return all_results
    
    def _compute_convergence_distribution(self) -> Dict:
        """Compute statistics on convergence across starts."""
        fvals = np.array([r.fun for r in self.all_runs])
        valid_fvals = fvals[fvals < 1e9]
        
        if len(valid_fvals) == 0:
            return {"n_valid": 0}
        
        return {
            "n_valid": int(len(valid_fvals)),
            "best": float(np.min(valid_fvals)),
            "worst": float(np.max(valid_fvals)),
            "mean": float(np.mean(valid_fvals)),
            "std": float(np.std(valid_fvals)),
            "median": float(np.median(valid_fvals)),
            "q25": float(np.percentile(valid_fvals, 25)),
            "q75": float(np.percentile(valid_fvals, 75))
        }


# ============================================================================
# Extended Result Class for Multistart
# ============================================================================

class MultistartResult:
    """
    Extended result class for multistart optimization.
    
    Provides additional analysis methods specific to multistart optimization
    while maintaining compatibility with the base Result class.
    """
    
    def __init__(self, result_dict: Dict):
        # Defaults for static type checkers
        self.algo_specific_info: Dict[str, Any] = {}
        self.posterior_llhs: Optional[np.ndarray] = None
        self.posterior_samples: Optional[np.ndarray] = None
        self.posterior_weights: Optional[np.ndarray] = None
        self.converged: bool = True

        for key, value in result_dict.items():
            setattr(self, key, value)
        
        # Ensure converged flag is present
        if not hasattr(self, 'converged'):
            self.converged = True
        
        # Ensure weights exist
        if self.posterior_weights is None:
            n = len(self.posterior_llhs) if self.posterior_llhs is not None else 1
            self.posterior_weights = np.ones(n) / n
    
    def get_best_parameters(self) -> Optional[np.ndarray]:
        """Return the best parameter set found."""
        if 'best_x' in self.algo_specific_info:
            return np.array(self.algo_specific_info['best_x'])
        if self.posterior_samples is None or self.posterior_llhs is None:
            return None
        return self.posterior_samples[np.argmax(self.posterior_llhs)]
    
    def get_best_likelihood(self) -> float:
        """Return the best likelihood found."""
        if 'best_fval' in self.algo_specific_info:
            return float(-self.algo_specific_info['best_fval'])
        if self.posterior_llhs is None:
            return float(-np.inf)
        return float(np.max(self.posterior_llhs))
    
    def get_convergence_stats(self) -> Dict:
        """Return convergence statistics."""
        if hasattr(self, 'algo_specific_info') and 'convergence_distribution' in self.algo_specific_info:
            return self.algo_specific_info['convergence_distribution']
        return {}
    
    def get_success_rate(self) -> float:
        """Return the success rate of optimization starts."""
        if hasattr(self, 'algo_specific_info') and 'success_rate' in self.algo_specific_info:
            return self.algo_specific_info['success_rate']
        return 1.0
    
    def get_n_basins(self, tol: float = 1.0) -> int:
        """
        Estimate number of distinct basins found.
        
        Two optima are in the same basin if their objective values differ by less than tol.
        """
        if hasattr(self, 'algo_specific_info') and 'all_fvals' in self.algo_specific_info:
            fvals = np.array(self.algo_specific_info['all_fvals'])
            fvals = fvals[fvals < 1e9]  # Valid only
            if len(fvals) == 0:
                return 0
            
            # Simple clustering by objective value
            sorted_fvals = np.sort(fvals)
            n_basins = 1
            for i in range(1, len(sorted_fvals)):
                if sorted_fvals[i] - sorted_fvals[i-1] > tol:
                    n_basins += 1
            return n_basins
        return 1