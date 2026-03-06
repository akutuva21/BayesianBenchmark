"""
Standalone Demo: Multistart Optimization Benchmarking

This demo shows how to use the multistart optimization framework with
synthetic test problems. It doesn't require PEtab/pypesto and can be
run directly to test the optimization infrastructure.

Run with:
    python demo_multistart.py
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import time
import warnings


# ============================================================================
# Simple Model Problem Class (no PEtab dependency)
# ============================================================================

class SimpleModelProblem:
    """
    Simple model problem for testing multistart optimization.
    
    This class mimics the interface of the PEtab-based ModelProblem class
    but uses pure Python/SciPy for ODE solving.
    """
    
    def __init__(self, model_name: str = "test"):
        self.model_name = model_name
        self.n_fun_calls = 0
        self.bounds: List[Tuple[float, float]] = []
        self.n_dim: int = 0
        self.true_params: Optional[np.ndarray] = None
        self.data: np.ndarray = np.array([])
        self.time_points: np.ndarray = np.array([])
        self.noise_std: float = 1.0
    
    def initialize(self):
        """Initialize the model - to be overridden by subclasses."""
        raise NotImplementedError
    
    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Simulate the model with given parameters."""
        raise NotImplementedError
    
    def log_likelihood_wrapper(self, x: np.ndarray, mode: str = "pos") -> float:
        """
        Compute negative log-likelihood (for minimization).
        
        mode="pos" returns positive nll (for minimization)
        mode="neg" returns negative nll (actual log-likelihood)
        """
        self.n_fun_calls += 1
        
        try:
            sim = self.simulate(x)
            
            if np.any(np.isnan(sim)) or np.any(np.isinf(sim)):
                return 1e10 if mode == "pos" else -1e10
            
            # Gaussian likelihood
            residuals = (sim - self.data) / self.noise_std
            nll = 0.5 * np.sum(residuals ** 2)
            
            if mode == "pos":
                return nll
            else:
                return -nll
                
        except Exception:
            return 1e10 if mode == "pos" else -1e10


class MichaelisMentenProblem(SimpleModelProblem):
    """
    Michaelis-Menten enzyme kinetics model.
    
    d[S]/dt = -Vmax * [S] / (Km + [S])
    d[P]/dt = Vmax * [S] / (Km + [S])
    
    Parameters: Vmax, Km (both in log10 scale for optimization)
    """
    
    def __init__(self, 
                 Vmax_true: float = 1.0,
                 Km_true: float = 0.5,
                 S0: float = 2.0,
                 noise_std: float = 0.05):
        super().__init__("Michaelis_Menten")
        self.Vmax_true = Vmax_true
        self.Km_true = Km_true
        self.S0 = S0
        self.noise_std = noise_std
    
    def initialize(self):
        # Parameters: log10(Vmax), log10(Km)
        self.bounds = [(-2, 1), (-2, 1)]  # log10 scale: 0.01 to 10
        self.n_dim = 2
        self.true_params = np.array([np.log10(self.Vmax_true), np.log10(self.Km_true)])
        
        # Time points
        self.time_points = np.linspace(0, 10, 50)
        
        # Generate synthetic data
        true_sim = self.simulate(self.true_params)
        np.random.seed(42)
        self.data = true_sim + np.random.normal(0, self.noise_std, true_sim.shape)
    
    def _ode_rhs(self, y: np.ndarray, t: float, Vmax: float, Km: float) -> np.ndarray:
        S, P = y
        rate = Vmax * S / (Km + S) if S > 0 else 0
        return np.array([-rate, rate])
    
    def simulate(self, params: np.ndarray) -> np.ndarray:
        # Transform from log10 to linear
        Vmax = 10 ** params[0]
        Km = 10 ** params[1]
        
        y0 = np.array([self.S0, 0.0])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = odeint(self._ode_rhs, y0, self.time_points, args=(Vmax, Km))
        
        # Return both S and P
        return sol.flatten()


class NegativeFeedbackOscillator(SimpleModelProblem):
    """
    Negative feedback oscillator (Goodwin model variant).
    
    dx1/dt = v0 / (1 + (x3/K)^n) - k1*x1
    dx2/dt = k2*x1 - k3*x2
    dx3/dt = k4*x2 - k5*x3
    
    This is a challenging problem with multimodal likelihood surface.
    """
    
    def __init__(self):
        super().__init__("Neg_Feed_Oscillate")
        self.noise_std = 0.1
        
        # True parameters (in log10 scale for rates, linear for Hill coefficient)
        self.true_values = {
            'v0': 1.0,    # Production rate
            'K': 1.0,     # Hill constant
            'n': 4.0,     # Hill coefficient (linear)
            'k1': 0.1,    # Degradation rates
            'k2': 0.5,
            'k3': 0.1,
            'k4': 0.5,
            'k5': 0.1
        }
    
    def initialize(self):
        # Bounds in optimization space (log10 for rates, linear for n)
        self.bounds = [
            (-1, 1),     # log10(v0): 0.1 to 10
            (-1, 1),     # log10(K): 0.1 to 10
            (1, 10),     # n: 1 to 10 (linear)
            (-2, 0),     # log10(k1): 0.01 to 1
            (-1, 1),     # log10(k2): 0.1 to 10
            (-2, 0),     # log10(k3)
            (-1, 1),     # log10(k4)
            (-2, 0),     # log10(k5)
        ]
        self.n_dim = 8
        
        # True parameters in optimization space
        tv = self.true_values
        self.true_params = np.array([
            np.log10(tv['v0']),
            np.log10(tv['K']),
            tv['n'],  # Linear
            np.log10(tv['k1']),
            np.log10(tv['k2']),
            np.log10(tv['k3']),
            np.log10(tv['k4']),
            np.log10(tv['k5'])
        ])
        
        # Time points
        self.time_points = np.linspace(0, 200, 100)
        
        # Generate synthetic data
        true_sim = self.simulate(self.true_params)
        np.random.seed(42)
        self.data = true_sim + np.random.normal(0, self.noise_std, true_sim.shape)
    
    def _ode_rhs(self, y: np.ndarray, t: float, params: np.ndarray) -> np.ndarray:
        x1, x2, x3 = y
        v0, K, n, k1, k2, k3, k4, k5 = params
        
        # Hill function repression
        hill = v0 / (1 + (x3 / K) ** n)
        
        dx1 = hill - k1 * x1
        dx2 = k2 * x1 - k3 * x2
        dx3 = k4 * x2 - k5 * x3
        
        return np.array([dx1, dx2, dx3])
    
    def _transform_params(self, opt_params: np.ndarray) -> np.ndarray:
        """Transform from optimization space to natural space."""
        return np.array([
            10 ** opt_params[0],  # v0
            10 ** opt_params[1],  # K
            opt_params[2],        # n (linear)
            10 ** opt_params[3],  # k1
            10 ** opt_params[4],  # k2
            10 ** opt_params[5],  # k3
            10 ** opt_params[6],  # k4
            10 ** opt_params[7],  # k5
        ])
    
    def simulate(self, params: np.ndarray) -> np.ndarray:
        opt_params = params
        params = self._transform_params(opt_params)
        y0 = np.array([1.0, 1.0, 1.0])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = odeint(self._ode_rhs, y0, self.time_points, args=(params,),
                        rtol=1e-8, atol=1e-10, mxstep=10000)
        
        return sol.flatten()


# ============================================================================
# Multistart Optimizer (simplified version for demo)
# ============================================================================

class DemoMultistartOptimizer:
    """
    Simplified multistart optimizer for the demo.
    """
    
    def __init__(
        self,
        model_problem: SimpleModelProblem,
        n_starts: int = 20,
        method: str = 'l-bfgs-b',
        seed: int = 42
    ):
        self.model = model_problem
        self.n_starts = n_starts
        self.method = method
        self.seed = seed
        self.results = []
    
    def _generate_start_points(self) -> np.ndarray:
        """Generate Latin Hypercube starting points."""
        from scipy.stats import qmc
        
        lbs = np.array([b[0] for b in self.model.bounds])
        ubs = np.array([b[1] for b in self.model.bounds])
        
        # Compatibility across SciPy versions for RNG arg name
        import inspect
        # Try common argument names for SciPy qmc (seed vs random_state)
        try:
            sampler = qmc.LatinHypercube(d=self.model.n_dim, seed=self.seed)  # type: ignore[arg-type]
        except TypeError:
            sampler = qmc.LatinHypercube(d=self.model.n_dim, random_state=self.seed)  # type: ignore[arg-type]
        samples = sampler.random(n=self.n_starts)
        return qmc.scale(samples, l_bounds=lbs, u_bounds=ubs)
    
    def _objective(self, x: np.ndarray) -> float:
        return self.model.log_likelihood_wrapper(x, mode="pos")
    
    def run(self) -> Dict:
        """Run multistart optimization."""
        start_points = self._generate_start_points()
        self.results = []
        
        print(f"Running multistart {self.method} with {self.n_starts} starts...")
        
        for i, x0 in enumerate(start_points):
            print(f"  Start {i+1}/{self.n_starts}", end='\r')
            
            start_time = time.time()
            n_feval = [0]
            
            def wrapped(x):
                n_feval[0] += 1
                return self._objective(x)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    if self.method == 'l-bfgs-b':
                        result = minimize(wrapped, x0, method='L-BFGS-B',
                                        bounds=self.model.bounds,
                                        options={'maxiter': 1000})
                    elif self.method == 'nelder-mead':
                        result = minimize(wrapped, x0, method='Nelder-Mead',
                                        options={'maxiter': 1000})
                    elif self.method == 'powell':
                        result = minimize(wrapped, x0, method='Powell',
                                        bounds=self.model.bounds,
                                        options={'maxiter': 1000})
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                
                elapsed = time.time() - start_time
                
                self.results.append({
                    'x': result.x,
                    'fun': result.fun,
                    'success': result.success,
                    'n_feval': n_feval[0],
                    'time': elapsed,
                    'start': x0
                })
                
            except Exception as e:
                self.results.append({
                    'x': x0,
                    'fun': 1e10,
                    'success': False,
                    'n_feval': n_feval[0],
                    'time': 0,
                    'start': x0,
                    'error': str(e)
                })
        
        print()  # Newline
        
        # Find best
        best_idx = np.argmin([r['fun'] for r in self.results])
        best = self.results[best_idx]
        
        # Create summary
        fvals = np.array([r['fun'] for r in self.results])
        valid_fvals = fvals[fvals < 1e9]
        
        summary = {
            'method': f'multistart_{self.method}',
            'n_starts': self.n_starts,
            'best_x': best['x'],
            'best_fun': best['fun'],
            'best_llh': -best['fun'],
            'total_feval': sum(r['n_feval'] for r in self.results),
            'total_time': sum(r['time'] for r in self.results),
            'success_rate': sum(1 for r in self.results if r['success']) / len(self.results),
            'n_valid': len(valid_fvals),
            'fval_mean': np.mean(valid_fvals) if len(valid_fvals) > 0 else np.nan,
            'fval_std': np.std(valid_fvals) if len(valid_fvals) > 0 else np.nan,
            'all_results': self.results
        }
        
        return summary


def run_demo():
    """
    Run a complete demonstration of multistart optimization.
    """
    print("=" * 70)
    print("MULTISTART OPTIMIZATION DEMO")
    print("=" * 70)
    
    # ========================================
    # Test 1: Michaelis-Menten (2 parameters)
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: Michaelis-Menten Model (2 parameters)")
    print("=" * 70)
    
    mm_problem = MichaelisMentenProblem()
    mm_problem.initialize()
    
    print(f"True parameters (log10 scale): {mm_problem.true_params}")
    print(f"Bounds: {mm_problem.bounds}")
    
    # Run different optimizers
    methods = ['l-bfgs-b', 'nelder-mead', 'powell']
    mm_results = {}
    
    for method in methods:
        print(f"\n--- {method.upper()} ---")
        optimizer = DemoMultistartOptimizer(
            mm_problem, 
            n_starts=20, 
            method=method,
            seed=42
        )
        mm_results[method] = optimizer.run()
        
        summary = mm_results[method]
        print(f"Best objective: {summary['best_fun']:.4f}")
        print(f"Best parameters: {summary['best_x']}")
        print(f"True parameters: {mm_problem.true_params}")
        print(f"Parameter error: {np.linalg.norm(summary['best_x'] - mm_problem.true_params):.4f}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Total function evaluations: {summary['total_feval']}")
        print(f"Total time: {summary['total_time']:.2f}s")
    
    # ========================================
    # Test 2: Negative Feedback Oscillator (8 parameters)
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 2: Negative Feedback Oscillator (8 parameters)")
    print("This is a challenging multimodal problem!")
    print("=" * 70)
    
    nfo_problem = NegativeFeedbackOscillator()
    nfo_problem.initialize()
    
    print(f"True parameters: {nfo_problem.true_params}")
    print(f"Number of parameters: {nfo_problem.n_dim}")
    
    # Run with more starts for the harder problem
    nfo_results = {}
    
    for method in ['l-bfgs-b', 'powell']:
        print(f"\n--- {method.upper()} ---")
        optimizer = DemoMultistartOptimizer(
            nfo_problem,
            n_starts=30,
            method=method,
            seed=42
        )
        nfo_results[method] = optimizer.run()
        
        summary = nfo_results[method]
        print(f"Best objective: {summary['best_fun']:.4f}")
        print(f"Parameter error: {np.linalg.norm(summary['best_x'] - nfo_problem.true_params):.4f}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Total function evaluations: {summary['total_feval']}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nMichaelis-Menten Results:")
    print("-" * 50)
    for method, result in mm_results.items():
        print(f"  {method:15s}: best_obj={result['best_fun']:8.4f}, "
              f"success={result['success_rate']*100:5.1f}%, "
              f"evals={result['total_feval']:5d}")
    
    print("\nNegative Feedback Oscillator Results:")
    print("-" * 50)
    for method, result in nfo_results.items():
        print(f"  {method:15s}: best_obj={result['best_fun']:8.4f}, "
              f"success={result['success_rate']*100:5.1f}%, "
              f"evals={result['total_feval']:5d}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    return mm_results, nfo_results


if __name__ == "__main__":
    run_demo()