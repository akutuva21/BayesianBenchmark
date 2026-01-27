"""
Run Model Calibration Script - Extended with Multistart Optimization

This script extends the original run_model_calibration.py to support
multistart local optimization and global optimization methods for
benchmarking against Bayesian inference methods.

Usage:
    python run_model_calibration.py -m multistart_lbfgsb -p Hopf -n 100 -s 42 -o results/
    python run_model_calibration.py -m differential_evolution -p Michaelis_Menten -s 1 -o results/
    
Available methods:
    Bayesian (original):
        - ptmcmc: Parallel Tempering MCMC
        - smc: Sequential Monte Carlo  
        - pmc: Preconditioned Monte Carlo
    
    Multistart local (new):
        - multistart_lbfgsb: L-BFGS-B
        - multistart_nm: Nelder-Mead
        - multistart_powell: Powell
        - multistart_trust: Trust-region constrained
        
    Global (new):
        - differential_evolution: Differential Evolution
        - dual_annealing: Dual Annealing
        - basinhopping: Basin Hopping
"""

import os
import gc
import gzip
import pickle
import argparse
import numpy as np
from typing import Optional

# Suppress OpenMP threading for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"


def get_sampler(args, mod_prob):
    """
    Factory function to create the appropriate sampler based on method.
    
    Returns the sampler instance ready for initialization.
    """
    method = args.method
    
    # Original Bayesian methods
    if method == "ptmcmc":
        from pestosampler import pestoSampler
        return pestoSampler(
            args.seed,
            args.n_ensemble,
            mod_prob,
            args.n_cpus,
            method,
            args.n_iter,
            args.n_chains
        )
    
    elif method in ["smc", "pmc"]:
        from pocosampler import pocoSampler
        return pocoSampler(
            args.seed,
            args.n_ensemble,
            mod_prob,
            args.n_cpus,
            method
        )
    
    # Multistart local optimization methods
    elif method.startswith("multistart_"):
        from multistart_sampler import MultistartSampler
        
        # Parse the local optimizer name
        optimizer_map = {
            "multistart_lbfgsb": "l-bfgs-b",
            "multistart_nm": "nelder-mead",
            "multistart_powell": "powell",
            "multistart_trust": "trust-constr",
            "multistart_bfgs": "bfgs"
        }
        
        if method not in optimizer_map:
            raise ValueError(f"Unknown multistart method: {method}")
        
        local_method = optimizer_map[method]
        
        return MultistartSampler(
            seed=args.seed,
            n_starts=args.n_starts,
            model_problem=mod_prob,
            n_cpus=args.n_cpus,
            method=local_method,
            sampler=args.start_sampler,
            maxiter=args.maxiter
        )
    
    # Global optimization methods
    elif method in ["differential_evolution", "dual_annealing", "basinhopping"]:
        from multistart_sampler import MultistartSampler
        
        return MultistartSampler(
            seed=args.seed,
            n_starts=1,  # Global methods don't use multiple starts
            model_problem=mod_prob,
            n_cpus=args.n_cpus,
            method=method,
            sampler=args.start_sampler,
            maxiter=args.maxiter
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_model_calibration(args):
    """Main function to run model calibration."""
    print("=" * 60)
    print(f"Running Model Calibration")
    print(f"  Problem: {args.problem}")
    print(f"  Method: {args.method}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # Import ModelProblem
    from modelproblem import ModelProblem
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set random seed
    np.random.seed(seed=args.seed)
    
    # Initialize model problem
    mod_prob = ModelProblem(args.problem)
    mod_prob.initialize()
    
    print(f"Model initialized with {mod_prob.n_dim} parameters")
    print(f"Bounds: {mod_prob.bounds}")
    
    # Create and run sampler
    sampler = get_sampler(args, mod_prob)
    sampler.initialize()
    results = sampler.run()
    
    # Save results
    results_fname = f"{args.output_dir}/{args.problem}_{args.method}_{args.seed}seed.pkl"
    
    # Use gzip compression for large results
    gc.disable()
    try:
        gc.collect()
        with gzip.open(results_fname, "w") as fp:
            pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        gc.enable()
    
    print(f"\nResults saved to {results_fname}")
    
    # Print summary
    print_results_summary(results)
    
    return results


def print_results_summary(results: dict):
    """Print a summary of the calibration results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Method: {results.get('method', 'unknown')}")
    print(f"Converged: {results.get('converged', 'N/A')}")
    print(f"Total function calls: {results.get('n_fun_calls', 'N/A')}")
    
    if 'posterior_llhs' in results:
        llhs = results['posterior_llhs']
        print(f"Best log-likelihood: {np.max(llhs):.4f}")
        print(f"Mean log-likelihood: {np.mean(llhs):.4f}")
    
    if 'algo_specific_info' in results:
        info = results['algo_specific_info']
        if 'success_rate' in info:
            print(f"Success rate: {info['success_rate']*100:.1f}%")
        if 'total_time' in info:
            print(f"Total time: {info['total_time']:.2f}s")
        if 'convergence_distribution' in info:
            conv = info['convergence_distribution']
            if 'best' in conv:
                print(f"Best objective: {conv['best']:.4f}")
            if 'n_valid' in conv:
                print(f"Valid runs: {conv['n_valid']}")
    
    print("=" * 60)


def run_benchmark_comparison(args):
    """
    Run a full benchmark comparison across multiple methods.
    
    This runs the same problem with multiple optimization methods
    for benchmarking purposes.
    """
    from modelproblem import ModelProblem
    
    methods = [
        "multistart_lbfgsb",
        "multistart_nm", 
        "multistart_powell",
        "differential_evolution",
    ]
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method}...")
        print(f"{'='*60}")
        
        args.method = method
        
        try:
            results = run_model_calibration(args)
            all_results[method] = results
        except Exception as e:
            print(f"Error running {method}: {e}")
            all_results[method] = {"error": str(e)}
    
    # Save comparison results
    comparison_fname = f"{args.output_dir}/{args.problem}_comparison_{args.seed}seed.pkl"
    with gzip.open(comparison_fname, "w") as fp:
        pickle.dump(all_results, fp)
    
    print(f"\nComparison results saved to {comparison_fname}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model calibration with various optimization methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run multistart L-BFGS-B optimization
  python run_model_calibration.py -m multistart_lbfgsb -p Hopf -n 50 -s 42

  # Run differential evolution
  python run_model_calibration.py -m differential_evolution -p Michaelis_Menten -s 1

  # Run original PT-MCMC (requires pestosampler.py)
  python run_model_calibration.py -m ptmcmc -p Hopf -n 100 -s 1 -i 10000 -w 4
        """
    )
    
    # Basic arguments
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument("-m", "--method", type=str, 
                        choices={
                            # Original Bayesian methods
                            "pmc", "smc", "ptmcmc",
                            # Multistart local methods
                            "multistart_lbfgsb", "multistart_nm", 
                            "multistart_powell", "multistart_trust", "multistart_bfgs",
                            # Global methods
                            "differential_evolution", "dual_annealing", "basinhopping",
                            # Benchmark mode
                            "benchmark"
                        },
                        default="multistart_lbfgsb",
                        help="Optimization/inference method to use")
    parser.add_argument("-p", "--problem", type=str, required=True,
                        help="Problem name (directory containing PEtab model)")
    parser.add_argument("-o", "--output_dir", type=str, default="results",
                        help="Output directory for results")
    
    # Bayesian-specific arguments
    parser.add_argument("-n", "--n_ensemble", type=int, default=100,
                        help="Number of posterior samples (Bayesian methods)")
    parser.add_argument("-c", "--n_cpus", type=int, default=1,
                        help="Number of CPUs for parallel methods")
    parser.add_argument("-i", "--n_iter", type=int, default=10000,
                        help="Number of MCMC iterations (PT-MCMC only)")
    parser.add_argument("-w", "--n_chains", type=int, default=4,
                        help="Number of temperature chains (PT-MCMC only)")
    
    # Multistart-specific arguments
    parser.add_argument("--n_starts", type=int, default=50,
                        help="Number of starting points for multistart optimization")
    parser.add_argument("--start_sampler", type=str, default="lhs",
                        choices={"lhs", "sobol", "uniform"},
                        help="Starting point sampling strategy")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="Maximum iterations per local optimization")
    
    args = parser.parse_args()
    
    # Handle benchmark mode
    if args.method == "benchmark":
        run_benchmark_comparison(args)
    else:
        run_model_calibration(args)