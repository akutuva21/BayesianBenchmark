import os
import numpy as np
from scipy.stats import qmc
from typing import Any
from bayesianinference import BayesianInference
from modelproblem import ModelProblem

# Optional dependency: pypesto. Import defensively to keep lint checks happy
try:
    import pypesto.sample as sample  # type: ignore
    from pypesto.sample.geweke_test import burn_in_by_sequential_geweke  # type: ignore
except Exception:
    sample: Any = None
    def burn_in_by_sequential_geweke(chain: np.ndarray) -> int:
        # Fallback stub for environments without pypesto; returns 0 burn-in
        return 0

if sample is None:
    raise ImportError("pestoSampler requires pypesto. Install with: pip install pypesto")

class pestoSampler(BayesianInference):
	def __init__(
			self, 
			seed: int, 
			n_ensemble: int, 
			model_problem: ModelProblem,
			n_cpus: int,
			method: str,
			n_iter: int,
			n_chains: int
			):
		super().__init__(seed, n_ensemble, model_problem, n_cpus, method)
		self.n_iter = n_iter
		self.n_chains = n_chains

	def initialize(self):
		mod_prob = self.model_problem

		lbs = [x[0] for x in mod_prob.bounds]
		ubs = [x[1] for x in mod_prob.bounds]
		# Compatibility across SciPy versions for RNG arg name
		# Try common argument names for SciPy qmc (seed vs random_state)
		try:
			lhs = qmc.LatinHypercube(d=mod_prob.n_dim, seed=self.seed)  # type: ignore[arg-type]
		except TypeError:
			lhs = qmc.LatinHypercube(d=mod_prob.n_dim, random_state=self.seed)  # type: ignore[arg-type]
		scale_x0 = lhs.random(n=self.n_chains)
		x0 = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)
		#print(f"Original x0: {x0}")

		x0_fail = False
		# verify that x0 works
		for x in x0:
			fval = self.model_problem.log_likelihood_wrapper(x)
			#print(fval)
			if fval == 1e10:
				x0_fail = True
				break
		
		# re-sample if necessary
		if x0_fail:
			max_tries = int(1e4)
			n_tries = 0
			while (n_tries < max_tries) and x0_fail:
				n_tries += 1
				scale_x0 = lhs.random(n=self.n_chains)
				x0 = qmc.scale(scale_x0, l_bounds=lbs, u_bounds=ubs)
				#print(f"Try No.{n_tries}, New x0: {x0}")
				new_fvals = np.array([self.model_problem.log_likelihood_wrapper(x) for x in x0])
				#print(f"\t Fvals: {new_fvals}")
				#print("check: ", new_fvals == 1e10)
				if np.any(new_fvals == 1e10):
					x0_fail = True
				else:
					x0_fail = False
				#print(x0_fail)

		#raise ValueError
		# reset total n_fun_calls
		mod_prob.n_fun_calls = 0
		self.x0 = list(x0)
		sampler = sample.AdaptiveParallelTemperingSampler(
			internal_sampler=sample.AdaptiveMetropolisSampler(),
			n_chains=self.n_chains
			)
	
		sampler.initialize(mod_prob.problem, list(x0))
			
		for internal_sampler in sampler.samplers:
			internal_sampler.neglogpost = self.model_problem.log_likelihood_wrapper
		
		self.sampler = sampler

	# Courtesy of ChatGPT
	def largest_step_size(self, L, N):
		# Start from the largest possible step size (L-1) and work backwards
		for i in range((L - 1) // (N - 1), 0, -1):
			# Check if this step size allows exactly N samples within the bounds of L
			if (N - 1) * i < L:
				return i
		return None  # Return None if no valid step size is found


	def create_posterior_ensemble(self):
		sampler = self.sampler
		samples = sampler.get_samples()
		n_iter = self.n_iter

		# get the lowest temperature chain index
		# Note: remember that beta is the inverse of the temperature so 
		# we want the chain with the max beta
		ch_idx = np.argmax(samples.betas)
		chain = np.array(samples.trace_x[ch_idx, :, :])
		burn_in_idx = burn_in_by_sequential_geweke(chain)
		diff = (n_iter - burn_in_idx)
		converged = True
		if diff < self.n_ensemble:
			converged = False

			# make empty lists to avert downstream errors
			posterior_samples = np.empty((self.n_ensemble, self.model_problem.n_dim))
			posterior_llhs = np.empty(self.n_ensemble)
			posterior_priors = np.empty(self.n_ensemble)
		else:
			step_size = self.largest_step_size(diff, self.n_ensemble)
			
			trim_trace_x = samples.trace_x[ch_idx, burn_in_idx:, :]
			trim_trace_llhs = -1*samples.trace_neglogpost[ch_idx, burn_in_idx:]
			trim_trace_priors = samples.trace_neglogprior[ch_idx, burn_in_idx:]

			posterior_samples = trim_trace_x[::step_size, :][:self.n_ensemble, :]
			posterior_llhs = trim_trace_llhs[::step_size][:self.n_ensemble]
			posterior_priors = trim_trace_priors[::step_size][:self.n_ensemble]
		
		all_results = {}
		all_results["converged"] = converged
		all_results["posterior_samples"] = posterior_samples
		all_results["posterior_llhs"] = posterior_llhs
		all_results["posterior_priors"] = np.array(posterior_priors, dtype=float)
		return burn_in_idx, all_results


	def process_results(self):
		sampler = self.sampler
		algo_specific_info = {}
		algo_specific_info["betas"] = sampler.betas
		bi_idx, all_results = self.create_posterior_ensemble()
		algo_specific_info["burn_in_idx"] = bi_idx

		all_results["seed"] = self.seed
		all_results["n_ensemble"] = self.n_ensemble
		all_results["method"] = self.method
		all_results["problem"] = self.model_problem.model_name

		all_results["n_iter"] = self.n_iter+1
		#all_results["iters"] = np.array(range(self.n_iter+1), dtype=float)
		all_results["n_chains"] = self.n_chains
		
		all_samples = np.array([x.trace_x for x in sampler.samplers])
		all_samples = np.swapaxes(all_samples, 0, 1)
		#all_weights = np.ones(shape=all_samples.shape[:-1])
		all_llhs = -1*np.array([x.trace_neglogpost for x in sampler.samplers])
		all_llhs = np.swapaxes(all_llhs, 0, 1)
		all_priors = np.array([x.trace_neglogprior for x in sampler.samplers], dtype=float)
		all_priors = np.swapaxes(all_priors, 0, 1)
		
		# downsample all traces for storage
		downsample_step = 100
		all_results["sample_step"] = downsample_step
		all_results["all_samples"] = all_samples[::downsample_step, :, :]
		#all_results["all_weights"] = all_weights
		all_results["all_llhs"] = all_llhs[::downsample_step, :]
		all_results["all_priors"] = all_priors[::downsample_step, :]

		n_fun_calls = self.model_problem.n_fun_calls
		all_results["n_fun_calls"] = n_fun_calls
		all_results["algo_specific_info"] = algo_specific_info
		return all_results
			
			
	def run(self):
		sampler = self.sampler
		sampler.sample(n_samples=self.n_iter)
		results = self.process_results()
		return results