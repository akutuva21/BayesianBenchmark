import os
import petab
from petab.v1.parameters import get_priors_from_df
from scipy.stats import uniform, norm
import pypesto.objective.roadrunner as pypesto_rr
class ModelProblem():
	def __init__(self, model_string):
		self.model_name = model_string

	def initialize(self):
		model_name =  self.model_name

		# Find project root (look for models/ directory)
		if os.path.exists(f"./models/{model_name}"):
			petab_yaml = f"./models/{model_name}/{model_name}.yaml"
		elif os.path.exists(f"../models/{model_name}"):
			petab_yaml = f"../models/{model_name}/{model_name}.yaml"
		else:
			raise FileNotFoundError(f"Could not find model directory for {model_name}")

		petab_problem = petab.v1.Problem.from_yaml(petab_yaml)
		importer = pypesto_rr.PetabImporterRR(petab_problem)
		problem = importer.create_problem()

		# set tolerances for ode solver
		if self.model_name == "Hopf":
			#solver_options = pypesto_rr.SolverOptions(
			#	integrator="rk45"
			#	)
			#problem.objective.solver_options = solver_options
			pass
		else:
			solver_options = pypesto_rr.SolverOptions(
				relative_tolerance = 1e-16,
				absolute_tolerance = 1e-8
				)
			problem.objective.solver_options = solver_options

		self.problem = problem
		self.petab_problem = petab_problem
		prior_info = get_priors_from_df(petab_problem.parameter_df,
								  mode="objective")
		self.prior_info = prior_info
		self.n_dim = len(prior_info)
		self.bounds = [x[3] for x in prior_info]
		self.n_fun_calls = 0

	def create_poco_priors(self):
		prior_list = []

		# the list returned from get_priors_from_df is always
		# in the following order:
		# 1) prior type (string)
		# 2) prior parameters (tuple)
		# 3) parameter scale (string - log or linear)
		# 4) parameter bounds (tuple)
		# ignore #3 and #4 
		for info in self.prior_info:
			type, prior_pars, _, _ = info
			if "uniform" in type.lower():
				lb, ub = prior_pars
				prior = uniform(loc=lb, scale=ub-lb)
			elif "normal" in type.lower():
				mean, std = prior_pars
				prior = norm(loc=mean, scale=std)
			prior_list.append(prior)
		return prior_list


	def log_likelihood_wrapper(self, x, mode="pos"):
		try:
			result = self.problem.objective(x, mode="mode_fun", return_dict=True)
			fval = result["fval"]
		except:
			fval = 1e10

		if mode == "neg":
			fval = -1*fval
		# ! IMPORTANT: self.n_fun_calls only tracks total number of function calls
		# ! when using PT-MCMC since it does NOT run in parallel
		# ! You can only use this with pocoMC when n_cpus = 1
		self.n_fun_calls += 1
		return fval