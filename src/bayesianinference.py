from abc import ABC, abstractmethod
from typing import Any
from modelproblem import ModelProblem

class BayesianInference(ABC):
    def __init__(
        self,
        seed: int,
        n_ensemble: int,
        model_problem: ModelProblem,
        n_cpus: int,
        method: str
        ):
        self.seed = seed
        self.n_ensemble = n_ensemble
        self.model_problem = model_problem
        self.n_cpus = n_cpus
        self.method = method

    @abstractmethod
    def initialize(self) -> Any:
        """Initialize the sampler. Return value is sampler-specific."""
        pass

    @abstractmethod
    def process_results(self) -> Any:
        """Process and return results in a sampler-specific format."""
        pass

    @abstractmethod
    def run(self) -> Any:
        """Run the sampler and return the results."""
        pass
