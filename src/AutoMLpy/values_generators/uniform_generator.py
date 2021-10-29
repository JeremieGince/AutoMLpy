import numpy as np
from typing import Tuple, Union
from .values_generator import ValuesGenerator


class UniformGenerator(ValuesGenerator):
	def __init__(
			self,
			bounds: Tuple[Union[int, float], Union[int, float]],
			resolution: int = 1_000,
			seed: int = None,
	):
		super(UniformGenerator, self).__init__(bounds, resolution, seed)
		self.generator = np.random.default_rng(seed=seed)

	def __call__(self, n: int = 1):
		return self.generator.uniform(self.bounds[0], self.bounds[1], size=n)





