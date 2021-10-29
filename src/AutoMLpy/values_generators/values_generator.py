import numpy as np
from typing import Tuple, Union


class ValuesGenerator:
	def __init__(
			self,
			bounds: Tuple[Union[int, float], Union[int, float]],
			resolution: int = 1_000,
			seed: int = None,
	):
		self.bounds = bounds
		self.resolution = resolution
		self.seed = seed

	def __call__(self, n: int = 1):
		raise NotImplementedError()







