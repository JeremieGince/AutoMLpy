import functools
import operator
from typing import Iterable, Tuple, Union

import numpy as np
from .tools import try_prod_overflow


class XXSpace:
	def __init__(
			self,
			values_matrix: Iterable,
			use_umap_for_high_dimensional_space: bool = False,
			**kwargs
	):
		self.values_matrix = values_matrix
		self._sizes = [len(iterable) for iterable in values_matrix]
		try:
			self._length = int(try_prod_overflow(self._sizes))
			self._is_finite = True
		except ValueError:
			self._length = np.inf
			self._is_finite = False

		self._is_using_umap = use_umap_for_high_dimensional_space
		self._kwargs = kwargs
		self._set_default_kwargs_()
		self.reducer = None
		self._xx_space_umap: XXSpace = None
		if self._is_using_umap:
			self._init_reducer_()
			self._fit_reducer_()
			if not self._xx_space_umap.is_finite:
				raise ValueError(
					"Umap space is too large. Consider decrease umap_resolution_space or umap_n_components."
				)

	@property
	def is_finite(self):
		return self._is_finite

	@property
	def shape(self) -> tuple:
		if self._is_using_umap:
			return self._xx_space_umap.shape
		return len(self), len(self._sizes)

	@property
	def transformed_space(self) -> 'XXSpace':
		if self._is_using_umap:
			return self._xx_space_umap
		return self

	def __len__(self):
		if self._is_using_umap:
			return len(self._xx_space_umap)
		return self._length

	def __getitem__(self, item):
		if isinstance(item, Iterable):
			if any([isinstance(a, slice) for a in item]):
				raise NotImplementedError("Nested slice is not implemented yet.")
			return np.stack(list(map(self.__getitem__, item)))
		elif isinstance(item, slice):
			ifnone = lambda a, b: b if a is None else a
			if item.stop is None and not self.is_finite:
				raise ValueError("If getitem slice.stop is None, the current space must be finite.")
			indexes = list(range(ifnone(item.start, 0), ifnone(item.stop, len(self)), ifnone(item.step, 1)))
			return np.stack(list(map(self.get_single_item, indexes)))
		elif isinstance(item, (int, np.int32, np.int64, np.int)):
			return self.get_single_item(item)
		else:
			raise ValueError(f"The current item type ({type(item)}) is not recognized. Must be Iterable, slice or int.")

	def get_single_item(self, item: int):
		assert np.asarray(item).ndim == 0, "item must be a scalar."
		indices = [
			int((item / functools.reduce(operator.mul, self._sizes[i + 1:], 1)) % self._sizes[i])
			for i in range(len(self._sizes))
		]
		return np.array([self.values_matrix[i][idx] for i, idx in enumerate(indices)])

	def get_random_subspace(
			self,
			nb_points: int,
			re_indexes: bool = True
	) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
		if self.is_finite:
			assert nb_points <= self._length
			indexes = np.sort(np.random.randint(0, self._length, size=nb_points))
			points = self[indexes]
		else:
			if re_indexes:
				raise NotImplementedError("re_indexes is not implement yet for non-finite space.")
			elements = np.asarray([np.random.choice(a, size=nb_points, replace=True) for a in self.values_matrix])
			points = elements.transpose()
		if re_indexes:
			return points, indexes
		else:
			return points

	def _set_default_kwargs_(self):
		self._kwargs.setdefault("nb_workers", 1)
		if self._is_using_umap:
			self._kwargs.setdefault("umap_n_components", 2)
			self._kwargs.setdefault("umap_metric", "euclidean")
			self._kwargs.setdefault("umap_random_state", 42)
			self._kwargs.setdefault("low_memory", False)
			self._kwargs.setdefault("nb_max_umap_fit_data", 1_000)
			self._kwargs.setdefault("umap_resolution_space", 50)

	def _init_reducer_(self):
		import umap
		self.reducer = umap.UMAP(
			n_components=self._kwargs["umap_n_components"],
			metric=self._kwargs["umap_metric"],
			random_state=self._kwargs["umap_random_state"],
			low_memory=self._kwargs["low_memory"],
			n_jobs=self._kwargs["nb_workers"],
		)

	def _fit_reducer_(self):
		fit_data = self.get_random_subspace(
			min(self._length, self._kwargs["nb_max_umap_fit_data"]),
			re_indexes=False
		)
		fit_data_embeds = self.reducer.fit_transform(fit_data)
		values_matrix_umap = [
			np.linspace(d.min(), d.max(), num=self._kwargs["umap_resolution_space"])
			for i, d in enumerate(fit_data_embeds.transpose())
		]
		self._xx_space_umap = XXSpace(values_matrix_umap, use_umap_for_high_dimensional_space=False, **self._kwargs)

	def transform(self, X: Iterable):
		X = np.asarray(X)
		if self._is_using_umap:
			if X.ndim == 1:
				return self.reducer.transform([X])[0]
			else:
				return self.reducer.transform(X)
		else:
			return X

	def inverse_transform(self, X: Iterable):
		X = np.asarray(X)
		if self._is_using_umap:
			if X.ndim == 1:
				return self.reducer.inverse_transform([X])[0]
			else:
				return self.reducer.inverse_transform(X)
		else:
			return X

	def to_numpy(self):
		return self[:]


if __name__ == '__main__':
	import time

	start_time = time.time()
	values = [
		# np.linspace(0, 1, num=1_000)
		np.arange(0, 1_000)
		for _ in range(16)
	]
	xx = XXSpace(
		values,
		use_umap_for_high_dimensional_space=True,
		umap_resolution_space=1_000,
		nb_max_umap_fit_data=1_000
	)
	print(f"Elapsed time construct: {time.time() - start_time:.2f} [s]")
	start_time = time.time()
	x0 = xx[0:22, 26:34:2]
	print(f"{x0.shape = }, {x0 = }")
	print(f"Elapsed time get: {time.time() - start_time:.2f} [s]")
