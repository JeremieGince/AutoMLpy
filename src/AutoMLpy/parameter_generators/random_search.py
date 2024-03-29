import numpy as np
from typing import Dict, Union, List, Iterable
from .parameter_generator import ParameterGenerator


class RandomHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 **kwargs):
        """
        Used to generate the hyper-parameter (hp) score_space, generate random trial parameter for the exploration and get the
        best set of hp of the hp score_space according to the current exploration.

        Parameters
        ----------
        values_dict:
            A dictionary which contained all the possible values of each hyper-parameter
            used to generate the exploring score_space.
        kwargs: {
                    xi (float): Exploration parameter. Must be in [0, 1]. default: 0.1.
                    Lambda (float): Default: 1.0.
                    bandwidth (float): lenght_scale of the RBF kernel used in self.gpr. Default: 1.0.
                    max_itr (int): Max iteration of the gpo. Default: 30.
                    max_seconds (int): Max seconds that the gpo can take to make its optimisation. Default: 60**2.
        }

        """
        super(RandomHpSearch, self).__init__(values_dict, **kwargs)
        self._idx_choices = list(range(self.xx.shape[0]))
        np.random.shuffle(self._idx_choices)

    @ParameterGenerator.Decorators.increment_counters
    def get_trial_param(self, worker_id: int = 0) -> Dict[Union[str, int], object]:
        """
        Returned a set of trial parameter.
        """
        idx = self._idx_choices.pop(0)

        if len(self._idx_choices) == 0:
            self._idx_choices = list(range(self.xx.shape[0]))
            np.random.shuffle(self._idx_choices)

        t_param = self.convert_subspace_to_param(self.xx[idx])
        return t_param
