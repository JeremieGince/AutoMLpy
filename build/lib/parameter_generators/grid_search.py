from typing import Dict, Union, List, Iterable
from AutoMLpy.parameter_generators.parameter_generator import ParameterGenerator


class GridHpSearch(ParameterGenerator):
    def __init__(self,
                 values_dict: Union[Dict[Union[int, str], List[Union[int, float]]],
                                    Dict[Union[int, str], Iterable]],
                 **kwargs):
        super(GridHpSearch, self).__init__(values_dict, **kwargs)
        self.idx = 0

    def reset(self):
        super(GridHpSearch, self).reset()
        self.idx = 0

    def __len__(self):
        return max(self.max_itr, len(self.xx))

    @ParameterGenerator.Decorators.increment_counters
    def get_trial_param(self):
        self.idx = self.current_itr % len(self.xx)
        t_param = self.convert_subspace_to_param(self.xx[self.idx])
        return t_param
