import enum
from .grid_search import GridHpSearch
from .random_search import RandomHpSearch
from .gp_search import GPOHpSearch
from .forest_search import RandomForestEpsilonGreedySearch
from .parameter_generator import ParameterGenerator


class SearchType(enum.Enum):
    __order__ = "Grid Random GPO Forest"
    Grid = 0
    Random = 1
    GPO = 2
    Forest = 3


search_type_2_type = {
    SearchType.Grid: GridHpSearch,
    SearchType.Random: RandomHpSearch,
    SearchType.GPO: GPOHpSearch,
    SearchType.Forest: RandomForestEpsilonGreedySearch,
}
