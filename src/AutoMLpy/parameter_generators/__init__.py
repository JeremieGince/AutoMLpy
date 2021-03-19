import enum
from .grid_search import GridHpSearch
from .random_search import RandomHpSearch
from .gp_search import GPOHpSearch
from .parameter_generator import ParameterGenerator


class SearchType(enum.Enum):
    __order__ = "Grid Random GPO"
    Grid = 0
    Random = 1
    GPO = 2


search_type_2_type = {
    SearchType.Grid: GridHpSearch,
    SearchType.Random: RandomHpSearch,
    SearchType.GPO: GPOHpSearch,
}
