import enum
from src.parameter_generators.grid_search import GridHpSearch
from src.parameter_generators.random_search import RandomHpSearch
from src.parameter_generators.gp_search import GPOHpSearch


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
