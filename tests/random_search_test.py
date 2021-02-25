import unittest
import time
from src.random_search import RandomHpSearch


class TestRandomHpSearch(unittest.TestCase):
    def test_bool_elapse_time(self):
        rnGen = RandomHpSearch({}, max_itr=1, max_seconds=3)
        self.assertEqual(bool(rnGen), True)
        time.sleep(3)
        self.assertEqual(bool(rnGen), False, f"gen elapse time: {rnGen.elapse_time}")


if __name__ == '__main__':
    unittest.main()
