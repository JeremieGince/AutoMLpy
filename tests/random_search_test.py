import unittest
import time
from AutoMLpy.parameter_generators.random_search import RandomHpSearch


class TestRandomHpSearch(unittest.TestCase):
    def test_bool_elapse_time(self):
        rnGen = RandomHpSearch({}, max_itr=1, max_seconds=3)
        self.assertEqual(bool(rnGen), True)
        time.sleep(3)
        self.assertEqual(bool(rnGen), False, f"gen elapse time: {rnGen.elapse_time}")

    def test_increment_counters(self):
        rnGen = RandomHpSearch({"x0": range(0, 10)}, max_itr=1)
        self.assertEqual(rnGen.current_itr, 0)
        self.assertEqual(rnGen.elapse_time_per_iteration, {})
        time.sleep(0.5)
        rnGen.get_trial_param()

        self.assertEqual(rnGen.current_itr, 1, f"rnGen.current_itr: {rnGen.current_itr}")
        self.assertTrue(rnGen.last_itr_elapse_time > 0, f"rnGen.last_itr_elapse_time: {rnGen.last_itr_elapse_time}")
        self.assertTrue(rnGen.elapse_time_per_iteration[rnGen.current_itr-1] > 0,
                        f"rnGen.elapse_time_per_iteration[rnGen.current_itr]: "
                        f"{rnGen.elapse_time_per_iteration[rnGen.current_itr-1]}")


if __name__ == '__main__':
    unittest.main()
