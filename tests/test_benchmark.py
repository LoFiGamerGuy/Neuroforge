import unittest

from neuroforge.benchmark import benchmark
from neuroforge.scenarios import TEST_SCENARIOS, TRAIN_SCENARIOS


class BenchmarkTests(unittest.TestCase):
    def test_all_required_policies_and_metrics(self):
        summary,raw=benchmark([1],5)
        self.assertEqual(set(summary),{"oracle","rules","statistical","classifier","gru"})
        for metrics in summary.values():
            for key in ("completion_rate","severe_failure_rate","total_cost_mean","completion_time_p95"):
                self.assertIn(key,metrics)
        self.assertTrue(raw)

    def test_holdouts_are_structurally_disjoint(self):
        self.assertTrue({s.family for s in TRAIN_SCENARIOS}.isdisjoint({s.family for s in TEST_SCENARIOS}))
        self.assertTrue({s.mechanism for s in TRAIN_SCENARIOS}.isdisjoint({s.mechanism for s in TEST_SCENARIOS}))


if __name__ == "__main__": unittest.main()
