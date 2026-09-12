import unittest
import numpy as np

from neuroforge.benchmark import benchmark, collect_training
from neuroforge.policies import GRUPolicy
from neuroforge.scenarios import TEST_SCENARIOS, TRAIN_SCENARIOS


class BenchmarkTests(unittest.TestCase):
    def test_all_required_policies_and_metrics(self):
        summary,raw=benchmark([1],5)
        self.assertEqual(set(summary),{"oracle","rules","statistical","classifier","gru","gru_no_memory"})
        for metrics in summary.values():
            for key in ("completion_rate","severe_failure_rate","total_cost_mean","completion_time_p95"):
                self.assertIn(key,metrics)
            self.assertIn("seed_cluster_ci95",metrics)
        self.assertTrue(raw)

    def test_holdouts_are_structurally_disjoint(self):
        self.assertTrue({s.family for s in TRAIN_SCENARIOS}.isdisjoint({s.family for s in TEST_SCENARIOS}))
        self.assertTrue({s.mechanism for s in TRAIN_SCENARIOS}.isdisjoint({s.mechanism for s in TEST_SCENARIOS}))

    def test_training_targets_cover_all_actions(self):
        sequences=collect_training([1],12)
        self.assertEqual({int(y) for seq in sequences for _,y in seq},{0,1,2})

    def test_gru_recurrent_core_is_trained(self):
        sequences=collect_training([1],12); model=GRUPolicy(seed=5)
        before=model.params["Uz"].copy(); model.fit(sequences,epochs=2)
        self.assertFalse(np.array_equal(before,model.params["Uz"]))


if __name__ == "__main__": unittest.main()
