import unittest

from neuroforge.environment import WorkflowEnv
from neuroforge.scenarios import TEST_SCENARIOS
from neuroforge.types import Action


class EnvironmentTests(unittest.TestCase):
    def test_determinism(self):
        def rollout():
            e=WorkflowEnv(TEST_SCENARIOS[0],123); obs=e.reset(); xs=[]
            while not e.done:
                xs.append(obs.vector().tolist()); obs,_=e.step(Action.WAIT)
            return xs,e.result()
        self.assertEqual(rollout(),rollout())

    def test_actions_change_trajectory(self):
        a=WorkflowEnv(TEST_SCENARIOS[2],9); b=WorkflowEnv(TEST_SCENARIOS[2],9)
        a.reset(); b.reset()
        for _ in range(12): a.step(Action.WAIT)
        for _ in range(9): b.step(Action.WAIT)
        b.step(Action.ESCALATE)
        self.assertNotEqual(a.result(),b.result())

    def test_observation_has_no_hidden_label(self):
        fields=set(WorkflowEnv(TEST_SCENARIOS[0],1).reset().__dataclass_fields__)
        self.assertNotIn("condition",fields); self.assertNotIn("hidden",fields)


if __name__ == "__main__": unittest.main()

