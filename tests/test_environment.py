import unittest
import numpy as np

from neuroforge.environment import WorkflowEnv
from neuroforge.scenarios import TEST_SCENARIOS
from neuroforge.types import Action
from neuroforge.policies import OraclePolicy


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

    def test_identical_current_observation_requires_different_history_action(self):
        retry=TEST_SCENARIOS[-2]; escalate=TEST_SCENARIOS[-1]
        a=WorkflowEnv(retry,77); b=WorkflowEnv(escalate,77); oa=a.reset(); ob=b.reset()
        early=[]
        for _ in range(retry.onset+1):
            early.append((oa.context_signal,ob.context_signal)); oa,_=a.step(Action.WAIT); ob,_=b.step(Action.WAIT)
        self.assertIn((1.0,-1.0),early)
        self.assertTrue(np.array_equal(oa.vector(),ob.vector()))
        pa=OraclePolicy(lambda:a.hidden_condition); pb=OraclePolicy(lambda:b.hidden_condition)
        self.assertEqual(pa.act(oa),Action.RETRY); self.assertEqual(pb.act(ob),Action.ESCALATE)


if __name__ == "__main__": unittest.main()
