import unittest
import numpy as np
from src.diffusion_model import DiffusionModel

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = DiffusionModel()
    def test_setup(self):
        self.assertIsNotNone(self.model)
        # generate random parameters and test that they're within range
        self.model.randomly_select_parameters()
        self.assertTrue(0.5 <= self.model.a <= 2, f"Boundary a={self.model.a} outside range")
        self.assertTrue(0.5 <= self.model.v <= 2, f"Drift rate v={self.model.v} outside range")
        self.assertTrue(0.1 <= self.model.t0 <= 0.5, f"Non-decision time t0={self.model.t0} outside range")
    def test_forward(self):
        a, v, t = 1.0, 1.0, 0.3 # random params
        R_pred, M_pred, V_pred = self.model.forward(a, v, t)
        self.assertTrue(0 < R_pred < 1, f"Response time R_pred={R_pred} outside range")
        self.assertTrue(M_pred > t, f"Mean response time M_pred={M_pred} not greater than t={t}")
        self.assertTrue(V_pred > 0, f"Variance {V_pred} not positive")
    def test_simulate_trial_data(self):
        a, v, t = 1.0, 1.0, 0.3 # random params
        R_pred, M_pred, V_pred = self.model.forward(a, v, t)
        n_trials = 10
        acc, rt = self.model.simulate_trial_data(a, v, t, n_trials)
        self.assertEqual(len(acc), n_trials)
        self.assertEqual(len(rt), n_trials)
        self.assertAlmostEqual(np.sum(acc) / n_trials, R_pred, places=1)
        self.assertAlmostEqual(np.mean(rt), M_pred, places=1)
        self.assertAlmostEqual(np.var(rt), V_pred, places=1)