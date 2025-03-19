import unittest
import numpy as np
from src.diffusion_model import DiffusionModel
from src.simulation import Simulation

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = DiffusionModel()
        self.sim = Simulation()
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
        a, v, t = 1.0, 1.0, 0.3  # Test parameters
        R_pred, M_pred, V_pred = self.model.forward(a, v, t)
        n_trials = 1000 
        
        acc, rt = self.model.simulate_trial_data(a, v, t, n_trials)
        self.assertEqual(len(acc), n_trials)
        self.assertEqual(len(rt), n_trials)
        

        self.assertAlmostEqual(np.mean(acc), R_pred, places=1)
        self.assertAlmostEqual(np.mean(rt), M_pred, places=1)
        self.assertAlmostEqual(np.var(rt), V_pred, delta=V_pred*0.2)
    def test_simulation(self):
        n_trials = 1000  
        n_simulations = 10
        results = self.sim.run_multiple_simulations(n_trials, n_simulations)
        self.assertTrue(len(results) > 0, "No results returned from simulation")
        
        # Use more appropriate tolerances for bias tests
        self.assertAlmostEqual(np.mean(results['a_bias']), 0, delta=0.1)
        self.assertAlmostEqual(np.mean(results['v_bias']), 0, delta=0.1)
        self.assertAlmostEqual(np.mean(results['t0_bias']), 0, delta=0.1)
    
    def test_squared_error(self):
        # Test with increasing sample sizes
        simulation = Simulation()
        results = simulation.run_all_simulations()
        
        # Group by sample size and calculate mean squared error
        mse_by_sample = results.groupby('n_trials')['a_squared_error'].mean()
        
        # Check that MSE decreases with larger sample sizes
        self.assertGreater(mse_by_sample.loc[10], mse_by_sample.loc[40], "Boundary squared error should decrease with larger sample")
        self.assertGreater(mse_by_sample.loc[40], mse_by_sample.loc[4000], "Boundary squared error should decrease with larger sample")
if __name__ == "__main__":
    unittest.main()