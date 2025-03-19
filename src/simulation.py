# simulate experiment with diffusion model
import numpy as np
from diffusion_model import DiffusionModel

class Simulation:
    def __init__(self):
        self.diffusion_model = DiffusionModel()
        self.n_trails = [10, 40, 4000]
    def simulate_data(self, a, v, t, n_trials):
        for i in range(n_trials):
            self.diffusion_model.randomly_select_parameters()
            R_pred, M_pred, V_pred = self.diffusion_model.forward(a, v, t)
            