import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from datetime import datetime

class DiffusionModel:
    def __init__(self):
        self.a = None
        self.v = None
        self.t0 = None

    def randomly_select_parameters(self):
        self.a = np.random.uniform(0.5, 2) # boundary separation, cautiousness of decision making
        self.v = np.random.uniform(0.5, 2) # drift rate, speed of evidence accumulation
        self.t0 = np.random.uniform(0.1, 0.5) # non-decision time, speed of response
        return self.a, self.v, self.t0

    def forward(self, a, v, t0):
        # parameters: a, v, t0
        # forward equation to compute summary stats from parameters
        y = np.exp(-a * v) 

        R_pred = 1 / (y + 1) # predicted accuracy
        M_pred = t0 + (a / (2 * v)) * ((1 - y) / (1 + y)) # predicted mean response time
        V_pred = (a / (2 * v**3)) * ((1 - 2*a*v*y - y**2) / ((y + 1)**2)) # predicted variance of RT
        return R_pred, M_pred, V_pred

    def inverse(self, R_obs, M_obs, V_obs):
        # inverse equation to estimate parameters from summary stats
        # visualize spread of data

        # edge cases
        if R_obs <= 0.5 or R_obs >= 1.0:
            raise ValueError("Accuracy must be between 0.5 and 1.0")
        if V_obs <= 0:
            raise ValueError("Variance must be positive")
        
        # log transform to visualize spread
        L = np.log(R_obs / (1 - R_obs))

        # est drift rate
        v_term = (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / V_obs
        v_est = np.sign(R_obs - 0.5) * np.sqrt(abs(v_term))
        # est boundary
        a_est = L / v_est
        # est non-decision time
        t_term = (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
        t_est = M_obs - t_term

        return a_est, v_est, t_est
    def simulate_trial_data(self, a, v, t0, n_trials):
        # simulate data for a single trial from model parameters
        # returns array of accuracy and RT's
        # ex: acc = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        # ex: rt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        R_pred, M_pred, V_pred = self.forward(a, v, t0)

        # Accuracy ~ Binomial(R_pred, n_trials)
        # Generate binary accuracy outcomes (0 or 1)
        acc = np.random.binomial(1, R_pred, n_trials)
        
        # RT ~ Normal(M_pred, V_pred)
        # Generate RT values for correct and incorrect trials
        # add sampling variability
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred/n_trials))
        if n_trials > 1:
            V_obs = V_pred * (n_trials -1) / np.random.chisquare(n_trials - 1)
        else:
            V_obs = V_pred
        
        rt = np.random.normal(M_obs, np.sqrt(V_obs), n_trials)
        # Ensure all RTs are at least t0
        rt = np.maximum(rt, t0)
        
        return acc, rt
