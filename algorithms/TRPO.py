import math
import numpy as np
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.optim import Adam
from algorithms.base import BaseAgent
from utils.misc import soft_update, hard_update
from utils.models import PolicyNetwork, ValueNetwork
from utils.models import GaussianPolicy, DoubleQNetwork
import matplotlib.pyplot as plt

from alfred.utils.plots import plot_curves
from alfred.utils.recorder import remove_nones

class TRPO(BaseAgent):
    def __init__(self, observation_space, action_space, config, logger, n_critics):
        super().__init__(observation_space, action_space, config)

        self.gamma = config.gamma
        self.lam = config.gae_lambda
        self.delta = config.trust_region_delta
        self.damping_coeff = config.conjugate_gradient_damping
        self.cg_iterations = config.cg_iterations
        self.max_backtracking_steps = config.max_backtracking_steps
        self.backtracking_coeff = config.backtracking_coeff

        self.policy = PolicyNetwork(self.obs_dim, self.act_dim, config.hidden_size, action_space).to(self.device)
        self.value_function = ValueNetwork(self.obs_dim, config.hidden_size).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.lr)
        self.value_optimizer = Adam(self.value_function.parameters(), lr=config.lr)
