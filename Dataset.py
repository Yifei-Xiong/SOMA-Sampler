# simulators for different datasets

import os
import torch


class Task:
    def __init__(self, task_type, seed=0, eps=1.0, n_component=10, device=torch.device('cpu'), args=None):
        if os.name == 'posix':  # input File Save Path here
            self.FileSavePath = ""
        else:
            self.FileSavePath = ""
        self.task_type = task_type
        self.dim_theta = 0
        self.dim_x = 0
        self.seed = seed
        self.eps = eps
        self.n_component = n_component
        self.device = device
        self.true_theta = None
        self.args = args
        self.target_density = None
        self.calc_weight_by_summary = False  # whether this task can calculate weight by summary
        self.calc_weight_fast = False  # whether this task can calculate weight using fast method

        if self.task_type == 1:
            # toy example
            assert self.n_component == 2
            self.n_dim = 1
            base_dist_type = 'beta'
            # base_dist_type = 'exponential'
            if base_dist_type == 'beta':
                self.proposal_dist = torch.distributions.Beta(torch.tensor([10.]).to(self.device), torch.tensor([10.]).to(self.device))
            elif base_dist_type == 'exponential':
                self.proposal_dist = torch.distributions.Exponential(torch.tensor([1.]).to(self.device))
            else:
                raise NotImplementedError
            self.lap_params = torch.tensor(0.5 / self.eps).to(self.device)
            self.sdp = torch.tensor([0.5], device=self.device)

        elif self.task_type == 2:
            # perturbed histograms
            self.n_dim = 1
            self.n_bins = args.hist_bin
            self.bin_range = [0, 1]
            self.bin_edges = torch.linspace(0, 1, steps=self.n_bins + 1, device=self.device)
            # true density: beta(10, 10)
            self.true_density = torch.distributions.Beta(torch.tensor([10.]).to(self.device), torch.tensor([10.]).to(self.device))
            # proposal: Unif(0, 1)
            self.proposal_dist = torch.distributions.Uniform(torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device))
            data = self.true_density.sample((self.n_component,)).squeeze()
            counts = torch.histc(data, bins=self.n_bins, min=self.bin_range[0], max=self.bin_range[1])
            lap_dist = torch.distributions.Laplace(torch.tensor([0.]), torch.tensor(2.0 / self.eps))
            noise = lap_dist.sample((self.n_bins,)).squeeze().to(self.device)
            self.obs = (counts + noise) / self.n_component
            self.calc_weight_by_summary = True

        else:
            raise NotImplementedError

    def get_log_weight(self, input):
        # input shape: (batch, n_component, n_dim)
        # output shape: (batch,)
        batch_size = input.shape[0]
        if self.task_type == 1:
            s = torch.mean(input, dim=1)  # shape: (batch, n_dim)
            log_weights = - (torch.abs(s - self.sdp) / self.lap_params).squeeze(-1)  # shape: (batch,)
            return log_weights
        elif self.task_type == 2:
            log_weights = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                probs = torch.histc(input[i], bins=self.n_bins, min=self.bin_range[0], max=self.bin_range[1]) / self.n_component
                log_weights[i] = torch.sum(- torch.abs(probs - self.obs) * self.eps * self.n_component / 2.0)
            return log_weights
        else:
            raise NotImplementedError

    def get_log_weight_fast(self, current_state, proposed_state):
        if self.task_type == 1:
            raise NotImplementedError
        elif self.task_type == 2:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_summary(self, input):
        # input shape: (n_component, n_dim)
        # output shape: (n_component, n_summary)
        batch_size = input.shape[0]
        if self.task_type == 1:
            raise NotImplementedError
        elif self.task_type == 2:
            summaries = torch.zeros(batch_size, self.n_bins, device=self.device)
            for i in range(batch_size):
                summaries[i] = torch.histc(input[i], bins=self.n_bins, min=self.bin_range[0], max=self.bin_range[1]) / self.n_component
            return summaries
        else:
            raise NotImplementedError

    def get_log_weight_by_summary(self, summary):
        # input shape: (batch, n_summary)
        # output shape: (batch,)
        batch_size = summary.shape[0]
        if self.task_type == 1:
            raise NotImplementedError
        elif self.task_type == 2:
            log_weights = torch.sum(- torch.abs(summary - self.obs) * self.eps * self.n_component / 2.0, dim=1)
            return log_weights
        else:
            raise NotImplementedError
