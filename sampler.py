# Implements of the SOMA, Ran-IMwG and Sys-IMwG

import torch
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(self, n_components, task, device=torch.device('cpu'), tqdm=True):
        self.n_components = n_components
        self.n_dim = task.n_dim
        self.proposal_dist = task.proposal_dist
        self.weight_func = task.get_log_weight
        self.calc_weight_by_summary = task.calc_weight_by_summary
        if self.calc_weight_by_summary:
            self.summary_func = task.get_summary
            self.combind_func = task.get_log_weight_by_summary
        self.calc_weight_fast = task.calc_weight_fast
        if self.calc_weight_fast:
            self.weight_fast_func = task.get_log_weight_fast
        self.device = device
        self.tqdm = tqdm
        self.postfix_update = 50
        self.unif_dist = torch.distributions.Uniform(torch.tensor([0.], device=device), torch.tensor([1.], device=device))
        self.early_stop = True  # early stop when both chains are coupled
        self.index_record = False  # record the index of the component to update
        self.index_history = []  # record the index history

    @abstractmethod
    def fit(self, init_state, n_iter):
        pass

    @abstractmethod
    def coupling_fit(self, init_state1, init_state2, n_iter):
        pass


class SomaSampler(Sampler):

    def fit(self, init_state, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        replace_idx = torch.arange(self.n_components).view(-1, 1, 1).long().expand(self.n_components, 1, self.n_dim).to(self.device)
        state = init_state
        all_states = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)  # store all states
        current_log_weight = self.weight_func(state.unsqueeze(0), **func_kwargs)
        n_accept = 0

        if separable:
            # calculate summary for each input
            summary = self.summary_func(state, **func_kwargs)  # shape: (n_components, n_summary)

        for i in range_generator:
            # propose new samples from proposal_dist
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)

            if separable:
                summary_sum = summary.sum(dim=0)  # shape: (n_summary,)
                new_samples_summary = self.summary_func(new_samples, **func_kwargs)
                new_samples_summary_expand = new_samples_summary.expand(self.n_components, -1)
                summary_sum_update = summary_sum.unsqueeze(0).repeat(self.n_components, 1)  # shape: (n_components, n_summary)
                summary_sum_update[torch.arange(self.n_components), :] -= (summary - new_samples_summary_expand)

                # calculate weights
                log_weights = self.combind_func(summary_sum_update, **func_kwargs)
            else:
                if not self.calc_weight_fast:
                    # mix new samples with current samples
                    combined_samples = state.unsqueeze(0).expand(self.n_components, -1, -1).clone()
                    new_samples_expand = new_samples.reshape(1, 1, -1).expand(self.n_components, 1, -1)
                    combined_samples.scatter_(1, replace_idx, new_samples_expand)  # shape: (n_components, n_components, n_dim)

                    # calculate weights
                    log_weights = self.weight_func(combined_samples, **func_kwargs)
                else:
                    log_weights = self.weight_fast_func(state, new_samples, **func_kwargs)

            weights_normalizer = torch.max(log_weights)
            weights = torch.exp(log_weights - weights_normalizer)

            # resample index
            resamp_idx = torch.multinomial(weights, 1, replacement=True)
            if self.index_record:
                self.index_history.append(resamp_idx)

            # accept or reject
            if log_weights[resamp_idx] >= current_log_weight:
                # accept with probability 1
                state[resamp_idx] = new_samples
                current_log_weight = log_weights[resamp_idx]
                n_accept += 1
                if separable:
                    summary[resamp_idx] = new_samples_summary
            else:
                current_weight = torch.exp(current_log_weight - weights_normalizer)
                accept_prob = torch.sum(weights) / (torch.sum(weights) + current_weight - weights[resamp_idx])
                if accept_prob > self.unif_dist.sample((1,)):
                    state[resamp_idx] = new_samples
                    current_log_weight = log_weights[resamp_idx]
                    n_accept += 1
                    if separable:
                        summary[resamp_idx] = new_samples_summary

            all_states[i] = state
            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate': n_accept / (i + 1)})

        return all_states, n_accept / n_iter

    def coupling_fit(self, init_state1, init_state2, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        replace_idx = torch.arange(self.n_components).view(-1, 1, 1).long().expand(self.n_components, 1, self.n_dim).to(self.device)
        distance = torch.zeros(n_iter)

        # Initialize two chains from different initial states
        state1 = init_state1
        state2 = init_state2

        # Store all states
        all_states1 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)
        all_states2 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)

        current_log_weight1 = self.weight_func(state1.unsqueeze(0), **func_kwargs)
        current_log_weight2 = self.weight_func(state2.unsqueeze(0), **func_kwargs)

        n_accept1 = 0
        n_accept2 = 0

        coupled = False  # Track whether the chains have coupled
        coupling_time = n_iter  # Set the initial coupling time as the number of iterations

        if separable:
            summary1 = self.summary_func(state1, **func_kwargs)  # shape: (n_components, n_summary)
            summary2 = self.summary_func(state2, **func_kwargs)

        for i in range_generator:
            chain1_acc_flag = False
            chain2_acc_flag = False

            # Propose new samples for both chains
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)

            if separable:
                summary_sum1 = summary1.sum(dim=0)  # shape: (n_summary,)
                summary_sum2 = summary2.sum(dim=0)

                new_samples_summary = self.summary_func(new_samples, **func_kwargs)
                new_samples_summary_expand = new_samples_summary.expand(self.n_components, -1)
                summary_sum_update1 = summary_sum1.unsqueeze(0).repeat(self.n_components, 1)  # shape: (n_components, n_summary)
                summary_sum_update2 = summary_sum2.unsqueeze(0).repeat(self.n_components, 1)
                summary_sum_update1[torch.arange(self.n_components), :] -= (summary1 - new_samples_summary_expand)
                summary_sum_update2[torch.arange(self.n_components), :] -= (summary2 - new_samples_summary_expand)

                # calculate weights
                log_weights1 = self.combind_func(summary_sum_update1, **func_kwargs)
                log_weights2 = self.combind_func(summary_sum_update2, **func_kwargs)
            else:
                if not self.calc_weight_fast:
                    combined_samples1 = state1.unsqueeze(0).expand(self.n_components, -1, -1).clone()
                    combined_samples2 = state2.unsqueeze(0).expand(self.n_components, -1, -1).clone()
                    new_samples_expand = new_samples.reshape(1, 1, -1).expand(self.n_components, 1, -1)
                    combined_samples1.scatter_(1, replace_idx, new_samples_expand)  # shape: (n_components, n_components, n_dim)
                    combined_samples2.scatter_(1, replace_idx, new_samples_expand)

                    # calculate weights
                    log_weights1 = self.weight_func(combined_samples1, **func_kwargs)
                    log_weights2 = self.weight_func(combined_samples2, **func_kwargs)
                else:
                    log_weights1 = self.weight_fast_func(state1, new_samples, **func_kwargs)
                    log_weights2 = self.weight_fast_func(state2, new_samples, **func_kwargs)

            weights_normalizer1 = torch.logsumexp(log_weights1, dim=0)
            weights_normalizer2 = torch.logsumexp(log_weights2, dim=0)
            unif_sample = self.unif_dist.sample((1,))
            weights1 = torch.exp(log_weights1 - weights_normalizer1)
            weights2 = torch.exp(log_weights2 - weights_normalizer2)
            shared_weights = torch.min(weights1, weights2)
            sum_shared_weights = torch.sum(shared_weights)

            # select index via maximal coupling
            if unif_sample < sum_shared_weights:
                resamp_idx = torch.multinomial(shared_weights, 1)
                resamp_idx1 = resamp_idx
                resamp_idx2 = resamp_idx
            else:
                remaining_weights1 = weights1 - shared_weights
                remaining_weights2 = weights2 - shared_weights

                cunsum1 = torch.cumsum(remaining_weights1, dim=0) + sum_shared_weights
                cunsum2 = torch.cumsum(remaining_weights2, dim=0) + sum_shared_weights
                resamp_idx1 = torch.sum(cunsum1 < unif_sample).long()
                resamp_idx2 = torch.sum(cunsum2 < unif_sample).long()

                # avoid overflows
                if resamp_idx1 == self.n_components:
                    resamp_idx1 = self.n_components - 1
                if resamp_idx2 == self.n_components:
                    resamp_idx2 = self.n_components - 1

            if coupled:
                resamp_idx2 = resamp_idx1

            # Accept or reject for chain 1
            unif_sample_acc = self.unif_dist.sample((1,))
            if log_weights1[resamp_idx1] >= current_log_weight1:
                state1[resamp_idx1] = new_samples
                current_log_weight1 = log_weights1[resamp_idx1]
                n_accept1 += 1
                chain1_acc_flag = True
                if separable:
                    summary1[resamp_idx1] = new_samples_summary
            else:
                current_weight1 = torch.exp(current_log_weight1 - weights_normalizer1)
                accept_prob1 = 1.0 / (1.0 + current_weight1 - weights1[resamp_idx1])  # since torch.sum(weights1) = 1
                if accept_prob1 > unif_sample_acc:
                    state1[resamp_idx1] = new_samples
                    current_log_weight1 = log_weights1[resamp_idx1]
                    n_accept1 += 1
                    chain1_acc_flag = True
                    if separable:
                        summary1[resamp_idx1] = new_samples_summary

            # Accept or reject for chain 2 (unless already coupled)
            if not coupled:
                if log_weights2[resamp_idx2] >= current_log_weight2:
                    state2[resamp_idx2] = new_samples
                    current_log_weight2 = log_weights2[resamp_idx2]
                    n_accept2 += 1
                    chain2_acc_flag = True
                    if separable:
                        summary2[resamp_idx2] = new_samples_summary
                else:
                    current_weight2 = torch.exp(current_log_weight2 - weights_normalizer2)
                    accept_prob2 = 1.0 / (1.0 + current_weight2 - weights2[resamp_idx2])  # since torch.sum(weights2) = 1
                    if accept_prob2 > unif_sample_acc:
                        state2[resamp_idx2] = new_samples
                        current_log_weight2 = log_weights2[resamp_idx2]
                        n_accept2 += 1
                        chain2_acc_flag = True
                        if separable:
                            summary2[resamp_idx2] = new_samples_summary
            else:
                state2 = state1.clone()

            # Check if the chains have coupled
            if torch.all(state1 == state2):
                if not coupled:
                    # print('Chains have coupled at iteration %d' % i)
                    coupled = True
                    coupling_time = i
                    if self.early_stop:
                        all_states1[i:] = state1
                        all_states2[i:] = state2
                        distance[i:] = 0
                        return all_states1, all_states2, coupling_time, n_accept1 / coupling_time, n_accept2 / coupling_time, distance

            if (not coupled) and chain1_acc_flag and chain2_acc_flag:
                # swap the state to the minimun uncoupled state
                uncoupled_idx = torch.where(state1 != state2)[0][0]
                state1[resamp_idx1] = state1[uncoupled_idx].clone()
                state1[uncoupled_idx] = new_samples
                state2[resamp_idx2] = state2[uncoupled_idx].clone()
                state2[uncoupled_idx] = new_samples

            all_states1[i] = state1
            all_states2[i] = state2
            distance[i] = torch.sum(state1 != state2).cpu()

            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate1': n_accept1 / (i + 1), 'accept_rate2': n_accept2 / (i + 1),
                                             'coupled': coupled, 'diff_num': distance[i].cpu().numpy()})

        return all_states1, all_states2, coupling_time, n_accept1 / n_iter, n_accept2 / n_iter, distance


class RandomScanSampler(Sampler):

    def fit(self, init_state, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        state = init_state
        all_states = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)  # store all states
        current_log_weight = self.weight_func(state.unsqueeze(0), **func_kwargs)
        n_accept = 0

        if separable:
            # calculate summary for each input
            summary = self.summary_func(state, **func_kwargs)  # shape: (n_components, n_summary)

        for i in range_generator:
            # propose new samples from proposal_dist
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)
            resamp_idx = torch.multinomial(torch.ones(self.n_components), 1)
            if self.index_record:
                self.index_history.append(resamp_idx)

            if separable:
                summary_sum = summary.sum(dim=0)  # shape: (n_summary,)
                new_samples_summary = self.summary_func(new_samples, **func_kwargs)
                new_log_weights = self.combind_func(summary_sum - summary[resamp_idx] + new_samples_summary, **func_kwargs)
            else:
                new_state = state.clone()
                new_state[resamp_idx] = new_samples

                # calculate weights
                new_log_weights = self.weight_func(new_state.unsqueeze(0), **func_kwargs)

            # accept or reject
            if new_log_weights >= current_log_weight:
                # accept with probability 1
                state[resamp_idx] = new_samples
                current_log_weight = new_log_weights
                n_accept += 1
                if separable:
                    summary[resamp_idx] = new_samples_summary
            else:
                accept_prob = torch.exp(new_log_weights - current_log_weight)
                if accept_prob > self.unif_dist.sample((1,)):
                    state[resamp_idx] = new_samples
                    current_log_weight = new_log_weights
                    n_accept += 1
                    if separable:
                        summary[resamp_idx] = new_samples_summary

            all_states[i] = state
            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate': n_accept / (i + 1)})

        return all_states, n_accept / n_iter

    def coupling_fit(self, init_state1, init_state2, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        distance = torch.zeros(n_iter)

        # Initialize two chains from different initial states
        state1 = init_state1
        state2 = init_state2

        # Store all states
        all_states1 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)
        all_states2 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)

        current_log_weight1 = self.weight_func(state1.unsqueeze(0), **func_kwargs)
        current_log_weight2 = self.weight_func(state2.unsqueeze(0), **func_kwargs)

        n_accept1 = 0
        n_accept2 = 0

        coupled = False  # Track whether the chains have coupled
        coupling_time = n_iter  # Set the initial coupling time as the number of iterations

        if separable:
            summary1 = self.summary_func(state1, **func_kwargs)  # shape: (n_components, n_summary)
            summary2 = self.summary_func(state2, **func_kwargs)

        for i in range_generator:
            # Propose new samples for both chains
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)
            resamp_idx = torch.multinomial(torch.ones(self.n_components), 1)

            if separable:
                summary_sum1 = summary1.sum(dim=0)  # shape: (n_summary,)
                summary_sum2 = summary2.sum(dim=0)

                new_samples_summary = self.summary_func(new_samples, **func_kwargs)
                new_log_weights1 = self.combind_func(summary_sum1 - summary1[resamp_idx] + new_samples_summary, **func_kwargs)
                new_log_weights2 = self.combind_func(summary_sum2 - summary2[resamp_idx] + new_samples_summary, **func_kwargs)
            else:
                new_state1 = state1.clone()
                new_state2 = state2.clone()

                new_state1[resamp_idx] = new_samples
                new_state2[resamp_idx] = new_samples

                new_log_weights1 = self.weight_func(new_state1.unsqueeze(0), **func_kwargs)
                new_log_weights2 = self.weight_func(new_state2.unsqueeze(0), **func_kwargs)

            # Accept or reject for chain 1
            unif_sample = self.unif_dist.sample((1,))
            if new_log_weights1 >= current_log_weight1:
                state1[resamp_idx] = new_samples
                current_log_weight1 = new_log_weights1
                n_accept1 += 1
                if separable:
                    summary1[resamp_idx] = new_samples_summary
            else:
                accept_prob1 = torch.exp(new_log_weights1 - current_log_weight1)
                if accept_prob1 > unif_sample:
                    state1[resamp_idx] = new_samples
                    current_log_weight1 = new_log_weights1
                    n_accept1 += 1
                    if separable:
                        summary1[resamp_idx] = new_samples_summary

            # Accept or reject for chain 2 (unless already coupled)
            if not coupled:
                if new_log_weights2 >= current_log_weight2:
                    state2[resamp_idx] = new_samples
                    current_log_weight2 = new_log_weights2
                    n_accept2 += 1
                    if separable:
                        summary2[resamp_idx] = new_samples_summary
                else:
                    accept_prob2 = torch.exp(new_log_weights2 - current_log_weight2)
                    if accept_prob2 > unif_sample:
                        state2[resamp_idx] = new_samples
                        current_log_weight2 = new_log_weights2
                        n_accept2 += 1
                        if separable:
                            summary2[resamp_idx] = new_samples_summary
            else:
                state2 = state1.clone()

            # Check if the chains have coupled
            if torch.all(state1 == state2):
                if not coupled:
                    coupled = True
                    coupling_time = i
                    if self.early_stop:
                        all_states1[i:] = state1
                        all_states2[i:] = state2
                        distance[i:] = 0
                        return all_states1, all_states2, coupling_time, n_accept1 / coupling_time, n_accept2 / coupling_time, distance

            all_states1[i] = state1
            all_states2[i] = state2
            distance[i] = torch.sum(state1 != state2).cpu()

            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate1': n_accept1 / (i + 1), 'accept_rate2': n_accept2 / (i + 1),
                                             'coupled': coupled, 'diff_num': distance[i].cpu().numpy()})

        return all_states1, all_states2, coupling_time, n_accept1 / n_iter, n_accept2 / n_iter, distance


class SystematicScanSampler(Sampler):

    def fit(self, init_state, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        state = init_state
        all_states = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)  # store all states
        current_log_weight = self.weight_func(state.unsqueeze(0), **func_kwargs)
        n_accept = 0

        if separable:
            # calculate summary for each input
            summary = self.summary_func(state, **func_kwargs)  # shape: (n_components, n_summary)

        j = 0  # index to track which component to update

        for i in range_generator:

            # update j
            j = (j + 1) % self.n_components
            if self.index_record:
                self.index_history.append(j)

            # propose new samples from proposal_dist
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)

            if separable:
                summary_sum = summary.sum(dim=0)
                new_samples_summary = self.summary_func(new_samples, **func_kwargs)
                new_log_weights = self.combind_func(summary_sum - summary[j] + new_samples_summary, **func_kwargs)
            else:
                new_state = state.clone()
                new_state[j] = new_samples

                # calculate weights
                new_log_weights = self.weight_func(new_state.unsqueeze(0), **func_kwargs)

            # accept or reject
            if new_log_weights >= current_log_weight:
                # accept with probability 1
                state[j] = new_samples
                current_log_weight = new_log_weights
                n_accept += 1
                if separable:
                    summary[j] = new_samples_summary
            else:
                accept_prob = torch.exp(new_log_weights - current_log_weight)
                if accept_prob > self.unif_dist.sample((1,)):
                    state[j] = new_samples
                    current_log_weight = new_log_weights
                    n_accept += 1
                    if separable:
                        summary[j] = new_samples_summary

            all_states[i] = state
            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate': n_accept / (self.n_components * (i + 1))})

        return all_states, n_accept / n_iter

    def coupling_fit(self, init_state1, init_state2, n_iter, separable=True, **func_kwargs):
        separable = False if not self.calc_weight_by_summary else separable
        range_generator = range(n_iter) if not self.tqdm else tqdm(range(n_iter))
        distance = torch.zeros(n_iter)

        # Initialize two chains from different initial states
        state1 = init_state1
        state2 = init_state2

        # Store all states
        all_states1 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)
        all_states2 = torch.zeros(n_iter, self.n_components, self.n_dim).to(self.device)

        current_log_weight1 = self.weight_func(state1.unsqueeze(0), **func_kwargs)
        current_log_weight2 = self.weight_func(state2.unsqueeze(0), **func_kwargs)

        n_accept1 = 0
        n_accept2 = 0

        coupled = False  # Track whether the chains have coupled
        coupling_time = n_iter  # Set the initial coupling time as the number of iterations

        if separable:
            summary1 = self.summary_func(state1, **func_kwargs)  # shape: (n_components, n_summary)
            summary2 = self.summary_func(state2, **func_kwargs)

        j = 0  # index to track which component to update

        for i in range_generator:

            # upadte j
            j = (j + 1) % self.n_components

            # Propose new samples for both chains
            new_samples = self.proposal_dist.sample()  # shape: (1, n_dim)

            if separable:
                summary_sum1 = summary1.sum(dim=0)  # shape: (n_summary,)
                summary_sum2 = summary2.sum(dim=0)

                new_samples_summary = self.summary_func(new_samples, **func_kwargs)

                new_log_weights1 = self.combind_func(summary_sum1 - summary1[j] + new_samples_summary, **func_kwargs)
                new_log_weights2 = self.combind_func(summary_sum2 - summary2[j] + new_samples_summary, **func_kwargs)
            else:
                new_state1 = state1.clone()
                new_state2 = state2.clone()

                new_state1[j] = new_samples
                new_state2[j] = new_samples

                new_log_weights1 = self.weight_func(new_state1.unsqueeze(0), **func_kwargs)
                new_log_weights2 = self.weight_func(new_state2.unsqueeze(0), **func_kwargs)

            # Accept or reject for chain 1
            unif_sample = self.unif_dist.sample((1,))
            if new_log_weights1 >= current_log_weight1:
                state1[j] = new_samples
                current_log_weight1 = new_log_weights1
                n_accept1 += 1
                if separable:
                    summary1[j] = new_samples_summary
            else:
                accept_prob1 = torch.exp(new_log_weights1 - current_log_weight1)
                if accept_prob1 > unif_sample:
                    state1[j] = new_samples
                    current_log_weight1 = new_log_weights1
                    n_accept1 += 1
                    if separable:
                        summary1[j] = new_samples_summary

            # Accept or reject for chain 2 (unless already coupled)
            if not coupled:
                if new_log_weights2 >= current_log_weight2:
                    state2[j] = new_samples
                    current_log_weight2 = new_log_weights2
                    n_accept2 += 1
                    if separable:
                        summary2[j] = new_samples_summary
                else:
                    accept_prob2 = torch.exp(new_log_weights2 - current_log_weight2)
                    if accept_prob2 > unif_sample:
                        state2[j] = new_samples
                        current_log_weight2 = new_log_weights2
                        n_accept2 += 1
                        if separable:
                            summary2[j] = new_samples_summary
            else:
                state2 = state1.clone()

            # Check if the chains have coupled
            if torch.all(state1 == state2):
                if not coupled:
                    coupled = True
                    coupling_time = i
                    if self.early_stop:
                        all_states1[i:] = state1
                        all_states2[i:] = state2
                        distance[i:] = 0
                        return all_states1, all_states2, coupling_time, n_accept1 / coupling_time, n_accept2 / coupling_time, distance

            all_states1[i] = state1
            all_states2[i] = state2
            distance[i] = torch.sum(state1 != state2).cpu()

            if i % (n_iter // self.postfix_update) == 0 and self.tqdm:
                range_generator.set_postfix({'accept_rate1': n_accept1 / (i + 1), 'accept_rate2': n_accept2 / (i + 1),
                                             'coupled': coupled, 'diff_num': distance[i].cpu().numpy()})

        return all_states1, all_states2, coupling_time, n_accept1 / n_iter, n_accept2 / n_iter, distance
