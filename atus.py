import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import random
import copy
import argparse
import yaml
from torch.xpu import device
from tqdm import tqdm
from scipy.stats import wasserstein_distance_nd, wasserstein_distance, laplace


def multinomial_by_inverse_sampling(weights):
    cdf = torch.cumsum(weights, dim=0)
    u = torch.rand(1, device=weights.device) * cdf[-1]
    return torch.searchsorted(cdf, u)

def multinomial_by_gumbel(weights):
    log_weights = torch.log(weights + 1e-9)
    uniform_samples = torch.rand_like(log_weights)
    gumbel_noise = -torch.log(-torch.log(uniform_samples + 1e-9))
    return torch.argmax(log_weights + gumbel_noise)

def gen_data(theta, num_samples):
    # theta: batch * dim_theta
    # num_samples: int
    # return: batch * num_samples * p
    dir_dist = torch.distributions.Dirichlet(theta)
    return dir_dist.sample((num_samples, )).transpose(0, 1)  # return untruncated data

def gen_s(data_input, clamp_lower=0.0006944, clamp_upper=1.):
    # data: batch * n * p
    # return: batch * dim_s=p
    batch = data_input.shape[0]
    # truncation
    data_input_trunc = data_input.clone()
    data_input_trunc.clamp_(min=clamp_lower, max=clamp_upper)  # shape: (batch, n, p)
    return torch.sum(torch.log(data_input_trunc), dim=1)  # shape: (batch, dim_s)

def main(args, config):
    # init args
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    FileSavePath = config['FileSavePath_linux'] if os.name == 'posix' else config['FileSavePath_win']
    # print("File Save Path: " + FileSavePath)
    method_colors = ['#EE9B00', '#01899D', '#565B7F']
    params_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    if args.gpu == 0:
        args.device = torch.device('cpu')
        # print('using cpu')
    else:
        args.device = torch.device("cuda:" + str(args.gpu - 1))
        # print('using gpu: %d' % args.gpu)
    ckpts_folder_name = "ckpts"
    if args.data == 0:
        print("load combined data")
        args.ncomp = 6656  # for ATUS data, n = 6656
        ckpt_str = ""
    elif args.data == 1:
        print("load male data")
        args.ncomp = 3528  # for ATUS data, n = 3528
        ckpt_str = "_male"
    elif args.data == 2:
        print("load female data")
        args.ncomp = 3128  # for ATUS data, n = 3128
        ckpt_str = "_female"
    else:
        raise NotImplementedError("No data with id %d." % args.data)
    ModelInfo = "mk+" + args.mkstr + "_eps+" + str(args.eps) + "_ncomp+" + str(args.ncomp) + "_me+" + \
                str(args.method) + "_data+" + str(args.data) + "_step+" + str(args.step) + "_seed+" + str(args.seed)
    if args.method == 1:
        sampler_name = "SOMA"
    elif args.method == 2:
        sampler_name = "Random-scan"
    elif args.method == 3:
        sampler_name = "Systematic-scan"
    else:
        raise NotImplementedError
    print("using sampler: " + sampler_name)

    # define simulator
    p = 3  # dimension of X, 3
    n = args.ncomp  # number_of_records, 100
    dim_theta = 3

    # a = np.min(x_true)  # Find truncation level
    a = 0.0006944
    args.scale = -p * np.log(a) / (n * args.eps)


    # generate noisy summaries
    if False:
        xF = pd.read_csv("data/female.csv")
        xM = pd.read_csv("data/male.csv")
        x_true = pd.concat([xF, xM]).sample(n=6656, random_state=args.seed).to_numpy()
        print("current seed: %d" % args.seed)
        laplace_dist = torch.distributions.Laplace(torch.tensor([0.], device=args.device),
                                                   torch.tensor(args.scale, device=args.device))
        noise_SS = laplace_dist.sample((p,)).cpu().squeeze().numpy()  # p * 1
        nSS = torch.from_numpy(np.mean(np.log(a * (x_true < a) + x_true * (x_true >= a)), axis=0) + noise_SS)

        # save noisy summaries
        torch.save(nSS, FileSavePath + os.sep + 'reference_theta' + os.sep + 'atus_sdp_obs_eps_' + str(args.eps) + '.pt')
        raise NotImplementedError("Noisy summaries for ATUS data (epsilon = %.3f) have been generated and saved." % args.eps)


    unif_dist = torch.distributions.Uniform(0, 1)

    # load data
    if args.eps == 1.0:
        sdp_obs = torch.load(FileSavePath + os.sep + 'reference_theta' + os.sep + 'atus' + ckpt_str + '_sdp_obs_eps_1.0.pt', weights_only=True, map_location=args.device)
    elif args.eps == 10.0:
        sdp_obs = torch.load(FileSavePath + os.sep + 'reference_theta' + os.sep + 'atus' + ckpt_str + '_sdp_obs_eps_10.0.pt', weights_only=True, map_location=args.device)
    else:
        raise NotImplementedError("Noisy summaries for ATUS data (epsilon = %.3f) have not been generated." % args.eps)
    prior_concentration = torch.tensor([1.0]).to(args.device)
    prior_rate = torch.tensor([0.1]).to(args.device)
    prior_params = {'concentration': prior_concentration, 'rate': prior_rate}
    prior_dist = torch.distributions.Gamma(prior_concentration, prior_rate)

    # fit
    if args.enable_fit:
        # init chain state
        theta_all = torch.zeros(args.step, dim_theta).to(args.device)
        acc_rate_all = torch.zeros(args.step)
        # theta = prior_dist.sample((dim_theta,)).reshape(1, -1).to(args.device)  # shape: 1 * dim_theta
        theta = (torch.exp(sdp_obs) / torch.sum(torch.exp(sdp_obs)) * (prior_concentration / prior_rate * dim_theta).reshape(1, -1)).to(args.device)  # initial value
        # theta = true_theta.to(args.device)
        data = gen_data(theta, n)

        range_generator = tqdm(range(args.step)) if args.tqdm else range(args.step)
        update_data_time = 0.0
        update_theta_time = 0.0
        for i in range_generator:
            data, acc_rate = update_data(data, theta, unif_dist, sdp_obs, args)  # shape: 1 * n * p
            theta_mcmc_steps = 100
            for j in range(theta_mcmc_steps):
                theta = update_theta(data, prior_dist, theta, unif_dist)
            theta_all[i] = theta
            acc_rate_all[i] = acc_rate
            if args.tqdm:
                range_generator.set_description('acc_rate: %.6f' % (torch.sum(acc_rate_all) / (i + 1)))

        # save all states
        if args.dbg2 == "_fit_coupling":
            if args.seed < 11000:
                torch.save(theta_all, FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_state_all.pt")
        else:
            # pass
            torch.save(theta_all, FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_state_all.pt")
        if args.dbg2 == "_fit_only":
            return torch.mean(acc_rate_all)

def update_data(data, theta, unif_dist, sdp_obs, args):
    # data: n * p
    # theta: dim_theta
    # return: n * p
    current_s = gen_s(data.transpose(0, 1)) / args.ncomp  # n * dim_s
    current_s_sum = torch.sum(current_s, dim=0)
    proposed_data = gen_data(theta, data.shape[1])  # 1 * n * (p+2)
    proposed_s = gen_s(proposed_data.transpose(0, 1)) / args.ncomp  # n * dim_s
    acc_times = 0
    rej_times = 0
    if args.method == 1:
        # SOMA update
        current_log_weight = torch.sum(-torch.abs(current_s_sum - sdp_obs)) / args.scale
        for i in range(args.ncomp):
            summary_sum_update = -current_s + current_s_sum + proposed_s[i]
            log_weights = torch.sum(-torch.abs(summary_sum_update - sdp_obs), dim=1) / args.scale
            weights_normalizer = torch.max(log_weights)
            weights = torch.exp(log_weights - weights_normalizer)

            # resamp_idx = torch.multinomial(weights, 1, replacement=True)
            resamp_idx = multinomial_by_inverse_sampling(weights)

            # accept or reject
            if log_weights[resamp_idx] >= current_log_weight:
                # accept with probability 1
                data[0, resamp_idx] = proposed_data[0, i]
                current_s_sum = current_s_sum - current_s[resamp_idx] + proposed_s[i]
                current_s[resamp_idx] = proposed_s[i]
                current_log_weight = log_weights[resamp_idx]
                acc_times += 1
                continue
            else:
                current_weight = torch.exp(current_log_weight - weights_normalizer)
                sum_weights = torch.sum(weights)
                accept_prob = sum_weights / (sum_weights + current_weight - weights[resamp_idx])
                if accept_prob > unif_dist.sample():
                    data[0, resamp_idx] = proposed_data[0, i]
                    current_s_sum = current_s_sum - current_s[resamp_idx] + proposed_s[i]
                    current_s[resamp_idx] = proposed_s[i]
                    current_log_weight = log_weights[resamp_idx]
                    acc_times += 1
                    continue
                else:
                    rej_times += 1

    elif args.method == 2:
        # random-scan update
        random_index = torch.multinomial(torch.ones(args.ncomp), args.ncomp, replacement=True)
        for i in range(args.ncomp):
            weight_diff = torch.sum(torch.abs(current_s_sum - sdp_obs) - torch.abs(current_s_sum - current_s[random_index[i]] + proposed_s[i] - sdp_obs))
            # accept if weight_diff >= 0
            if weight_diff >= 0:
                data[0, random_index[i]] = proposed_data[0, i]
                current_s_sum = current_s_sum - current_s[random_index[i]] + proposed_s[i]
                current_s[random_index[i]] = proposed_s[i]
                acc_times += 1
            else:
                if torch.exp(weight_diff / args.scale) > unif_dist.sample():
                    data[0, random_index[i]] = proposed_data[0, i]
                    current_s_sum = current_s_sum - current_s[random_index[i]] + proposed_s[i]
                    current_s[random_index[i]] = proposed_s[i]
                    acc_times += 1
                else:
                    rej_times += 1

    elif args.method == 3:
        # systematic-scan update
        for i in range(args.ncomp):
            weight_diff = torch.sum(torch.abs(current_s_sum - sdp_obs) - torch.abs(current_s_sum - current_s[i] + proposed_s[i] - sdp_obs))
            # accept if weight_diff >= 0
            if weight_diff >= 0:
                data[0, i] = proposed_data[0, i]
                current_s_sum = current_s_sum - current_s[i] + proposed_s[i]
                current_s[i] = proposed_s[i]
                acc_times += 1
            else:
                if torch.exp(weight_diff / args.scale) > unif_dist.sample():
                    data[0, i] = proposed_data[0, i]
                    current_s_sum = current_s_sum - current_s[i] + proposed_s[i]
                    current_s[i] = proposed_s[i]
                    acc_times += 1
                else:
                    rej_times += 1

    return data, acc_times / (acc_times + rej_times)

def conjugate_update_theta(data_input, prior_params, gen_num=1):
    xy_input = data_input.squeeze(0)
    xtx = xy_input[:, :-1].t() @ xy_input[:, :-1]
    xty = (xy_input[:, :-1].t() @ xy_input[:, -1]).reshape(-1, 1)
    yty = (xy_input[:, -1].t() @ xy_input[:, -1]).reshape(1, 1)
    Lambda = xtx + prior_params['lambda']
    invLambda = torch.inverse(Lambda)
    mu = invLambda @ (xty + prior_params['lambda'] @ prior_params['mu0'])
    an = prior_params['a0'] + xy_input.shape[0] / 2
    bn = prior_params['b0'] + 0.5 * (yty + prior_params['mu0'].t() @ prior_params['lambda'] @ prior_params['mu0']
                            - mu.t() @ Lambda @ mu).squeeze(0)
    sigma2 = 1 / torch.distributions.Gamma(an, bn).sample((gen_num,))
    # sigma2 = torch.zeros_like(sigma2)
    # sigma2 += 2.0
    normal_sample = torch.normal(0, 1, size=(gen_num, prior_params['mu0'].shape[0]), device=xy_input.device)
    beta = normal_sample @ torch.linalg.cholesky(invLambda).t() * torch.sqrt(sigma2) + mu.t()
    return torch.cat((beta, sigma2), dim=1)

def update_theta(data, prior_dist, theta, unif_dist):
    # sample theta from posterior using slice sampling
    # data shape: 1 * n * p
    k = data.shape[2]  # p
    device = data.device
    w = torch.ones(k, device=data.device, dtype=data.dtype)  # window size

    # 1. calc slice height
    log_prior = prior_dist.log_prob(theta)
    alpha_dir_dist = torch.distributions.Dirichlet(theta)
    log_lik = alpha_dir_dist.log_prob(data)  # shape: 1 * n
    log_post = torch.sum(log_prior) + torch.sum(log_lik)
    log_u = torch.log(unif_dist.sample())
    fx0 = log_u + log_post

    # 2. Get [LL, RR]
    u = unif_dist.sample((1, k)).to(device)  # shape: 1 * k
    LL = theta - w * u
    RR = LL + w
    LL.clamp_(min=1e-9)  # ensure LL >= 0

    # 3. Stepping-out
    count = 0
    max_count = 50
    while True:
        x1_unif = unif_dist.sample((1, k)).to(device)  # shape: 1 * k
        x1 = LL + x1_unif * (RR - LL)

        log_prior_x1 = prior_dist.log_prob(x1)
        alpha_dir_dist_x1 = torch.distributions.Dirichlet(x1)
        log_lik_x1 = alpha_dir_dist_x1.log_prob(data)  # shape: 1 * n
        log_post_x1 = torch.sum(log_prior_x1) + torch.sum(log_lik_x1)

        if log_post_x1 > fx0:
            # print("Slice sampling accepted at iteration %d." % count)
            return x1
        else:
            LL = torch.where(x1 < theta, x1, LL)
            RR = torch.where(x1 >= theta, x1, RR)
            count += 1
            if count > max_count:
                return theta


def update_data_coupling(data1, data2, theta1, theta2, x0_dist, y_dist, unif_dist, sdp_obs, args):
    # data: 1 * n * (p+2)
    # theta: 1 * dim_theta
    # return: 1 * n * (p+2)
    current_s1 = gen_s(data1.transpose(0, 1)) / args.ncomp  # n * dim_s
    current_s_sum1 = torch.sum(current_s1, dim=0)
    current_s2 = gen_s(data2.transpose(0, 1)) / args.ncomp  # n * dim_s
    current_s_sum2 = torch.sum(current_s2, dim=0)

    # generate proposed data via coupling
    x0_sample = x0_dist.sample((1, args.ncomp))  # 1 * n * p
    x_sample = torch.concat((torch.ones(1, args.ncomp, 1, device=args.device), x0_sample), dim=2)  # batch * num_samples * (p + 1)
    y_mu1 = torch.bmm(x_sample, theta1[:, :-1].unsqueeze(2))
    y_mu2 = torch.bmm(x_sample, theta2[:, :-1].unsqueeze(2))
    y_std1 = torch.sqrt(theta1[:, -1]).reshape(-1, 1)
    y_std2 = torch.sqrt(theta2[:, -1]).reshape(-1, 1)

    # coupling of two normal
    y_sample1 = y_mu1 + y_dist.sample((1, args.ncomp)) * y_std1
    y_sample2 = torch.zeros_like(y_sample1)
    log_ratio = 0.5 * ((y_sample1 - y_mu1) / y_std1) ** 2 + torch.log(y_std1) - 0.5 * ((y_sample1 - y_mu2) / y_std2) ** 2 - torch.log(y_std2)
    y_update_idx = unif_dist.sample((1, args.ncomp)) < torch.exp(log_ratio.cpu()).squeeze(-1)
    coupled_proposal = torch.sum(y_update_idx).item()
    y_sample2[y_update_idx] = y_sample1[y_update_idx]
    count = 0
    while torch.sum(~y_update_idx) > 0:
        # print('remain %d' % torch.sum(~y_update_idx).item())
        y_sample2_proposal = y_mu2[~y_update_idx] + y_dist.sample((torch.sum(~y_update_idx),)) * y_std2
        log_ratio = (0.5 * ((y_sample2_proposal - y_mu2[~y_update_idx]) / y_std2) ** 2 + torch.log(y_std2)
                     - 0.5 * ((y_sample2_proposal - y_mu1[~y_update_idx]) / y_std1) ** 2 - torch.log(y_std1))
        acc_idx = (unif_dist.sample((torch.sum(~y_update_idx), 1)) > torch.exp(log_ratio.cpu())).squeeze(1)
        # print("acc num: %d" % torch.sum(acc_idx).item())
        count += 1
        if count > 100:
            y_sample2[~y_update_idx] = y_sample1[~y_update_idx]
            print("count > 100, fill %d samples with y_sample1" % torch.sum(~y_update_idx).item())
            break
        if torch.sum(acc_idx) > 0:
            y_sample2[0, torch.arange(args.ncomp).unsqueeze(0)[~(y_update_idx)][acc_idx]] = y_sample2_proposal[acc_idx]
            y_update_idx[~y_update_idx] = acc_idx.squeeze(-1)

    proposed_data1 = torch.cat((x_sample, y_sample1), dim=-1)
    proposed_data2 = torch.cat((x_sample, y_sample2), dim=-1)
    proposed_s1 = gen_s(proposed_data1.transpose(0, 1)) / args.ncomp  # n * dim_s
    proposed_s2 = gen_s(proposed_data2.transpose(0, 1)) / args.ncomp  # n * dim_s
    acc_times1 = 0
    acc_times2 = 0

    if args.method == 1:
        # SOMA update
        current_log_weight1 = torch.sum(-torch.abs(current_s_sum1 - sdp_obs)) / args.scale
        current_log_weight2 = torch.sum(-torch.abs(current_s_sum2 - sdp_obs)) / args.scale
        for i in range(args.ncomp):
            chain1_acc_flag = False
            chain2_acc_flag = False
            summary_sum_update1 = -current_s1 + current_s_sum1 + proposed_s1[i]
            log_weights1 = torch.sum(-torch.abs(summary_sum_update1 - sdp_obs), dim=1) / args.scale
            weights_normalizer1 = torch.logsumexp(log_weights1, dim=0)
            weights1 = torch.exp(log_weights1 - weights_normalizer1)

            summary_sum_update2 = -current_s2 + current_s_sum2 + proposed_s2[i]
            log_weights2 = torch.sum(-torch.abs(summary_sum_update2 - sdp_obs), dim=1) / args.scale
            weights_normalizer2 = torch.logsumexp(log_weights2, dim=0)
            weights2 = torch.exp(log_weights2 - weights_normalizer2)

            shared_weights = torch.min(weights1, weights2)
            sum_shared_weights = torch.sum(shared_weights)

            # select index via maximal coupling
            unif_sample = unif_dist.sample()
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
                if resamp_idx1 == args.ncomp:
                    resamp_idx1 = args.ncomp - 1
                if resamp_idx2 == args.ncomp:
                    resamp_idx2 = args.ncomp - 1

            # Accept or reject for chain 1
            unif_sample_acc = unif_dist.sample()
            if log_weights1[resamp_idx1] >= current_log_weight1:
                data1[0, resamp_idx1] = proposed_data1[0, i]
                current_s_sum1 = current_s_sum1 - current_s1[resamp_idx1] + proposed_s1[i]
                current_s1[resamp_idx1] = proposed_s1[i]
                current_log_weight1 = log_weights1[resamp_idx1]
                acc_times1 += 1
                chain1_acc_flag = True
            else:
                current_weight1 = torch.exp(current_log_weight1 - weights_normalizer1)
                accept_prob1 = 1. / (1. + current_weight1 - weights1[resamp_idx1])  # since sum(weights1) = 1
                if accept_prob1 > unif_sample_acc:
                    data1[0, resamp_idx1] = proposed_data1[0, i]
                    current_s_sum1 = current_s_sum1 - current_s1[resamp_idx1] + proposed_s1[i]
                    current_s1[resamp_idx1] = proposed_s1[i]
                    current_log_weight1 = log_weights1[resamp_idx1]
                    acc_times1 += 1
                    chain1_acc_flag = True

            # Accept or reject for chain 2
            if log_weights2[resamp_idx2] >= current_log_weight2:
                data2[0, resamp_idx2] = proposed_data2[0, i]
                current_s_sum2 = current_s_sum2 - current_s2[resamp_idx2] + proposed_s2[i]
                current_s2[resamp_idx2] = proposed_s2[i]
                current_log_weight2 = log_weights2[resamp_idx2]
                acc_times2 += 1
                chain2_acc_flag = True
            else:
                current_weight2 = torch.exp(current_log_weight2 - weights_normalizer2)
                accept_prob2 = 1. / (1. + current_weight2 - weights2[resamp_idx2])  # since sum(weights1) = 1
                if accept_prob2 > unif_sample_acc:
                    data2[0, resamp_idx2] = proposed_data2[0, i]
                    current_s_sum2 = current_s_sum2 - current_s2[resamp_idx2] + proposed_s2[i]
                    current_s2[resamp_idx2] = proposed_s2[i]
                    current_log_weight2 = log_weights2[resamp_idx2]
                    acc_times2 += 1
                    chain2_acc_flag = True

            if chain1_acc_flag and chain2_acc_flag and (not torch.all(data1[0, :, -1] == data2[0, :, -1])):
                uncoupled_idx = torch.where(data1[0, :, -1] != data2[0, :, -1])[0][0]
                data1[0, resamp_idx1] = data1[0, uncoupled_idx].clone()
                data1[0, uncoupled_idx] = proposed_data1[0, i]
                data2[0, resamp_idx2] = data2[0, uncoupled_idx].clone()
                data2[0, uncoupled_idx] = proposed_data2[0, i]

    elif args.method == 2:
        # random-scan update
        random_index = torch.multinomial(torch.ones(args.ncomp), args.ncomp, replacement=True)
        for i in range(args.ncomp):
            weight_diff1 = torch.sum(torch.abs(current_s_sum1 - sdp_obs) - torch.abs(current_s_sum1 - current_s1[random_index[i]] + proposed_s1[i] - sdp_obs))
            weight_diff2 = torch.sum(torch.abs(current_s_sum2 - sdp_obs) - torch.abs(current_s_sum2 - current_s2[random_index[i]] + proposed_s2[i] - sdp_obs))
            # accept if weight_diff >= 0
            unif_value = unif_dist.sample()
            if torch.exp(weight_diff1 / args.scale) > unif_value:
                data1[0, random_index[i]] = proposed_data1[0, i]
                current_s_sum1 = current_s_sum1 - current_s1[random_index[i]] + proposed_s1[i]
                current_s1[random_index[i]] = proposed_s1[i]
                acc_times1 += 1
            if torch.exp(weight_diff2 / args.scale) > unif_value:
                data2[0, random_index[i]] = proposed_data2[0, i]
                current_s_sum2 = current_s_sum2 - current_s2[random_index[i]] + proposed_s2[i]
                current_s2[random_index[i]] = proposed_s2[i]
                acc_times2 += 1

    elif args.method == 3:
        # systematic-scan update
        for i in range(args.ncomp):
            weight_diff1 = torch.sum(torch.abs(current_s_sum1 - sdp_obs) - torch.abs(current_s_sum1 - current_s1[i] + proposed_s1[i] - sdp_obs))
            weight_diff2 = torch.sum(torch.abs(current_s_sum2 - sdp_obs) - torch.abs(current_s_sum2 - current_s2[i] + proposed_s2[i] - sdp_obs))
            # accept if weight_diff >= 0
            unif_value = unif_dist.sample()
            if torch.exp(weight_diff1 / args.scale) > unif_value:
                data1[0, i] = proposed_data1[0, i]
                current_s_sum1 = current_s_sum1 - current_s1[i] + proposed_s1[i]
                current_s1[i] = proposed_s1[i]
                acc_times1 += 1
            if torch.exp(weight_diff2 / args.scale) > unif_value:
                data2[0, i] = proposed_data2[0, i]
                current_s_sum2 = current_s_sum2 - current_s2[i] + proposed_s2[i]
                current_s2[i] = proposed_s2[i]
                acc_times2 += 1

    return data1, data2, acc_times1 / args.ncomp, acc_times2 / args.ncomp, coupled_proposal

def conjugate_update_theta_coupling(data1, data2, unif_dist, prior_params):
    an = prior_params['a0'] + data1.shape[1] / 2
    mu_times_lambda = prior_params['lambda'] @ prior_params['mu0']
    mu_times_lambda_times_mu = prior_params['mu0'].t() @ prior_params['lambda'] @ prior_params['mu0']

    xy1 = data1.squeeze(0)
    Lambda1 = xy1[:, :-1].t() @ xy1[:, :-1] + prior_params['lambda']
    invLambda1 = torch.inverse(Lambda1)
    mu1 = invLambda1 @ ((xy1[:, :-1].t() @ xy1[:, -1]).reshape(-1, 1) + mu_times_lambda)
    bn1 = prior_params['b0'] + 0.5 * ((xy1[:, -1].t() @ xy1[:, -1]).reshape(1, 1) +
        mu_times_lambda_times_mu - mu1.t() @ Lambda1 @ mu1).squeeze(0)

    xy2 = data2.squeeze(0)
    Lambda2 = xy2[:, :-1].t() @ xy2[:, :-1] + prior_params['lambda']
    invLambda2 = torch.inverse(Lambda2)
    mu2 = invLambda2 @ ((xy2[:, :-1].t() @ xy2[:, -1]).reshape(-1, 1) + mu_times_lambda)
    bn2 = prior_params['b0'] + 0.5 * ((xy2[:, -1].t() @ xy2[:, -1]).reshape(1, 1) +
        mu_times_lambda_times_mu - mu2.t() @ Lambda2 @ mu2).squeeze(0)

    # step 1: max coupling for sigma
    gamma_dist1 = torch.distributions.Gamma(an, bn1)
    gamma_dist2 = torch.distributions.Gamma(an, bn2)
    gamma_samp1 = gamma_dist1.sample()
    if unif_dist.sample() <= torch.exp(gamma_dist2.log_prob(gamma_samp1) - gamma_dist1.log_prob(gamma_samp1)):
        gamma_samp2 = gamma_samp1
    else:
        batch_size = 10
        while True:
            gamma_samp2_batch = gamma_dist2.sample((batch_size,)).squeeze()
            acc_idx = unif_dist.sample((batch_size,)) > torch.exp(gamma_dist1.log_prob(gamma_samp2_batch) - gamma_dist2.log_prob(gamma_samp2_batch)).cpu()
            if torch.sum(acc_idx) > 0:
                gamma_samp2 = gamma_samp2_batch[acc_idx][0].reshape(-1)
                break
    sigma21 = 1 / gamma_samp1
    sigma22 = 1 / gamma_samp2

    # step 2: max coupling for beta
    normal_dist1 = torch.distributions.MultivariateNormal(mu1.squeeze(), sigma21 * invLambda1)
    normal_dist2 = torch.distributions.MultivariateNormal(mu2.squeeze(), sigma22 * invLambda2)
    normal_samp1 = normal_dist1.sample()
    if unif_dist.sample() <= torch.exp(normal_dist2.log_prob(normal_samp1) - normal_dist1.log_prob(normal_samp1)):
        normal_samp2 = normal_samp1
    else:
        batch_size = 10
        while True:
            normal_samp2_batch = normal_dist2.sample((batch_size,)).squeeze()
            acc_idx = unif_dist.sample((batch_size,)) > torch.exp(normal_dist1.log_prob(normal_samp2_batch) - normal_dist2.log_prob(normal_samp2_batch)).cpu()
            if torch.sum(acc_idx) > 0:
                normal_samp2 = normal_samp2_batch[acc_idx][0]
                break

    theta1 = torch.cat((normal_samp1, sigma21), dim=0).reshape(1, -1)
    theta2 = torch.cat((normal_samp2, sigma22), dim=0).reshape(1, -1)
    return theta1, theta2


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file path")
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--method', type=int, default=1, help='method')  # 1: SOMA; 2: random-scan; 3: systematic-scan
    parser.add_argument('--ncomp', type=int, default=6656, help='n_component')  # number of components
    parser.add_argument('--mkstr', type=str, default="atus", help='mark_str')  # mark string
    parser.add_argument('--step', type=int, default=2000, help='total_steps')  # total steps
    parser.add_argument('--data', type=int, default=2, help='dataset')  # 0: combined data; 1: male; 2: female
    parser.add_argument('--enable_fit', type=int, default=1, help='enable_mcmc_fit')  # 0: disable; 1: enable
    parser.add_argument('--enable_coupling', type=int, default=0, help='enable_coupling')  # 0: disable; 1: enable
    parser.add_argument('--seed', type=int, default=10000, help='random seed')  # default value: 0
    parser.add_argument('--eps', type=float, default=10.0, help='epsilon')
    parser.add_argument('--tqdm', type=int, default=1, help='enable_tqdm')
    parser.add_argument('--dbg1', type=int, default=0, help='debug_flag_1')
    parser.add_argument('--dbg2', type=str, default="", help='debug_flag_1')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    main(args, config)
