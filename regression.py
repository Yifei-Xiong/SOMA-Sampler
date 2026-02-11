# Code for privatized Bayesian linear regression

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from tqdm import tqdm


def gen_data(theta, num_samples, x0_dist, y_dist, x_sample=None):
    # theta: batch * dim_theta
    # num_samples: int
    # return: batch * num_samples * (p+2)
    batch = theta.shape[0]
    if x_sample is None:
        x0_sample = x0_dist.sample((batch, num_samples))  # batch * num_samples * p
        x_sample = torch.concat((torch.ones(batch, num_samples, 1, device=theta.device), x0_sample), dim=2)  # batch * num_samples * (p + 1)
    y_sample = torch.bmm(x_sample, theta[:, :-1].unsqueeze(2)) + \
               y_dist.sample((batch, num_samples)) * torch.sqrt(theta[:, -1]).reshape(-1, 1, 1)  # batch * num_samples * 1
    xy_sample = torch.cat((x_sample, y_sample), dim=2)
    return xy_sample  # return untruncated data


def gen_s(data_input, clamp_lower=-6., clamp_upper=6.):
    # data: batch * n * (p+2)
    # return: batch * dim_s
    batch = data_input.shape[0]
    # truncation & normalization
    xy_sample_norm = data_input.clone()
    xy_sample_norm[:, :, 1:].clamp_(min=clamp_lower, max=clamp_upper)
    xy_sample_norm[:, :, 1:].sub_(clamp_lower).mul_(2 / (clamp_upper - clamp_lower)).sub_(1.)
    x_sample_norm = xy_sample_norm[:, :, :-1]
    y_sample_norm = xy_sample_norm[:, :, -1].unsqueeze(2)
    xty_output = torch.sum(x_sample_norm * y_sample_norm, dim=1)
    yty_output = torch.sum(y_sample_norm * y_sample_norm, dim=1)
    xtx_output = torch.bmm(x_sample_norm.transpose(1, 2), x_sample_norm)
    return torch.concat((xty_output, yty_output, xtx_output.reshape(batch, -1)[:, [1, 2, 4, 5, 8]]), dim=1)


def gen_sdp(s_input, scale, gen_num=1):
    # input param: summary stat S, scale = delta / eps, generate number per sample
    # input shape (of s): batch * dim_s
    # output param: privatized Sdp
    # output shape: batch * gen_num * dim_s
    batch = s_input.shape[0]
    dim = s_input.shape[1]
    unif_result = torch.rand(batch, gen_num, dim, dtype=torch.float64)
    # inverse cumulative
    noise = - scale * torch.sgn(unif_result - 0.5) * torch.log(1 - 2 * (torch.abs(unif_result - 0.5)) + 1e-8)
    return noise.type(torch.float32).to(s_input.device) + s_input.reshape(batch, 1, -1).expand(batch, gen_num, -1)


def main(args, config):
    # init args
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    FileSavePath = config['FileSavePath_linux'] if os.name == 'posix' else config['FileSavePath_win']
    print("File Save Path: " + FileSavePath)
    method_colors = ['#EE9B00', '#01899D', '#565B7F']
    params_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    if args.gpu == 0:
        args.device = torch.device('cpu')
        print('using cpu')
    else:
        args.device = torch.device("cuda:" + str(args.gpu - 1))
        print('using gpu: %d' % args.gpu)
    print("manual seed: " + str(args.seed))
    print("mark str: " + args.mkstr)
    print("epsilon: %.3f" % args.eps)
    print("total steps: %d" % args.step)
    print("number of components: %d" % args.ncomp)
    ckpts_folder_name = "ckpts"
    ModelInfo = "mk+" + args.mkstr + "_eps+" + str(args.eps) + "_ncomp+" + str(args.ncomp) + "_me+" + \
                str(args.method) + "_data+" + str(args.data) + "_step+" + str(args.step) + "_seed+" + str(args.seed)
    if args.method == 1:
        sampler_name = "SOMA"
    elif args.method == 2:
        sampler_name = "Ran_IMwG"
    elif args.method == 3:
        sampler_name = "Sys_IMwG"
    else:
        raise NotImplementedError
    print("using sampler: " + sampler_name)

    # define simulator
    p = 2  # dimension of X, 2
    n = args.ncomp  # number_of_records, 100
    dim_theta = p + 2  # dimension of theta
    delta = ((p + 3) * p + 3) / n  # l1 sensitivity
    args.scale = delta / args.eps
    true_theta = torch.tensor([[-1.79, -2.89, -0.66, 1.13]], device=torch.device('cpu'))
    columns = ['$\\beta_0$', '$\\beta_1$', '$\\beta_2$', '$\\sigma^2$']
    x0_mean = torch.tensor([0.90, -1.17], device=args.device)
    x0_cov = torch.diag(torch.ones(p, ) * 1.).to(args.device)
    x0_dist = torch.distributions.MultivariateNormal(x0_mean, x0_cov, validate_args=False)
    y_dist = torch.distributions.MultivariateNormal(torch.tensor([0.], device=args.device),
                                                    torch.diag(torch.ones(1, )).to(args.device), validate_args=False)
    unif_dist = torch.distributions.Uniform(0, 1)

    # load data
    eps_type_list = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    eps_type = eps_type_list.index(args.eps)
    if args.ncomp == 100:
        sdp_obs_ckpt = torch.load(FileSavePath + os.sep + 'reference_theta' + os.sep + 'sdp_obs_ckpt_n+100.pt', weights_only=True)
    else:
        sdp_obs_ckpt = torch.load(FileSavePath + os.sep + 'reference_theta' + os.sep + 'sdp_obs_ckpt_n+20.pt', weights_only=True)
    sdp_obs = sdp_obs_ckpt[eps_type, 0].reshape(1, -1).to(args.device)
    prior_a0 = torch.tensor([10.0]).to(args.device)
    prior_b0 = torch.tensor([10.0]).to(args.device)
    prior_mu0 = torch.zeros((3, 1)).to(args.device)
    prior_lambda = torch.diag(torch.ones(3) / 2).to(args.device)
    prior_params = {'a0': prior_a0, 'b0': prior_b0, 'mu0': prior_mu0, 'lambda': prior_lambda}

    # fit
    if args.enable_fit:
        # init chain state
        theta_all = torch.zeros(args.step, dim_theta).to(args.device)
        acc_rate_all = torch.zeros(args.step)
        sigma2_init = (1 / torch.distributions.Gamma(prior_params['a0'], prior_params['b0']).sample()).to(args.device)  # inverse gamma
        beta_init = torch.distributions.MultivariateNormal(prior_params['mu0'].squeeze(),
                                                           sigma2_init * torch.inverse(prior_params['lambda'])).sample().to(args.device)
        theta = torch.cat((beta_init, sigma2_init), dim=0).reshape(1, -1)
        data = gen_data(theta, n, x0_dist, y_dist)

        range_generator = tqdm(range(args.step)) if args.tqdm else range(args.step)
        for i in range_generator:
            # update data
            data, acc_rate = update_data(data, theta, x0_dist, y_dist, unif_dist, sdp_obs, args)
            # conjugate update theta
            theta = conjugate_update_theta(data, prior_params)
            theta_all[i] = theta
            acc_rate_all[i] = acc_rate
            if args.tqdm:
                range_generator.set_description('acc_rate: %.3f' % (torch.sum(acc_rate_all) / (i + 1)))

        # save (last) state
        torch.save((data, theta, torch.mean(acc_rate_all)), FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_state.pt")

        # save all states
        torch.save(theta_all, FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_state_all.pt")

    if args.enable_fit:
        # trace plot of theta_all
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
        for j in range(dim_theta):
            ax.plot(theta_all[:, j].cpu().numpy(), label=columns[j], color=params_colors[j], linewidth=0.8)
        ax.legend(loc='upper right')
        # add true theta
        for j in range(dim_theta):
            ax.axhline(y=true_theta[0, j], color=params_colors[j], linestyle='--', linewidth=0.8)
        plt.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, args.step])
        ax.set_ylim([-8, 8])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Trace Plot of Parameters')
        plt.savefig(FileSavePath + os.sep + 'fig' + os.sep + 'traceplot_' + ModelInfo + '.png', dpi=300)

    if args.enable_fit:
        # acc rate plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        ax.scatter(range(args.step), acc_rate_all.cpu().numpy(), color=method_colors[args.method - 1], s=3.0)
        ax.legend([sampler_name], loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim([0, args.step])
        ax.set_ylim([-0.03, 1.03])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rate')
        plt.savefig(FileSavePath + os.sep + 'fig' + os.sep + 'accrate_' + ModelInfo + '.png', dpi=300)

    # coupling
    if args.enable_coupling:
        torch.manual_seed(args.seed + 10000)  # avoid state_init2 has some same components with state_init1
        torch.cuda.manual_seed(args.seed + 10000)
        np.random.seed(args.seed + 10000)

        # init chain state
        ModelInfo_modify = ("mk+" + args.mkstr + "_eps+" + str(args.eps) + "_ncomp+" + str(args.ncomp) +
                            "_me+1_data+" + str(args.data) + "_step+" + str(args.step) + "_seed+" + str(args.seed))

        data1, theta1, _ = torch.load(FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo_modify + "_state.pt", weights_only=True)
        theta1_all = torch.zeros(args.step, dim_theta).to(args.device)
        theta2_all = torch.zeros(args.step, dim_theta).to(args.device)
        acc_rate1_all = torch.zeros(args.step)
        acc_rate2_all = torch.zeros(args.step)
        sigma2_init = (1 / torch.distributions.Gamma(prior_params['a0'], prior_params['b0']).sample()).to(args.device)  # inverse gamma
        beta_init = torch.distributions.MultivariateNormal(prior_params['mu0'].squeeze(),
                                                           sigma2_init * torch.inverse(prior_params['lambda'])).sample().to(args.device)
        theta2 = torch.cat((beta_init, sigma2_init), dim=0).reshape(1, -1)
        data2 = gen_data(theta2, n, x0_dist, y_dist)

        range_generator = tqdm(range(args.step)) if args.tqdm else range(args.step)
        coupling_times = args.step
        for i in range_generator:
            # update data
            data1, data2, acc_rate1, acc_rate2, coupled_proposal = update_data_coupling(
                data1, data2, theta1, theta2, x0_dist, y_dist, unif_dist, sdp_obs, args)
            # conjugate update theta
            theta1, theta2 = conjugate_update_theta_coupling(data1, data2, unif_dist, prior_params)
            theta1_all[i] = theta1
            theta2_all[i] = theta2
            acc_rate1_all[i] = acc_rate1
            acc_rate2_all[i] = acc_rate2

            data_diff_count = torch.sum(~torch.all(data1 == data2, dim=-1))
            data_diff_val = torch.sum(torch.abs(data1 - data2))
            theta_diff_count = torch.sum(theta1 != theta2)
            data1_sdp_diff = torch.sum(torch.abs(gen_s(data1) - sdp_obs) / args.scale)
            data2_sdp_diff = torch.sum(torch.abs(gen_s(data2) - sdp_obs) / args.scale)
            if theta_diff_count == 0 and data_diff_count == 0:
                print("Chain coupled at iteration %d." % i)
                coupling_times = i
                break
            if args.tqdm:
                range_generator.set_description(
                    'acc1: %.3f, acc2: %.3f, theta diff: %d, data diff: %d, val: %.4f, prop diff: %d, d1df: %.4f, d2df: %.4f' %
                    (torch.sum(acc_rate1_all) / (i + 1), torch.sum(acc_rate2_all) / (i + 1),
                     theta_diff_count, data_diff_count, data_diff_val, coupled_proposal, data1_sdp_diff, data2_sdp_diff))
        if args.dbg2 == "_fit_coupling":
            return coupling_times
    pass


def update_data(data, theta, x0_dist, y_dist, unif_dist, sdp_obs, args):
    # data: 1 * n * (p+2)
    # theta: 1 * dim_theta
    # return: 1 * n * (p+2)
    current_s = gen_s(data.transpose(0, 1)) / args.ncomp  # n * dim_s
    current_s_sum = torch.sum(current_s, dim=0)
    proposed_data = gen_data(theta, data.shape[1], x0_dist, y_dist)  # 1 * n * (p+2)
    proposed_s = gen_s(proposed_data.transpose(0, 1)) / args.ncomp  # n * dim_s
    acc_times = 0
    if args.method == 1:
        # SOMA update
        current_log_weight = torch.sum(-torch.abs(current_s_sum - sdp_obs)) / args.scale
        for i in range(args.ncomp):
            summary_sum_update = -current_s + current_s_sum + proposed_s[i]
            log_weights = torch.sum(-torch.abs(summary_sum_update - sdp_obs), dim=1) / args.scale
            weights_normalizer = torch.max(log_weights)
            weights = torch.exp(log_weights - weights_normalizer)

            resamp_idx = torch.multinomial(weights, 1, replacement=True)

            if log_weights[resamp_idx] >= current_log_weight:
                # accept with probability 1
                data[0, resamp_idx] = proposed_data[0, i]
                current_s_sum = current_s_sum - current_s[resamp_idx] + proposed_s[i]
                current_s[resamp_idx] = proposed_s[i]
                current_log_weight = log_weights[resamp_idx]
                acc_times += 1
            else:
                current_weight = torch.exp(current_log_weight - weights_normalizer)
                accept_prob = torch.sum(weights) / (torch.sum(weights) + current_weight - weights[resamp_idx])
                if accept_prob > unif_dist.sample():
                    data[0, resamp_idx] = proposed_data[0, i]
                    current_s_sum = current_s_sum - current_s[resamp_idx] + proposed_s[i]
                    current_s[resamp_idx] = proposed_s[i]
                    current_log_weight = log_weights[resamp_idx]
                    acc_times += 1

    elif args.method == 2:
        # random-scan update
        random_index = torch.multinomial(torch.ones(args.ncomp), args.ncomp, replacement=True)
        for i in range(args.ncomp):
            weight_diff = torch.sum(
                torch.abs(current_s_sum - sdp_obs) - torch.abs(current_s_sum - current_s[random_index[i]] + proposed_s[i] - sdp_obs))
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

    elif args.method == 3:
        # systematic-scan update
        for i in range(args.ncomp):
            weight_diff = torch.sum(torch.abs(current_s_sum - sdp_obs) - torch.abs(current_s_sum - current_s[i] + proposed_s[i] - sdp_obs))
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

    return data, acc_times / args.ncomp


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
    normal_sample = torch.normal(0, 1, size=(gen_num, prior_params['mu0'].shape[0]), device=xy_input.device)
    beta = normal_sample @ torch.linalg.cholesky(invLambda).t() * torch.sqrt(sigma2) + mu.t()
    return torch.cat((beta, sigma2), dim=1)


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
        y_sample2_proposal = y_mu2[~y_update_idx] + y_dist.sample((torch.sum(~y_update_idx),)) * y_std2
        log_ratio = (0.5 * ((y_sample2_proposal - y_mu2[~y_update_idx]) / y_std2) ** 2 + torch.log(y_std2)
                     - 0.5 * ((y_sample2_proposal - y_mu1[~y_update_idx]) / y_std1) ** 2 - torch.log(y_std1))
        acc_idx = (unif_dist.sample((torch.sum(~y_update_idx), 1)) > torch.exp(log_ratio.cpu())).squeeze(1)
        count += 1
        if count > 100:
            y_sample2[~y_update_idx] = y_sample1[~y_update_idx]
            # print("count > 100, fill %d samples with y_sample1" % torch.sum(~y_update_idx).item())
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
            weight_diff1 = torch.sum(torch.abs(current_s_sum1 - sdp_obs) -
                                     torch.abs(current_s_sum1 - current_s1[random_index[i]] + proposed_s1[i] - sdp_obs))
            weight_diff2 = torch.sum(torch.abs(current_s_sum2 - sdp_obs) -
                                     torch.abs(current_s_sum2 - current_s2[random_index[i]] + proposed_s2[i] - sdp_obs))
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
            weight_diff1 = torch.sum(torch.abs(current_s_sum1 - sdp_obs) -
                                     torch.abs(current_s_sum1 - current_s1[i] + proposed_s1[i] - sdp_obs))
            weight_diff2 = torch.sum(torch.abs(current_s_sum2 - sdp_obs) -
                                     torch.abs(current_s_sum2 - current_s2[i] + proposed_s2[i] - sdp_obs))
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
            acc_idx = unif_dist.sample((batch_size,)) > torch.exp(
                gamma_dist1.log_prob(gamma_samp2_batch) - gamma_dist2.log_prob(gamma_samp2_batch)).cpu()
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
            acc_idx = unif_dist.sample((batch_size,)) > torch.exp(
                normal_dist1.log_prob(normal_samp2_batch) - normal_dist2.log_prob(normal_samp2_batch)).cpu()
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
    parser.add_argument('--ncomp', type=int, default=100, help='n_component')  # number of components
    parser.add_argument('--mkstr', type=str, default="regression", help='mark_str')  # mark string
    parser.add_argument('--step', type=int, default=10000, help='total_steps')  # total steps
    parser.add_argument('--data', type=int, default=3, help='dataset')  # 3: Bayesian Linear Regression
    parser.add_argument('--enable_fit', type=int, default=1, help='enable_mcmc_fit')  # 0: disable; 1: enable
    parser.add_argument('--enable_coupling', type=int, default=1, help='enable_coupling')  # 0: disable; 1: enable
    parser.add_argument('--seed', type=int, default=10000, help='random seed')  # default value: 0
    parser.add_argument('--eps', type=float, default=3.0, help='epsilon')
    parser.add_argument('--tqdm', type=int, default=1, help='enable_tqdm')
    parser.add_argument('--dbg1', type=int, default=0, help='debug_flag_1')
    parser.add_argument('--dbg2', type=str, default="", help='debug_flag_1')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    main(args, config)
