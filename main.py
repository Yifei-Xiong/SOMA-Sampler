# Code for synthetic example and perturbed histograms

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from tqdm import tqdm
from scipy.stats import wasserstein_distance

import Dataset
from sampler import SomaSampler, RandomScanSampler, SystematicScanSampler


def main(args, config):
    # init args
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    FileSavePath = config['FileSavePath_linux'] if os.name == 'posix' else config['FileSavePath_win']
    print("File Save Path: " + FileSavePath)
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
    ModelInfo = "mk+" + args.mkstr + "_eps+" + str(args.eps) + "_ncomp+" + str(args.ncomp) + "_me+" + \
                str(args.method) + "_data+" + str(args.data) + "_step+" + str(args.step) + "_seed+" + str(args.seed)
    ModelInfo_list = ["mk+" + args.mkstr + "_eps+" + str(args.eps) + "_ncomp+" + str(args.ncomp) + "_me+" + str(i) +
                      "_data+" + str(args.data) + "_step+" + str(args.step) + "_seed+" + str(args.seed) for i in [1, 2, 3]]

    # define simulator
    task_name_list = ['2d Synthetic Example', 'Perturbed Histograms']
    task_name = task_name_list[args.data - 1]
    print("task: " + task_name)
    ckpts_folder_name = "ckpts"
    task = Dataset.Task(args.data, seed=args.seed, eps=args.eps, n_component=args.ncomp, device=args.device, args=args)
    proposal = task.proposal_dist

    # define sampler
    if args.method == 1:
        sampler = SomaSampler(args.ncomp, task, args.device, args.tqdm)
        sampler_name = "SOMA"
    elif args.method == 2:
        sampler = RandomScanSampler(args.ncomp, task, args.device, args.tqdm)
        sampler_name = "Ran_IMwG"
    elif args.method == 3:
        sampler = SystematicScanSampler(args.ncomp, task, args.device, args.tqdm)
        sampler_name = "Sys_IMwG"
    else:
        raise NotImplementedError
    print("using sampler: " + sampler_name)

    # start sampling
    if args.enable_fit:
        state_init = torch.cat([proposal.sample((1,)) for _ in range(args.ncomp)], dim=0).to(args.device)
        all_states, acc_rate = sampler.fit(state_init, args.step)
        print("acceptance rate: %.3f" % acc_rate)
        torch.save((all_states[-1].clone(), acc_rate), FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + ".pt")

    # coupling
    if args.enable_coupling:
        torch.manual_seed(args.seed + 10000)  # avoid state_init2 has some same components with state_init1
        torch.cuda.manual_seed(args.seed + 10000)
        np.random.seed(args.seed + 10000)

        state_init1 = torch.cat([proposal.sample((1,)) for _ in range(args.ncomp)], dim=0).to(args.device)
        if task.target_density is not None:
            state_init2 = task.target_density.sample((1,)).to(args.device).t()
        else:
            state_init2, _ = torch.load(FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo_list[0] + ".pt", weights_only=True)
        all_states1, all_states2, coupling_time, acc_rate1, acc_rate2, distance = sampler.coupling_fit(state_init1, state_init2, args.step)
        print("acceptance rate 1: %.3f, acceptance rate 2: %.3f" % (acc_rate1, acc_rate2))
        torch.save((coupling_time, acc_rate1, acc_rate2, distance.to(torch.int16)),
                   FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_distance.pt")

    # ------ for fitting task ------

    # save quantile *.pt file
    if args.enable_fit and args.data == 2 and config['save_quantiles']:
        quantiles = torch.quantile(all_states, torch.tensor([0.05, 0.5, 0.95], device=args.device), dim=1).cpu()  # shape: (7, ncomp)
        torch.save(quantiles.half(), FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_quantiles.pt")

    # plot Ridgeline
    if args.enable_fit and args.data == 2 and config['plot_ridgeline']:  # perturbed histograms
        ridge_num = 10
        select_index = np.linspace(0, args.step, ridge_num + 1, dtype=int)[1:]
        fig, axes = plt.subplots(ridge_num, 1, figsize=(8, 10), sharex=True, sharey=True, constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.hist(all_states[select_index[i] - 1].cpu().numpy(), bins=task.n_bins, range=(task.bin_range[0], task.bin_range[1]),
                    color="skyblue", edgecolor="black", alpha=0.7, linewidth=1.5, density=False)
            ax.set_ylabel("iter: %d" % select_index[i], rotation=0)
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        if args.method == 1:
            axes[0].set_title("SOMA")
        elif args.method == 2:
            axes[0].set_title("Ran_IMwG")
        elif args.method == 3:
            axes[0].set_title("Sys_IMwG")
        plt.tight_layout()
        plt.savefig(FileSavePath + os.sep + "fig" + os.sep + "ridgeplot_" + ModelInfo + ".png", dpi=300)
        plt.close()

    # plot trace
    if args.enable_fit and args.data == 1 and config['plot_trace']:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(all_states[:, 0].cpu().numpy(), all_states[:, 1].cpu().numpy(), s=2)
        ax.plot(all_states[:, 0].cpu().numpy(), all_states[:, 1].cpu().numpy(), color='gray', alpha=0.5, linewidth=0.5)
        ax.set_xlabel("Coord 1")
        ax.set_ylabel("Coord 2")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title("{}, eps: {:.1f}, steps: {}, acc_prob: {:.1%}".format(sampler_name, args.eps, args.step, acc_rate))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(FileSavePath + os.sep + "fig" + os.sep + "trace_" + ModelInfo + ".png", dpi=300)
        # save all states
        torch.save(all_states, FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_trace.pt")

    # ------ for coupling task ------

    # save coupling wasserstein distance *.pt file
    if args.enable_coupling and config['save_wasserstein']:
        wasserstein_dist = torch.zeros(args.step)
        for i in range(args.step):
            wasserstein_dist[i] = wasserstein_distance(all_states1[i].cpu().squeeze(1).numpy(), all_states2[i].cpu().squeeze(1).numpy())
        torch.save(wasserstein_dist.half(), FileSavePath + os.sep + ckpts_folder_name + os.sep + ModelInfo + "_wasserstein.pt")

    # plot the chain distance
    if args.enable_coupling and config['plot_distance']:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(distance)
        if coupling_time < args.step:
            ax.axvline(x=coupling_time, color='r', linestyle='--')
        ax.set_xlabel("iter")
        ax.set_ylabel("chain distance")
        ax.set_ylim(0, args.ncomp)
        ax.set_xlim(1, args.step)
        ax.set_xscale('log')
        if args.method == 1:
            ax.set_title("SOMA")
        elif args.method == 2:
            ax.set_title("Ran-IMwG")
        elif args.method == 3:
            ax.set_title("Sys-IMwG")
        plt.tight_layout()
        plt.savefig(FileSavePath + os.sep + "fig" + os.sep + "coupling_dist_" + ModelInfo + ".png", dpi=300)
        plt.close()

    # finish
    print("Finish task: " + ModelInfo)


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file path")
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--method', type=int, default=1, help='method')  # 1: SOMA; 2: random-scan; 3: systematic-scan
    parser.add_argument('--ncomp', type=int, default=2, help='n_component')  # number of components
    parser.add_argument('--mkstr', type=str, default="", help='mark_str')  # mark string
    parser.add_argument('--step', type=int, default=10000, help='total_steps')  # total steps
    parser.add_argument('--data', type=int, default=1, help='dataset')  # 1: Synthetic Example, 2: Perturbed Histograms
    parser.add_argument('--enable_fit', type=int, default=1, help='enable_mcmc')  # 0: disable; 1: enable
    parser.add_argument('--enable_coupling', type=int, default=1, help='enable_test')  # 0: disable; 1: enable
    parser.add_argument('--seed', type=int, default=10000, help='random seed')  # default value: 0
    parser.add_argument('--eps', type=float, default=5.0, help='epsilon')
    parser.add_argument('--tqdm', type=int, default=1, help='enable_tqdm')
    parser.add_argument('--hist_bin', type=int, default=10, help='perturb_hist_nbin')
    parser.add_argument('--dbg1', type=int, default=0, help='debug_flag_1')
    parser.add_argument('--dbg2', type=str, default="", help='debug_flag_1')
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)

    main(args, config)
