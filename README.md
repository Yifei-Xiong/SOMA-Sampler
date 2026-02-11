# SOMA: A Novel Sampler for Bayesian Inference from Privatized Data

This repository provides the official implementation of **SOMA**, a specialized MCMC sampler designed for Bayesian inference when the available data has been privatized (e.g., via Differential Privacy mechanisms).

Arxiv link: [![arXiv](https://img.shields.io/badge/arXiv-2505.00635-b31b1b.svg)](https://arxiv.org/2505.00635)

## Repository Contents
- `sampler.py`: Core implementation. Contains abstract classes for SOMA and IMwG algorithms.
- `main.py`: Experiment on a synthetic example.
- `regression.py`: Bayesian Linear Regression experiments.
- `atus.py`: Real-world application using the American Time Use Survey (ATUS) dataset.
