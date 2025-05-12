---
tags: [prototype,docker, version 0.1.2,flwr_version 1.17]
dataset: [cifar]
framework: [torch]
---

# Federated Learning component for RAIDO:
This repository is the source code of RAIDO's Federated Learning component. It is built on Flower and Pytorch Frameworks.
It includes Docker Files for local and distributed Deployment of a Federation.


## project code structure
``` shell
FLOWER-APP/
├── data/                   # Local data sources
├── docker/                 # Dockerfiles for deployment
├── flowerapp/              # Main application code
│   ├── strategies/         # Custom federated learning strategies
│   │   
│   ├── tasks/              # Task definitions (e.g., classification, etc.)
│   │   ├── ImageClassification.py
│   │   └── Task.py
│   ├── client_app.py       # Client-side application logic
│   ├── models.py           # Machine learning models
│   ├── server_app.py       # Server-side application logic
│   ├── utils.py            # shared utilities for server_app and client_app
│   └── __init__.py
├── output/                 # Output results
│   └── results.json        # Training/evaluation results
├── index.html              
├── pyproject.toml          # Project metadata like dependencies and configs
└── README.md               
```
## Set up the project

The setup follows the Docker guide on: https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html

If you don't have Docker installed in your system you can download here: https://www.docker.com/get-started/

After you have installed Docker follow these steps to deploy and run the application:

First create the docker images, from inside the flower-app directory run:
```shell
cd docker
```
```shell
docker build -f clientapp.Dockerfile -t flwr_clientapp:0.0.3../
``` 
```shell
docker build -f serverapp.Dockerfile -t flwr_serverapp:0.0.3 ../

``` 
For a local deployment simply run: 

```shell
docker compose -f complete/compose.yml up --build -d
```

after the containers are started, you can start a Federated Learning run with: ( from linux terminal )

```shell
flwr run . local-deployment-docker --stream
```
or (recommended) use the index.html, run it with a web server.
(you can download a liver server here: https://simplewebserver.org/ or use liveserver plug-in for VScode)


# DOCUMENTATION

# ⚙️ Federated Learning Strategy Configuration Options

This document outlines all the available strategy and privacy configuration options for your federated learning setup. Options are grouped by strategy and feature.

---

##  common strategy Options ( FedAvg )

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.name` | Selected strategy for Federated Learning | `str` | `FedAvg` |
| `strategy.options.fraction_fit` | Fraction of clients used in `fit()` | `float`, 0.0–1.0 | `1.0` |
| `strategy.options.fraction_evaluate` | Fraction of clients used in `evaluate()` | `float`, 0.0–1.0 | `1.0` |
| `strategy.options.min_fit_clients` | Minimum number of clients in `fit()` | `int`, ≥ 1 | `2` |
| `strategy.options.min_evaluate_clients` | Minimum number of clients in `evaluate()` | `int`, ≥ 1 | `2` |
| `strategy.options.min_available_clients` | Minimum required available clients | `int`, ≥ 1 | `2` |
| `strategy.options.accept_failures` | Accept client failures | `bool` | `true` |

---

## FedAvgPlus

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedAvgPlus.options.lr_new` | Learning rate for new clients | `float`, > 0 | `0.005` |
| `strategy.FedAvgPlus.options.epochs_new` | Epochs for new clients | `int`, ≥ 0 | `1` |
| `strategy.FedAvgPlus.options.decay_round` | Round to decay new client influence | `int`, ≥ 0 | `1` |

---

## FedProx

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedProx.options.proximal_mu` | Proximal term coefficient | `float`, ≥ 0 | `0.5` |

---

## Bulyan

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.Bulyan.options.num_malicious_clients` | Assumed number of malicious clients | `int`, ≥ 0 | `1` |

---

## FedAvgM

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedAvgM.options.server_learning_rate` | Server-side learning rate | `float`, > 0 | `0.01` |
| `strategy.FedAvgM.options.server_momentum` | Server aggregation momentum | `float`, 0–1 | `0.7` |

---

## FaultTolerantFedAvg

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FaultTolerantFedAvg.options.min_completion_rate_fit` | Minimum fit completion rate | `float`, 0.0–1.0 | `0.5` |
| `strategy.FaultTolerantFedAvg.options.min_completion_rate_evaluate` | Minimum evaluate completion rate | `float`, 0.0–1.0 | `0.5` |

---

## FedAdagrad

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedAdagrad.options.eta` | Global learning rate | `float`, > 0 | `0.05` |
| `strategy.FedAdagrad.options.eta_l` | Local learning rate | `float`, > 0 | `0.05` |
| `strategy.FedAdagrad.options.tau` | Stability constant | `float`, > 0 | `1e-8` |

---

## FedOpt

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedOpt.options.eta` | Global learning rate | `float`, > 0 | `0.05` |
| `strategy.FedOpt.options.eta_l` | Local learning rate | `float`, > 0 | `0.05` |
| `strategy.FedOpt.options.tau` | Stability constant | `float`, > 0 | `1e-8` |
| `strategy.FedOpt.options.beta_1` | Momentum β₁ | `float`, 0–1 | `0.9` |
| `strategy.FedOpt.options.beta_2` | Momentum β₂ | `float`, 0–1 | `0.99` |

---

## FedYogi

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedYogi.options.eta` | Global learning rate | `float`, > 0 | `0.05` |
| `strategy.FedYogi.options.eta_l` | Local learning rate | `float`, > 0 | `0.05` |
| `strategy.FedYogi.options.tau` | Stability constant | `float`, > 0 | `1e-8` |
| `strategy.FedYogi.options.beta_1` | β₁ for Yogi optimizer | `float`, 0–1 | `0.9` |
| `strategy.FedYogi.options.beta_2` | β₂ for Yogi optimizer | `float`, 0–1 | `0.99` |

---

## QFedAvg

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.QFedAvg.options.q_param` | Fairness parameter `q` | `float`, > 0 | `0.2` |
| `strategy.QFedAvg.options.qffl_learning_rate` | Client learning rate | `float`, > 0 | `0.1` |

---

## FedTrimmedAvg

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedTrimmedAvg.options.beta` | Trimming parameter | `float`, 0–1 | `0.2` |

---

## FedXgbCyclic

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `strategy.FedXgbCyclic.options.num_evaluation_clients` | Evaluation clients | `str` or `int` | `""` |
| `strategy.FedXgbCyclic.options.num_fit_clients` | Fit clients | `str` or `int` | `""` |

---

## Differential Privacy (DP)

### Common DP Options

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `dp.flag` | Enable DP | `str`, `'true'` or `'false'` | `true` |
| `dp.side` | Apply DP on `client` or `server` | `str` | `client` |
| `dp.type` | Type of DP (`fixed` or `adaptive`) | `str` | `adaptive` |

### Fixed DP

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `dp.fixed.noise_multiplier` | Noise multiplier | `float`, > 0 | `0.1` |
| `dp.fixed.clipping_norm` | Gradient clipping norm | `float`, > 0 | `0.5` |
| `dp.fixed.num_sampled_clients` | Sampled clients | `int`, ≥ 1 | `2` |

### 🔁 Adaptive DP

| Name | Description | Type / Range | Default |
|------|-------------|---------------|---------|
| `dp.adaptive.noise_multiplier` | Noise multiplier | `float`, > 0 | `0.1` |
| `dp.adaptive.num_sampled_clients` | Sampled clients | `int`, ≥ 1 | `2` |
| `dp.adaptive.initial_clipping_norm` | Initial clipping norm | `float`, > 0 | `0.1` |
| `dp.adaptive.target_clipped_quantile` | Target clipping quantile | `float`, 0–1 | `0.5` |
| `dp.adaptive.clip_norm_lr` | Learning rate for clipping norm | `float`, > 0 | `0.2` |
| `dp.adaptive.clipped_count_stddev` | Stddev for clipped count noise | `str` or `float` | `" "` |