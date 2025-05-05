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

## OPTIONS
| Name | Description | Type / Range | Default Value |
|------|-------------|---------------|----------------|
| strategy.name | Selected strategy to run | str | FedAvgPlus |
| strategy.options.fraction_fit | Fraction of clients used in fit() | float, 0.0–1.0 | 1.0 |
| strategy.options.fraction_evaluate | Fraction of clients used in evaluate() | float, 0.0–1.0 | 1.0 |
| strategy.options.min_fit_clients | Minimum number of clients to be used in fit() | int, ≥ 1 | 2 |
| strategy.options.min_evaluate_clients | Minimum number of clients to be used in evaluate() | int, ≥ 1 | 2 |
| strategy.options.min_available_clients | Minimum number of available clients required | int, ≥ 1 | 2 |
| strategy.options.accept_failures | Accept failures during rounds | bool | true |
| strategy.FedAvgPlus.options.lr_new | Learning rate for newly joining clients | float, > 0 | 0.005 |
| strategy.FedAvgPlus.options.epochs_new | Epochs for new clients | int, ≥ 0 | 1 |
| strategy.FedAvgPlus.options.decay_round | Round at which to decay new client influence | int, ≥ 0 | 1 |
| strategy.FedProx.options.proximal_mu | Proximal term coefficient for FedProx | float, ≥ 0 | 0.5 |
| strategy.Bulyan.options.num_malicious_clients | Number of malicious clients assumed | int, ≥ 0 | 1 |
| strategy.FedAvgM.options.server_learning_rate | Server-side learning rate | float, > 0 | 0.01 |
| strategy.FedAvgM.options.server_momentum | Server momentum for aggregation | float, 0–1 | 0.7 |
| strategy.FaultTolerantFedAvg.options.min_completion_rate_fit | Min. client completion rate in fit() | float, 0.0–1.0 | 0.5 |
| strategy.FaultTolerantFedAvg.options.min_completion_rate_evaluate | Min. client completion rate in evaluate() | float, 0.0–1.0 | 0.5 |
| strategy.FedAdagrad.options.eta | Learning rate (global) | float, > 0 | 0.05 |
| strategy.FedAdagrad.options.eta_l | Learning rate (local) | float, > 0 | 0.05 |
| strategy.FedAdagrad.options.tau | Stability constant | float, > 0 | 1e-8 |
| strategy.FedOpt.options.eta | Global learning rate | float, > 0 | 0.05 |
| strategy.FedOpt.options.eta_l | Local learning rate | float, > 0 | 0.05 |
| strategy.FedOpt.options.tau | Stability constant | float, > 0 | 1e-8 |
| strategy.FedOpt.options.beta_1 | Momentum factor β₁ | float, 0–1 | 0.9 |
| strategy.FedOpt.options.beta_2 | Momentum factor β₂ | float, 0–1 | 0.99 |
| strategy.FedYogi.options.eta | Global learning rate | float, > 0 | 0.05 |
| strategy.FedYogi.options.eta_l | Local learning rate | float, > 0 | 0.05 |
| strategy.FedYogi.options.tau | Stability constant | float, > 0 | 1e-8 |
| strategy.FedYogi.options.beta_1 | Yogi optimizer β₁ | float, 0–1 | 0.9 |
| strategy.FedYogi.options.beta_2 | Yogi optimizer β₂ | float, 0–1 | 0.99 |
| strategy.QFedAvg.options.q_param | Fairness parameter q | float, > 0 | 0.2 |
| strategy.QFedAvg.options.qffl_learning_rate | Q-FFL client learning rate | float, > 0 | 0.1 |
| strategy.FedTrimmedAvg.options.beta | Trimming parameter for trimmed mean | float, 0–1 | 0.2 |
| strategy.FedXgbCyclic.options.num_evaluation_clients | Evaluation clients for XGBoost | str or int | "" |
| strategy.FedXgbCyclic.options.num_fit_clients | Fit clients for XGBoost | str or int | "" |
| dp.flag | Enable differential privacy | str, 'true' or 'false' | true |
| dp.side | Where to apply DP (client or server) | str | client |
| dp.type | DP method used | str | adaptive |
| dp.fixed.noise_multiplier | Noise added to gradients | float, > 0 | 0.1 |
| dp.fixed.clipping_norm | Gradient clipping threshold | float, > 0 | 0.5 |
| dp.fixed.num_sampled_clients | Number of sampled clients | int, ≥ 1 | 2 |
| dp.adaptive.noise_multiplier | Noise multiplier for DP | float, > 0 | 0.1 |
| dp.adaptive.num_sampled_clients | Number of clients sampled per round | int, ≥ 1 | 2 |
| dp.adaptive.initial_clipping_norm | Initial norm for adaptive clipping | float, > 0 | 0.1 |
| dp.adaptive.target_clipped_quantile | Target quantile for clipping | float, 0–1 | 0.5 |
| dp.adaptive.clip_norm_lr | Clipping norm learning rate | float, > 0 | 0.2 |
| dp.adaptive.clipped_count_stddev | Stddev for clipped count noise | str or float | " " |