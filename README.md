---
tags: [prototype, version 0.1.4 , flower version 1.17]
dataset: [cifar]
framework: [flower, torch]
---

# Federated Learning Component source code

This code is the source code for the RAIDO platform. It is built with Flower AI and Pytorch Frameworks.
It includes Docker Files for local and distributed Deployment of a Federation.
More information on the system can be found here.

## Set up the project

The setup follows the Docker guide on:

For a local deployment simply run: 
```shell
docker compose -f complete/compose.yml up --built -d
```

after the containers are started, you can start a Federated Learning run with:
```shell
flwr run . local-deployment-docker --stream
```


### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-tabular . && rm -rf flower && cd fl-tabular
```

This will create a new directory called `fl-tabular` containing the following files:

```shell
fl-tabular
├── fltabular
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fltabular` package.

```shell
# From a new python environment, run:
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!NOTE\]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=10
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.


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