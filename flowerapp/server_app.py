"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""
from flwr.common import ndarrays_to_parameters, Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flowerapp.my_strategy import CustomFedAvg
from flowerapp.utils import get_weights, set_weights, get_model , drop_empty_keys , validate_options
from typing import List, Tuple
import json
import importlib
from flowerapp.strategies.custom_strategy import CustomStrategy
import inspect
import flwr.server.strategy.fedprox
from flwr.common.config import unflatten_dict
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import (DifferentialPrivacyClientSideFixedClipping,
                                  DifferentialPrivacyClientSideAdaptiveClipping,
                                  DifferentialPrivacyServerSideAdaptiveClipping,
                                  DifferentialPrivacyServerSideFixedClipping)
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
import torch

from omegaconf import DictConfig
import numpy as np

def get_function_from_string(func_name):
    """Dynamically get a function from its name."""
    if not func_name:
        return None  # If no function is provided, return None
    
    # Try getting from global scope (if defined in the same script)
    if func_name in globals():
        return globals()[func_name]

    # Try importing dynamically (if it's in another module)
    try:
        module_name, function_name = func_name.rsplit(".", 1)  # Example: "module.function"
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError, ValueError):
        raise ValueError(f"Function '{func_name}' could not be found or imported.")
    
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(weight for weight, _ in metrics)
    return {
        k: sum(examples * metric[k] for examples, metric in metrics) / total_examples
        for k in metrics[0][1].keys()
    }

def average(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    return {k: np.mean([metric[k] for _, metric in metrics]) for k in metrics[0][1].keys()}


# def weighted_average(metrics):
#     """ Function that calculates weighted average of metrics, to be passed at: evaluate_metrics_aggregation_fn """
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     return {"accuracy": sum(accuracies) / sum(examples) }




def personalized_metrics(metrics: List[Tuple[int,Metrics]])-> Metrics:
    """ A function to run evaluation on all clients and print the results of each client to be passed at: evaluate_metrics_aggregation_fn """
    accuracies = [m["accuracy"] for _ , m in metrics]
    return{"accuracy": accuracies}

def on_fit_config(server_round: int)-> Metrics:
    """Adjust learning rate based on round, to be passed at: on_fit_config_fn """
    # for this to work, update config object at client up
    lr = 0.01
    if server_round > 2:
        lr = 0.005   
    return {"lr":lr}

# def get_evaluate_fn(testloader, device):
#     """Return a function that performs evaluation of the global model, to be passed at: evaluate_fn """
#     def evaluate(server_round,net,parameters_ndarrays,config):
        
#         set_weights(net, parameters_ndarrays)
#         net.to(device)
#         loss, accuracy = test(net,testloader,device)
#         return loss, {"cen_accuracy": accuracy}
    
#     return evaluate

def handle_fit_metrics(metrics: List[Tuple[int,Metrics]])-> Metrics:
    """ handle metrics from fit method in clients, passed at strategy callback: fit_metrics_aggregation_fn """
    b_values = []
    for _, m in metrics :
        my_metric_str=m["my_metric"]
        my_metric = json.loads(my_metric_str) # convert it back to dictionary
        b_values.append(my_metric["b"])
    return {"max_b":max(b_values)}


# from functools import lru_cache
# @lru_cache(maxsize=None) # for heavy loads and multiple users
def get_strategy(strategy_name: str ,  **strategy_opts):
     # strategies integrated in flower
    if strategy_name=="FedAvgPlus" or strategy_name=="Scaffold":
        module = __import__(f"flowerapp.strategies.{strategy_name}", fromlist=[strategy_name]) # custom strategies from custom_strategy module
    else:
        module = __import__("flwr.server.strategy", fromlist=[strategy_name])
    strategyClass= getattr(module, strategy_name)
    try:
        strategy_opts = validate_options( strategyClass,strategy_opts)
        strategy = strategyClass(**strategy_opts,
                                evaluate_metrics_aggregation_fn=weighted_average,
                                )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize strategy '{strategy_name}' with args {strategy_opts}: {e}")
    return strategy

#initialize app
app = ServerApp()


# Setup serverapp
@app.main()
def main(grid: Grid, context: Context) -> None:
    # create nested config Dictionary:
    cfg = DictConfig(drop_empty_keys(unflatten_dict(context.run_config)))

    # Initialize global model
    net = get_model(cfg.model.name,**cfg.model.options)
    pretrained = True if cfg.model.pre_trained.lower()=="true" else False
    if pretrained:
        net.load_state_dict(torch.load(cfg.model.path, weights_only=True))
        params=ndarrays_to_parameters(get_weights(net))
    else:
        params = ndarrays_to_parameters(get_weights(net))
        
    # Initialize Strategy:   
    strategy_name = cfg.strategy.name
    if strategy_name in cfg.strategy:
        strategy_opts = {**cfg.strategy.options, **cfg.strategy[strategy_name]["options"]} # pass common and class-specific strategy options
    else:
        strategy_opts={**cfg.strategy.options}

    strategy_opts["initial_parameters"] = params # pass initial parameters to strategy
    strategy = get_strategy(strategy_name,**strategy_opts)
    
    # read number of rounds:
    num_rounds = context.run_config["num-server-rounds"]
    
    # First Wrapper for custom messages:
    strategy = CustomStrategy(strategy=strategy,
                              num_rounds=num_rounds,
                              model=cfg.model.name,
                              run_id=context.run_id,
                              configs=cfg)


    # enable Differential Privacy:
    dp = cfg.dp
    if dp.flag.lower() == "true":
        if  dp.side == "client":
            if dp.type =="fixed":
                strategy = DifferentialPrivacyClientSideFixedClipping(
                    strategy,
                    **dp.fixed
                )
            else:
                strategy = DifferentialPrivacyClientSideAdaptiveClipping(
                    strategy,
                    **dp.adaptive
                )
        else:
            if dp.type =="fixed":
                strategy = DifferentialPrivacyServerSideFixedClipping(
                    strategy,
                    **dp.fixed
                )
            else:
                strategy = DifferentialPrivacyServerSideAdaptiveClipping(
                    strategy,
                    **dp.adaptive
                )
            

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    
    # create secure aggregation workflow if set in the configs.
    if cfg.secagg.flag.lower() == "true":
        workflow = DefaultWorkflow(
                fit_workflow=SecAggPlusWorkflow(
                num_shares=cfg.secagg.num_shares,
                reconstruction_threshold=cfg.secagg.reconstruction_threshold,
            )
        )
    else:
        # Create default train/evaluate workflow
        workflow = DefaultWorkflow()
    
    # Execute
    workflow(grid, context)