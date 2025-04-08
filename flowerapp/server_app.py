"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.common import ndarrays_to_parameters, Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flowerapp.my_strategy import CustomFedAvg
from flowerapp.task import get_weights, set_weights, test, get_model_class
from typing import List, Tuple
import json
import importlib
from flwr.server.strategy import (
    Bulyan,
    DPFedAvgAdaptive,
    DPFedAvgFixed,
    DifferentialPrivacyClientSideAdaptiveClipping,
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
    DifferentialPrivacyServerSideFixedClipping,
    FaultTolerantFedAvg,
    FedAdagrad,
    FedAdam,
    FedAvg,
    FedAvgAndroid,
    FedAvgM,
    FedMedian,
    FedOpt,
    FedProx,
    FedTrimmedAvg,
    FedXgbBagging,
    FedXgbCyclic,
    FedXgbNnAvg,
    FedYogi,
    Krum,
    QFedAvg,
    Strategy,
)
from flowerapp.custom_strategy import CustomStrategy 

# def get_strategy(strategy_name: str, **kwargs):
#     """Dynamically instantiate a strategy given its name."""
#     try:
#         strategy_class = globals().get(strategy_name)  # Get class from global namespace
#         if strategy_class and issubclass(strategy_class, Strategy):
#             return strategy_class(**kwargs)  # Instantiate with provided kwargs
#         else:
#             raise ValueError(f"Invalid strategy: {strategy_name}")
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
    
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

def weighted_average(metrics):
    """ Function that calculates weighted average of metrics, to be passed at: evaluate_metrics_aggregation_fn """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) }

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

def get_evaluate_fn(testloader, device):
    """Return a function that performs evaluation of the global model, to be passed at: evaluate_fn """
    def evaluate(server_round,net,parameters_ndarrays,config):
        
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net,testloader,device)
        return loss, {"cen_accuracy": accuracy}
    
    return evaluate

def handle_fit_metrics(metrics: List[Tuple[int,Metrics]])-> Metrics:
    """ handle metrics from fit method in clients, passed at strategy callback: fit_metrics_aggregation_fn """
    b_values = []
    for _, m in metrics :
        my_metric_str=m["my_metric"]
        my_metric = json.loads(my_metric_str) # convert it back to dictionary
        b_values.append(my_metric["b"])
    return {"max_b":max(b_values)}



# def server_fn(context: Context) -> ServerAppComponents:
#     #initalize global model
#     net=get_model_class(context.run_config["model"])
#     params = ndarrays_to_parameters(get_weights(net))


#     # Retrieve function names from configuration
#     eval_fn_name = context.run_config.get("evaluate_fn", "")
#     fit_fn_name = context.run_config.get("on_fit_config_fn", "")
#     eval_config_fn_name = context.run_config.get("on_evaluate_config_fn", "")
#     evaluate_metrics_aggregation_fn_name=context.run_config.get("evaluate_metrics_aggregation_fn", "")
    
#     # Dynamically get the functions (if they exist)
#     eval_fn = get_function_from_string(eval_fn_name)
#     fit_config_fn = get_function_from_string(fit_fn_name)
#     eval_config_fn = get_function_from_string(eval_config_fn_name)
#     evaluate_metrics_aggregation_fn=get_function_from_string(evaluate_metrics_aggregation_fn_name)
#     strategy_name = context.run_config["strategy"]
#     strategy_opts = {'initial_parameters':params,
#                      'evaluate_metrics_aggregation_fn':weighted_average}
    
#     strategy = get_strategy(strategy_name, **strategy_opts)

#     strategy = CustomStrategy(strategy=strategy)
#     num_rounds = context.run_config["num-server-rounds"] # options under app.config become available at context.run_config
    
#     config = ServerConfig(num_rounds=num_rounds)
    
#     return ServerAppComponents(config=config, strategy=strategy)

# def server_fn(context: Context) -> ServerAppComponents:
#     # Initialize global model
#     net = get_model_class(context.run_config["model"])
#     params = ndarrays_to_parameters(get_weights(net))

#     # Collect all optional strategy arguments from config
#     optional_strategy_keys = [
#         "evaluate_fn",
#         "on_fit_config_fn",
#         "on_evaluate_config_fn",
#         "initial_parameters",
#         "fit_metrics_aggregation_fn",
#         "evaluate_metrics_aggregation_fn",
#         "fraction_fit",
#         "fraction_evaluate",
#         "min_fit_clients",
#         "min_evaluate_clients",
#         "min_available_clients",
#         "accept_failures",
#         "inplace",
#         "proximal_mu"
        
#     ]

#     strategy_opts = {}

#     for key in optional_strategy_keys:
#         value = context.run_config.get(key, None)

#         # Convert string to actual function if it's a function path/name
#         if key.endswith("_fn") or "aggregation_fn" in key:
#             value = get_function_from_string(value) if value else None

#         # Skip if the value is empty or None
#         if value not in ("", None):
#             strategy_opts[key] = value

#     # Always include the initial model parameters
#     strategy_opts["initial_parameters"] = params

#     # Get strategy class and instantiate it
#     strategy_name = context.run_config["strategy"]
#     strategy = get_strategy(strategy_name, **strategy_opts)

#     # Wrap it if needed in a custom strategy
#     strategy = CustomStrategy(strategy=strategy)

#     num_rounds = context.run_config["num-server-rounds"]

#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(config=config, strategy=strategy)
import flwr
def get_strategy(name: str, return_class: bool = False, **kwargs):
    
    strategies = {
        "Bulyan": flwr.server.strategy.Bulyan,
        "DPFedAvgAdaptive": flwr.server.strategy.DPFedAvgAdaptive,
        "DPFedAvgFixed": flwr.server.strategy.DPFedAvgFixed,
        "DifferentialPrivacyClientSideAdaptiveClipping": flwr.server.strategy.DifferentialPrivacyClientSideAdaptiveClipping,
        "DifferentialPrivacyClientSideFixedClipping": flwr.server.strategy.DifferentialPrivacyClientSideFixedClipping,
        "DifferentialPrivacyServerSideAdaptiveClipping": flwr.server.strategy.DifferentialPrivacyServerSideAdaptiveClipping,
        "DifferentialPrivacyServerSideFixedClipping": flwr.server.strategy.DifferentialPrivacyServerSideFixedClipping,
        "FaultTolerantFedAvg": flwr.server.strategy.FaultTolerantFedAvg,
        "FedAdagrad": flwr.server.strategy.FedAdagrad,
        "FedAdam": flwr.server.strategy.FedAdam,
        "FedAvg": flwr.server.strategy.FedAvg,
        "FedAvgAndroid": flwr.server.strategy.FedAvgAndroid,
        "FedAvgM": flwr.server.strategy.FedAvgM,
        "FedMedian": flwr.server.strategy.FedMedian,
        "FedOpt": flwr.server.strategy.FedOpt,
        "FedProx": flwr.server.strategy.FedProx,
        "FedTrimmedAvg": flwr.server.strategy.FedTrimmedAvg,
        "FedXgbBagging": flwr.server.strategy.FedXgbBagging,
        "FedXgbCyclic": flwr.server.strategy.FedXgbCyclic,
        "FedXgbNnAvg": flwr.server.strategy.FedXgbNnAvg,
        "FedYogi": flwr.server.strategy.FedYogi,
        "Krum": flwr.server.strategy.Krum,
        "QFedAvg": flwr.server.strategy.QFedAvg,
        # Add more strategies here
    }
    cls = strategies.get(name)
    if return_class:
        return cls
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return cls(**kwargs)

import inspect

def server_fn(context: Context) -> ServerAppComponents:
    # Initialize model and parameters
    net = get_model_class(context.run_config["model"])
    params = ndarrays_to_parameters(get_weights(net))

    strategy_name = context.run_config["strategy"]

    # Step 1: Get the actual strategy class (not an instance)
    strategy_cls = get_strategy(strategy_name, return_class=True)  # You'll need to update get_strategy to support this

    # Step 2: Get the accepted parameter names for the constructor
    sig = inspect.signature(strategy_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Step 3: Build filtered strategy options from context.run_config
    strategy_opts = {}

    for key, value in context.run_config.items():
        if key in valid_params:
            # Convert strings representing functions to actual callables
            if key.endswith("_fn") or "aggregation_fn" in key:
                value = get_function_from_string(value) if value else None

            # Pass only valid, non-empty values
            if value not in (None, "", "null"):
                strategy_opts[key] = value

    # Always override initial_parameters
    strategy_opts["initial_parameters"] = params

    # Step 4: Instantiate strategy
    try:
        strategy = strategy_cls(**strategy_opts)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize strategy '{strategy_name}' with args {strategy_opts}: {e}")

    # Step 5: Wrap with CustomStrategy if needed
    strategy = CustomStrategy(strategy=strategy)

    # Step 6: Build server config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)

app = ServerApp(server_fn=server_fn)