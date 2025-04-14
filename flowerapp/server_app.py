"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""
from flwr.common import ndarrays_to_parameters, Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flowerapp.my_strategy import CustomFedAvg
from flowerapp.task import get_weights, set_weights, test, get_model_class
from typing import List, Tuple
import json
import importlib
from flowerapp.custom_strategy import CustomStrategy
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
import inspect
import flwr.server.strategy.fedavg

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


# from functools import lru_cache
# @lru_cache(maxsize=None) # for heavy loads and multiple users
def get_strategy(strategy_name: str):
    module = __import__("flwr.server.strategy", fromlist=[strategy_name])
    return getattr(module, strategy_name)

app = ServerApp()
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
import torch
@app.main()
def main(grid: Grid, context: Context) -> None:
    
    # Initialize global model
    net = get_model_class(context.run_config["model"])
    model_dict = True if context.run_config["model_dict"].lower()=="true" else False
    params = ndarrays_to_parameters(get_weights(net))
    if model_dict:
        PATH = context.run_config["model_dict_path"]
        net.load_state_dict(torch.load(PATH, weights_only=True))
        params=ndarrays_to_parameters(get_weights(net))
    else:
        params = ndarrays_to_parameters(get_weights(net))
        
    # Initialize Strategy    
    strategy_name = context.run_config["strategy"]
    strategy_cls = get_strategy(strategy_name)
    sig = inspect.signature(strategy_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    strategy_opts = {}
    for key, value in context.run_config.items():
        if key in valid_params:
            if key.endswith("_fn") or "aggregation_fn" in key:
                value = get_function_from_string(value) if value else None
                
            if value not in (None, "", "null"):
                strategy_opts[key] = value
    strategy_opts["initial_parameters"] = params
    try:
        strategy = strategy_cls(**strategy_opts,
                                evaluate_metrics_aggregation_fn=weighted_average,
                                )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize strategy '{strategy_name}' with args {strategy_opts}: {e}")
    num_rounds = context.run_config["num-server-rounds"]
    # Wrap with custom messages for strategies
    strategy = CustomStrategy(strategy=strategy,num_rounds=num_rounds,model=context.run_config["model"],context=context)


    # enable Differential Privacy if set in the configurations:
    dp = True if context.run_config["dp"].lower()=="true" else False
    if dp:
        noise_multiplier = context.run_config["noise-multiplier"]
        clipping_norm = context.run_config["clipping-norm"]
        print(clipping_norm)
        num_sampled_clients = context.run_config["num-sampled-clients"]
        
        strategy = DifferentialPrivacyClientSideFixedClipping(
            strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_sampled_clients
        )
        
 
    
    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    # create secure aggregation workflow if set in the configs.
    sec_agg=True if context.run_config["sec-agg"].lower()=="true" else False
    print(sec_agg)
    if sec_agg:
        workflow = DefaultWorkflow(
                fit_workflow=SecAggPlusWorkflow(
                num_shares=context.run_config["num-shares"],
                reconstruction_threshold=context.run_config["reconstruction-threshold"],
            )
        )
    else:
        workflow = DefaultWorkflow()
    
    # Execute
    workflow(grid, context)