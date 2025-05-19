
from flwr.common import FitRes, Parameters, parameters_to_ndarrays , Context , ndarray_to_bytes , bytes_to_ndarray
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from logging import WARNING
#import wandb
from datetime import datetime
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import json
class Scaffold(FedAvg):
    
    def __init__(self,server_control_variate: NDArrays = None  ,eta_g: float = 0.01,*args,**kwargs):
            
        super().__init__(*args,**kwargs)
        self.eta_g = eta_g
        self.server_control_variate = None
        self.current_weights = None
     
    def initialize_parameters(self, client_manager):
        parameters = super().initialize_parameters(client_manager) 
        self.current_weights = parameters_to_ndarrays (parameters) 
        self.server_control_variate = [np.zeros_like(p) for p in self.current_weights]
        return parameters
        
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        array_list = [arr.tolist() for arr in self.server_control_variate]
        config["server_control_variate"] = json.dumps(array_list)
        fit_ins = FitIns(parameters, config)
        for i, arr in enumerate(self.server_control_variate):
            print(f"Index {i}: shape = {arr.shape}, type = {type(arr)}")
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]
        # return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results,failures):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        # if not super().accept_failures and failures:
        #     return None, {}

        # if super().inplace:
        #     # Does in-place weighted average of results
        #     aggregated_ndarrays = aggregate_inplace(results)
        # else:
        #     # Convert results
        #     weights_results = [
        #         (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #         for _, fit_res in results
        #     ]
        #     aggregated_ndarrays = aggregate(weights_results)
        print(type(results))
        for res in results: print(type(res))
        weights_results = [
                (parameters_to_ndarrays(fit_res.parameters))
                for _, fit_res in results
            ]
        aggregated_ndarrays = scaffold_aggregate(self.current_weights,weights_results,self.eta_g)
        self.current_weights = aggregated_ndarrays # update server model
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        for i, (sc, cw) in enumerate(zip(self.server_control_variate, self.current_weights)):
            if sc.shape != cw.shape:
             print(f"Shape mismatch at layer {i}: control {sc.shape}, weight {cw.shape}")
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        self.server_control_variate = aggregate_controls(fit_metrics,self.server_control_variate)
        for i, arr in enumerate(self.server_control_variate):
            print(f"Layer {i} max: {np.max(arr)}")
        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        return super().evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)

import numpy as np
from functools import reduce
from typing import List, Tuple

def aggregate_controls(
    fit_metrics: List[Tuple[int, dict]],
    server_control: List[np.ndarray]
) ->List[np.ndarray]:
    # Extract number of examples and new controls
    samples, deltas_str = zip(*[
        (num_examples, metrics["control_variate"])
        for num_examples, metrics in fit_metrics
    ])
    deltas = [ [np.array(arr) for arr in json.loads(delta_json)] for delta_json in deltas_str]
    sampled_clients = len(deltas)
    # Compute average delta for each layer
    avg_delta_controls = [
        sum(client_deltas[i] for client_deltas in deltas) / sampled_clients
        for i in range(len(server_control))
    ]
    updated_server_control = [
        sc + dc for sc, dc in zip(server_control, avg_delta_controls)
    ]
    return updated_server_control

def scaffold_aggregate(
    current_weights: NDArrays,
    results: List[NDArrays],
    eta_g: float,
) -> NDArrays:
    """SCAFFOLD aggregation:
    x ← x + (ηg / |S|) * sum_i (y_i - x)
    
    Args:
        current_weights: Server model weights (x).
        results: List of (client_weights, num_examples) tuples.
        eta_g: Server learning rate.
    
    Returns:
        Updated server weights.
    """
    num_clients = len(results)
    
    # Compute the deltas: (y_i - x) for each client
    deltas_per_client = [
        [client_w - server_w for client_w, server_w in zip(client_weights, current_weights)]
        for client_weights in results]

    # Average the deltas across clients
    delta_avg: NDArrays = [
        reduce(np.add, layer_deltas) 
        for layer_deltas in zip(*deltas_per_client)
    ]

    # Apply the update: x ← x + ηg * avg_delta
    updated_weights: NDArrays = [
        server_w + eta_g / num_clients * delta_w for server_w, delta_w in zip(current_weights, delta_avg)
    ]
    
    return updated_weights