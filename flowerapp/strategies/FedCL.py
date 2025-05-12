
# from flwr.common import FitRes, Parameters, parameters_to_ndarrays , Context , ndarray_to_bytes , bytes_to_ndarray
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.strategy import FedAvg
# from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

# from logging import WARNING
# #import wandb
# from datetime import datetime

# from flwr.common import (
#     EvaluateIns,
#     EvaluateRes,
#     FitIns,
#     FitRes,
#     MetricsAggregationFn,
#     NDArrays,
#     Parameters,
#     Scalar,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )

# from flwr.common.logger import log
# from flwr.server.client_manager import ClientManager
# from flwr.server.client_proxy import ClientProxy
# import json
# from flowerapp.models import SimpleCNN
# class FedCL(FedAvg):
    
#     def __init__(self,D_proxy,beta: float = 0.5,*args,**kwargs):
            
#         super().__init__(*args,**kwargs)
#         self.beta = beta
#         self.importance_weights = None
#         self.D_proxy = None
#         super().evaluate_fn = compute_importance_weights

#     def initialize_parameters(self, client_manager):
#         parameters = super().initialize_parameters(client_manager)
#         self.importance_weights = parameters_to_ndarrays (parameters)
#         return parameters
        
#     def configure_fit(self, server_round, parameters, client_manager):
#         """Configure the next round of training."""
#         config = {}

#         array_list = [arr.tolist() for arr in self.importance_weights]
#         config["importance_weights"] = json.dumps(array_list)
#         fit_ins = FitIns(parameters, config)

#         # Sample clients
#         sample_size, min_num_clients = self.num_fit_clients(
#             client_manager.num_available()
#         )
#         clients = client_manager.sample(
#             num_clients=sample_size, min_num_clients=min_num_clients
#         )
#         return [(client, fit_ins) for client in clients]
#         # return super().configure_fit(server_round, parameters, client_manager)

#     def aggregate_fit(self, server_round, results,failures):
#         """Aggregate fit results using weighted average."""
#         if not results:
#             return None, {}
#         # Do not aggregate if there are failures and failures are not accepted
#         # if not super().accept_failures and failures:
#         #     return None, {}

#         # if super().inplace:
#         #     # Does in-place weighted average of results
#         #     aggregated_ndarrays = aggregate_inplace(results)
#         # else:
#         #     # Convert results
#         #     weights_results = [
#         #         (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#         #         for _, fit_res in results
#         #     ]
#         #     aggregated_ndarrays = aggregate(weights_results)
#         weights_results = [
#                 (parameters_to_ndarrays(fit_res.parameters))
#                 for _, fit_res in results
#             ]
#         aggregated_ndarrays = scaffold_aggregate(self.current_weights,weights_results,self.eta_g)
#         self.current_weights = aggregated_ndarrays # update server model
#         parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
#         for i, (sc, cw) in enumerate(zip(self.server_control_variate, self.current_weights)):
#             if sc.shape != cw.shape:
#              print(f"Shape mismatch at layer {i}: control {sc.shape}, weight {cw.shape}")
#         # Aggregate custom metrics if aggregation fn was provided
#         metrics_aggregated = {}
#         fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#         self.server_control_variate = aggregate_controls(fit_metrics,self.server_control_variate)
#         for i, arr in enumerate(self.server_control_variate):
#             print(f"Layer {i} max: {np.max(arr)}")
#         return parameters_aggregated, metrics_aggregated

#     def evaluate(self, server_round, parameters):
#         model=SimpleCNN()
#         omega = compute_importance_weights(model,self.D_proxy)
#         # update importance_weights
#         self.importance_weights = omega
        
#         return None

#     def aggregate_evaluate(self, server_round, results, failures):
#         return super().aggregate_evaluate(server_round, results, failures)

# import numpy as np
# from functools import reduce
# from typing import List, Tuple
# import torch

# def compute_importance_weights(model: torch.nn.Module, proxy_loader: torch.utils.data.DataLoader, device: torch.device):
#     model.to(device)
#     model.eval()
#     criterion = torch.nn.CrossEntropyLoss()  # or your custom loss

#     # Store squared gradients here
#     omega = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters() if param.requires_grad}

#     for x_k, y_k in proxy_loader:
#         x_k, y_k = x_k.to(device), y_k.to(device)

#         # Zero gradients
#         model.zero_grad()

#         # Forward pass
#         outputs = model(x_k)
#         loss = criterion(outputs, y_k)

#         # Backward pass
#         loss.backward()

#         # Accumulate squared gradients
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 omega[name] += (param.grad.detach() ** 2)

#     # Average over the number of samples
#     num_samples = len(proxy_loader.dataset)
#     for name in omega:
#         omega[name] /= num_samples

#     return omega  # This is your Ω: importance weight per parameter
