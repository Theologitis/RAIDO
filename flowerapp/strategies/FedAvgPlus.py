
from flwr.common import FitRes, Parameters, parameters_to_ndarrays , Context
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

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

class FedAvgPlus(FedAvg):
    
    def __init__(self,epochs_new,lr_new,decay_round,*args,**kwargs):
            
        super().__init__(*args,**kwargs)
        self.epochs_new = epochs_new
        self.lr_new = lr_new
        self.decay_round = decay_round
     
    def initialize_parameters(self, client_manager):
        return super().initialize_parameters(client_manager)   
        
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {}
        
        if server_round >= self.decay_round:
            
            config["lr"] = self.lr_new
            config["epochs"] = self.epochs_new
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results,failures):
        return super().aggregate_fit(server_round, results, failures)
    
    def evaluate(self, server_round, parameters):
        return super().evaluate(server_round, parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)