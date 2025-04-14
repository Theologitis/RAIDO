
from flwr.common import FitRes, Parameters, parameters_to_ndarrays , Context
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy , FedAvg
#import wandb
from datetime import datetime
import torch
import flowerapp.models
from flowerapp.task import get_model_class,set_weights
import json        
import os
import threading
class CustomStrategy(Strategy):
    
    def __init__(self,strategy:Strategy,num_rounds,model,context):
            
        super().__init__()
        self.strategy=strategy
        self.num_rounds=num_rounds
        self.results_to_save = {}
        self.model=model

    
    def initialize_parameters(self, client_manager):
        """Initialize parameters before training starts."""
        return self.strategy.initialize_parameters(client_manager)
    
    def configure_fit(self, server_round, parameters, client_manager):
        print(f'\n ROUND {server_round}: ')
        print('Training...')
        return self.strategy.configure_fit(server_round, parameters, client_manager)
        
    def aggregate_fit(self, server_round, results,failures):
        
        print(f'Aggregating...')
        
        parameters_aggregated, metrics_aggregated =self.strategy.aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrays
        ndarrays= parameters_to_ndarrays(parameters_aggregated)
        
        # # instantiate model
        model=get_model_class(self.model)
        set_weights(model, ndarrays)
        
        # # Save global model in the standard Pytorch way
        if server_round==self.num_rounds:
            torch.save(model.state_dict(),f"output/global_model.pth")
            print("model saved")
        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round, parameters):
        if server_round>0:
            #print(f'ROUND {server_round}: Evaluating...')
            print('Evaluating...')
        loss = self.strategy.evaluate(server_round, parameters)
        
        my_results = {"loss": loss}
        
        # self.results_to_save[server_round] = my_results

        # if server_round==self.num_rounds:
        #     threading.Thread(target=save_results, args=(self.results_to_save,), daemon=True).start()
        
        return loss #, metrics
    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Return evaluation instructions for clients."""
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round, results, failures):
        
        res = self.strategy.aggregate_evaluate(server_round, results, failures)
        
        accuracy_value = list(res[1].values())[0]*100
        
        print(f'weighted average Accuracy =  {accuracy_value:.2f}%')
            
        self.results_to_save["weighted average accuracy"] ={server_round: accuracy_value}
        
        if server_round == self.num_rounds:
            save_results(self.results_to_save)
            print(f'\nRun completed successfully with:\nweighted average Accuracy = {accuracy_value:.2f}%') 
        return res
    


## Helping Functions ##

def save_results(results):
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/results.json", 'w') as f:
            json.dump(results, f, indent=4)
        # with open("output/results.json.tmp", 'w') as tmp_file:
        #     json.dump(results, tmp_file, indent=4)
        # os.rename("output/results.json.tmp", "output/results.json")
    except Exception as e:
        print(f"[ERROR saving results.json] {e}")

def save_model(model,path):
    torch.save(model.state_dict(),path)