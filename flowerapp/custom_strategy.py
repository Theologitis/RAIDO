
from flwr.common import FitRes, Parameters, parameters_to_ndarrays , Context
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
#import wandb
from datetime import datetime

import json
#     "Bulyan",
#     "DPFedAvgAdaptive",
#     "DPFedAvgFixed",
#     "DifferentialPrivacyClientSideAdaptiveClipping",
#     "DifferentialPrivacyClientSideFixedClipping",
#     "DifferentialPrivacyServerSideAdaptiveClipping",
#     "DifferentialPrivacyServerSideFixedClipping",
#     "FaultTolerantFedAvg",
#     "FedAdagrad",
#     "FedAdam",
#     "FedAvg",
#     "FedAvgAndroid",
#     "FedAvgM",
#     "FedMedian",
#     "FedOpt",
#     "FedProx",
#     "FedTrimmedAvg",
#     "FedXgbBagging",
#     "FedXgbCyclic",
#     "FedXgbNnAvg",
#     "FedYogi",
#     "Krum",
#     "QFedAvg",
#     "Strategy",
class CustomStrategy(Strategy):
    
    def __init__(self,strategy:Strategy):
            
        super().__init__()
        self.strategy=strategy
        #print('Federated Learning started successfully\n')
        # results to be saved
        self.results_to_save = {}
        
        #Log the metrics/results to W&B
        #name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        #wandb.init(project="flower-run-simulation-tutorial", name=f"custom-strategy-{name}")
        
        
    def aggregate_fit(self, server_round, results,failures):
        
        parameters_aggregated, metrics_aggregated =self.strategy.aggregate_fit(server_round, results, failures)
        print(f'Aggregating...')
        # convert parameters to ndarrays
        ndarrays= parameters_to_ndarrays(parameters_aggregated)
        
        # # instantiate model
        # model_class=get_model_class(context.run_config['model_class'])
        # print(model_class)
        # model = model_class()
        # print(model)
        # set_weights(model, ndarrays)
        
        # # Save global model in the standard Pytorch way
        # torch.save(model.state_dict(),f"global_model_round{server_round}")
        
        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round, parameters):
        if server_round>0:
            #print(f'ROUND {server_round}: Evaluating...')
            print('Evaluating...')
        loss = self.strategy.evaluate(server_round, parameters)
        
        my_results = {"loss": loss}
        
        self.results_to_save[server_round] = my_results
        
        with open("results.json",'w') as json_file:
            json.dump(self.results_to_save,json_file,indent=4)
        
        # Log(push) to W&B
        #wandb.log(my_results, step=server_round)
        
        return loss #, metrics
    def configure_fit(self, server_round, parameters, client_manager):
        print(f'\nROUND {server_round}: ')
        print('Training...')
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, server_round, results, failures):
        res = self.strategy.aggregate_evaluate(server_round, results, failures)
        accuracy_value = list(res[1].values())[0]*100
        print(f'weighted average Accuracy =  {accuracy_value:.2f}%')
        if server_round==2:
            print(f'\nRun completed successfully with:\nweighted average Accuracy = {accuracy_value:.2f}%')
        return res
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Return evaluation instructions for clients."""
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)  

    def initialize_parameters(self, client_manager):
        """Initialize parameters before training starts."""
        return self.strategy.initialize_parameters(client_manager)

