from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
import random, json
import torch
from flowerapp.task import (
    get_weights,
    set_weights,
    train,
    load_data_loc,
    test,
    get_model_class
)

class FlowerClient(NumPyClient):
    " Default class of Flower: implements client logic, inherits from NumpyClient"
    
    def __init__(self, net, trainloader, testloader,epochs,lr,context):
        self.context=context
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = epochs
        self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr=lr
        
    def fit(self, parameters, config):
        print('Training...')
        set_weights(self.net, parameters)
        all_metrics=train(self.net, self.trainloader,self.local_epochs,self.lr)
        return get_weights(self.net), len(self.trainloader), {}
    
    # def fit(self, parameters, config): # config is the return Metric of on_fit_config_fn Callback
    #     set_weights(self.net, parameters)
        
    #     train_loss = train(
    #         self.net,
    #         self.trainloader,
    #         self.local_epochs,
    #         self.device,
    #         #config['lr']
    #     )
 
    #     fit_metrics = self.client_state.configs_records["fit_metrics"]
    #     if "train_loss_history" not in fit_metrics:
    #         fit_metrics["train_loss_history"] = [train_loss]
    #     else:
    #         fit_metrics["train_loss_history"].append(train_loss)
        
        
    #     return (  # These are the arguments that client send back to the server and they are the most important.
    #         get_weights(self.net), # the updated model
    #         len(self.trainloader.dataset), # N: the number of samples train on in this round, to use them in weighting the different models e.g.
    #         {"train_loss": train_loss}, # metrics: it is what is returned to fit_merics_aggregate
    #     )

    def evaluate(self, parameters, config):
        print('Evaluating...')
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        print(f'Accuracy:{100*accuracy:.2f}%')
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    "Default function of Flower: initializes FlowerClient inside the ClientApp"
    partition_id = context.node_config["partition-id"]
    
    train_loader,test_loader=load_data_loc('data',partition_id)
    net=get_model_class(context.run_config["model"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    epochs=context.run_config["epochs"]
    lr=context.run_config["lr"]
    return FlowerClient(net, train_loader, test_loader,epochs,lr,context).to_client()

app = ClientApp(
    client_fn=client_fn,
    mods=[
        #secaggplus_mod,
        #fixedclipping_mod,
    ],
)