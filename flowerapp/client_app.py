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
    get_model_class,
    load_data_sim
)

class FlowerClient(NumPyClient):
    " Default class of Flower: implements client logic, inherits from NumpyClient"
    
    def __init__(self, net, trainloader, testloader,epochs,lr,device,context):
        self.context=context
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = epochs
        self.device= device
        self.lr=lr
        
    def fit(self, parameters, config):
        print('Training...')
        set_weights(self.net, parameters)
        if config != {}:
            self.lr = config["lr"]
            self.local_epochs = config["epochs"]
        all_metrics = train(self.net, self.trainloader, self.local_epochs, self.device,self.lr)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print('Evaluating...')
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader,self.device)
        print(f'Accuracy:{100*accuracy:.2f}%')
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    "Default function of Flower: initializes FlowerClient inside the ClientApp"
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch_size"]
    #train_loader,test_loader=load_data_loc('data',partition_id)
    train_loader,test_loader = load_data_sim(partition_id,num_partitions,batch_size)
    net = get_model_class(context.run_config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = context.run_config["epochs"]
    lr = context.run_config["lr"]

    return FlowerClient(net, train_loader, test_loader,epochs,lr,device,context).to_client()

app = ClientApp(
    client_fn=client_fn,
    mods=[
        #secaggplus_mod,
        #fixedclipping_mod,
    ],
)