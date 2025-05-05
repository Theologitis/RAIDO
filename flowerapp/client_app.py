from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from flwr.client.mod import fixedclipping_mod, secaggplus_mod , adaptiveclipping_mod
import random, json
import torch
from flowerapp.utils import (
    get_weights,
    set_weights,
    get_model,
    drop_empty_keys

)
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
# from omegaconf import DictConfig
class FlowerClient(NumPyClient):
    " Default class of Flower: implements client logic, inherits from NumpyClient"
    
    def __init__(self, net, trainloader, testloader,epochs,lr,device,context,task):
        self.context=context
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = epochs
        self.device= device
        self.lr = lr
        self.task = task
        
    def fit(self, parameters, config):
        print('Training...')
        set_weights(self.task.model, parameters)
        # update parameters if passed from strategy in config:
        self.lr = config.get("lr",self.lr)
        self.local_epochs = config.get("epochs",self.local_epochs)
        proximal_mu = config.get("proximal_mu", 0.0)
        all_metrics = self.task.train(self.trainloader, self.local_epochs,self.lr,proximal_mu)
        return get_weights(self.task.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print('Evaluating...')
        set_weights(self.task.model, parameters)
        loss, accuracy = self.task.test(self.testloader)
        print(f'Accuracy:{100*accuracy:.2f}%')
        return loss, len(self.testloader), {"accuracy": accuracy}

def get_task(task_name: str):
    module = __import__(f"flowerapp.tasks.{task_name}", fromlist=[task_name])
    return getattr(module, task_name)

mods= []

def client_fn(context: Context):
    "Setup ClientApp"
    partition_id = context.node_config["partition-id"] 
    num_partitions = context.node_config["num-partitions"]
    
    cfg = DictConfig(drop_empty_keys(unflatten_dict(context.run_config)))
    
    batch_size = cfg.train.batch_size # batch size
    net = get_model(cfg.model.name,**cfg.model.options) # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device
    task = get_task(cfg.task.name) # task e.g. ImageClassification
    task = task(net,device)
    train_loader,test_loader = task.load_data(partition_id,num_partitions,batch_size) #train and test dataloaders
    # train_loader, test_loader = task.load_data('data')
    epochs = cfg.train.epochs # epochs
    lr = cfg.train.lr # learning rate
    
    # check if Differential privacy is enabled and add the mod
    dp = cfg.dp
    if dp.flag.lower() == "true":
        if dp.side == "client":
            if dp.type == "fixed":
                mods.append(fixedclipping_mod)
            else:
                mods.append(adaptiveclipping_mod)
            
    # check if secure aggregation is enabled and add the mod
    sec_agg = True if cfg.dp.flag.lower()=="true" else False
    if sec_agg:
        mods.append(secaggplus_mod)
        
    return FlowerClient(net, train_loader, test_loader, epochs, lr, device, context, task).to_client()

app = ClientApp(
        client_fn=client_fn,
        mods=mods,
)