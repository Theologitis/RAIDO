from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord , ndarray_to_bytes , bytes_to_ndarray
from flwr.client.mod import fixedclipping_mod, secaggplus_mod , adaptiveclipping_mod
import random, json
import torch
from flowerapp.utils import (
    get_weights,
    set_weights,
    get_model,
    drop_empty_keys,
    check_compatibility
)
from flwr.common import ArrayRecord
import numpy as np
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
import json
import os

class ScaffoldClient(NumPyClient):
    " Default class of Flower: implements client logic, inherits from NumpyClient"

    def __init__(self, net, trainloader, testloader,epochs,lr,device,context: Context,task):
        self.context=context
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = epochs
        self.device= device
        self.lr = lr
        self.task = task
        self.client_state = context.state

        if "control_variate" not in self.client_state:
            params=get_weights(self.task.model)
            self.client_state["control_variate"] = ArrayRecord([np.zeros_like(p) for p in params])


    def fit(self, parameters, config):
        print('Training...')
        set_weights(self.task.model, parameters) # set new global model as task's model
        # update parameters if passed from strategy in config:
        self.lr = config.get("lr",self.lr) # for FedAvgPlus
        self.local_epochs = config.get("epochs",self.local_epochs) # for FedAvgPlus
        array_list = json.loads(config["server_control_variate"]) # For Scaffold
        server_control_variate = [np.array(arr) for arr in array_list]
        _,all_metrics = self.task.train_scaffold(self.trainloader, 5,self.lr,server_control_variate,self.client_state["control_variate"].to_numpy_ndarrays()) # train on local train dataset
        client_delta = [a - b for a, b in zip(all_metrics,self.client_state["control_variate"].to_numpy_ndarrays())]
        #print(f"max delta in clinet{max([delta.max() for delta in client_delta])}")
        self.client_state["control_variate"] = ArrayRecord(all_metrics)
        array_list = [arr.tolist() for arr in client_delta]
        print(array_list.shape)
        all_metrics_str = json.dumps(array_list)
        #for i, arr in enumerate(all_metrics):
            #print(f"Control variate shape [{i}]: {arr.shape}, dtype: {arr.dtype}")
        return get_weights(self.task.model), len(self.trainloader), {"control_variate":all_metrics_str}

    def evaluate(self, parameters, config):
        print('Evaluating...')
        set_weights(self.task.model, parameters) # set new global model as task's model
        loss, metrics = self.task.test(self.testloader)  # test on local test dataset
        #print(f'Accuracy:{100*accuracy:.2f}%')
        return loss, len(self.testloader), metrics