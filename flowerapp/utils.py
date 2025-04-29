"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
#from flwr_datasets import FederatedDataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
#from flwr_datasets.partitioner import IidPartitioner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os
from pathlib import Path
#from torchvision.models import resnet18
import flwr as fl
import yaml
import flowerapp.models

def get_model_class(class_name):
    ModelClass= getattr(flowerapp.models, class_name, None)
    net=ModelClass()
    return net

def get_model(model_name,**model_options):
    import flowerapp.models  # ensure the module is loaded

    ModelClass = getattr(flowerapp.models, model_name, None)
    if ModelClass is None:
        raise ValueError(f"Model class '{model_name}' not found in flowerapp.models.")
    
    try:
        model = ModelClass(**model_options)
    except TypeError as e:
        raise ValueError(f"Error instantiating model '{model_name}': {e}")
    
    return model

def get_optimizer(optim_name,**optim_options):
    import flowerapp.models  # ensure the module is loaded

    optimClass = getattr(torch.optim, optim_name, None)
    if optimClass is None:
        raise ValueError(f"Model class '{optim_name}' not found in flowerapp.models.")
    
    try:
        optimizer = optimClass(**optim_options)
    except TypeError as e:
        raise ValueError(f"Error instantiating model '{optim_name}': {e}")
    
    return optimizer

def get_criterion(criterion_name,**criterion_options):
    import flowerapp.models  # ensure the module is loaded

    criterionClass = getattr(torch.nn, criterion_name, None)
    if criterionClass is None:
        raise ValueError(f"Model class '{criterion_name}' not found in flowerapp.models.")
    
    try:
        criterion = criterionClass(**criterion_options)
    except TypeError as e:
        raise ValueError(f"Error instantiating model '{criterion_name}': {e}")
    
    return criterion

class CIFAR10Custom(Dataset):
    """
    Custom CIFAR-10 dataset implementation for federated learning.
    Handles data loading from partitioned files and applies transformations.
    
    Args:
        root (str): Path to dataset directory
        train (bool): Whether to load training or test data
        transform (callable): Transformations to apply to images
    """
    def __init__(self, root, train=True, transform=None,max_samples=100,start_index=0):
        self.root = root
        self.transform = transform
        self.train = train
        self.data = []
        self.labels = []
        self.max_samples = max_samples
        self.start_index = start_index
        # Modified to handle single batch file for training
        batch_files = ["data_batch"] if train else ["test_batch"]
       
        # Load and process batch files
        for batch_file in batch_files:
            batch_path = os.path.join('data', batch_file) # /data is the mounted volumes directory in the container

            if not os.path.exists(batch_path): # Check if file exists
                raise FileNotFoundError(f"Dataset file {batch_path} not found!")
            

            with open(batch_path, "rb") as f: # Load batch data
                batch = pickle.load(f, encoding="bytes")
               

                # Verify data format
                if b"data" in batch and b"labels" in batch:
                    self.data.append(batch[b"data"][start_index:self.max_samples+start_index])
                    self.labels.extend(batch[b"labels"][start_index:self.max_samples+start_index])
                else:
                    raise KeyError(f"Expected keys b'data' and b'labels' not found in {batch_file}")

        # Convert to float and normalize
        self.data = torch.tensor(np.vstack(self.data).reshape(-1, 3, 32, 32), dtype=torch.float32) / 255.0
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (transformed_image, label)
        """
        import torchvision.transforms as transforms
        img, label = self.data[idx], self.labels[idx]

        # Convert tensor to PIL Image for transformations    
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
    
        img = self.transform(img) # Apply transformations

        return img, label

def load_data_loc(client_data_path,part_id):
    """
    Prepares data loaders for training and testing.
    
    Args:
        client_data_path (str): Path to client's data directory
    
    Returns:
        tuple: (train_loader, test_loader)
    """

    # # Verify data directory exists
    # if not os.path.exists(client_data_path):
    #     raise FileNotFoundError(f"Client data folder not found: {client_data_path}")

    # Define data transformations
    transform = Compose([
        ToTensor(),                                             # Convert images to PyTorch tensors
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    # Normalize RGB channels
    ])
    
    # Create dataset instances
    N=1000
    trainset = CIFAR10Custom(client_data_path, train=True,transform=transform,max_samples=N)
    testset = CIFAR10Custom(client_data_path, train=False,transform=transform,max_samples=N+100*part_id,start_index=part_id*N)
    
    return DataLoader(trainset, batch_size=256, shuffle=True), DataLoader(testset, batch_size=32)


# def train(model, trainloader, epochs=1,device="cpu",lr=0.01):
#     """
#     Trains the model on the provided data and tracks metrics.
    
#     Args:
#         model (nn.Module): Neural network model to train
#         trainloader (DataLoader): DataLoader containing training data
#         epochs (int): Number of training epochs
    
#     Returns:
#         list: List of tuples containing (loss, accuracy) for each epoch
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     model.to(device)
    
#     loss_fn = nn.CrossEntropyLoss() # Define loss function
#     optimizer = optim.Adam(model.parameters(), lr) # Define optimizer and learning rate
    
#     model.train()
#     all_metrics = []
    
#     for epoch in range(epochs): # Training loop over epochs
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in trainloader: # Batch training loop
#             print(type(inputs), inputs.shape)
#             inputs, labels = inputs.to(device), labels.to(device) # Move data to the same device as the model (GPU/CPU)
   
#             # Forward pass
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)
            
#             # Backward pass and optimization
#             optimizer.zero_grad() # Clear previous gradients
#             loss.backward()       # Compute gradients
#             optimizer.step()      # Update weights
            
#             # Track metrics
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
        
#         # Calculate epoch metrics
#         epoch_loss = running_loss / len(trainloader)
#         epoch_accuracy = 100 * correct / total
        
#         # Store and print metrics
#         all_metrics.append((epoch_loss, epoch_accuracy))        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
#     # model.cpu()
#     return all_metrics


# def test(net, testloader,device):
#     """
#     Evaluates the model on test data.
    
#     Args:
#         net (nn.Module): Neural network model to evaluate
#         testloader (DataLoader): DataLoader containing test data
    
#     Returns:
#         tuple: (average_loss, accuracy)
#     """
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net.to(DEVICE)
    
#     criterion = torch.nn.CrossEntropyLoss()
#     correct = 0
#     total = 0
#     loss = 0.0

#     # Disable gradient computation for evaluation
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE) # Move data to the same device as the model (GPU/CPU)  
#             outputs = net(images) # Forward pass
#             loss += criterion(outputs, labels).item() # Accumulate loss
#             total += labels.size(0) # Track accuracy
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
#     return loss /len(testloader.dataset), correct / total


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays


# from flwr/examples/flowertune-llm.git:
def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

