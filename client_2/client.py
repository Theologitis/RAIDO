# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:00:33 2024

@author: stylelev
"""

from pathlib import Path  
from collections import OrderedDict  
import sys
import threading  
import time
import os
import itertools
import collections

import torch  
import flwr as fl  
import warnings

sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add the parent directory to system path
from RaidoFL import load_data, load_model, train, test, MODEL_NAME # Imports from RaidoFL.py
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

def waiting_animation():
    """
    Displays a real-time waiting animation with elapsed time counter.
    
    Features:
    - Shows minutes:seconds format
    - Updates every second
    - Provides visual feedback during server connection
    """
    start_time = time.time()

    # Loop until stop flag set by another thread
    while not getattr(waiting_animation, "stop", False):
        elapsed_time = int(time.time() - start_time)
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60        
        print(f"\r[{minutes:02d}:{seconds:02d}] : Awaiting the server's reply to the train message", end="") # \r returns cursor to start of line to update in place
        sys.stdout.flush()  # Immediate display update
        time.sleep(1)  # Wait 1 second before next update
    print("\nAll clients connected. Starting federated learning...")

def wait_for_server_ready():
    """
    Initializes server connection process with visual feedback.
    
    Returns:
        threading.Thread: Animation thread object
    """
    print("[CLIENT] Initializing connection...")
    time.sleep(2)  
    animation_thread = threading.Thread(target=waiting_animation)
    animation_thread.daemon = True  # Ensure thread doesn't prevent program exit
    animation_thread.start()
    
    return animation_thread

def set_parameters(model, parameters):
    """
    Updates the model's parameters with those received from the server.
    
    Args:
        model (torch.nn.Module): PyTorch model to update
        parameters (list): List of numpy arrays containing new parameters
    
    Returns:
        torch.nn.Module: Updated PyTorch model
    
    """
    params_dict = zip(model.state_dict().keys(), parameters)   # Create parameter dictionary -> names - values
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})  # Convert numpy arrays to PyTorch, maintain parameter order
    model.load_state_dict(state_dict, strict=True)  # Update model with new parameters, ensuring all parameters are present

    return model

class FlowerClient(fl.client.NumPyClient):
  
    def __init__(self, client_id, animation_thread=None):
        """
        Initialize client with unique ID and required components.
        
        Args:
            client_id (str): Unique identifier for this client
            animation_thread (threading.Thread, optional): Thread for waiting animation
        
        """
        self.client_id = client_id
        self.animation_thread = animation_thread

        self.net = load_model()
        
        # Data path of each client
        base_path = Path(__file__).resolve().parent
        client_data_path = base_path / "data"
        
        print("\n" + "="*85 + "\n")  # Print header
        self.trainloader, self.testloader = load_data(client_data_path) # Load training and test data
               
        CLIENT_ID = int(os.path.basename(os.getcwd()).split('_')[-1]) # Take client ID from directory name
        
        self._analyze_data_distribution(CLIENT_ID, client_data_path)   # Analyze and verify data distribution
        print("\n" + "="*85 + "\n")

    def _analyze_data_distribution(self, CLIENT_ID, data_path):
        """
        Performs detailed analysis of local dataset distribution.
        
        Args:
            CLIENT_ID (int): Client identifier
            data_path (Path): Path to client's data directory
        
        Analyzes and prints:
        - Data source location
        - Sample of first labels
        - Label distribution
        - Dataset size
        """
        size = len(self.trainloader)
        first_labels = []
        
        # Collect labels from first batch
        for batch in self.trainloader:
            _, labels = batch
            first_labels.extend(labels.tolist())
            if len(first_labels) >= size:
                first_labels = first_labels[:size]
                break
        
        # Print detailed data analysis
        print(f"\nClient {CLIENT_ID} is loading data from: {data_path}")
        print(f"\nFirst 10 labels for Client {CLIENT_ID}: {first_labels[:10]}")
        label_counts = collections.Counter(first_labels)
        print(f"\nLabel distribution for Client {CLIENT_ID}: {label_counts}")
        print(f"\nClient {CLIENT_ID} dataset size: {size} samples")

    def get_parameters(self, config):
        """
        Retrieves current model parameters for server synchronization.
        
        Args:
            config (dict): Configuration parameters (unused but required by Flower)
        
        Returns:
            list: Model parameters as numpy arrays
        """
        print(f"[CLIENT {self.client_id}] Preparing to send model parameters to server...")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """
        Performs local training with global parameters.
        
        Args:
            parameters (list): Model parameters from server
            config (dict): Training configuration
        
        Returns:
            tuple: (updated parameters, number of samples, metrics)
        """
        # Stop waiting animation if still running
        if self.animation_thread and not getattr(waiting_animation, "stop", False):
            waiting_animation.stop = True
            time.sleep(1)  # Allow animation to clean up
            
        print(f"\n[CLIENT {self.client_id}] Starting local training...")
        
        set_parameters(self.net, parameters) # Update local model with global parameters
        
        epoch_metrics = train(self.net, self.trainloader, epochs=5)   # Local training for 5 epochs
        
        print(f"\n[CLIENT {self.client_id}] Local training completed. Sending updated parameters back to server.")

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluates global model performance on local test data.
        
        Args:
            parameters (list): Global model parameters
            config (dict): Evaluation configuration
        
        Returns:
            tuple: (loss value, number of samples, metrics dictionary)
        """
        print(f"\n[CLIENT {self.client_id}] Preparing for GLOBAL model evaluation...")
        
        set_parameters(self.net, parameters) # Update model with global parameters
        
        loss, accuracy = test(self.net, self.testloader) # Evaluation
        
        print(f"\n[CLIENT {self.client_id}] Evaluation complete. Global Loss: {loss:.4f}, Global Accuracy: {accuracy:.4f}")

        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}

def main():
    # Extract client ID from directory name
    client_folder = Path(__file__).resolve().parent
    client_id = client_folder.name.split('_')[-1]
    print(f"\nStarting client {client_id}")
    
    animation_thread = wait_for_server_ready() # Initialize connection with waiting animation
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",  # Local server address
        client=FlowerClient(client_id, animation_thread),
    )

if __name__ == "__main__":
    main()