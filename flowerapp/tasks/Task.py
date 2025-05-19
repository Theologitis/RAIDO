from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Task(ABC):
    """
    Abstract class for tasks such as image classification, etc.
    All tasks should implement the required methods for data loading, training, and testing.
    """
    def __init__(self, model, device=None):
        """
        Initialize the Task instance.
        
        Args:
            model (nn.Module): The model for the task.
            device (str or torch.device, optional): Device to run on (defaults to CUDA if available).
        """
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = None
        self.proximal_mu = 0.0

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        Abstract method to load the data for training and testing.
        
        Args:
            *args: Arguments for data loading
            **kwargs: Keyword arguments for data loading
        
        Returns:
            tuple: (trainloader, testloader) - DataLoader objects for training and testing
        """
        pass

    @abstractmethod
    def train(self, trainloader, epochs=1, lr=0.001):
        """
        Abstract method for training the model.

        Args:
            trainloader (DataLoader): The DataLoader containing the training data
            epochs (int): The number of epochs to train
            lr (float): Learning rate

        Returns:
            float: The average loss for the training process
        """
        pass

    @abstractmethod
    def test(self, testloader):
        """
        Abstract method for testing the model.

        Args:
            testloader (DataLoader): The DataLoader containing the test data

        Returns:
            tuple: (float) test loss, (float) test accuracy
        """
        pass
