import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 , resnet152
class CNNModel1(nn.Module):
    """
    A simple CNN suitable for CIFAR-10.
    
    Architecture details:
    1. First Convolutional Layer:
       - Input: 3 channels (RGB)
       - Output: 6 feature maps
       - Kernel size: 5x5
    2. First MaxPool Layer (2x2)
    3. Second Convolutional Layer:
       - Input: 6 channels
       - Output: 16 feature maps
       - Kernel size: 5x5
    4. Second MaxPool Layer (2x2)
    5. Fully Connected Layers:
       - 16*5*5 -> 120 -> 84 -> 10 (output classes)
    """    
    def __init__(self) -> None:
        super(CNNModel1, self).__init__()
        #self.conv1 = nn.Conv2d(128,14,5)
        self.conv1 = nn.Conv2d(3, 6, 5) # 5x5 kernel, (in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2) # MaxPool for dimension reduction, (kernel_size, stride)
        self.conv2 = nn.Conv2d(6, 16, 5) # 5x5 kernel 
        # Fully connected layers for classification
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Flattened conv output -> 120
        self.fc2 = nn.Linear(120, 84) # 120 -> 84
        self.fc3 = nn.Linear(84, 10)  # 84 -> 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply first conv layer and max pooling
        x = self.pool(F.relu(self.conv2(x))) # Apply second conv layer and max pooling
        x = x.view(-1, 16 * 5 * 5) # Flatten the tensor for fully connected layers, Reshape to (batch_size, 16*5*5)
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=16 * 5 * 5, hidden_dims=[120, 84], output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        #x = torch.flatten(x, start_dim=1)  # Flatten tensor for FC layers more flexible instead of x.view()
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, input_dim=16 * 5 * 5, hidden_dims=[64, 32], output_dim=10):
        super(SmallCNN, self).__init__()

        # Fully connected layers with different hidden dimensions
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # First hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # Output layer

    def forward(self, x):
        # Flatten the input tensor for fully connected layers (like the first model)
        x = x.view(x.size(0), -1)  # Flatten the input tensor

        # Pass through the fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)          # Output layer (no activation function here)

        return x

class CustomModel(nn.Module):
    """ Custom model with ResNet18 architecture, compatible with the Net2 style """
    def __init__(self):
        super(CustomModel, self).__init__()
        # Load pre-trained ResNet18
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False,weights=None)
        # Modify the final fully connected layer to output 10 classes
        self.model.fc = nn.Linear(in_features=512, out_features=10)
        
    def forward(self, x):
        # Forward pass through the ResNet model
        return self.model(x)
    
class Net(nn.Module):
    """ Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz') """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ResNetModel(nn.Module):
    """
    Adaptation of ResNet18 architecture for CIFAR-10 classification.
    ResNet18 includes skip connections to help with gradient flow during training.
    
    The model is modified by:
    1. Loading the standard ResNet18 architecture
    2. Replacing the final fully connected layer to match CIFAR-10's 10 classes
    3. Removing pretrained weights (set pretrained=False)
    """
    def __init__(self) -> None:
        super(ResNetModel, self).__init__()
        self.model = resnet18(pretrained=False) # Load ResNet18 model without pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Adjust for CIFAR-10 classes

    def forward(self, x):
        return self.model(x) # Forward pass through the entire ResNet architecture
