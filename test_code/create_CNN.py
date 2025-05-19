# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomCNN(nn.Module):
#     def __init__(self, conv_layers, fc_layers, input_channels=3, num_classes=10,
#                  dropout_prob=0.5, batch_norm=True, l2_regularization=0.0, l1_regularization=0.0):
#         """
#         Args:
#             conv_layers (list of dict): List of dictionaries, where each dictionary represents a convolutional layer.
#                                         Keys: 'out_channels', 'kernel_size', 'stride', 'padding', 'activation'.
#             fc_layers (list of int): List of integers representing the number of units in each fully connected layer.
#             input_channels (int): Number of input channels (e.g., 3 for RGB images).
#             num_classes (int): Number of output classes (default 10 for classification).
#             dropout_prob (float): Dropout probability for regularization. Default is 0.5.
#             batch_norm (bool): Whether to include Batch Normalization after each convolutional layer.
#             l2_regularization (float): L2 regularization (weight decay) coefficient for the optimizer.
#             l1_regularization (float): L1 regularization (lasso) coefficient for the optimizer.
#         """
#         super(CustomCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv_layers = self._create_conv_layers(conv_layers, input_channels, batch_norm, dropout_prob)
        
#         # Fully connected layers
#         self.fc_layers = self._create_fc_layers(fc_layers, dropout_prob)
        
#         # Final output layer (fully connected)
#         self.fc_out = nn.Linear(fc_layers[-1], num_classes)
        
#         # Store L1 and L2 regularization terms
#         self.l2_regularization = l2_regularization
#         self.l1_regularization = l1_regularization
        
#     def _create_conv_layers(self, conv_layers, input_channels, batch_norm, dropout_prob):
#         """
#         Create the convolutional layers with optional batch normalization and dropout.
#         """
#         layers = []
#         in_channels = input_channels
        
#         for layer in conv_layers:
#             out_channels = layer['out_channels']
#             kernel_size = layer['kernel_size']
#             stride = layer['stride']
#             padding = layer['padding']
#             activation = layer['activation']
            
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
#             if batch_norm:
#                 layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization
#             if activation == 'relu':
#                 layers.append(nn.ReLU(inplace=True))
#             elif activation == 'leaky_relu':
#                 layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
#             elif activation == 'sigmoid':
#                 layers.append(nn.Sigmoid())
#             elif activation == 'tanh':
#                 layers.append(nn.Tanh())
#             if dropout_prob > 0:
#                 layers.append(nn.Dropout(dropout_prob))  # Dropout layer
                
#             in_channels = out_channels
        
#         return nn.Sequential(*layers)
    
#     def _create_fc_layers(self, fc_layers, dropout_prob):
#         """
#         Create the fully connected layers with optional dropout.
#         """
#         layers = []
        
#         # Convert the 2D input from conv layers to 1D
#         in_features = self._get_conv_output(torch.zeros(1, 3, 32, 32))  # Example input size
        
#         for out_features in fc_layers:
#             layers.append(nn.Linear(in_features, out_features))
#             layers.append(nn.ReLU(inplace=True))  # Add ReLU activation to fully connected layers
#             if dropout_prob > 0:
#                 layers.append(nn.Dropout(dropout_prob))  # Dropout layer
#             in_features = out_features
            
#         return nn.Sequential(*layers)
    
#     def _get_conv_output(self, shape):
#         """
#         Get the output size after passing through the convolutional layers.
#         This is used to determine the input size for the first fully connected layer.
#         """
#         with torch.no_grad():
#             output = self.conv_layers(shape)
#         return output.view(1, -1).size(1)
    
#     def forward(self, x):
#         """
#         Forward pass through the network.
#         """
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
#         x = self.fc_layers(x)
#         x = self.fc_out(x)
#         return x

#     def get_l1_regularization_loss(self):
#         """
#         Computes the L1 regularization loss based on the model's parameters (Lasso).
#         """
#         l1_loss = 0.0
#         for param in self.parameters():
#             l1_loss += torch.sum(torch.abs(param))
#         return self.l1_regularization * l1_loss

#     def get_l2_regularization_loss(self):
#         """
#         Computes the L2 regularization loss based on the model's parameters.
#         """
#         l2_loss = 0.0
#         for param in self.parameters():
#             l2_loss += torch.norm(param, 2)
#         return self.l2_regularization * l2_loss

# # Example usage:
# # Define a CNN architecture with regularization (L1 and L2)
# conv_layers = [
#     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
#     {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
#     {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'}
# ]

# fc_layers = [128, 64]  # Fully connected layers

# # Create the model with L1 regularization (lasso), L2 regularization (ridge), and dropout
# model = CustomCNN(conv_layers, fc_layers, input_channels=3, num_classes=10, 
#                   dropout_prob=0.5, batch_norm=True, l2_regularization=0.01, l1_regularization=0.001)

# # Print the model architecture
# print(model)
import cloudpickle
from flowerapp.tasks.ImageClassification import ImageClassification

task = ImageClassification()
with open("model.pkl","wb") as f:
    cloudpickle.dump(task,f)