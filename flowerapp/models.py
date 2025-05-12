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
        self.model = resnet18(weights=None) # Load ResNet18 model without pretrained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Adjust for CIFAR-10 classes

    def forward(self, x):
        return self.model(x) # Forward pass through the entire ResNet architecture


### new models ###
class TimeSeriesLSTM(nn.Module):
    """
    LSTM-based model for time series classification with multiple pathways.
    Updated to work with non-sequence (tabular) data.
    
    Architecture details:
    1. Input Preprocessing:
       - Batch normalization for feature scaling
       
    2. Three Parallel Pathways:
       a. Direct Pathway (for simple patterns):
          - Linear layer: input_dim -> hidden_dim
       b. Deep Pathway (for complex patterns):
          - Linear layer 1: input_dim -> hidden_dim
          - Linear layer 2: hidden_dim -> hidden_dim
       c. LSTM Pathway (for temporal patterns):
          - Bidirectional LSTM: input_dim -> hidden_dim*2
          - Number of layers: configurable
          - Dropout: applied between LSTM layers
          
    3. Feature Combination:
       - Concatenates outputs from all three pathways
       - Linear layer: hidden_dim*4 -> hidden_dim
       
    4. Output Processing:
       - Dropout for regularization
       - Linear layer: hidden_dim -> output_dim (number of classes)
       
    5. Weight Initialization:
       - LSTM: Kaiming for input weights, Orthogonal for hidden weights
       - Linear layers: Kaiming normal initialization
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.3):
        super(TimeSeriesLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # Multiple model pathways for better feature extraction
        
        # 1. Direct pathway for simple patterns
        self.direct_fc = nn.Linear(input_dim, hidden_dim)
        
        # 2. Deep pathway for complex patterns
        self.deep_fc1 = nn.Linear(input_dim, hidden_dim)
        self.deep_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. LSTM pathway for temporal patterns
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Combine pathways
        self.combine_fc = nn.Linear(hidden_dim * 4, hidden_dim)  # 2*hidden_dim from bidirectional LSTM + 2*hidden_dim from direct & deep
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply better initialization to improve learning"""
        # LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # FC layers
        nn.init.kaiming_normal_(self.direct_fc.weight)
        nn.init.kaiming_normal_(self.deep_fc1.weight)
        nn.init.kaiming_normal_(self.deep_fc2.weight)
        nn.init.kaiming_normal_(self.combine_fc.weight)
        nn.init.kaiming_normal_(self.output_fc.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply batch normalization
        if len(x.shape) == 2:
            # For non-sequence (tabular) data: (batch_size, features)
            x_normalized = self.batch_norm(x)
            
            # Direct pathway
            direct_out = F.relu(self.direct_fc(x_normalized))
            
            # Deep pathway
            deep_out = F.relu(self.deep_fc1(x_normalized))
            deep_out = F.relu(self.deep_fc2(deep_out))
            
            # For non-sequence data, create a single-step sequence for LSTM
            lstm_input = x_normalized.unsqueeze(1)  # (batch_size, 1, features)
            
            # Initialize LSTM state
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            
            # LSTM pathway
            lstm_out, _ = self.lstm(lstm_input, (h0, c0))
            lstm_out = lstm_out[:, -1, :]  # Get last time step
            
            # Combine all pathways
            combined = torch.cat([direct_out, deep_out, lstm_out], dim=1)
            
            # Final processing
            out = F.relu(self.combine_fc(combined))
            out = self.dropout(out)
            out = self.output_fc(out)
            
            return out
        else:
            # Handle sequence data if needed (for future compatibility)
            # This part shouldn't be reached with the current dataset
            print(f"Warning: Unexpected input shape {x.shape}. Expected 2D input.")
            
            # Reshape for batch norm
            x_reshaped = x.reshape(-1, self.input_dim)
            x_normalized = self.batch_norm(x_reshaped)
            x_normalized = x_normalized.reshape(x.shape)
            
            # Process with LSTM
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            
            lstm_out, _ = self.lstm(x_normalized, (h0, c0))
            lstm_out = lstm_out[:, -1, :]
            
            # Final processing
            out = self.dropout(lstm_out)
            out = self.output_fc(out)
            
            return out


class TimeSeriesCNN(nn.Module):
    """
    CNN-based model for time series classification.
    Designed to handle both sequential data and non-sequential (tabular) data.
    
    Architecture details:
    1. Input Handling:
       - Reshapes tabular data (batch_size, features) to (batch_size, features, 1)
       - Transposes sequential data if needed to (batch_size, features, seq_length)
       
    2. Feature Extraction:
       a. First Convolutional Layer:
          - Input: input_dim channels
          - Output: 32 feature maps
          - Kernel size: 3 with padding=1
          - Activation: ReLU
       b. Second Convolutional Layer:
          - Input: 32 channels
          - Output: 64 feature maps
          - Kernel size: 3 with padding=1
          - Activation: ReLU
       
    3. Dimension Reduction:
       - Adaptive Average Pooling (global pooling)
       - Reduces temporal dimension to 1 regardless of input sequence length
       
    4. Classification Layers:
       a. First Fully Connected Layer:
          - Input: 64 features
          - Output: 32 features
          - Activation: ReLU
       b. Dropout Layer:
          - Rate: 0.3 (30% of neurons randomly disabled during training)
       c. Output Layer:
          - Input: 32 features
          - Output: num_classes (prediction logits)
    
    Args:
        input_dim (int): Number of features in the input time series (default=8)
        seq_length (int): Length of input sequence (default=1)
        num_classes (int): Number of output classes (default=2)
    """
    # def __init__(self, input_dim=16, seq_length=100, num_classes=2):
    def __init__(self, input_dim=8, seq_length=1, num_classes=2):
        super(TimeSeriesCNN, self).__init__()
        
        self.input_dim = input_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Global pooling to handle any sequence length
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # For tabular data (batch_size, features)
        if len(x.shape) == 2:
            batch_size = x.size(0)
            # Reshape to (batch_size, input_dim, 1)
            x = x.view(batch_size, self.input_dim, 1)
        elif x.size(1) != self.input_dim:
            # If shape is (batch_size, seq_len, input_dim)
            x = x.transpose(1, 2)
            
        x = F.relu(self.conv1(x))  # First convolutional layer
        x = F.relu(self.conv2(x))  # Second convolutional layer

        x = self.global_pool(x)    # Global pooling instead of max pooling

        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class TemperatureForecastModel(nn.Module):
    
    def __init__(self, input_dim=8, hidden_dim=32, output_dim=7, num_layers=2, dropout=0.3):
        super(TemperatureForecastModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Check the input shape and handle it appropriately
        if len(x.shape) == 2:
            # For non-sequence data: (batch_size, features)
            x_normalized = self.batch_norm(x)
            
            # Create a single-step sequence for LSTM
            lstm_input = x_normalized.unsqueeze(1)  # (batch_size, 1, features)
        else:
            # Handle the case when x already has 3 dimensions (batch_size, seq_len, features)
            # This is likely what's causing the reshape error
            if x.size(2) == self.input_dim:
                # If the last dimension matches input_dim, we're good
                x_reshaped = x.reshape(-1, self.input_dim)
                x_normalized = self.batch_norm(x_reshaped)
                lstm_input = x_normalized.reshape(batch_size, x.size(1), self.input_dim)
            else:
                # If dimensions are in a different order, try to handle that
                # Print the shape for debugging
                print(f"Warning: Unexpected input shape {x.shape}, expected last dim to be {self.input_dim}")
                
                # Try to flatten and then reshape appropriately
                flattened = x.reshape(batch_size, -1)
                # If the flattened size is divisible by input_dim, we can reshape
                if flattened.size(1) % self.input_dim == 0:
                    seq_len = flattened.size(1) // self.input_dim
                    lstm_input = flattened.reshape(batch_size, seq_len, self.input_dim)
                else:
                    # As a last resort, truncate or pad to make it fit
                    print(f"Warning: Input size {flattened.size(1)} not divisible by input_dim {self.input_dim}")
                    pad_size = (self.input_dim - (flattened.size(1) % self.input_dim)) % self.input_dim
                    if pad_size > 0:
                        padding = torch.zeros(batch_size, pad_size, device=x.device)
                        flattened = torch.cat([flattened, padding], dim=1)
                    seq_len = flattened.size(1) // self.input_dim
                    lstm_input = flattened.reshape(batch_size, seq_len, self.input_dim)
        
        # Initialize LSTM state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM pathway
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        
        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Final fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class ChronosTemperatureForecastModel(nn.Module):
    """
    Temperature forecasting model that uses Amazon Chronos for forecasting.
    
    This model integrates Amazon Chronos within the PyTorch framework to enable
    federated learning with advanced time series forecasting capabilities.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden layers (used for PyTorch fallback only)
        output_dim (int): Number of output values (1 for temperature)
        batch_norm (nn.BatchNorm1d): Batch normalization layer for inputs
        chronos_enabled (bool): Whether Chronos is available and enabled
    """
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.3):
        super(ChronosTemperatureForecastModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.chronos_enabled = False
        
        # Create batch_norm for compatibility with existing code
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        # Try to import and initialize Chronos
        try:
            # Import Amazon Chronos
            from amazon.chronos import ChronosPredictor, ChronosConfig
            
            print("Successfully imported Amazon Chronos!")
            
            # Initialize Chronos configuration
            chronos_config = ChronosConfig(
                freq="H",  # Hourly frequency
                prediction_length=24,  # 24-hour prediction
                context_length=72,  # 3 days of context
                num_layers=2,
                num_cells=64,
                dropout_rate=0.1
            )
            
            # Initialize Chronos predictor
            self.chronos_model = ChronosPredictor(config=chronos_config)
            self.chronos_enabled = True
            print("Chronos model initialized and enabled for forecasting")
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing Amazon Chronos: {e}")
            print("Using PyTorch fallback model instead.")
            self.fallback_model = True
            
            # Create a simple LSTM-based model as fallback
            self.lstm = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply proper initialization to improve learning (fallback model only)"""
        # LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # FC layers
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
    
    def forward(self, x):
        """
        Forward pass for the model.
        Handles both Chronos and PyTorch fallback paths.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                             or (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: Predicted temperature values
        """
        batch_size = x.size(0)
        
        # If Chronos is enabled, use it for predictions
        if self.chronos_enabled:
            try:
                # Format input data for Chronos
                # For this example, we assume x contains the features to predict with
                
                # Create normalized features for Chronos
                x_normalized = self.batch_norm(x) if len(x.shape) == 2 else self.batch_norm(x.reshape(-1, self.input_dim)).reshape(x.shape)
                
                # Convert torch tensor to numpy for Chronos
                numpy_features = x_normalized.detach().cpu().numpy()
                
                # Call Chronos prediction
                # This is a simplified example - in practice, you might need to 
                # format the data differently based on Chronos requirements
                predictions = self.chronos_model.predict(numpy_features)
                
                # Convert predictions back to PyTorch tensor
                return torch.tensor(predictions, device=x.device, dtype=torch.float32)
                
            except Exception as e:
                print(f"Error during Chronos prediction: {e}")
                print("Falling back to PyTorch model for this batch")
                # If there's an error, fall back to PyTorch model for this batch
        
        # Use fallback PyTorch model (LSTM)
        # Handle input data format
        if len(x.shape) == 2:
            # For non-sequence data: (batch_size, features)
            x_normalized = self.batch_norm(x)
            
            # Create a single-step sequence for LSTM
            lstm_input = x_normalized.unsqueeze(1)  # (batch_size, 1, features)
        else:
            # Handle sequence data
            if x.size(2) == self.input_dim:
                x_reshaped = x.reshape(-1, self.input_dim)
                x_normalized = self.batch_norm(x_reshaped)
                lstm_input = x_normalized.reshape(batch_size, x.size(1), self.input_dim)
            else:
                # Handle unexpected shape
                print(f"Warning: Unexpected input shape {x.shape}")
                flattened = x.reshape(batch_size, -1)
                if flattened.size(1) % self.input_dim == 0:
                    seq_len = flattened.size(1) // self.input_dim
                    lstm_input = flattened.reshape(batch_size, seq_len, self.input_dim)
                else:
                    # Pad to match dimensions
                    pad_size = (self.input_dim - (flattened.size(1) % self.input_dim)) % self.input_dim
                    if pad_size > 0:
                        padding = torch.zeros(batch_size, pad_size, device=x.device)
                        flattened = torch.cat([flattened, padding], dim=1)
                    seq_len = flattened.size(1) // self.input_dim
                    lstm_input = flattened.reshape(batch_size, seq_len, self.input_dim)
        
        # Initialize LSTM state
        h0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM pathway
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        
        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Final fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
