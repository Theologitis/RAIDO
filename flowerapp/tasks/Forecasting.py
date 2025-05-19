
from flowerapp.tasks.Task import Task
from pathlib import Path
from flowerapp.tasks.TimeSeriesClassification import ExcelTimeSeriesDataset
from torch.utils.data import DataLoader
import torch
import time
import torch.nn as nn
from torch.optim import Adam


class Forecasting(Task): 
    
    def __init__(self,model,device):
        self.model=model
        self.device = device
        self.model.to(device)
        return
    
    def load_data(self, client_data_path, batch_size=32, config={}):
        MODEL_NAME = "TemperatureForecastModel"
        # Find all Excel files in the data directory
        data_path = Path(client_data_path)
        excel_files = list(data_path.glob('*.xlsx')) + list(data_path.glob('*.xls'))
        
        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {client_data_path}")
        
        # Look specifically for SH_Load_vars_labeled.xlsx - this is critical for temperature forecasting
        sh_files = [f for f in excel_files if "SH_Load_vars_labeled" in f.name]
        
        if not sh_files and MODEL_NAME == "TemperatureForecastModel":
            raise FileNotFoundError(f"SH_Load_vars_labeled.xlsx not found in {client_data_path}. This file is required for temperature forecasting.")
        
        # Use the SH_Load_vars_labeled file if found, otherwise use first Excel file
        file_path = sh_files[0] if sh_files else excel_files[0]
        print(f"Loading time series data from: {file_path}")
        
        try:
            # Check if we're doing temperature forecasting
            is_temp_forecast = MODEL_NAME == "TemperatureForecastModel"
            
            if is_temp_forecast:
                print("="*80)
                print("PREPARING TEMPERATURE FORECASTING MODEL WITH SH_LOAD_VARS_LABELED DATA")
                print("="*80)
        #######
        
            # Create datasets
            trainset = ExcelTimeSeriesDataset(
                file_path=file_path,
                train=True,
                train_ratio=0.8,  # 80% for training, 20% for testing
                verbose=True
            )
            
            
            testset = ExcelTimeSeriesDataset(
                file_path=file_path,
                train=False,
                train_ratio=0.8,
                verbose=False
            )
            
            
            # Get number of features from the dataset
            num_features = trainset.features.shape[1]
            print(f"Feature tensor shape: {trainset.features.shape}")
            
            
            # Handle dataset with too few samples
            min_batch_size = 1
            if len(trainset) < batch_size or len(testset) < batch_size:
                new_batch_size = max(min_batch_size, min(len(trainset), len(testset)))
                print(f"Adjusting batch size from {batch_size} to {new_batch_size} due to small dataset size")
                batch_size = new_batch_size
            
            # Create data loaders
            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False  # Keep all samples even if they don't form complete batches
            )
            
            testloader = DataLoader(
                testset,
                batch_size=batch_size,
                drop_last=False
            )
            
            # For debugging: check the shape of a batch
            for inputs, labels in trainloader:
                print(f"Batch input shape: {inputs.shape}")
                print(f"Batch label shape: {labels.shape}")
                print(f"Batch label dtype: {labels.dtype}")
                break
            
            return trainloader, testloader
        
        except Exception as e:
            
            print(f"Error loading time series data: {e}")
            # Add more detailed error information
            import traceback
            print("Detailed error traceback:")
            traceback.print_exc()
            raise
    
    def train(self, trainloader, epochs, lr , proximal_mu):
        torch.manual_seed(int(time.time()) % 10000)
        self.model.train()
        
        loss_fn = nn.MSELoss()
        optimizer = Adam(self.model.parameters(), lr=0.001)
        all_metrics = []

        for epoch in range(epochs):
            running_loss = 0.0
            sum_squared_error = 0.0
            sum_absolute_error = 0.0
            n_samples = 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()

                outputs = self.model(inputs)
                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)

                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)

                with torch.no_grad():
                    pred = outputs.view(-1)
                    target = labels.view(-1)
                    sum_squared_error += ((pred - target) ** 2).sum().item()
                    sum_absolute_error += (pred - target).abs().sum().item()
                    n_samples += labels.size(0)

            epoch_loss = running_loss / len(trainloader.dataset)
            rmse = (sum_squared_error / n_samples) ** 0.5
            mae = sum_absolute_error / n_samples

            print(f"[Regression] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C")
            all_metrics.append((epoch_loss, rmse, mae))
            return all_metrics
    
    def test(self,testloader):
        self.model.eval()
        criterion = nn.MSELoss(reduction='sum')
        total_loss = 0.0
        sum_squared_error = 0.0
        sum_absolute_error = 0.0
        total_samples = 0

        torch.manual_seed(42)

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if outputs.shape != labels.shape:
                    outputs = outputs.view(labels.shape)

                loss = criterion(outputs, labels).item()
                total_loss += loss

                pred = outputs.view(-1)
                target = labels.view(-1)
                sum_squared_error += ((pred - target) ** 2).sum().item()
                sum_absolute_error += (pred - target).abs().sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        rmse = (sum_squared_error / total_samples) ** 0.5
        mae = sum_absolute_error / total_samples

        print(f"[Regression Test] Loss: {avg_loss:.4f} | RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C")
        return avg_loss, rmse
    