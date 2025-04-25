import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from flowerapp.tasks.Task import Task
import time
from torch.nn import CrossEntropyLoss , MSELoss
from torch.optim import Adam

class ExcelTimeSeriesDataset(Dataset):
    """
    Custom dataset for loading time series data from Excel files.
    Handle different dataset structures dynamically.
    (Pilot 1: Weather, Pilot 1: Consumption)
    
    Functions:
    - __init__: Initialize the dataset, load Excel data, preprocess, and split into train/test
    - _preprocess_load_full_data: Apply specific preprocessing for "load_full_data_15min_interval_NW_sample_test" dataset
    - _identify_target_column: Automatically identify the target column based on naming patterns
    - _preprocess_data: Clean and preprocess data including handling non-numeric data and NaN values
    - _apply_dimension_adjustment: Adjust feature dimensions to match configuration requirements
    - __len__: Return the number of samples in the dataset
    - __getitem__: Return a specific sample by index
    """
    def __init__(self, file_path, target_column=None, 
                feature_columns=None, train=True, train_ratio=0.8, 
                transform=None, verbose=True, dataset_type=None, input_dim=7):
        
        """
        Initialize the dataset with Excel data and preprocess it.
        
        Inputs:
            file_path (str): Path to the Excel file
            target_column (str, optional): Name of the target column, auto-detected if None
            feature_columns (list, optional): List of column names to use as features, uses all non-target/non-timestamp if None
            train (bool): Whether to use training portion of the data (True) or test portion (False)
            train_ratio (float): Ratio of data to use for training (0.0-1.0)
            transform (callable, optional): Optional transform to apply to the data
            verbose (bool): Whether to print detailed processing information
            dataset_type (str, optional): Specific dataset type for custom preprocessing
            
        Returns:
            Initialized dataset with processed features and labels as tensors
        """

        self.transform = transform
        self.verbose = verbose
        self.dataset_type = dataset_type
        self.input_dim = input_dim
        # Load Excel 
        try:
            xl_file = pd.ExcelFile(file_path)
            
            # All sheet names
            sheet_names = xl_file.sheet_names
            
            # Load data from each sheet and concatenate
            all_data = []
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(xl_file, sheet_name=sheet)
                    if len(df) > 0:  # Only add non-empty sheets
                        all_data.append(df)
                except Exception as e:
                    if self.verbose:
                        print(f"Error reading sheet '{sheet}': {e}")
            
            if not all_data:
                raise ValueError(f"No valid data found in any sheet of {file_path}")
            
            self.data = pd.concat(all_data, ignore_index=True) # Concatenate all sheets

        except Exception as e:
            if self.verbose:
                print(f"Error loading Excel file: {e}")
            raise
        
        # Apply dataset-specific preprocessing
        if self.dataset_type == "load_full_data_15min_interval_NW_sample_test":
            self._preprocess_load_full_data()
        
        # Identify the timestamp column (Unnamed: 0)
        self.timestamp_col = None
        for col in self.data.columns:
            if 'Unnamed' in col and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.timestamp_col = col
                if self.verbose:
                    print(f"Detected timestamp column: {self.timestamp_col}")
                break
            # Check for explicit timestamp columns
            elif any(time_col in col.lower() for time_col in ['time', 'date', 'timestamp']) and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.timestamp_col = col
                if self.verbose:
                    print(f"Detected timestamp column: {self.timestamp_col}")
                break
        
        # Identify the target column if not specified
        if target_column is None:
            self.target_column = self._identify_target_column(target_column)
        else:
            self.target_column = target_column
            
        if self.verbose:
            print(f"Using target column: {self.target_column}")
        
        # If feature columns not specified use all columns except target and timestamp
        if feature_columns is None:
            self.feature_columns = [col for col in self.data.columns 
                                   if col != self.target_column and col != self.timestamp_col]
        else:
            self.feature_columns = feature_columns
        
        print(f"Using {len(self.feature_columns)} feature columns")
        
        # Clean and preprocess data
        self._preprocess_data()
        
        # Sort data by timestamp
        if self.timestamp_col:
            self.data = self.data.sort_values(by=self.timestamp_col)
            print(f"Data sorted by timestamp column: {self.timestamp_col}")
        
        # Extract features and labels
        self.features = self.data[self.feature_columns].values
        self.labels = self.data[self.target_column].values
        
        # Apply padding or truncation if needed to match config dimensions
        self._apply_dimension_adjustment()
        
        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Split into train and test
        total_samples = len(self.features)
        train_size = int(total_samples * train_ratio)
        
        if train:
            self.features = self.features[:train_size]
            self.labels = self.labels[:train_size]
            print(f"Created training dataset with {len(self.features)} samples")
        else:
            self.features = self.features[train_size:]
            self.labels = self.labels[train_size:]
            print(f"Created testing dataset with {len(self.features)} samples")
    
    def _preprocess_load_full_data(self):
        """
        Apply specific preprocessing for load_full_data_15min_interval dataset.
        
        Inputs:
            None (uses self.data from the class)
            
        Returns:
            None (modifies self.data, self.timestamp_col, and self.target_column in place)
        """

        if self.verbose:
            print("Applying preprocessing for load_full_data_15min_interval dataset")

        # Hhandle datetime column
        datetime_cols = [col for col in self.data.columns if 'datetime' in col.lower()]
        if datetime_cols:
            self.timestamp_col = datetime_cols[0]
            print(f"Using {self.timestamp_col} as timestamp column")
            
            # Ensure datetime column has correct format
            try:
                self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
            except:
                print(f"Could not convert {self.timestamp_col} to datetime format")
        
        # Check f the dataset has a specific target column structure
        # Look for columns like 'class', 'label', 'target' etc.
        label_candidates = [col for col in self.data.columns 
                          if any(label_name in col.lower() 
                               for label_name in ['class', 'label', 'target', 'category'])]
        
        if label_candidates:
            # Use the first candidate as the target
            self.target_column = label_candidates[0]
            print(f"Selected {self.target_column} as target column from available candidates")
    
    def _identify_target_column(self, target_column):
        """
        Automatically identify the most appropriate target column.
        
        Inputs:
            target_column (str): User-specified target column, if any
            
        Returns:
            str: Name of the identified target column
        """

        # Check if the specified target column exists
        if target_column in self.data.columns:
            return target_column
        
        # Find dataset-specific target column 
        if self.dataset_type == "load_full_data_15min_interval_NW_sample_test":
            # Check specific naming patterns for this dataset
            label_candidates = [col for col in self.data.columns 
                              if any(label_name in col.lower() 
                                   for label_name in ['class', 'label', 'target', 'category', 'anomaly'])]
            
            if label_candidates:
                return label_candidates[0]
        
        # If 'Label' column exists but not lowercase 'label' use the capitalized version
        if 'Label' in self.data.columns and 'label' not in self.data.columns:
            return 'Label'
        
        # Check for common target column names
        possible_targets = ['label', 'target', 'class', 'category', 'y', 'output', 'Label', 'TARGET', 'Class', 'anomaly', 'Anomaly']
        for col in possible_targets:
            if col in self.data.columns:
                return col
        
        # Check for columns with not so unique values (maybe labels)
        for col in self.data.columns:
            if col.lower() in ['id', 'index', 'time', 'date', 'timestamp'] or col == self.timestamp_col:
                continue  # Skip non-label columns
                
            try:
                num_unique = self.data[col].nunique()
                if 2 <= num_unique <= 10:  # Reasonable number of classes for classification
                    return col
            except:
                continue
        
        # If still don't have a target column use the last column
        return self.data.columns[-1]
    
    def _preprocess_data(self):
        """
        Clean and preprocess data including handling non-numeric features and NaN values.
        
        Inputs:
            None (uses self.data, self.feature_columns, and self.target_column from the class)
            
        Returns:
            None (modifies self.data and self.feature_columns in place)
        """

        # Dont use timestamp column as feature
        if self.timestamp_col and self.timestamp_col in self.feature_columns:
            self.feature_columns.remove(self.timestamp_col)
            print(f"Removed timestamp column '{self.timestamp_col}' from features")
            
        # Convert all feature columns to numeric
        for col in self.feature_columns[:]:  # Create a copy to safely modify during iteration
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric")
                except:
                    print(f"Failed to convert column '{col}'. Dropping this column.")
                    self.feature_columns.remove(col)
        
        # Convert target column to numeric if needed
        if not pd.api.types.is_numeric_dtype(self.data[self.target_column]):
            # Map unique values to integers
            unique_values = self.data[self.target_column].unique()
            value_to_int = {val: i for i, val in enumerate(unique_values)}
            self.data[self.target_column] = self.data[self.target_column].map(value_to_int)
            print(f"Mapped target values: {value_to_int}")
        
        # Handle NaN values
        nan_count = self.data.isnull().sum().sum()
        if nan_count > 0:
            print(f"Filling {nan_count} NaN values with zeros")
            self.data = self.data.fillna(0)
    
    def _apply_dimension_adjustment(self):
        """
        Adjust feature dimensions to match configuration requirements by padding or truncating.
        
        Inputs:
            None (uses self.features from the class and config.get("ts_input_dim"))
            
        Returns:
            None (modifies self.features in place)
        """

        config_input_dim = self.input_dim
        actual_dim = self.features.shape[1]
        
        if config_input_dim and actual_dim != config_input_dim:
            print(f"Feature dimension mismatch: data has {actual_dim}, config expects {config_input_dim}")
            if actual_dim < config_input_dim:
                # Pad with zeros to match expected dimensions
                padding = np.zeros((self.features.shape[0], config_input_dim - actual_dim))
                self.features = np.concatenate([self.features, padding], axis=1)
                print(f"Padded features with zeros to match config dimensions: {self.features.shape}")
            else:
                # Truncate to match expected dimensions
                self.features = self.features[:, :config_input_dim]
                print(f"Truncated features to match config dimensions: {self.features.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label
    
    def prepare_temperature_forecast(self, target_column='airTemperature', target_hour=14, lookback_days=7):
        """
        Prepares the dataset for temperature forecasting.
        Directly uses SH_Load_vars_labeled for forecasting.
        
        Args:
            target_column (str): Name of the temperature column
            target_hour (int): Hour of the day to forecast temperature for (0-23)
            lookback_days (int): Number of days of history to use for prediction
            
        Returns:
            bool: Success or failure
        """
        from datetime import datetime, timedelta
        
        print(f"Preparing dataset for temperature forecasting at {target_hour}:00")
        
        # Identify temperature column
        self.temp_col = None
        for col in self.data.columns:
            if col.lower() in ['airtemperature', 'air temperature', 'temperature'] or (target_column.lower() in col.lower()):
                self.temp_col = col
                print(f"Using temperature column: {self.temp_col}")
                break
                
        if not self.temp_col:
            print("Warning: Could not find air temperature column. Looking for columns with 'temp' or 'air'")
            for col in self.data.columns:
                if 'temp' in col.lower() or 'air' in col.lower():
                    self.temp_col = col
                    print(f"Using alternative temperature column: {self.temp_col}")
                    break
                    
        if not self.temp_col:
            print("Error: Could not find any suitable temperature column")
            return False
        
        # Ensure timestamp column is datetime
        if self.timestamp_col and self.timestamp_col in self.data.columns:
            try:
                self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col])
                print(f"Converted {self.timestamp_col} to datetime format")
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                return False
        else:
            print("Error: No timestamp column found for forecasting")
            return False
        
        # Sort data by timestamp
        self.data = self.data.sort_values(by=self.timestamp_col)
        
        # Add time-based features
        self.data['hour'] = self.data[self.timestamp_col].dt.hour
        self.data['day_of_week'] = self.data[self.timestamp_col].dt.dayofweek
        self.data['month'] = self.data[self.timestamp_col].dt.month
        self.data['day_of_year'] = self.data[self.timestamp_col].dt.dayofyear
        
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_week'] / 7)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_week'] / 7)
        
        print("Created time-based features for forecasting")
        
        # Select only numeric columns for features
        feature_cols = [col for col in self.data.columns 
                    if col != self.timestamp_col 
                    and col != self.temp_col  # Exclude the target column from features
                    and pd.api.types.is_numeric_dtype(self.data[col])]
        
        print(f"Using {len(feature_cols)} feature columns for temperature forecasting")
        
        # Determine data sampling frequency
        time_diffs = self.data[self.timestamp_col].diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            sampling_mins = median_diff.total_seconds() / 60
            print(f"Detected data sampling frequency: approximately every {sampling_mins:.1f} minutes")
            
            samples_per_day = int(24 * 60 / sampling_mins)
        else:
            samples_per_day = 96
            print("Could not determine sampling frequency. Assuming 15-minute intervals (96 samples per day)")
        
        lookback_size = lookback_days * samples_per_day
        print(f"Using lookback window of {lookback_days} days ({lookback_size} samples)")
        
        X_list = []
        y_list = []
        
        data_sorted = self.data.copy()
        
        min_required = lookback_size + samples_per_day + 1
        
        if len(data_sorted) < min_required:
            print(f"Warning: Not enough data for forecasting. Need at least {min_required} samples.")
            lookback_size = max(1, int(len(data_sorted) * 0.7))
            print(f"Reducing lookback window to {lookback_size} samples")
        
        # Create forecast samples
        for i in range(len(data_sorted) - lookback_size - 1):
            end_idx = i + lookback_size
            end_time = data_sorted.iloc[end_idx][self.timestamp_col]
            
            target_day = end_time.date() + timedelta(days=1)
            target_time = datetime.combine(target_day, datetime.min.time()) + timedelta(hours=target_hour)
            
            target_idx = None
            closest_diff = timedelta(days=1)  
            
            # Find the closest time point to our target
            for j in range(end_idx + 1, len(data_sorted)):
                time_diff = abs(data_sorted.iloc[j][self.timestamp_col] - target_time)
                if time_diff < closest_diff:
                    closest_diff = time_diff
                    target_idx = j
                    if time_diff < timedelta(minutes=30):
                        break
            
            if target_idx is None or closest_diff > timedelta(hours=1):
                continue
            
            # Extract feature window
            window_data = data_sorted.iloc[i:end_idx][feature_cols].values
            target_temp = data_sorted.iloc[target_idx][self.temp_col]
            
            if pd.isna(target_temp):
                continue
            
            # Use a simplified approach for features - just the most recent data point
            # This avoids the LSTM sequence handling complexity
            X_list.append(window_data[-1])  # Just use the most recent time point
            y_list.append(target_temp)
        
        if len(X_list) > 0:
            print(f"Created {len(X_list)} samples for temperature forecasting")
            self.features = np.array(X_list)
            self.labels = np.array(y_list)
            
            # Print feature shape information
            print(f"Feature shape: {self.features.shape}")
            if len(self.features) > 0:
                print(f"Each sample has {self.features[0].size} values")
            
            # Convert to tensors with appropriate dtypes
            self.features = torch.tensor(self.features, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.float32)  # Float for regression
            
            return True
        else:
            print("Error: Could not create any valid forecast samples")
            return False
        

    def forecast_next_day_temperature(self, model, device="cpu"):

        from datetime import datetime, timedelta
        
        if not hasattr(self, 'temp_col') or not self.temp_col:
            print("Error: Dataset not configured for temperature forecasting")
            return None, None
        
        data_sorted = self.data.sort_values(by=self.timestamp_col)
        
        last_timestamp = data_sorted.iloc[-1][self.timestamp_col]
        
        forecast_date = last_timestamp.date() + timedelta(days=1)
        forecast_datetime = datetime.combine(forecast_date, datetime.min.time()) + timedelta(hours=14)
        
        if hasattr(self, 'features') and len(self.features) > 0:
            most_recent_features = torch.tensor(self.features[-1], dtype=torch.float32).unsqueeze(0).to(device)
            
            model.eval()
            
            with torch.no_grad():
                prediction = model(most_recent_features)
                
            forecasted_temp = prediction.item()
            
            print(f"\n===== TEMPERATURE FORECAST =====")
            print(f"Forecast for: {forecast_datetime}")
            print(f"Predicted temperature: {forecasted_temp:.1f}°C")
            print(f"==================================\n")
            
            return forecasted_temp, forecast_datetime
        else:
            print("Error: No forecast data available")
            return None, None
    def ensure_feature_dimensions(self, target_dim):
        """
        Ensures the feature tensor has the specified number of dimensions.
        Pads or truncates features as needed.
        
        Args:
            target_dim (int): Target number of features
            
        Returns:
            None (modifies self.features in place)
        """
        if not hasattr(self, 'features') or len(self.features) == 0:
            print("Warning: No features to adjust dimensions")
            return
            
        current_dim = self.features.shape[1]
        
        if current_dim == target_dim:
            print(f"Feature dimensions already match target: {target_dim}")
            return
            
        print(f"Adjusting feature dimensions from {current_dim} to {target_dim}")
        
        if current_dim < target_dim:
            # Need to pad
            padding = torch.zeros((self.features.shape[0], target_dim - current_dim), 
                                dtype=self.features.dtype, 
                                device=self.features.device)
            self.features = torch.cat([self.features, padding], dim=1)
            print(f"Padded features to shape: {self.features.shape}")
        else:
            # Need to truncate
            self.features = self.features[:, :target_dim]
            print(f"Truncated features to shape: {self.features.shape}")


# class TimeSeriesClassification(Task):
    
#     def __init__(self, model, device=None):
#         """
#         Initialize the ImageClassification task.

#         Args:
#             model (nn.Module): The PyTorch model for image classification.
#             device (str or torch.device, optional): Device to run on. Defaults to CUDA if available.
#         """
#         self.model = model
#         self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
#         self.model.to(self.device)
#         self.optimizer = None  # Will be initialized during training
#         self.config = {} # FederatedDataset instance
        
    def load_time_series_data(client_data_path, batch_size=32,config={}):
        """
        Loads time series data from Excel files for federated learning.
        Specifically targets SH_Load_vars_labeled.xlsx for temperature forecasting.
        
        Args:
            client_data_path (str): Path to client's data directory
            batch_size (int): Batch size for DataLoader
            
        Returns:
            tuple: (train_loader, test_loader, num_features, num_classes)
        """
        ## CHeck if available and read xlsx file
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
            
            # For temperature forecasting, we need special preparation
            if is_temp_forecast:
                print("Preparing dataset for temperature forecasting...")
                # Get forecast parameters from config
                target_hour = config.get("forecast_target_hour", 14)
                lookback_days = config.get("forecast_lookback_days", 7)
                
                # Prepare the dataset for forecasting
                success = trainset.prepare_temperature_forecast(
                    target_column='airTemperature',
                    target_hour=target_hour,
                    lookback_days=lookback_days
                )
                
                if not success:
                    raise RuntimeError("Failed to prepare temperature forecast data. Check if the data contains temperature readings.")
            
            testset = ExcelTimeSeriesDataset(
                file_path=file_path,
                train=False,
                train_ratio=0.8,
                verbose=False
            )
            
            # Also prepare test set for temperature forecasting
            if is_temp_forecast:
                success = testset.prepare_temperature_forecast(
                    target_column='airTemperature',
                    target_hour=config.get("forecast_target_hour", 14),
                    lookback_days=config.get("forecast_lookback_days", 7)
                )
                if not success:
                    print("Warning: Could not prepare test set for temperature forecasting.")
            
            # Get number of features from the dataset
            num_features = trainset.features.shape[1]
            print(f"Feature tensor shape: {trainset.features.shape}")
            
            # Check if we need to use fixed dimensions for TemperatureForecastModel
            if MODEL_NAME == "TemperatureForecastModel":
                # Get fixed dimension from config or use default
                fixed_dim = config.get("ts_input_dim", 8)
                
                # Always ensure we're using exactly 8 features for temperature forecasting
                # This ensures consistent model parameters across all clients
                if num_features != fixed_dim:
                    print(f"Adjusting feature dimensions from {num_features} to {fixed_dim} for temperature forecasting")
                    trainset.ensure_feature_dimensions(fixed_dim)
                    testset.ensure_feature_dimensions(fixed_dim)
                    num_features = fixed_dim
                    
                # Update config with the fixed dimension
                config["ts_input_dim"] = fixed_dim
                print(f"Set fixed input dimension to {fixed_dim} for temperature forecasting")
            
            # Update the config with the actual feature count
            config["ts_input_dim"] = num_features
            print(f"Setting ts_input_dim in config to: {num_features}")
            
            # Get number of classes from unique labels
            if is_temp_forecast:
                # For regression problem (temperature forecasting)
                unique_labels = 1  # Single continuous output
                num_classes = 1
                # Ensure labels are float for regression
                trainset.labels = trainset.labels.float()
                testset.labels = testset.labels.float()
            else:
                # For classification problem
                unique_labels = torch.unique(trainset.labels)
                num_classes = len(unique_labels)
                # Ensure labels are long for classification
                trainset.labels = trainset.labels.long()
                testset.labels = testset.labels.long()
            
            # Update config with detected class count
            config["ts_output_dim"] = num_classes
            config["ts_num_classes"] = num_classes
            print(f"Setting output_dim to: {num_classes}")
            
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
            
            return trainloader, testloader, num_features, num_classes
        
        except Exception as e:
            print(f"Error loading time series data: {e}")
            # Add more detailed error information
            import traceback
            print("Detailed error traceback:")
            traceback.print_exc()
            raise
#     def train(self, trainloader: DataLoader, epochs=1, lr=0.001):
#         """
#         Trains the model on the provided data and tracks metrics.
#         Handles temperature forecasting with proper data types.
        
#         Args:
#             model (nn.Module): Neural network model to train
#             trainloader (DataLoader): DataLoader containing training data
#             epochs (int): Number of training epochs
        
#         Returns:
#             list: List of tuples containing metrics for each epoch
#         """
#         # Set random seed (time)
#         seed = int(time.time()) % 10000
#         torch.manual_seed(seed)
        
#         # Check if we're doing regression (temperature forecasting) or classification
#         #is_regression = MODEL_NAME == "TemperatureForecastModel"
#         is_regression = True
#         # Define appropriate loss function
#         if is_regression:
#             loss_fn = MSELoss()  # Mean Squared Error for regression
#             print("Using MSE loss for temperature regression")
#         else:
#             loss_fn = CrossEntropyLoss()  # Cross Entropy for classification
        
#         # Define optimizer
#         optimizer = Adam(self.model.parameters(), lr=lr)
        
#         self.model.train()
#         all_metrics = []
        
#         for epoch in range(epochs):
#             running_loss = 0.0
            
#             # For classification metrics
#             correct = 0
#             total = 0
            
#             # For regression metrics
#             if is_regression:
#                 sum_squared_error = 0.0
#                 sum_absolute_error = 0.0
#                 n_samples = 0
            
#             for inputs, labels in trainloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 # Forward pass
#                 outputs = self.model(inputs)
                
#                 # For regression, ensure outputs and labels have same shape
#                 if is_regression:
#                     # Make sure both are properly shaped for MSE loss
#                     if outputs.shape != labels.shape:
#                         outputs = outputs.view(labels.shape)
                
#                 # Calculate loss
#                 loss = loss_fn(outputs, labels)
                
#                 # Backward pass and optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
                
#                 # Track metrics
#                 running_loss += loss.item() * labels.size(0)  # Weighted by batch size
                
#                 if is_regression:
#                     # Track regression metrics
#                     with torch.no_grad():
#                         pred = outputs.view(-1)
#                         target = labels.view(-1)
#                         squared_error = ((pred - target) ** 2).sum().item()
#                         absolute_error = (pred - target).abs().sum().item()
                        
#                         sum_squared_error += squared_error
#                         sum_absolute_error += absolute_error
#                         n_samples += labels.size(0)
#                 else:
#                     # Track classification metrics
#                     _, predicted = torch.max(outputs, 1)
#                     correct += (predicted == labels).sum().item()
#                     total += labels.size(0)
            
#             # Calculate epoch metrics
#             epoch_loss = running_loss / len(trainloader.dataset)
            
#             if is_regression:
#                 # Calculate regression metrics
#                 rmse = (sum_squared_error / n_samples) ** 0.5  # Root Mean Squared Error
#                 mae = sum_absolute_error / n_samples  # Mean Absolute Error
                
#                 print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C")
#                 all_metrics.append((epoch_loss, rmse, mae))
#             else:
#                 # Calculate classification metrics
#                 epoch_accuracy = 100 * correct / total
                
#                 print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
#                 all_metrics.append((epoch_loss, epoch_accuracy))
        
#         return all_metrics

        
#     def test(self, model, testloader):
#         """
#         Evaluates the model on test data.
#         Updated to handle both classification and regression tasks.
        
#         Args:
#             model (nn.Module): Neural network model to evaluate
#             testloader (DataLoader): DataLoader containing test data
        
#         Returns:
#             tuple: For classification: (average_loss, accuracy)
#                 For regression: (average_loss, rmse, mae)
#         """
#         # Check if we're doing regression or classification
#         #is_regression = MODEL_NAME == "TemperatureForecastModel"
#         is_regression = True
#         # Define appropriate loss function
#         if is_regression:
#             criterion = MSELoss(reduction='sum')  # Sum reduction to calculate average later
#         else:
#             criterion = CrossEntropyLoss(reduction='sum')
#         model.eval()
        
#         total_loss = 0.0
        
#         # For classification
#         total_correct = 0
#         total_samples = 0
        
#         # For regression
#         if is_regression:
#             sum_squared_error = 0.0
#             sum_absolute_error = 0.0
        
#         # Fix random seed for reproducibility during evaluation
#         torch.manual_seed(42)
        
#         # Disable gradient computation for evaluation
#         with torch.no_grad():
#             for inputs, labels in testloader:
#                 batch_size = labels.size(0)
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 # Forward pass
#                 outputs = model(inputs)
                
#                 # For regression, ensure outputs and labels have same shape
#                 if is_regression:
#                     if outputs.shape != labels.shape:
#                         outputs = outputs.view(labels.shape)
                
#                 # Calculate loss
#                 loss = criterion(outputs, labels).item()
#                 total_loss += loss
                
#                 if is_regression:
#                     # Calculate regression metrics
#                     pred = outputs.view(-1)
#                     target = labels.view(-1)
#                     squared_error = ((pred - target) ** 2).sum().item()
#                     absolute_error = (pred - target).abs().sum().item()
                    
#                     sum_squared_error += squared_error
#                     sum_absolute_error += absolute_error
#                     total_samples += batch_size
#                 else:
#                     # Calculate classification metrics
#                     _, predicted = torch.max(outputs, 1)
#                     total_correct += (predicted == labels).sum().item()
#                     total_samples += batch_size
        
#         # Calculate global metrics
#         avg_loss = total_loss / total_samples
        
#         if is_regression:
#             # Return regression metrics
#             rmse = (sum_squared_error / total_samples) ** 0.5
#             mae = sum_absolute_error / total_samples
#             return avg_loss, rmse, mae
#         else:
#             # Return classification metrics
#             accuracy = total_correct / total_samples
#             return avg_loss, accuracy

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
class TimeSeriesClassification(Task):
    def __init__(self, model, device=None):
        self.model=model
        self.device = device

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
            
            # For temperature forecasting, we need special preparation
            if is_temp_forecast:
                print("Preparing dataset for temperature forecasting...")
                # Get forecast parameters from config
                target_hour = config.get("forecast_target_hour", 14)
                lookback_days = config.get("forecast_lookback_days", 7)
                
                # Prepare the dataset for forecasting
                success = trainset.prepare_temperature_forecast(
                    target_column='airTemperature',
                    target_hour=target_hour,
                    lookback_days=lookback_days
                )
                
                if not success:
                    raise RuntimeError("Failed to prepare temperature forecast data. Check if the data contains temperature readings.")
            
            testset = ExcelTimeSeriesDataset(
                file_path=file_path,
                train=False,
                train_ratio=0.8,
                verbose=False
            )
            
            # Also prepare test set for temperature forecasting
            if is_temp_forecast:
                success = testset.prepare_temperature_forecast(
                    target_column='airTemperature',
                    target_hour=config.get("forecast_target_hour", 14),
                    lookback_days=config.get("forecast_lookback_days", 7)
                )
                if not success:
                    print("Warning: Could not prepare test set for temperature forecasting.")
            
            # Get number of features from the dataset
            num_features = trainset.features.shape[1]
            print(f"Feature tensor shape: {trainset.features.shape}")
            
            # Check if we need to use fixed dimensions for TemperatureForecastModel
            if MODEL_NAME == "TemperatureForecastModel":
                # Get fixed dimension from config or use default
                fixed_dim = config.get("ts_input_dim", 8)
                
                # Always ensure we're using exactly 8 features for temperature forecasting
                # This ensures consistent model parameters across all clients
                if num_features != fixed_dim:
                    print(f"Adjusting feature dimensions from {num_features} to {fixed_dim} for temperature forecasting")
                    trainset.ensure_feature_dimensions(fixed_dim)
                    testset.ensure_feature_dimensions(fixed_dim)
                    num_features = fixed_dim
                    
                # Update config with the fixed dimension
                config["ts_input_dim"] = fixed_dim
                print(f"Set fixed input dimension to {fixed_dim} for temperature forecasting")
            
            # Update the config with the actual feature count
            config["ts_input_dim"] = num_features
            print(f"Setting ts_input_dim in config to: {num_features}")
            
            # Get number of classes from unique labels
            if is_temp_forecast:
                # For regression problem (temperature forecasting)
                unique_labels = 1  # Single continuous output
                num_classes = 1
                # Ensure labels are float for regression
                trainset.labels = trainset.labels.float()
                testset.labels = testset.labels.float()
            else:
                # For classification problem
                unique_labels = torch.unique(trainset.labels)
                num_classes = len(unique_labels)
                # Ensure labels are long for classification
                trainset.labels = trainset.labels.long()
                testset.labels = testset.labels.long()
            
            # Update config with detected class count
            config["ts_output_dim"] = num_classes
            config["ts_num_classes"] = num_classes
            print(f"Setting output_dim to: {num_classes}")
            
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

    def train(self, trainloader, epochs=1, lr=0.001):
        """
        Trains the model on the provided data and tracks metrics.
        Handles temperature forecasting with proper data types.
        
        Args:
            model (nn.Module): Neural network model to train
            trainloader (DataLoader): DataLoader containing training data
            epochs (int): Number of training epochs
            lr (float): learning rate
        
        Returns:
            list: List of tuples containing metrics for each epoch
        """
        # Set random seed (time)
        seed = int(time.time()) % 10000
        torch.manual_seed(seed)
        
        # Check if we're doing regression (temperature forecasting) or classification
        #is_regression = MODEL_NAME == "TemperatureForecastModel"
        is_regression = True
        # Define appropriate loss function
        if is_regression:
            loss_fn = MSELoss()  # Mean Squared Error for regression
            print("Using MSE loss for temperature regression")
        else:
            loss_fn = CrossEntropyLoss()  # Cross Entropy for classification
        
        # Define optimizer
        optimizer = Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        all_metrics = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            # For classification metrics
            correct = 0
            total = 0
            
            # For regression metrics
            if is_regression:
                sum_squared_error = 0.0
                sum_absolute_error = 0.0
                n_samples = 0
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # For regression, ensure outputs and labels have same shape
                if is_regression:
                    # Make sure both are properly shaped for MSE loss
                    if outputs.shape != labels.shape:
                        outputs = outputs.view(labels.shape)
                
                # Calculate loss
                loss = loss_fn(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                running_loss += loss.item() * labels.size(0)  # Weighted by batch size
                
                if is_regression:
                    # Track regression metrics
                    with torch.no_grad():
                        pred = outputs.view(-1)
                        target = labels.view(-1)
                        squared_error = ((pred - target) ** 2).sum().item()
                        absolute_error = (pred - target).abs().sum().item()
                        
                        sum_squared_error += squared_error
                        sum_absolute_error += absolute_error
                        n_samples += labels.size(0)
                else:
                    # Track classification metrics
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(trainloader.dataset)
            
            if is_regression:
                # Calculate regression metrics
                rmse = (sum_squared_error / n_samples) ** 0.5  # Root Mean Squared Error
                mae = sum_absolute_error / n_samples  # Mean Absolute Error
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C")
                all_metrics.append((epoch_loss, rmse, mae))
            else:
                # Calculate classification metrics
                epoch_accuracy = 100 * correct / total
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
                all_metrics.append((epoch_loss, epoch_accuracy))
                
        return all_metrics

    def test(self, testloader):
        """
        Evaluates the model on test data.
        Updated to handle both classification and regression tasks.
        
        Args:
            model (nn.Module): Neural network model to evaluate
            testloader (DataLoader): DataLoader containing test data
        
        Returns:
            tuple: For classification: (average_loss, accuracy)
                For regression: (average_loss, rmse, mae)
        """
        # Check if we're doing regression or classification
        #is_regression = MODEL_NAME == "TemperatureForecastModel"
        is_regression = True
        # Define appropriate loss function
        if is_regression:
            criterion = MSELoss(reduction='sum')  # Sum reduction to calculate average later
        else:
            criterion = CrossEntropyLoss(reduction='sum')
        self.model.eval()
        
        total_loss = 0.0
        
        # For classification
        total_correct = 0
        total_samples = 0
        
        # For regression
        if is_regression:
            sum_squared_error = 0.0
            sum_absolute_error = 0.0
        
        # Fix random seed for reproducibility during evaluation
        torch.manual_seed(42)
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, labels in testloader:
                batch_size = labels.size(0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # For regression, ensure outputs and labels have same shape
                if is_regression:
                    if outputs.shape != labels.shape:
                        outputs = outputs.view(labels.shape)
                
                # Calculate loss
                loss = criterion(outputs, labels).item()
                total_loss += loss
                
                if is_regression:
                    # Calculate regression metrics
                    pred = outputs.view(-1)
                    target = labels.view(-1)
                    squared_error = ((pred - target) ** 2).sum().item()
                    absolute_error = (pred - target).abs().sum().item()
                    
                    sum_squared_error += squared_error
                    sum_absolute_error += absolute_error
                    total_samples += batch_size
                else:
                    # Calculate classification metrics
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += batch_size
        
        # Calculate global metrics
        avg_loss = total_loss / total_samples
        
        if is_regression:
            # Return regression metrics
            rmse = (sum_squared_error / total_samples) ** 0.5
            mae = sum_absolute_error / total_samples
            return avg_loss, rmse, mae
        else:
            # Return classification metrics
            accuracy = total_correct / total_samples
            return avg_loss, accuracy