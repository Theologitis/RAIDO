import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam , SGD
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flowerapp.tasks.Task import Task
from flowerapp.utils import get_weights ,get_optimizer

fds=None # Cache FederatedDataset

class ImageClassification(Task):
    
    def __init__(self, model, device=None):
        """
        Initialize the ImageClassification task.

        Args:
            model (nn.Module): The PyTorch model for image classification.
            device (str or torch.device, optional): Device to run on. Defaults to CUDA if available.
        """
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = None  # Will be initialized during training
        self.fds = None  # FederatedDataset instance

    def load_data(self, partition_id: int, num_partitions: int, batch_size: int):
        """Load partitioned CIFAR-10 data."""
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        def apply_transforms(batch):
            batch["img"] = torch.stack([transform(img) for img in batch["img"]])
            batch["label"] = torch.tensor(batch["label"])
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

        return trainloader, testloader

    def train(self, trainloader: DataLoader, epochs=1, lr=0.001, proximal_mu=0.0, server_control_variate=None,client_control_variate=None):
        """Train the model."""
        self.model.train()
        self.optimizer = get_optimizer("SGD",self.model.parameters(), lr=lr)
        global_weights = get_weights(self.model)
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch in trainloader:
                
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # === Add Proximal Regularization === for FedProx strategy
                if proximal_mu > 0.0:
                    prox_term = 0.0
                    for param, global_param in zip(get_weights(self.model), global_weights):
                        prox_term += ((param - global_param) ** 2).sum()

                    loss += (proximal_mu / 2) * prox_term
                # ========================================== #
               
                loss.backward()
                # max_grad = max(p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None)
                # #print(f"Max gradient before correction: {max_grad}")
                # #=== Add control variant === for Scaffold strategy
                if server_control_variate is not None and client_control_variate is not None:
                    with torch.no_grad():
                        for i, param in enumerate(self.model.parameters()):
                            if param.grad is not None:
                                correction = torch.tensor(server_control_variate[i], device=self.device) - torch.tensor(client_control_variate[i], device=self.device)
                                param.grad += torch.clamp(correction, -1.0, 1.0)
                # ========================================== #
                max_grad = max(p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None)
                #print(f"Max gradient after correction: {max_grad}")
                self.optimizer.step()
                running_loss += loss.item()
                for param in self.model.parameters():
                    if torch.isnan(param.grad).any():
                        print("NaN in gradients!")
                        break
            avg_loss = running_loss / len(trainloader)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            # print(f"Epoch {epoch+1}, Gradients: {param.grad.max()}")
            updated_weights = get_weights(self.model)
            control_variate = [
                client_control_variate[i] - server_control_variate[i] + 
                (global_weights[i] - updated_weights[i]) / (lr * epochs)
                for i in range(len(updated_weights))
            ]
        return avg_loss , control_variate
    
    def train_scaffold(self, trainloader: DataLoader, local_steps=5, lr=0.001, server_control_variate=None,client_control_variate=None):
        """Train the model."""
        self.model.train()
        self.optimizer = get_optimizer("SGD",self.model.parameters(), lr=lr)
        global_weights = get_weights(self.model)
        
        running_loss = 0.0
        local_steps = 5

        for i, batch in enumerate(trainloader):
            if i >= local_steps:
                break

            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # max_grad = max(p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None)
            # #print(f"Max gradient before correction: {max_grad}")
            # #=== Add control variant === for Scaffold strategy
            if server_control_variate is not None and client_control_variate is not None:
                for i, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        scv = torch.tensor(server_control_variate[i], device=self.device)
                        ccv = torch.tensor(client_control_variate[i], device=self.device)
                        
                        # Check shape compatibility
                        if param.grad.shape != scv.shape:
                            raise ValueError(f"Shape mismatch: grad {param.grad.shape}, server_cv {scv.shape}")
                        if param.grad.shape != ccv.shape:
                            raise ValueError(f"Shape mismatch: grad {param.grad.shape}, client_cv {ccv.shape}")
                        
                        correction = scv - ccv
                        param.grad += torch.clamp(correction, -1.0, 1.0)
            # ========================================== #
            
            max_grad = max(p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None)
            #print(f"Max gradient after correction: {max_grad}")
            self.optimizer.step()
            running_loss += loss.item()
            for param in self.model.parameters():
                if torch.isnan(param.grad).any():
                    print("NaN in gradients!")
                    break
        avg_loss = running_loss / len(trainloader)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        # print(f"Epoch {epoch+1}, Gradients: {param.grad.max()}")
        updated_weights = get_weights(self.model)
        control_variate = [
            client_control_variate[i] - server_control_variate[i] + 
            (global_weights[i] - updated_weights[i]) / (lr * local_steps)
            for i in range(len(updated_weights))
        ]
        return avg_loss , control_variate


    def test(self, testloader: DataLoader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()

        accuracy = total_correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader)
        return avg_loss, {"accuracy":accuracy}
# import cloudpickle
# import torchvision.models as models
# model = models.resnet18(pretrained=True)
# task = ImageClassification(model,None)
# with open("model.pkl","wb") as f:
#     cloudpickle.dump(task,f)