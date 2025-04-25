import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flowerapp.tasks.Task import Task
from flowerapp.task import get_weights
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
        if self.fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            
            self.fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = self.fds.load_partition(partition_id)
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

    def train(self, trainloader: DataLoader, epochs=1, lr=0.001, proximal_mu=0.0):
        """Train the model."""
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        global_weights = get_weights(self.model)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in trainloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # === Add Proximal Regularization ===
                prox_term = 0.0
                for param, global_param in zip(get_weights(self.model), global_weights):
                    prox_term += ((param - global_param) ** 2).sum()

                loss += (proximal_mu / 2) * prox_term
                # ========================================== #
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(trainloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return avg_loss

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
        return avg_loss, accuracy
    
