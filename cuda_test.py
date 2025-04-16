import torch
import torchvision.models as models

# ✅ This should return True
print("CUDA available:", torch.cuda.is_available())

# ✅ Send model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18().to(device)

# ✅ Send data to GPU
x = torch.randn(64, 3, 224, 224).to(device)

# Forward pass to see memory used
with torch.no_grad():
    output = model(x)

print("Memory allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
print("Max memory allocated:", torch.cuda.max_memory_allocated() / 1e6, "MB")