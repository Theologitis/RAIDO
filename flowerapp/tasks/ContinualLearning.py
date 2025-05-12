import argparse
import torch
import torch.optim.lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from avalanche.benchmarks import SplitMNIST
from avalanche.models import MlpVAE
from avalanche.training.supervised import VAETraining
from avalanche.training.plugins import GenerativeReplayPlugin
from flowerapp.tasks.Task import Task
from torch.utils.data import ConcatDataset, DataLoader

class ContinualLearning(Task):
    
    def __init__(self,model,device):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model= MlpVAE((1, 28, 28), nhid=2, device=device)
        return
    
    def load_data(self,path=None,batch_size=None):
        # Create SplitMNIST benchmark (10 tasks)
        benchmark = SplitMNIST(n_experiences=10, seed=1234)

        # Combine all training datasets from each experience
        full_train_dataset = ConcatDataset([exp.dataset for exp in benchmark.train_stream])

        # Combine all test datasets from each experience
        full_test_dataset = ConcatDataset([exp.dataset for exp in benchmark.test_stream])

        # Create PyTorch-style DataLoaders
        train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(full_test_dataset, batch_size=64, shuffle=False)
        return train_loader , test_loader
    
    def train(self,trainloader,epochs,lr,proximal_mu):
        

        # --- BENCHMARK CREATION
        benchmark = SplitMNIST(n_experiences=10, seed=1234)
        # ---------

        # MODEL CREATION
        model = self.model
        device = self.device

        # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
        cl_strategy = VAETraining(
            model,
            torch.optim.Adam(model.parameters(), lr=lr),
            train_mb_size=100,
            train_epochs=epochs,
            device=device,
            plugins=[GenerativeReplayPlugin()],
        )

        # TRAINING LOOP
        print("Starting experiment...")
        f, axarr = plt.subplots(benchmark.n_experiences, 10)
        k = 0
        for experience in benchmark.train_stream:
            print("Start of experience ", experience.current_experience)
            cl_strategy.train(experience)
            print("Training completed")

            samples = model.generate(10)
            samples = samples.detach().cpu().numpy()

            for j in range(10):
                axarr[k, j].imshow(samples[j, 0], cmap="gray")
                axarr[k, 4].set_title("Generated images for experience " + str(k))
            np.vectorize(lambda ax: ax.axis("off"))(axarr)
            k += 1

        f.subplots_adjust(hspace=1.2)
        plt.savefig('data')
        plt.show()
        return {"loss":0.2}
    
    def test(self,testloader=None):
        return 0.0,0.0