from renyi_vib import *
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='../input/data',
    train=False,
    download=True,
    transform=transform
)
print(f"train set size {len(train_data)}, validation set size {len(val_data)}")

train_dataloader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True)

vib = DeepVIB(784, 10, z_dim).to(device)
print("Device: ", device)
# Training
import time 
#from collections import defaultdict
start_time = time.time()
vib.fit(train_dataloader,val_dataloader)
# put Deep VIB into train mode 