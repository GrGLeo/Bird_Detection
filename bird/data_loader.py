import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os


data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
train_path = os.path.join(data_path,"training")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root=train_path, transform=data_transforms)

