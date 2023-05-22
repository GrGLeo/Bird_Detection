import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os


BATCH_SIZE = 32
def data_load():
    """
    Return the DataLoader object for the train,val and test datasets.
    """
    # Instanciate path
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
    train_path = os.path.join(data_path,"train")
    val_path = os.path.join(data_path,"val")
    test_path = os.path.join(data_path,"test")

    # Transform with data augmentation for train
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # Transform for val and test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load from folder
    train_dataset = ImageFolder(root=train_path, transform=train_transforms)
    val_dataset = ImageFolder(root=val_path, transform=test_transform)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    # DataLoader 
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return (trainloader,valloader,testloader)