from model import Model,  calc_metrics, batch_calc_metrics
from data_loader import data_load
import time
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# Constant
MODEL_PRETRAINED = True
MODEL_LOAD = False
EPOCH = 10
LEARNING_RATE = 1e-4
PATIENCE = 5

# Load data
train,val,test = data_load()

# Model instanciation
if MODEL_PRETRAINED:
    model = torchvision.models.efficientnet_v2_s(pretrained=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 25) 
else:    
    model = Model()
if MODEL_LOAD:
    model.load_state_dict(torch.load(r"D:\Coding\bird_detection\Bird_Detection\models\934584.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # use the GPU
else:
    device = torch.device("cpu")   # use the CPU

# Move the model to the GPU
model.to(device)


model.fit(train,val,EPOCH,learning_rate=LEARNING_RATE,patience=PATIENCE)

# Save model
path_to_save = rf"D:\Coding\bird_detection\Bird_Detection\models\{str(time.time())[-6:]}.pth"
torch.save(model.state_dict(),path_to_save)