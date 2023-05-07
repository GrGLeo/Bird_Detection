from model import Model,  AlexNet
from data_loader import data_load
import time

import torch
import torch.nn as nn
import torch.optim as optim

train,val,test = data_load()
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=1e-4)

# Early_Stopping
best_val_loss = float('inf')
epoch_since_improvement = 0
patience = 5

for epoch in range(10):
    for i, data in enumerate(train, 0):
        # Time step
        start_time = time.time()
        # Get the inputs
        inputs, labels = data
        # Zero the parameters gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        # Time step
        end_time = time.time()
        step_time = end_time - start_time
        print(f'Step {i}/{len(train)} -- loss:{round(train_loss,4)} -- step time: {step_time}')
    
    with torch.no_grad():
        inputs, labels = val
        y_val_pred = model(inputs)
        val_loss = criterion(y_val_pred, labels).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_since_improvement = 0
    else:
        epoch_since_improvement+=1
    
    if epoch_since_improvement >=patience:
        print(f"Early Stopping after {epoch+1}")
        break
    
    print(f'Epoch: {epoch} -- loss:{round(train_loss,4)} val_loss:{round(val_loss,4)}')

