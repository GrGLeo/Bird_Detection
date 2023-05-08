from model import Model,  calc_metrics, batch_calc_metrics
from data_loader import data_load
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# Constant
EPOCH = 10
LEARNING_RATE = 1e-4

# Load data
train,val,test = data_load()

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # use the GPU
else:
    device = torch.device("cpu")   # use the CPU

# Move the model to the GPU
model.to(device)

# Early_Stopping
best_val_loss = float('inf')
epoch_since_improvement = 0
patience = 5

for epoch in range(10):
    # Progress bar
    progress_bar = tqdm(total=len(train),desc="Training",position=0)
    epoch_loss = 0
    epoch_acc = 0
    for i, data in enumerate(train, 0):
        # Time step
        start_time = time.time()
        # Get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero the parameters gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # Metrics
        batch_loss = loss.item()
        accuracy = batch_calc_metrics(outputs,labels)
        # Epoch metrics
        epoch_loss += batch_loss
        epoch_acc += accuracy
        # Time step
        end_time = time.time()
        step_time = end_time - start_time
        progress_bar.set_postfix({"loss":round(loss.item(),4),"acc":accuracy})
        progress_bar.update(1)
    progress_bar.close()
    
    # Calculate loss & val_loss
    with torch.no_grad():
        # train
        epoch_loss /= len(train)
        epoch_acc /= len(train)
        # val
        val_loss,val_acc = calc_metrics(model,criterion,val)

    # Early Stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_since_improvement = 0
    else:
        epoch_since_improvement+=1
    
    if epoch_since_improvement >=patience:
        print(f"Early Stopping after {epoch+1}")
        break
    
    print(f'Epoch: {epoch} -- loss:{round(epoch_loss,4)} -- acc:{round(epoch_acc,2)} -- val_loss:{round(val_loss,4)} -- val_acc:{round(val_acc,2)}')

# Save model
path_to_save = rf"D:\Coding\bird_detection\Bird_Detection\models\{str(time.time())[:5]}.pth"
torch.save(model.state_dict(),path_to_save)