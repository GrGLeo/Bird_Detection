from model import Model,  AlexNet
from data_loader import data_load
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# Load data
train,val,test = data_load()
val_inputs, val_labels = [], []
for inputs, labels in val:
    val_inputs.append(inputs)
    val_labels.append(labels)
val_inputs = torch.cat(val_inputs)
val_labels = torch.cat(val_labels) 

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=1e-4)

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
        # Time step
        end_time = time.time()
        step_time = end_time - start_time
        progress_bar.set_postfix({"loss":round(loss.item(),4),"step time":step_time})
        progress_bar.update(1)
    progress_bar.close()
    
    with torch.no_grad():
        val_inputs.to(device)
        val_labels.to(device)
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
    
    print(f'Epoch: {epoch} -- loss:{round(loss.item(),4)} val_loss:{round(val_loss,4)}')

