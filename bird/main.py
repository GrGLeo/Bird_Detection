from model import Model,  calc_accuracy
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
val_inputs, val_labels = [], []
train_inputs, train_labels = [], []
for inputs, labels in train:
    train_inputs.append(inputs)
    train_labels.append(labels)
for inputs, labels in val:
    val_inputs.append(inputs)
    val_labels.append(labels)
train_inputs = torch.cat(train_inputs)
train_labels = torch.cat(train_labels)
val_inputs = torch.cat(val_inputs)
val_labels = torch.cat(val_labels) 

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # use the GPU
else:
    device = torch.device("cpu")   # use the CPU

# Early_Stopping
best_val_loss = float('inf')
epoch_since_improvement = 0
patience = 5

for epoch in range(1):
    # Move the model to the GPU
    model.to(device)
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
        # Metrics
        batch_loss = loss.item()
        accuracy = calc_accuracy(model,inputs,labels)
        # Time step
        end_time = time.time()
        step_time = end_time - start_time
        progress_bar.set_postfix({"loss":round(loss.item(),4),"acc":accuracy})
        progress_bar.update(1)
    progress_bar.close()
    
    # Calculate loss & val_loss
    model.to("CPU")
    with torch.no_grad():
        # loss
        y_train_pred = model(train_inputs)
        loss = criterion(y_train_pred, train_labels).item()
        acc = calc_accuracy(model,train_inputs,train_labels)
        # val_loss
        y_val_pred = model(val_inputs)
        val_loss = criterion(y_val_pred, val_labels).item()
        val_acc = calc_accuracy(model, val_inputs, val_labels)

    # Early Stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_since_improvement = 0
    else:
        epoch_since_improvement+=1
    
    if epoch_since_improvement >=patience:
        print(f"Early Stopping after {epoch+1}")
        break
    
    print(f'Epoch: {epoch} -- loss:{round(loss,4)} -- acc:{round(acc,2)} -- val_loss:{round(val_loss,4)} -- val_acc:{round(val_acc,2)}')

# Save model
path_to_save = rf"D:\Coding\bird_detection\Bird_Detection\models\{str(time.time())[:5]}.pth"
torch.save(model.state_dict(),path_to_save)