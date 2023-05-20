import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision

from tqdm import tqdm
import time
import os


class Model(nn.Module):
    def __init__(self,device="cuda"):
        super(Model,self).__init__()
        self.device = device
        self.model = torchvision.models.efficientnet_v2_s(pretrained=True)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, 25) 
        self.to(device)
    
    def fit(self,train,val,epochs,learning_rate=1e-4,early_stop=False,patience=0):
        # Verify model is in training mode
        assert self.model.training, "Set model to train()"

        # Instanciate loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.parameters(),lr=learning_rate)

        # Early Stop parameter
        best_val_loss = float('inf')
        epoch_since_improvement = 0

        # Epoch loop
        self.model.train()
        for epoch in range(epochs):
            # Progress bar
            progress_bar = tqdm(total=len(train),desc="Training",position=0)
            epoch_loss = 0
            epoch_acc = 0 
            
            # Batch loop
            for inputs, labels in train:
                # Get the inputs
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Zero the parameters gradients
                self.optimizer.zero_grad()
                # Forward + Backward + Optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                # Metrics
                batch_loss = loss.item()
                accuracy = self.batch_calc_metrics(outputs,labels)
                # Epoch metrics
                epoch_loss += batch_loss
                epoch_acc += accuracy
                # Progress bar
                progress_bar.set_postfix({"loss":round(loss.item(),4),"acc":accuracy})
                progress_bar.update(1)
            progress_bar.close()
            
            # Calculate loss & val_loss
            self.model.eval()
            with torch.no_grad():
                # train
                epoch_loss /= len(train)
                epoch_acc /= len(train)
                # val
                val_loss, val_acc = self.calc_metrics(val)
            
            # Early Stopping logic
            if early_stop:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epoch_since_improvement = 0
                    else:
                        epoch_since_improvement+=1
                    
                    if epoch_since_improvement >=patience:
                        print(f"Early Stopping after {epoch+1}")
                        break

            print(f'Epoch: {epoch} -- loss:{round(epoch_loss,4)} -- acc:{round(epoch_acc,2)}\
                   -- val_loss:{round(val_loss,4)} -- val_acc:{round(val_acc,2)}')
            
        # Save model
        path_to_save = rf"D:\Coding\bird_detection\Bird_Detection\models\{str(time.time())[-6:]}.pth"
        torch.save(self.model.state_dict(),path_to_save)

    def calc_metrics(self,data):
        data_loss = 0
        correct = 0
        total = 0
        for i,batch in enumerate(data,0):
            inputs, labels = batch
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = self.model(inputs)
            batch_loss = self.criterion(outputs,labels)
            batch_loss = batch_loss.item()
            _, predicted = torch.max(outputs.data,1)
            data_loss += batch_loss
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = (correct/total) * 100
        loss = data_loss/(i+1)
        return loss,accuracy
    
    def batch_calc_metrics(self,outputs,labels):
            _,predicted = torch.max(outputs.data,1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = (correct/total) * 100
            return accuracy
    

    def prediction(self, X):
        self.model.eval()
        y_pred = self.model.forward(X)

        labels_name = os.listdir(r"D:\Coding\bird_detection\Bird_Detection\data\train")
        idx = torch.argmax(y_pred)
        probabilities = F.softmax(y_pred, dim=1)
        confidence = round(probabilities[0][idx].item()*100,2)

        labels = labels_name[torch.argmax(y_pred)]

        self.model.train()
        return labels, confidence
    
    
         