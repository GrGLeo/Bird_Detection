import torch.nn as nn
import torch.optim as optim
import torch

from tqdm import tqdm
import time
import os


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), 
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,25)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x
    
    def fit(self,train,val,epochs,learning_rate=1e-4,early_stop=False,patience=0,device="cuda"):
        # Verify model is in training mode
        assert self.training, "Set model to train()"

        # Instanciate loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.parameters(),lr=learning_rate)

        # Early Stop parameter
        best_val_loss = float('inf')
        epoch_since_improvement = 0

        # Epoch loop
        self.train()
        for epoch in range(epochs):
            # Progress bar
            progress_bar = tqdm(total=len(train),desc="Training",position=0)
            epoch_loss = 0
            epoch_acc = 0 
            
            # Batch loop
            for i, data in enumerate(train, 0):
                # Get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the parameters gradients
                self.optimizer.zero_grad()
                # Forward + Backward + Optimize
                outputs = self(inputs)
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
            self.eval()
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
        torch.save(self.state_dict(),path_to_save)

    def calc_metrics(self,data):
        data_loss = 0
        correct = 0
        total = 0
        for i,batch in enumerate(data,0):
            inputs, labels = batch
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = self(inputs)
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
        self.eval()
        y_pred = self.forward(X)

        labels_name = os.listdir(r"D:\Coding\bird_detection\Bird_Detection\data\train")

        labels = labels_name[torch.argmax(y_pred)]

        self.train()
        return labels
    
         