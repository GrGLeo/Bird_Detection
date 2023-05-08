import torch.nn as nn
import torch


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

def calc_metrics(model,loss_fc,data):
    data_loss = 0
    correct = 0
    total = 0
    for i,batch in enumerate(data,0):
        inputs, labels = batch
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        outputs = model(inputs)
        batch_loss = loss_fc(outputs,labels)
        batch_loss = batch_loss.item()
        _, predicted = torch.max(outputs.data,1)
        data_loss += batch_loss
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = (correct/total) * 100
    loss = data_loss/(i+1)
    return loss,accuracy
    
def batch_calc_metrics(outputs,labels):
        _,predicted = torch.max(outputs.data,1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = (correct/total) * 100
        return accuracy