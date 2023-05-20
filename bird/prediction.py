import torch
import torch.nn as nn
import torchvision
import os


def prediction(model, X):
    model.eval()
    y_pred = model.forward(X)

    labels_name = os.listdir(r"D:\Coding\bird_detection\Bird_Detection\data\train")

    label = labels_name[torch.argmax(y_pred)]

    return label

model = torchvision.models.efficientnet_v2_s(pretrained=True)
model = model.load_state_dict(torch.load(r"D:\Coding\bird_detection\Bird_Detection\models\934584.pth"))


