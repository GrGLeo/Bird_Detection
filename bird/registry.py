import torch
import os
import time
import mlflow

def save_model(model):
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(save_path, "models",str(time.time())[-6:])
    torch.save(model.state_dict(),save_path)

def log_runs(lr,loss,val_loss,acc,val_acc):
     with mlflow.start_run():
        mlflow.log_param("learning_rate", lr)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("val_acc", val_acc)