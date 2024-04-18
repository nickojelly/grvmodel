import torch
import os
import wandb
from rnn_tools.rnn_classes import Races

def model_saver(model, optimizer, epoch, loss, hidden_state_dict,train_state_dict, model_name = None):
    
    pathtofolder = "C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model"
    # model_name = wandb.run.name
    if not model_name:
        model_name = "test NZ GRU saver"
    isExist = os.path.exists(
        f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/"
    )
    if isExist:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
                "db_train":train_state_dict,
            },
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{epoch}.pt",
        )
    else:
        print("created path")
        os.makedirs(
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
                "db_train":train_state_dict,
            },
            f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{epoch}.pt",
        )

def model_saver_linux(model, optimizer, epoch, loss, hidden_state_dict,train_state_dict, model_name = None):
    
    pathtofolder = "C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model"
    # model_name = wandb.run.name
    if not model_name:
        model_name = "nz_model"
    isExist = os.path.exists(
        f"models/"
    )
    if isExist:
        pass
    else:
        print("created path")
        os.makedirs(
            f"models/{model_name}/"
        )
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
                "db_train":train_state_dict,
            },
            f"models/{model_name}/{model_name}_{epoch}.pt",
        )

def model_saver_wandb(model, optimizer, epoch, loss, hidden_state_dict,train_state_dict, model_name = None):
    
    pathtofolder = "C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model"
    model_name = wandb.run.name
    if not model_name:
        model_name = "test NZ GRU saver"
    isExist = os.path.exists(
        f"models/savedmodel/{model_name}/"
    )
    if isExist:
        pass
    else:
        print("created path")
        os.makedirs(
            f"models/savedmodel/{model_name}/"
        )
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim": optimizer.state_dict(),
                "loss": loss,
                "db":hidden_state_dict,
                "db_train":train_state_dict,
            },
            f"models/savedmodel/{model_name}/{model_name}_{epoch}.pt",
        )

def model_saver_second(model, optimizer, epoch, loss, hidden_state_dict,train_state_dict,raceDB:Races, model_name = None,path="second_models"):
    
    pathtofolder = "second_models/"
    model_name = wandb.run.name
    if not model_name:
        model_name = "test NZ GRU saver"
    isExist = os.path.exists(
        f"models/{path}/{model_name}/"
    )
    if not isExist:
        print("created path")
        os.makedirs(
            f"models/{path}/{model_name}/"
        )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss,
            "db":hidden_state_dict,
            "db_train":train_state_dict,
            'hidden_ins': raceDB.hidden_ins_dict,
            'output': raceDB.output_dict,
        },
        f"models/{path}/{model_name}/{model_name}_{epoch}.pt",
    )

def model_saver_second_profit(model,profit_model, optimizer, epoch, loss, hidden_state_dict,train_state_dict,raceDB:Races, model_name = None,path="second_models"):
    
    pathtofolder = "models/second_models/"
    model_name = wandb.run.name
    if not model_name:
        model_name = "test NZ GRU saver"
    isExist = os.path.exists(
        f"models/{path}/{model_name}/"
    )
    if not isExist:
        print("created path")
        os.makedirs(
            f"models/{path}/{model_name}/"
        )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            'profit_model': profit_model.state_dict(),
            "optim": optimizer.state_dict(),
            "loss": loss,
            "db":hidden_state_dict,
            "db_train":train_state_dict,
            # 'hidden_ins': raceDB.hidden_ins_dict,
            # 'output': raceDB.output_dict,
        },
        f"models/{path}/{model_name}/{model_name}_{epoch}.pt",
    )