import torch
with torch.profiler.profile() as profiler:
        pass

import pickle
import pandas as pd
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.profiler import profile
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
import pprint
import matplotlib.pyplot as plt
import wandb

CUDA_LAUNCH_BLOCKING=1

class GRV:
    # class to store training data

    # reading data from pickle
    file = open(r"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/DATA/total_list_w_price_bf.npy", "rb")
    data = pickle.load(file)
    # seperate out classes from inputs
    raceIDs, inputs, classes, prices, win_price, margins, betfairSP = zip(*data)
    # removing nan from inputs and convert to float
    inputs_df = pd.DataFrame(inputs)
    inputs_df.fillna(value=-1, inplace=True)
    inputs = inputs_df.values.tolist()
    inputs = [[float(i) for i in j] for j in inputs]

    # data
    training_data = []

    def make_training_data(self):
        excluded = 0
        for i in range(len(self.inputs)):
            if len(self.classes[i]) == 8:
                self.training_data.append(
                    [
                        np.array(self.inputs[i]),
                        np.array(self.classes[i]),
                        np.array(self.prices[i]),
                        np.array(self.margins[i]),
                        np.array(self.betfairSP[i]),
                    ]
                )
            else:
                adjustedList = self.classes[i] + ([8] * (8 - len(self.classes[i])))
                adjustedListP = self.prices[i] + ([0] * (8 - len(self.prices[i])))
                adjustedListM = self.margins[i] + ([100] * (8 - len(self.margins[i])))
                adjustedListSP = self.margins[i] + ([0] * (8 - len(self.betfairSP[i])))
                self.training_data.append(
                    [
                        np.array(self.inputs[i]),
                        np.array(adjustedList),
                        np.array(adjustedListP),
                        np.array(adjustedListM),
                        np.array(adjustedListSP)
                    ]
                )
                if len(adjustedList) != 8:
                    print(adjustedList)
        np.save("training_data.npy", self.training_data)
        print("excluded = ", excluded)


def custom_MSE(output, target):
    sorts = torch.argsort(target)
    out = sorts.narrow(1,0,3)
    ohe = torch.nn.functional.one_hot(out, num_classes=8).sum(1)
    out_first3 = ohe*output
    target_ohe = ohe*target
    loss = torch.mean((1+abs(out_first3-target_ohe))**2)
    # loss = torch.sum(((out_first3-target_ohe))**2)
    return loss
    

def make_loader(dataset, config, train=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config["validation_split"] * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    if train:
        dataset_sampler = SubsetRandomSampler(train_indices)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            sampler=dataset_sampler
        )
    else:
        dataset_sampler = SubsetRandomSampler(val_indices)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            sampler=dataset_sampler,
        )

    return loader


def build_network(f1_layer_size, f2_layer_size, dropout, num_layers=2):

    if num_layers == 2:
        network = nn.Sequential(  # fully-connected, dual hidden layer
            nn.Linear(120, f1_layer_size),
            nn.ReLU(),
            nn.Linear(f1_layer_size, f2_layer_size),
            nn.ReLU(),
            nn.Linear(f2_layer_size, 8),
            nn.Softmax(dim=1),
        )

    else:
        network = nn.Sequential(  # fully-connected, dual hidden layer
            nn.Linear(120, f1_layer_size),
            nn.ReLU(),
            nn.Linear(f1_layer_size, f2_layer_size),
            nn.ReLU(),
            nn.Linear(f2_layer_size, f2_layer_size),
            nn.ReLU(),
            nn.Linear(f2_layer_size, 8),
            nn.Softmax(dim=1),
        )

    return network

def make(config, dataset):
    # Make the data

    train_loader = make_loader(dataset, config, train=True)
    test_loader = make_loader(dataset, config, train=False)
    # Make the model
    # model = Net().to(device)
    model = build_network(
        config["f1_layer_size"], config["f2_layer_size"], config["dropout"], config["num_layers"]
    )

    loss_functions = {
        "Huber":nn.HuberLoss(),
        "MSE":nn.MSELoss(),
        "L1":nn.SmoothL1Loss(reduction='sum', beta=0.25),
        "BCE":nn.CrossEntropyLoss(),
        "Custom":custom_MSE,
        "KL":nn.KLDivLoss(reduction='batchmean')
    }
    # Make the loss and optimizer
    #  criterion = nn.NLLLoss()
    loss_f = loss_functions[config['loss']]
    criterion = loss_f
    optimizer = config["optimizer"]

    if optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config["learning_rate"], momentum=0.9
        )
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    # optimizer = torch.optim.Adam(
    #    model.parameters(), lr=config["learning_rate"])

    return model, train_loader, test_loader, criterion, optimizer

def test(model, test_loader, batch_ct):
    model.eval()
    classL, predL, maxL, correctL, priceP, priceR, bfPriceR, pred_odds, model_outputs = [], [], [], [], [], [], [], [], []
    # Run the model on some test examples
    with torch.no_grad():
        correct, total, max_sum, max_w_sum, profit, bfprofit, bfnotavail = 0, 0, 0, 0, 0,0,0
        value_pick_correct, value_pick_profit = 0, 0
        num_bets = 0
        for images, labels, prices, bfspPrices in test_loader:

            outputs = model(images)
            prices = prices[0,].tolist()
            bfspPrices = bfspPrices[0,].tolist()

            max, predicted = torch.max(outputs.data, 1)
            _, real = torch.max(labels.data, 1)

            prediction = predicted.item()
            real_item = real.item()

            predL.append(prediction)
            maxL.append(max.item())

            total += labels.size(0)
            correct += prediction == real_item

            correctL.append(int(prediction == real_item))
            classL.append(real_item)

            priceR.append(prices[real_item])
            priceP.append(prices[prediction])
            bfPriceR.append(bfspPrices[real_item])
            # print(outputs.data.flatten().tolist())

            predicted_odds = [
                1 / ((x + 10**-7)) for x in outputs.data.flatten().tolist()
            ]

            pred_odds.append(predicted_odds)
            model_outputs.append(outputs.data.flatten().tolist())

            if prices[real_item] > (predicted_odds[real_item] * 1.5):
                value_pick_correct += 1
                value_pick_profit += prices[real_item]

            bets = [x > (y * 1.5) for x, y in zip(prices, predicted_odds)]
            num_bets += sum(bets)

            value_pick_profit += -sum(bets)

            if prediction == real_item:
                max_sum += max
                profit += prices[real_item]
                if bfspPrices[real_item]:
                    bfprofit += bfspPrices[real_item]
                else:
                    bfprofit += prices[real_item]
                    bfnotavail += 1
            else:
                max_w_sum += max

            profit += -1
            bfprofit += -1

            # print(f"{correct=}")

        # print(f"Accuracy of the model on the {total} " +
        #       f"test images: {100 * correct / total}%" +
        #       f"profit: {profit}"+
        #       f"profit: {value_pick_profit}")

        # wandb.log(
        #     {
        #         "test_accuracy": correct / total,
        #         "correct_conf": max_sum / correct,
        #         "incorrect_conf": (max_w_sum) / (total - correct),
        #         "profit": profit,
        #         "bfprofit": bfprofit,
        #         "bfnotavail": bfnotavail,
        #         "value_pick_roi": value_pick_profit / num_bets,
        #         "num_bets_per": num_bets / total,
        #     }
        # )

        # logdf = pd.DataFrame(
        #     data={
        #         "class": classL,
        #         "pred": predL,
        #         "max": maxL,
        #         "correct": correctL,
        #         "priceR": priceR,
        #         "priceP": priceP,
        #         "bets": sum(bets),
        #         "pred_odds": pred_odds,
        #         "model_outputs": model_outputs,
        #         "bfodds" : bfPriceR
        #     }
        # )
        # table = wandb.Table(dataframe=logdf)
        # wandb.log({"table_key": table})
        # classCounts = logdf["class"].value_counts()
        # predCounts = logdf["pred"].value_counts()
        # boxplot = logdf.boxplot(column=['priceR'],by='correct')
        # print(classCounts, predCounts)
        # boxplot
        # plt.savefig("boxplot.png")
        # wandb.log({"boxplot":boxplot})

    # Save the model in the exchangeable ONNX format
    # pathtofolder = 'C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model'
    # model_name = wandb.run.name
    # isExist = os.path.exists(f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/")
    # if isExist:
    #     torch.save(model.state_dict(), f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{batch_ct}.pt")
    # else:
    #     print("created path")
    #     os.makedirs(f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/")
    #     torch.save(model.state_dict(), f"C:/Users/Nick/Documents/GitHub/grvmodel/Python/pytorch/New Model/savedmodel/{model_name}/{model_name}_{batch_ct}.pt")

def train(model, loader,test_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    #wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    print(f"{model.is_cuda=}")
    raw_inputs = True
    if config['loss'] == "KL":

        for epoch in tqdm(range(config.epochs)):
            for _, (images, labels, _ , _) in enumerate(loader):

                loss = train_batch_lsftmax(images, labels, model, optimizer, criterion, btch_count=batch_ct, raw_inputs=True)
                example_ct +=  len(images)
                batch_ct += 1

                # Report metrics every 25th batch
                if ((batch_ct + 1) % 250) == 0:
                    train_log(loss, example_ct, epoch)

            if epoch %10 ==0:
                test(model,test_loader, epoch)

    else:
        for epoch in tqdm(range(config.epochs)):
            for _, (images, labels, _ , _) in enumerate(loader):

                loss = train_batch(images, labels, model, optimizer, criterion, btch_count=batch_ct, raw_inputs=True)
                example_ct +=  len(images)
                batch_ct += 1

                # Report metrics every 25th batch
                if ((batch_ct + 1) % 250) == 0:
                    train_log(loss, example_ct, epoch)

            if epoch %10 ==0:
                test(model,test_loader, epoch)

def train_batch(images, labels, model, optimizer, criterion, btch_count=0, raw_inputs=True):
    images, labels = images, labels
    

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels.float())
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    pass
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    #print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def train_batch_lsftmax(images, labels, model, optimizer, criterion, btch_count=0, raw_inputs=True):
    images, labels = images, labels
    

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(F.log_softmax(outputs), labels.float())
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def model_pipeline(config=None):
    dataset = my_dataset
    # tell wandb to get started
    with wandb.init(project="debug", config=config):
      # access all HPs through wandb.config, so logging matches execution!
      wandb.define_metric("loss", summary="min")
      wandb.define_metric("test_accuracy", summary="max")
      wandb.define_metric("bfprofit", summary="max")
      config = wandb.config
      pprint.pprint(config)
      pprint.pprint(config.epochs)
      print(config)

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config, dataset)
      model = model.to(device)
      print(model)

      # and use them to train the model
      train(model, train_loader,test_loader, criterion, optimizer, config)

      # and test its final performance
      #test(model, test_loader)

    return model

def model_no_wandb(config=None):
    dataset = my_dataset
    model, train_loader, test_loader, criterion, optimizer = make(config, dataset)
    model = model.to(device)
    print(model)

    # and use them to train the model
    train(model, train_loader,test_loader, criterion, optimizer, config)

    # and test its final performance
    #test(model, test_loader)

    return model



def setup_data():
    global device, my_dataloader,my_dataset
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    if REBUILD_DATA:
        grv = GRV()
        grv.make_training_data()


    softmin = nn.Softmin(dim=1)

    # dataset setup
    training_data = grv.training_data

    X = torch.Tensor([i[0] for i in training_data])
    Y = torch.Tensor([i[1] for i in training_data])
    P = torch.Tensor([i[2] for i in training_data])
    Y_m = softmin(torch.Tensor([i[3] for i in training_data]))
    bfSP = torch.Tensor([i[4] for i in training_data])

    # Generate winner only class
    Y_w = []
    for i in Y:
        n = np.zeros(8)
        index = torch.argmin(i)
        n[index] = float(1)
        Y_w.append(n)

    Y_w = torch.tensor([i for i in Y_w])
    X = X.to(device)
    Y_w = Y_w.to(device)


    Y_m = Y_m.to(device)
    P = P.to(device)
    bfSP = bfSP.to(device)
    bfSP = torch.nan_to_num(bfSP, nan=0)
    my_dataset = TensorDataset(X, Y_m, P, bfSP)
    my_dataloader = DataLoader(my_dataset)
    return my_dataloader,my_dataset

    
REBUILD_DATA = True

if __name__=='__main__':
    with torch.profiler.profile() as profiler:
        pass
    print('hello')
    os.chdir(r'C:\Users\Nick\Documents\GitHub\grvmodel\Python\pytorch\New Model')
    setup_data()
    wandb.login()

    normal_config = {'batch_size': 32, 'dropout': 0.3, 'epochs': 1, 'f1_layer_size': 64, 'f2_layer_size': 64, 'learning_rate': 0.00047085841644517234, 'loss': 'L1', 'num_layers': 3, 'optimizer': 'adam', 'validation_split': 0.1}
    model = model_no_wandb(config=normal_config)

    # with torch.autograd.profiler.profile(with_stack=True, profile_memory=True) as prof:

    #     normal_config = {'batch_size': 32, 'dropout': 0.3, 'epochs': 10, 'f1_layer_size': 64, 'f2_layer_size': 64, 'learning_rate': 0.00047085841644517234, 'loss': 'L1', 'num_layers': 3, 'optimizer': 'adam', 'validation_split': 0.1}


    #     model = model_no_wandb(config=normal_config)
    #     print('finished')
    # print('outside')
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

    