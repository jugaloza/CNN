import torch.utils
import torch.utils.data
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from Network import CNN
import argparse
from torch.utils.data import Subset

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-e","--epochs",type=int,default=5,help="Specify number of epochs for training")
    ap.add_argument("-d","--dataset",type=str,default=5,help="Specify MNIST or FashionMNIST dataset for training")
    args = vars(ap.parse_args())

    n_epochs = args["epochs"]
    dataset = args["dataset"]

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "MNIST":
        trainData = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    else:
        trainData = torchvision.datasets.FashionMNIST(root="./data",train=True,download=True,transform=transform)
    
    subset_train = Subset(trainData, indices=range(len(trainData)//30))
    
    trainDl = torch.utils.data.DataLoader(subset_train,batch_size=8,shuffle=True)
    
    model = CNN()

    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Total Parameters : ",total_params)

    for epoch in tqdm.tqdm(range(n_epochs)):
        model.train()
        train_loss = 0

        for x,y in trainDl:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            res = model(x)

            loss = loss_fn(res,y)

            train_loss += loss.item()
            loss.backward()

            optim.step()

        print(f"Training Loss at epoch {epoch + 1} is {train_loss/len(trainDl)}")

        if ((epoch+1) % 5 == 0):
            torch.save(model.state_dict(),f"weights/epoch_{epoch+1}.pt")

    
