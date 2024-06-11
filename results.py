from Network import CNN
import torch
import torchvision
from torchvision import transforms
from sklearn.metrics import classification_report,confusion_matrix
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import argparse

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-e","--epochs",type=int,default=5,help="Specify number of epochs for training")
    ap.add_argument("-d","--dataset",type=str,default=5,help="Specify MNIST or FashionMNIST dataset for training")
    ap.add_argument("-w","--weights",type=str,default=5,help="Specify best weights file path for testing")
    args = vars(ap.parse_args())

    n_epochs = args["epochs"]
    dataset = args["dataset"]
    weights_path = args["weights"]
    
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "MNIST":
        classes = ('0', '1', '2', '3', '4', 
           '5', '6', '7', '8', '9')
        
        testData = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
    else:
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        testData = torchvision.datasets.FashionMNIST(root="./data",train=False,download=True,transform=transform)
    
    
    
    testSubsetData = Subset(testData,indices=range(len(testData)//10))

    testDl = torch.utils.data.DataLoader(testSubsetData,batch_size=8,shuffle=False)

    Figure_name = dataset+"ConfustionMatrix.png"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN()

    weights = torch.load(weights_path)
    model.load_state_dict(weights)

    model.to(device)

    model.eval()

    predicted_values = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm.tqdm(testDl):
            images = images.to(device)
            res = model(images)

            _,res = torch.max(res,dim=1)
            predicted_values.extend(res.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        

    print(classification_report(all_labels,predicted_values))

    conf_matrix = confusion_matrix(all_labels,predicted_values)

    plt.figure(figsize=(14,14))
    sns.heatmap(conf_matrix,fmt="d",annot=True,xticklabels=classes,yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("GroundTruth")
    plt.title("Confusion matrix")
    plt.show()
    
    plt.savefig(Figure_name)