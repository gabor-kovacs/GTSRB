import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import  numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

from dataset import GTSRB, get_test, base_transforms
from model import Net

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = get_test()
    test_dataset = GTSRB(test, base_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(f"./out/model_best.pth"))
    model.eval()

    y_true = []
    y_pred = []

    correct = 0
    for data, labels in test_loader:
        data = data.to(device) 
        labels = labels.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] 

        y_pred.extend(pred.data.cpu().numpy())
        y_true.extend(labels.data.view_as(pred).cpu().numpy())
        
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    print(f"Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")


    # plot confusion matrix
    signnames = pd.read_csv(Path(Path(__file__).parent.absolute(), "signnames.csv"))
    classes = signnames["SignName"]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (36,21))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('./out/confusion_matrix.png')

if __name__ == "__main__":
  test()

