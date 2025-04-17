import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class  DataProcess:
    def __init__(self,path,sheet):
        self.df = pd.read_excel(path,sheet_name=sheet)
        self.normlized_df = self.df.iloc[:,:4]
        for feature_name in self.normlized_df.columns:
            max_value = self.normlized_df[feature_name].max()
            min_value = self.normlized_df[feature_name].min()
            self.normlized_df[feature_name] = (self.normlized_df[feature_name] - min_value) / (max_value - min_value)

    def data_split(self,train_batch_size,test_batch_size,Y_column:str):
        x = self.normlized_df.iloc[:,:4]
        y = self.df.loc[:,Y_column]
        data = pd.concat([x,y],axis=1)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=2025, stratify=data[Y_column])
        x_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
        y_train = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32)
        x_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
        y_test = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float32)
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)
        return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(30)
        
    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def acc_count(y_pred, y_true):
    y_pred = torch.round(y_pred)
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    return accuracy

def train_loop(model, train_loader, lr, epoch_num):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_acc = []
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            target = target.unsqueeze(1)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            acc = acc_count(output, target)
            epoch_acc.append(acc)
        print(f'Epoch {epoch+1}, Loss: {loss.item():.3f}, train_ccuracy: {np.mean(epoch_acc):.3f}')
        epoch_acc.clear()
    return model

def test_loop(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.unsqueeze(1)
            output = model(data)
            acc = acc_count(output, target)
    return acc

if __name__ == "__main__":
    path = "biyelunwen/形貌数据.xlsx"
    sheet_name = 'Sheet1'
    train_batch_size = 32
    test_batch_size = 100
    lr = 0.01
    epoch_num = 50
    Y_column = 'Mixture'
    data_process = DataProcess(path, sheet_name)
    train_loader, test_loader = data_process.data_split(train_batch_size, test_batch_size, Y_column)
    model = MLP()
    trained_model = train_loop(model, train_loader, lr, epoch_num)
    accuracy = test_loop(trained_model, test_loader)
    print(f'test accuracy: {accuracy:.3f}')
    # torch.save(trained_model.state_dict(), "biyelunwen/model_save/Mixture_model.pth")

