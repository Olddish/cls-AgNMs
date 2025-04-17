import torch
import torch.nn as nn
from nn_train import DataProcess, MLP


def tpr_caculate(y_pred, y_true, num_classes=4):
    y_true = y_true.view(-1)
    y_pred = torch.argmax(y_pred, dim=1)
    true_positive = torch.zeros(4, dtype=torch.float32)
    total_positive = torch.zeros(4, dtype=torch.float32)
    
    for i in range(num_classes):
        true_positive[i] = ((y_pred == i) & (y_true == i)).sum().item()
        total_positive[i] = (y_true == i).sum().item()
    tpr = true_positive / total_positive
    
    return tpr


def ensemble_model(model1, model2, model3, model4, loader, weights):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    with torch.no_grad():
        for data, target in loader:
            output1 = model1(data)
            output2 = model2(data)
            output3 = model3(data)
            output4 = model4(data)
            combined_output = torch.concat((output1, output2, output3, output4), dim=1)
            weighted_output = torch.softmax(combined_output * weights, dim=1)
    return weighted_output, target


if __name__ == "__main__":
    path = "biyelunwen/形貌数据.xlsx"
    sheet_name = 'Sheet1'
    train_batch_size = 200
    test_batch_size = 100
    Y_column = 'multi_label'
    data_process = DataProcess(path, sheet_name)
    train_loader, test_loader = data_process.data_split(train_batch_size, test_batch_size, Y_column)

    AgNW_model = MLP()
    AgNW_model.load_state_dict(torch.load("biyelunwen/model_save/AgNW_model.pth"))
    AgNP_model = MLP()
    AgNP_model.load_state_dict(torch.load("biyelunwen/model_save/AgNP_model.pth"))
    LNP_model = MLP()
    LNP_model.load_state_dict(torch.load("biyelunwen/model_save/L-NP_model.pth"))
    Mixture_model = MLP()
    Mixture_model.load_state_dict(torch.load("biyelunwen/model_save/Mixture_model.pth"))

    weights = torch.tensor([1.1, 1.0, 0.8, 1.0], dtype=torch.float32) #AgNW,LNP,AgNP,Mixture
    train_output,train_labels = ensemble_model(AgNW_model, LNP_model, AgNP_model, Mixture_model, train_loader, weights)
    test_output,test_labels = ensemble_model(AgNW_model, LNP_model, AgNP_model, Mixture_model, test_loader, weights)
    all_output = torch.concat((train_output, test_output), dim=0)
    all_labels = torch.concat((train_labels, test_labels), dim=0)
    tpr = tpr_caculate(all_output, all_labels, num_classes=4)
    print("AgNW tpr: ", tpr[0].item())
    print("LNP tpr: ", tpr[1].item())
    print("AgNP tpr: ", tpr[2].item())
    print("Mixture tpr: ", tpr[3].item())
    


