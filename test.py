import torch 
import numpy as np 
import pandas as pd 

def test_acc_Top1(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
        #correct += torch.eq(pred, y).sum().item()
    return correct / total
def test_acc_Top5(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total
def make_top1_top5_table(verbose=True, loader=testloader):
    hist, modeldict, modellist = [], {}, ['vit', 'distillvit', 'tnt', 'dtnt']
    df = pd.DataFrame(index =['Top1', 'Top5'], columns = modellist)  
    for i in range(len(modellist)):
        STUDENT_MODEL_PATH = f"/kaggle/input/models/{datasetname}-studentmodel_{modellist[i]}.pt"
        modeldict[modellist[i]], *_ = load_from_checkpoint(STUDENT_MODEL_PATH)
        top1 = test_acc_Top1(modeldict[modellist[i]], loader)*100
        top5 = test_acc_Top5(modeldict[modellist[i]], loader)*100
        modeldict[modellist[i]] = (top1, top5)
        log = f'Model Name :  {modellist[i]}, \tTop-1 Acc: {top1:.5f}, \tTop-5 Acc: {top5:.5f}'
        if verbose==True:
            print(log)
        hist.append(log)
    df = pd.DataFrame(modeldict, index=['top1', 'Top5'])
    return df, hist