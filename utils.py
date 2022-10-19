import torch.backends.cudnn as cudnn
import warnings,copy, random, time, glob, math, torch, argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms 
import matplotlib.pyplot as plt 
from PIL import Image 
from torch.utils.tensorboard import SummaryWriter
logdir = r'/kaggle/outputs/'
writer = SummaryWriter(log_dir=logdir, flush_secs=10)

### Visualizers
def PrintImagefromDataloader(iterator, i):
    dataiter = iter(iterator)
    images, labels = dataiter.next()
    plt.imshow(np.transpose(images[i].cpu().detach().numpy(), (1, 2, 0)))
    plt.show()
def GetImage(iterator):
#     warnings.filterwarnings("ignore", category=UserWarning) 
    samples = iter(iterator)
    data, label = samples.next()
    return data
def plot_graph(paramlist, picname):
    plt.figure(figsize=(12,3))
    plt.plot(paramlist)
    plt.title(label=f'Best Acc : {max(paramlist)*100}%')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(picname, dpi=300, bbox_inches='tight')

### load and save models
def load_from_checkpoint(PATH):
    checkpoint = torch.load(PATH)
    last_epoch = checkpoint['epoch']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    best_test_acc = checkpoint['acc']
    last_lr = checkpoint['last_lr']
    return model, optimizer, last_epoch, best_test_acc, last_lr
def save_to_checkpoint(epoch, model, scheduler, optimizer, test_acc, last_lr, PATH):
    torch.save({'epoch': epoch, 'model': model,'optimizer': optimizer,'acc': test_acc, 'last_lr': last_lr}, PATH)

### Metrics
def calc_acc_top5(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
def calc_acc(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
def cutmix_calc_acc(outputs, targets, data):
    _, preds = torch.max(outputs, dim=1)
    num = data.size(0)
    if isinstance(targets, (tuple, list)):
        targets1, targets2, lam = targets
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        acc = (lam * correct1 + (1 - lam) * correct2) / num
    else:
        correct_ = preds.eq(targets).sum().item()
        acc = correct_ / num  
    return acc

### Training
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs