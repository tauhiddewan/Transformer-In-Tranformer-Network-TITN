import torch
import argparse
import torch.nn as nn
from models import VIT, distillVIT, TNT, DTNT
from IPython.display import FileLink
from zipfile import ZipFile
from pathlib import Path
from torchsummary import summary
from datasets import mnist
torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_bs")
    parser.add_argument("test_bs")
    args = parser.parse_args()
    using_cutmix = False
    train_bs, test_bs = int(args.train_bs), int(args.test_bs)
    trainloader, testloader = mnist(train_bs=train_bs, test_bs=test_bs).returnloader()
    train_criterion = CutMixCriterion(reduction='mean') if using_cutmix  else nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss()
    S_EPOCHS = 10
    s_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(s_optimizer, T_max=S_EPOCHS, eta_min=eta_min)
    s_hist, logs = train_loop(last_epoch=last_epoch, epochs=S_EPOCHS, best_test_acc=best_test_acc, model=studentmodel, 
                        trainiterator=trainloader, testiterator=testloader, traincriteron=train_criterion, 
                        testcriteron=test_criterion, optimizer=s_optimizer, scheduler=s_scheduler, device=device, 
                        SAVEPATH=SAVEPATH, teachertrain=teachertrain)
    # print(len(trainloader))
    