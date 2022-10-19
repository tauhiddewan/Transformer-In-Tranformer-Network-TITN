from utils import *
from torch.utils.tensorboard import SummaryWriter


def teacher_train_one_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i, (x, y) in enumerate(iterator):
        x = x.to(device, non_blocking=True)
        if isinstance(y, (tuple, list)):
            y1, y2, lam = y
            y = (y1.to(device, non_blocking=True), y2.to(device, non_blocking=True), lam)
        else:
            y = y.to(device, non_blocking=True)
        optimizer.zero_grad() 
        y_pred = model(x) 
        loss = criterion(preds=y_pred, targets=y)
        loss.backward()
        optimizer.step()
        #Old - >acc = calculate_accuracy(y_pred, y)
        acc = cutmix_calc_acc(outputs=y_pred, targets=y, data=x) #review
        epoch_loss += loss.item()
        epoch_acc += acc #check  
    return epoch_loss / len(iterator), epoch_acc / len(iterator), optimizer.param_groups[0]["lr"]
def student_train_one_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i, (x, y) in enumerate(iterator):
        x = x.to(device, non_blocking=True)
        if isinstance(y, (tuple, list)):
            y1, y2, lam = y
            y = (y1.to(device, non_blocking=True), y2.to(device, non_blocking=True), lam)
        else:
            y = y.to(device, non_blocking=True)
        optimizer.zero_grad() 
        y_pred = model(x) 
#         loss = criterion(y_pred, y)
        loss = criterion(x, y_pred, y)
        loss.backward()
        optimizer.step()
        y_cls_pred, y_dist_pred = y_pred
        #Old - >acc = calculate_accuracy(y_pred, y)
        acc = cutmix_calc_acc(outputs=y_cls_pred, targets=y, data=x) #review
        epoch_loss += loss.item()
        epoch_acc += acc #check
    return epoch_loss / len(iterator), epoch_acc / len(iterator), optimizer.param_groups[0]["lr"]
def test_one_epoch(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred= model(x)
            loss = criterion(y_pred, y)
            acc = calc_acc(y_pred=y_pred, y=y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def train_loop(last_epoch, epochs, best_test_acc, model, trainiterator, testiterator, traincriteron, 
               testcriteron, optimizer, scheduler, device, SAVEPATH, teachertrain=False):
    history, logs = [[], [], [], []], []
    
    for epoch in range(last_epoch+1, last_epoch+epochs+1):
#         warnings.filterwarnings("ignore", category=UserWarning)
        start_time = time.monotonic()
        if teachertrain:
            train_loss, train_acc, last_lr = teacher_train_one_epoch(model=model, iterator=trainiterator, criterion=traincriteron, 
                                                    optimizer=optimizer, device=device)
        else:
            train_loss, train_acc, last_lr = student_train_one_epoch(model=model, iterator=trainiterator, criterion=traincriteron, 
                                                    optimizer=optimizer, device=device)
        test_loss, test_acc = test_one_epoch(model=model, iterator=testiterator, criterion=testcriteron, device=device)
        scheduler.step()
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        log = f'Epoch: {epoch:04d} ({epoch_mins}m {epoch_secs}s) | Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f} | Train Acc: {train_acc*100:.5f}, Test Acc: {test_acc*100:.5f}'
        print(log)
#         print(f'Epoch: {epoch:04d} ({epoch_mins}m {epoch_secs}s), lr = {optimizer.param_groups[0]["lr"]} | Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f} | Train Acc: {train_acc*100:.5f}, Test Acc: {test_acc*100:.5f}')
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Loss/Valid", test_loss, epoch)
        writer.add_scalar("Acc/Valid", test_acc, epoch)
        history[0].append(train_loss) 
        history[1].append(train_acc)
        history[2].append(test_loss)
        history[3].append(test_acc)
        logs.append(log)
        if test_acc>best_test_acc:
            best_test_acc = test_acc
            save_to_checkpoint(epoch=epoch, model=model, optimizer=optimizer,
                               scheduler=scheduler, test_acc=test_acc, PATH=SAVEPATH, last_lr=last_lr)
    return history, logs