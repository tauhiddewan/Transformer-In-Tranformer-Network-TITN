# Loading the Dataset and Augmentation
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from cutmix_utils import  CutMixCollator, CutMixCriterion
class cifar10:
    def __init__(self, train_bs=1024, test_bs=1024, alpha = 1, using_cutmix = True, ROOT = './dataset'):
        self.collator = CutMixCollator(alpha=alpha) if using_cutmix else None
        self.train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128), 
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.train_data = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=self.train_transforms)
        self.test_data = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True, transform=self.test_transforms)
        self.trainloader = DataLoader(self.train_data, shuffle=True, batch_size = train_bs, num_workers=2, 
                                      collate_fn=self.collator, pin_memory=True)
        self.testloader = DataLoader(self.test_data, shuffle=False, batch_size = test_bs, num_workers=2, pin_memory=True)
    def returnloader(self):
        print(f'\nDataset = Cifar10. Number of training examples: {len(self.train_data)}, Number of testing examples: {len(self.test_data)}')
        print(f'Creating Mini-Batches. Number of Training Batch : {len(self.trainloader)}, Number of Testing  Batch : {len(self.testloader)}')
        return self.trainloader, self.testloader
class cifar100:
    def __init__(self, train_bs=1024, test_bs=1024, alpha = 1, using_cutmix = True, ROOT = './dataset'):
        self.collator = CutMixCollator(alpha=alpha) if using_cutmix else None
        self.train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128), 
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        self.train_data = torchvision.datasets.CIFAR100(root=ROOT, train=True, download=True, transform=self.train_transforms)
        self.test_data = torchvision.datasets.CIFAR100(root=ROOT, train=False, download=True, transform=self.test_transforms)
        self.trainloader = DataLoader(self.train_data,shuffle=True ,batch_size = train_bs, num_workers=2, 
                                      collate_fn=self.collator, pin_memory=True)
        self.testloader = DataLoader(self.test_data, shuffle=False, batch_size = test_bs, num_workers=2, pin_memory=True)
        
    def returnloader(self):
        print(f'\nDataset = Cifar100. Number of training examples: {len(self.train_data)}, Number of testing examples: {len(self.test_data)}')
        print(f'Creating Mini-Batches. Number of Training Batch : {len(self.trainloader)}, Number of Testing  Batch : {len(self.testloader)}')
        return self.trainloader, self.testloader
class mnist:
    def __init__(self, train_bs=1024, test_bs=1024, alpha = 1, using_cutmix = False, ROOT = './dataset'):
        self.collator = CutMixCollator(alpha=alpha) if using_cutmix else None
        self.train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.RandomCrop(32, padding=4, fill=128),
                                                    transforms.RandomRotation(30),
                                                    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
                                                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))])
        self.test_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307,), (0.3081,))])
        
        self.train_data = torchvision.datasets.MNIST(root=ROOT, train=True, download=True, transform=self.train_transforms)
        self.test_data = torchvision.datasets.MNIST(root=ROOT, train=False, download=True, transform=self.test_transforms)
        self.trainloader = DataLoader(self.train_data,shuffle=True ,batch_size = train_bs, 
                                      num_workers=2, collate_fn=self.collator, pin_memory=True)
        self.testloader = DataLoader(self.test_data, shuffle=False, batch_size = test_bs, num_workers=2, pin_memory=True)
    def returnloader(self):
        print(f'\nDataset = Mnist. Number of training examples: {len(self.train_data)}, Number of testing examples: {len(self.test_data)}')
        print(f'Creating Mini-Batches. Number of Training Batch : {len(self.trainloader)}, Number of Testing  Batch : {len(self.testloader)}')
        return self.trainloader, self.testloader
 