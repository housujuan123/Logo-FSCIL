import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
#from .autoaugment import AutoAugImageNetPolicy
from .autoaugment import CIFAR10Policy, Cutout

class MiniImageNet(Dataset):

    def __init__(self, root='/data/zhoudw/FSCIL', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None,autoaug=1):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'images')
        self.SPLIT_PATH = os.path.join(root, 'split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            addr, wnid = l.split(',')
            addr = addr.split('/')
            name = addr[1]+'/'+addr[2]
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        
        image_size = 64
        if autoaug==0:
            #do not use autoaug.
            if train:
                
                self.transform = transforms.Compose([
                    transforms.Resize(72),  #72
                    transforms.RandomCrop(image_size, padding=4),  #64
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),  #
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            #use autoaug.
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(72),  #
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),    # add AutoAug
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),  #
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, data2label, index_path):
        index=[]
        #print(data2label)
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[1]+'/'+line.split('/')[2])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            #print(img_path)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        #select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets,path


class MiniImageNet_concate(Dataset):
    def __init__(self, train,x1,y1,x2,y2):
        
        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            
        
        self.data=x1+x2
        self.targets=y1+y2
        print(len(self.data),len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets

if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '/data/zhoudw/FSCIL'
    batch_size_base = 400
    trainset = MiniImageNet(root=dataroot, train=True, transform=None, index_path=txt_path)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    print(trainloader.dataset.data.shape)
    