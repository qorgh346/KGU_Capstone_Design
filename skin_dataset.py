import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import json
import sys
import random

class SkinDataset(Dataset):
    def __init__(self,type,root,phase,transform):
        self.phase = phase
        self.root = root
        self.transforms = transform
        self.type = type
        self.labels = os.listdir(root)
        self.dataset = list()
        for label in self.labels:
            root_dir = os.path.join(root,label)
            [self.dataset.append(os.path.join(root_dir,file)) for file in os.listdir(root_dir)]
        # print(self.dataset)
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        filename = self.dataset[index]
        image = Image.open(filename)
        label = int(filename.split('\\')[1])
        image = self.transforms(image)
        label = torch.LongTensor([label])
        return image,label

    def __len__(self):
        if self.type in ['미세각질','탈모']:
            print(' : ', len(self.dataset))
            return len(self.dataset)

        return len(self.dataset)//2


if __name__ == '__main__':
    trainModel_order = ['피지과다', '탈모', '미세각질', '모낭홍반농포', '비듬', '모낭사이홍반']
    transforms_train = transforms.Compose([
        transforms.Resize([int(600), int(600)], interpolation=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda x: x.rotate(90)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_val = transforms.Compose([
        transforms.Resize([int(600), int(600)], interpolation=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train_data_list = [
    #     SkinDataset('../new_dataset/img_dataset/train_dataset/{}'.format(i),phase='train', transform=transforms_train)
    #     for i in trainModel_order]
    #

    dataset = SkinDataset('모낭사이홍반','../new_dataset/img_dataset/train_dataset/{}'.format('모낭사이홍반'), phase='val', transform=transforms_train)
    dataset.__getitem__(5)
    a = DataLoader(dataset,batch_size=4,shuffle=True)
    print(len(a))