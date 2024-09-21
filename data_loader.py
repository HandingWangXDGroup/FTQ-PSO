import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset
import random
import os
import numpy as np
from glob import glob
from PIL import Image



def default_fn(file):
    img = Image.open(file).convert("RGB")
    return img

class ImageNet(Dataset):
    def __init__(self, data_folder, label_path='', transform=None, default_fn=default_fn):
        data = []
        classes = {}
        if label_path:
            with open(label_path, 'r') as fr:
                lines = fr.readlines()
                # lines = fr.read()
                labels = []
                for line in lines:
                    img_name, label = line.split()
                    data.append((os.path.join(data_folder, img_name), int(label)))
                    labels.append(int(label))
                # num_classes=len(set(labels))
                for i in range(len(set(labels))):
                    classes[i] = i
        else:
            data_list = []
            for pth in data_folder:
                if "COCO" in pth or 'VOC' in pth:
                    file_list = glob(os.path.join(pth, "*.jpg"))
                    random.shuffle(file_list)
                    file_list = file_list[:5000]
                    data_list.extend(file_list)
                else:
                    current_list = []
                    for subfolder in os.listdir(pth):
                        if "SUN397" in pth:
                            file_list = glob(os.path.join(pth, subfolder, "*.jpg"))
                            current_list.extend(file_list)
                        else:
                            file_list = glob(os.path.join(pth, subfolder, "*.JPEG"))
                            current_list.extend(file_list)
                    random.shuffle(current_list)
                    current_list = current_list[:5000]
                    data_list.extend(current_list)
            data = data_list
        self.data = data
        self.label_path = label_path
        self.classes = classes
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        if self.label_path:
            file, label = self.data[index]
        else:
            file = self.data[index]
            label = 1
        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class COCOVOCDataset(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            data.append((file_path, 1))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = self.default_fn(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label




def get_dataset(args, only_train = True):

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    if args.data_name == 'imagenet':
        if "linux" in args.system:
            path = '/mnt/jfs/wangdonghua/dataset/ImageNet/'
        elif "win" in args.system:
            path = 'D:/DataSource/ImageNet/'
        traindir = os.path.join(path, 'ImageNet10k')
        valdir = os.path.join(path, 'val')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if args.image_numbers < 10000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:args.image_numbers]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif args.data_name == 'coco':
        if "linux" in args.system:
            path = '/mnt/jfs/wangdonghua/dataset/COCO/train2014/'
        elif "win" in args.system:
            path = "D:/DataSource/COCO2014/train2014/"
        train_transform = transforms.Compose([
            transforms.Resize(int(args.image_size * 1.143)),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(path, train_transform)
        if args.image_numbers < 50000:
            random.seed(1024)
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:args.image_numbers]
            train_dataset = Subset(train_dataset, sample_indices)
    elif args.data_name == 'voc':
        if "linux" in args.system:
            path = '/mnt/jfs/wangdonghua/dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
        elif "win" in args.system:
            path = "D:/DataSource/VOCPASCAL/VOCtrainval/VOCdevkit/VOC2007/JPEGImages/"
        train_transform = transforms.Compose([
            transforms.Resize(int(args.image_size * 1.143)),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(path, train_transform)
        if args.image_numbers < 50000:
            random.seed(1024)
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:args.image_numbers]
            train_dataset = Subset(train_dataset, sample_indices)
    elif args.data_name == 'sun397':
        if "linux" in args.system:
            path = '/mnt/jfs/wangdonghua/dataset/transfer/SUN397/'
        elif "win" in args.system:
            path = 'D:/DataSource/ImageNet/'
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'test')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if args.image_numbers < 50000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:args.image_numbers]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif args.data_name == 'mixed':
        if "linux" in args.system:
            path = [
                    '/mnt/jfs/wangdonghua/dataset/transfer/SUN397/',
                    '/mnt/jfs/wangdonghua/dataset/COCO/train2014/',
                    '/mnt/jfs/wangdonghua/dataset/ImageNet/ImageNet10k/',
                    '/mnt/jfs/wangdonghua/dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
                    ]
        elif "win" in args.system:
            path = [r'D:\DataSource\ImageNet\train',
                    r'D:\DataSource\COCO2014\train2014',
                    r'D:\DataSource\VOCPASCAL\VOCtrainval\VOCdevkit\VOC2007\JPEGImages',
                    r'D:\DataSource\AttackTransfer\SUN397\train'
                    ]
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ])

        # dataset
        if only_train:
            train_dataset = ImageNet(data_folder=path, transform=train_transform)
            if args.image_numbers < 50000:
                random.seed(1024)
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:args.image_numbers]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            pass
    return train_dataset
