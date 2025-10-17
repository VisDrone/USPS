from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import numpy as np
from randaugment import RandAugment
import random
import os
import sys
import pickle
import torch
import torchvision
import os
from copy import deepcopy


class TinyImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(TinyImageNet, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

def get_tiny_image_net_datasets(args, train_transform, test_transform, train_classes=range(20),
                       open_set_classes=range(20, 200), balance_open_set_eval=False, ):
    tin_train_root_dir = args.data_root+'/tiny-imagenet-200/train'  # TinyImageNet Train
    tin_val_root_dir = args.data_root+'/tiny-imagenet-200/val/images'
    # Init train dataset and subsample training classes
    train_dataset_whole = TinyImageNet(root=tin_train_root_dir, transform=train_transform)
    train_dataset_whole = subsample_classes_tinyimagenet(train_dataset_whole, include_classes=train_classes)


    # Get test set for known classes
    test_dataset_known = TinyImageNet(root=tin_val_root_dir, transform=test_transform)
    test_dataset_known = subsample_classes_tinyimagenet(test_dataset_known, include_classes=train_classes)



    train_dataset_unknown = TinyImageNet(root=tin_train_root_dir, transform=train_transform)
    train_dataset_unknown = subsample_classes_tinyimagenet(train_dataset_unknown, include_classes=open_set_classes)

    test_dataset_unknown = TinyImageNet(root=tin_val_root_dir, transform=test_transform)
    test_dataset_unknown = subsample_classes_tinyimagenet(test_dataset_unknown, include_classes=open_set_classes)



    all_datasets = {
        'train': train_dataset_whole,
        'test_known': test_dataset_known,
        'train_unknown': train_dataset_unknown,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_cifar_plus_datasets(args, train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10)):

    cifar_10_root = args.data_root

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes_cifar(train_dataset_whole, include_classes=train_classes)


    # Split into training and validation sets
    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    # val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes_cifar(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR100(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes_cifar(test_dataset_unknown, include_classes=open_set_classes)

    # Get trainset for unknown classes
    train_dataset_unknown = CustomCIFAR100(root=cifar_10_root, transform=test_transform, train=True)
    train_dataset_unknown = subsample_classes_cifar(train_dataset_unknown, include_classes=open_set_classes)

    # if balance_open_set_eval:
    #     test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)


    all_datasets = {
        'train': train_dataset_whole,
        'test_known': test_dataset_known,
        'train_unknown': train_dataset_unknown,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_cifar_10_100_datasets(args, train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10)):

    cifar_10_root = args.data_root
    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes_cifar(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    # val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes_cifar(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR100(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes_cifar(test_dataset_unknown, include_classes=open_set_classes)

    # Get trainset for unknown classes
    train_dataset_unknown = CustomCIFAR100(root=cifar_10_root, transform=test_transform, train=True)
    train_dataset_unknown = subsample_classes_cifar(train_dataset_unknown, include_classes=open_set_classes)

    # if balance_open_set_eval:
    #     test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    all_datasets = {
        'train': train_dataset_whole,
        'test_known': test_dataset_known,
        'train_unknown': train_dataset_unknown,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


def get_cifar_10_10_datasets(args, train_transform, test_transform, train_classes=range(4),
                       open_set_classes=range(4, 10)):





    cifar_10_root = args.data_root

    # Init train dataset and subsample training classes
    train_dataset_whole = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes_cifar(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    # val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes_cifar(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes_cifar(test_dataset_unknown, include_classes=open_set_classes)

    # Get trainset for unknown classes
    train_dataset_unknown = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=True)
    train_dataset_unknown = subsample_classes_cifar(train_dataset_unknown, include_classes=open_set_classes)

    # if balance_open_set_eval:
    #     test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    all_datasets = {
        'train': train_dataset_whole,
        'test_known': test_dataset_known,
        'train_unknown': train_dataset_unknown,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets
# ------------------------
# 预处理函数
# ------------------------

def get_transform(transform_type, image_size=32, args=None):

    if transform_type == 'pytorch-cifar':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'pytorch-tinyimagenet':

        mean = (0.43701944, 0.4077677, 0.36182693)
        std = (0.18368854, 0.18001619, 0.1800104)

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'rand-augment':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_transform.transforms.insert(0, RandAugment(1, 9, args=args))

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)

# ------------------------
# 工具函数
# ------------------------

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):

    if dataset in ('cifar-10-10', 'mnist', 'svhn'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset == 'cifar-10-100':
        train_classes = [x for x in range(10)]
        open_set_classes = [x for x in range(100)]

    elif dataset == 'cifar-100':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(100) if x not in train_classes]

    elif dataset == 'cifar+10':
        train_classes = [0,1,8,9]
        open_set_classes = osr_splits[dataset][split_idx]

    elif dataset == 'cifar+50':
        train_classes = [0,1,8,9]
        open_set_classes = osr_splits[dataset][split_idx]

    elif dataset in ('tinyimagenet'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(200) if x not in train_classes]


    else:

        raise NotImplementedError

    return train_classes, open_set_classes

def subsample_dataset_cifar(dataset, idxs):

    dataset.data = dataset.data[idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_classes_cifar(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset_cifar(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset



def subsample_dataset_tinyimagenet(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes_tinyimagenet(dataset, include_classes=range(20)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset_tinyimagenet(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split_tinyimagenet(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset_cifar(train_dataset, train_idxs)
    val_dataset = subsample_dataset_cifar(val_dataset, val_idxs)

    return train_dataset, val_dataset


def create_val_img_folder_tinyimagenet(root):
    '''
    This method is responsible for separating validation images into separate sub folders
    Run this before running TinyImageNet experiments

    :param root: Root dir for TinyImageNet, e.g /work/sagar/datasets/tinyimagenet/tiny-imagenet-200/
    '''
    dataset_dir = os.path.join(root)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

# ------------------------
# 字典
# ------------------------
get_dataset_funcs = {
    'cifar-10-100': get_cifar_10_100_datasets,
    'cifar-10-10': get_cifar_10_10_datasets,
    'cifar+10': get_cifar_plus_datasets,
    'cifar+50': get_cifar_plus_datasets,
    'tinyimagenet': get_tiny_image_net_datasets,

}


# ------------------------
# 用于外部调用
# ------------------------

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)



    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](args, train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes)
    else:
        raise NotImplementedError

    return datasets


osr_splits = {

    'cifar-10-10': [
        [0, 1, 2, 4, 5, 9],
        [ 0, 3, 5, 7, 8, 9],
        [ 0, 1, 5, 6, 7, 8],
        [ 3, 4, 5, 7, 8, 9],
        [ 0, 1, 2, 3, 7, 8]
    ],

    'cifar-10-100': [
        list(range(100)),
        list(range(100)),
        list(range(100)),
        list(range(100)),
        list(range(100))
    ],

'cifar+10': [
        [49, 74,  0, 50,  2, 31, 43, 83, 92, 77],
        [ 49, 74,  0, 50,  2, 31, 43, 83, 92, 77],
        [ 49, 74,  0, 50,  2, 31, 43, 83, 92, 77],
        [ 49, 74,  0, 50,  2, 31, 43, 83, 92, 77],
        [ 49, 74,  0, 50,  2, 31, 43, 83, 92, 77]
    ],

'cifar+50': [
        [9, 74,  0, 50,  2, 31, 43, 83, 92, 77,  1, 95, 32, 87,  5, 70, 73, 51,
        16, 39, 52, 11, 19, 71, 47, 38, 20, 94, 53, 82, 80, 68, 12, 37, 67, 86,
        57, 97, 98, 15, 33, 24, 27, 79, 18, 35, 61, 56, 84, 25],
        [9, 74,  0, 50,  2, 31, 43, 83, 92, 77,  1, 95, 32, 87,  5, 70, 73, 51,
        16, 39, 52, 11, 19, 71, 47, 38, 20, 94, 53, 82, 80, 68, 12, 37, 67, 86,
        57, 97, 98, 15, 33, 24, 27, 79, 18, 35, 61, 56, 84, 25],
        [ 9, 74,  0, 50,  2, 31, 43, 83, 92, 77,  1, 95, 32, 87,  5, 70, 73, 51,
        16, 39, 52, 11, 19, 71, 47, 38, 20, 94, 53, 82, 80, 68, 12, 37, 67, 86,
        57, 97, 98, 15, 33, 24, 27, 79, 18, 35, 61, 56, 84, 25],
        [9, 74,  0, 50,  2, 31, 43, 83, 92, 77,  1, 95, 32, 87,  5, 70, 73, 51,
        16, 39, 52, 11, 19, 71, 47, 38, 20, 94, 53, 82, 80, 68, 12, 37, 67, 86,
        57, 97, 98, 15, 33, 24, 27, 79, 18, 35, 61, 56, 84, 25],
        [ 9, 74,  0, 50,  2, 31, 43, 83, 92, 77,  1, 95, 32, 87,  5, 70, 73, 51,
        16, 39, 52, 11, 19, 71, 47, 38, 20, 94, 53, 82, 80, 68, 12, 37, 67, 86,
        57, 97, 98, 15, 33, 24, 27, 79, 18, 35, 61, 56, 84, 25]
    ],




'cifar-100': [
        [8, 13, 48, 58, 90, 41, 69, 81, 85, 89],

    ],



    'tinyimagenet': [
        [108, 147, 17, 58, 193, 123, 72, 144, 75, 167, 134, 14, 81, 171, 44, 197, 152, 66, 1, 133],
        [198, 161, 91, 59, 57, 134, 61, 184, 90, 35, 29, 23, 199, 38, 133, 19, 186, 18, 85, 67],
        [177, 0, 119, 26, 78, 80, 191, 46, 134, 92, 31, 152, 27, 60, 114, 50, 51, 133, 162, 93],
        [98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190],
        [95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170, 102, 177, 108, 169]
    ],
}


