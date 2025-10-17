#######################################   Description   #######################################################
####  Goal: This script aims to prepare dataset used in USPS.
####  Requirement: For cifar10, cifar+50, mnist, and svhn, it requires "./data/XXX_train_features_foldX.pth",
####               "./data/XXX_test_closed_features_foldX.pth", "./data/XXX_test_open_features_foldX.pth",
####               "./data/XXX_embedding_foldX.pth",
####               For CUB, AwA, aPaY, and SUN, it requires ./xlsa17/data which can be obtained from https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
####  Output:
#######################################   Description   #######################################################

######################  import standard library
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans
import os
import random



class DATA_PREPROCESS(object):
    def __init__(self, opt):


        if (opt.dataset == "cifar10") or (opt.dataset == "mnist") or (opt.dataset == "svhn"):
            self.read_cifar10_like(opt)
        if opt.dataset == "CUB":
            self.read_matdataset_CUB(opt)
        if opt.dataset == "AwA":
            self.read_matdataset_AwA(opt)
        if opt.dataset == "aPaY":
            self.read_matdataset_aPaY(opt)
        if opt.dataset == "SUN":
            self.read_matdataset_SUN(opt)

        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]


    def read_cifar10_like(self, opt):

        if (opt.dataset == "cifar10") :
            self.dataset = "cifar-10-10"
        if (opt.dataset == "mnist"):
            self.dataset = "mnist"
        if (opt.dataset == "svhn"):
            self.dataset = "svhn"

        ########################################   read dataset   ##########################################################


        SAVE_PARA_PATH = './backbone/results/Softmax_' + self.dataset + "_split" + str(
            opt.split) + '_train_known_set.pth'
        train_feature, train_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = './backbone/results/Softmax_' + self.dataset + "_split" + str(
            opt.split) + '_test_known_set.pth'
        test_feature, test_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = './backbone/results/Softmax_' + self.dataset + "_split" + str(
            opt.split) + '_open_set.pth'
        open_feature, _ = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = './backbone/results/Softmax_' + self.dataset + "_split" + str(
            opt.split) + '_word_embeddings.pth'
        att = torch.load(SAVE_PARA_PATH)


        self.split_unknown_class = torch.tensor([0, 1, 2, 3])
        self.split_known_class = torch.tensor([i for i in range(10) if i not in self.split_unknown_class])

        for kk in range(len(self.split_known_class)):
            self.split_known_class[kk] = self.split_known_class[kk] + 1


        self.nclass = 10
        self.split_known = self.split_known_class


        ########################################   preprocess & setting   ##########################################################

        self.attribute = (att-torch.min(att))/(torch.max(att)-torch.min(att))
        self.train_feature = train_feature
        self.train_label = train_label

        self.test_unseen_feature = open_feature
        self.test_unseen_label = torch.ones(open_feature.shape[0])*len(self.split_known_class)

        self.test_seen_feature = test_feature
        self.test_seen_label = test_label

        self.seenclasses = self.split_known_class
        self.unseenclasses = self.split_unknown_class

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = self.train_label
        self.train_att = self.attribute
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class

    def read_matdataset_SUN(self, opt):

        ########################################   read dataset & split   ##########################################################

        matcontent = sio.loadmat("./data/xlsa17/data/" + "SUN" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "SUN" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 717
        self.all_att_num = 102

        if opt.split == 0:
            split_known_class = list(range(50))#split 0
        if opt.split == 1:
            split_known_class = list(range(300,350)) #split 1
        if opt.split == 2:
            split_known_class = list(range(600,650))#split2

        self.split_unknown = [i for i in range(1, 717 + 1) if i not in split_known_class]

        index = torch.randperm(self.all_att_num)
        chosen_att = index[:self.all_att_num]


        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1

        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._image_id_label = {}

        awa_split_ratio = 4/5
        random.seed(6666)

        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i])*awa_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])


        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids
        ########################################   preprocess & setting   ##########################################################
        self.attribute = torch.from_numpy(matcontent['original_att'].T).float()[:,torch.as_tensor(chosen_att, dtype=torch.long)]
        self.attribute = (self.attribute-torch.min(self.attribute))/(torch.max(self.attribute)-torch.min(self.attribute))

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)
        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.tensor([self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = self.train_label
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class

    def read_matdataset_aPaY(self, opt):

        ########################################   read dataset & split  ##########################################################

        matcontent = sio.loadmat("./data/xlsa17/data/" + "APY" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "APY" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 32
        self.all_att_num = 64

        if opt.split == 0:
            split_known_class = [ 7,  9, 15, 16, 20,  2, 27, 26, 14, 25]#split 0
        if opt.split == 1:
            split_known_class = [2, 11, 7, 20, 9, 17, 31, 6, 13, 24] #split 1
        if opt.split == 2:
            split_known_class = [15, 22, 24, 28, 19, 26,  0, 31,  7,  1]#split2
        if opt.split == 3:
            split_known_class = [3, 8, 20, 27, 1, 28, 6, 25, 14, 7]#split3
        if opt.split == 4:
            split_known_class = [24, 14, 4, 27, 9, 0, 3, 2, 11, 26]  # split4



        self.split_unknown = [i for i in range(1, 32 + 1) if i not in split_known_class]

        index = torch.randperm(self.all_att_num)
        chosen_att = index[:self.all_att_num]


        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1



        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._image_id_label = {}

        awa_split_ratio = 4/5
        random.seed(6666)

        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i])*awa_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])


        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids

        ########################################   preprocess & setting   ##########################################################

        self.attribute = torch.from_numpy(matcontent['original_att'].T).float()[:,torch.as_tensor(chosen_att, dtype=torch.long)]
        self.attribute = (self.attribute-torch.min(self.attribute))/(torch.max(self.attribute)-torch.min(self.attribute))




        scaler = preprocessing.MinMaxScaler()


        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)
        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.tensor([self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())).shape)
        # exit(0)
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = self.train_label
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class


    def read_matdataset_AwA(self, opt):

        ########################################   read dataset & split  ##########################################################

        matcontent = sio.loadmat("./data/xlsa17/data/" + "AWA2" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "AWA2" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 50
        self.all_att_num = 85

        if opt.split == 0:
            split_known_class = [48, 35, 25, 8, 24, 33, 20, 7, 15, 12, 2, 21, 16, 5, 36, 3, 40, 43, 31, 23]  # split0
        if opt.split == 1:
            split_known_class = [15, 11, 34, 32, 20, 37, 38, 18, 39, 0, 45, 35, 41, 25, 31, 46, 5, 12, 30, 33]  # split1
        if opt.split == 2:
            split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split2
        if opt.split == 3:
            split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split3
        if opt.split == 4:
            split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split4



        self.split_unknown = [i for i in range(1, 50 + 1) if i not in split_known_class]

        index = torch.randperm(self.all_att_num)
        chosen_att = index[:self.all_att_num]

        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1

        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._image_id_label = {}

        awa_split_ratio = 4/5
        random.seed(6666)

        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i])*awa_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])


        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids

        ########################################   preprocess & setting   ##########################################################

        self.attribute = torch.from_numpy(matcontent['original_att'].T).float()[:,torch.as_tensor(chosen_att, dtype=torch.long)]
        self.attribute = (self.attribute-torch.min(self.attribute))/(torch.max(self.attribute)-torch.min(self.attribute))




        scaler = preprocessing.MinMaxScaler()


        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)
        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.tensor([self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())).shape)
        # exit(0)
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = self.train_label
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class


    def read_matdataset_CUB(self, opt):

        ########################################   read dataset & split  ##########################################################

        matcontent = sio.loadmat("./data/xlsa17/data/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")

        #################################################################################################################


        if opt.split == 4:
            split_known_class = [144, 73, 84, 182, 24, 29, 10, 48, 75, 1, 112, 19, 183, 135, 101, 95, 192, 163, 98, 14]#split4

        if opt.split == 3:
            split_known_class = [42, 99, 184, 121, 113, 17, 93, 157, 173, 192, 27, 35, 36, 103, 56, 145, 80, 85, 47, 89]#split3

        if opt.split == 2:
            split_known_class = [48, 49, 20, 114, 78, 157, 29, 59, 199, 177, 58, 76, 101, 168, 63, 53, 36, 6, 12, 182]#split2

        if opt.split == 1:
            split_known_class = [124, 55, 146, 112, 157, 29, 12, 63, 136, 186, 13, 94, 189, 47, 41, 182, 154, 200, 52, 106]#split1

        if opt.split == 0:
            split_known_class = [20, 166, 1, 101, 23, 84, 70, 57, 29, 102, 155, 97, 48, 53, 183, 110, 142, 106, 122, 15] #split0

        self.split_unknown = [i for i in range(1, 200 + 1) if i not in split_known_class]


        chosen_att =[228, 160, 148, 191, 125, 153,   4, 194, 298,  35,  19,  38, 214, 203,
         81, 219,  77, 293, 174,  12, 143, 152, 283, 234,  31, 142, 164, 123,
        245,  90, 193,  82, 118, 117, 115, 244, 201,  26, 252, 281,  43,  51,
        231, 155,  53, 292, 286, 141,  71, 129]




        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1


        self.root = "../data/CUB_200_2011/"
        self.nclass = 200
        self.split_known = split_known_class
        self.classes_file = os.path.join(self.root, 'classes.txt')  # <class_id> <class_name>
        self.att_file = os.path.join(self.root, 'attributes.txt')
        self.classes2att_file = os.path.join(self.root, 'class_attribute_labels_continuous.txt')
        self.image_class_labels_file = os.path.join(self.root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.image_attribute_labels_file = os.path.join(self.root, 'image_attribute_labels.txt')
        self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>

        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._image_id_label = {}

        self._train_test_split()
        self._get_id_to_label()
        self._open_split()

        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids - 1
        test_seen_loc = self._test_ids - 1
        test_unseen_loc = self._open_ids - 1


        ########################################   preprocess & setting   ##########################################################
        self.attribute = torch.from_numpy(matcontent['original_att'].T).float()[:,torch.tensor(chosen_att, dtype=torch.long)]/100

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)
        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.tensor([self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())))
        # print(torch.from_numpy(np.unique(self.train_label.numpy())).shape)
        # exit(0)
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = self.train_label
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]
        self.train_cls_num = self.ntrain_class
        self.test_cls_num  = self.ntest_class

        # print(self.train_att.shape)
        # exit(0)

    def _open_split(self):

        flag = 1
        flag2 = 0

        while(flag):
            for k in range(len(self._train_ids)):
                if int(self._image_id_label[self._train_ids[k]]) not in self.split_known:
                    self._train_ids.remove(self._train_ids[k])
                    flag2 = 1
                    break
            if flag2 == 0:
                flag = 0
            flag2 = 0



        flag = 1
        flag2 = 0

        while (flag):
            for k in range(len(self._test_ids)):
                if int(self._image_id_label[self._test_ids[k]]) not in self.split_known:

                    if int(self._image_id_label[self._test_ids[k]]) in self.split_unknown:
                        self._open_ids.append(self._test_ids[k])
                    self._test_ids.remove(self._test_ids[k])
                    flag2 = 1
                    break
            if flag2 == 0:
                flag = 0
            flag2 = 0



    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label