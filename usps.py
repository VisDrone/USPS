#######################################   Description   #######################################################
####  Goal: This script aims to construct unknown prototypes and evaluate a open set model based on that.
####  Running: python3 usps.py --dataset cifar10 --split 3
####  Requirement: gan_gxx.pth, CIFAR10 samples(deep feature and label pairs), CIFAR10 class-semantic embeddings. See details in ./data/dataloader.py
####  Output: evaluation resutls.
#######################################   Description   #######################################################

######################  import standard library
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import sys
import numpy as np
import time
import torch.nn.functional as F
from sklearn.cluster import KMeans
import copy
import time
import sklearn
from torch.nn import DataParallel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from models import model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

######################  import self-constructed code file
from data import data_preprocess
import hyperparameters

######################  running command analysis
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--split', type=int, default=None, help='split id')
parser.add_argument('--gpu', type=int, default=0, help='split id')


opt = parser.parse_args()

class USPS:
    def __init__(self, opt):

        data = opt.data

        self.device = opt.device
        self.data = data
        self.nclass = data.seenclasses.size(0)
        self.input_dim = opt.resSize
        self.known_class_semantics = data.train_att.to(self.device)
        self.semantics_dim = self.known_class_semantics.shape[1]
        self.syn_feature_all = None
        self.syn_label_all = None
        
        self.adversarial_training_batchsize = opt.adversarial_training_batchsize
        self.adversarial_training_epoch = opt.adversarial_training_epoch
        self.epoch_detector_initial = opt.initialization_training_epoch
        self.lr = opt.lr
        self.lr_prototype = opt.lr_prototype
        self.milestone_prototype = opt.milestone_prototype

        self.incremental_num = opt.incremental_num
        self.epoch_prototype = opt.epoch_prototype
        self.prototype_num = 0
        self.prototype_per_incremental = opt.prototype_per_incremental
        self.samples_per_prototype = opt.samples_per_prototype

        self.prototype = None
        self.maxmargin = opt.maxmargin
        self.minmargin = opt.minmargin
        self.dissim = opt.dissim
        self.sim = opt.sim
        self.diversity_weight = opt.diversity_weight

        self.netG = model.MLP_G(opt)
        SAVE_PARA_PATH = './checkpoints/' + "gan_g" +str(opt.epoch_cgan)+ '.pth'
        self.netG.load_state_dict(torch.load(SAVE_PARA_PATH))
        self.netG.to(self.device)


        self.best_roc = 0
        self.best_oscr = 0
        self.best_f1 = 0


        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass + self.nclass * (self.prototype_per_incremental*(self.incremental_num-1))).to(self.device)
        self.model.apply(weights_init)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9,
                                         weight_decay=5e-4,
                                         nesterov=True)

        data.train_feature = data.train_feature.to(self.device)
        data.train_label = data.train_label.to(self.device)
        data.train_att = data.train_att.to(self.device)
        data.test_seen_feature = data.test_seen_feature.to(self.device)
        data.test_seen_label = data.test_seen_label.to(self.device)
        data.test_unseen_feature = data.test_unseen_feature.to(self.device)
        data.test_unseen_label = data.test_unseen_label.to(self.device)

        self.trainset = FeatDataset(data=[data.train_feature, data.train_label])
        self.traindataloader = DataLoader(self.trainset, batch_size=self.adversarial_training_batchsize, shuffle=True, num_workers=0)

        self.testset = FeatDataset(data=[data.test_seen_feature, data.test_seen_label])
        self.testdataloader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=0)

        self.openset = FeatDataset(data=[data.test_unseen_feature, data.test_unseen_label])
        self.opendataloader = DataLoader(self.openset, batch_size=128, shuffle=False, num_workers=0)

        # construct unknown prototypes and start adversarial training.
        self.fit()


    def fit(self):



        for iter in range(self.incremental_num):

            print("iter: %d,  boundary_nclass: %d" % (iter, self.prototype_num))

            if iter > 0:
                training_epoch = self.adversarial_training_epoch
            else:
                training_epoch = self.epoch_detector_initial
                training_epoch = 1

            for epoch in range(training_epoch):

                if iter > 0:

                    self.trainset = FeatDataset(data=[self.data.train_feature, self.data.train_label])
                    self.traindataloader = DataLoader(self.trainset, batch_size=self.adversarial_training_epoch, shuffle=True,
                                                      num_workers=0)

                train_correct = AverageMeter()
                for batch_idx, samples in enumerate(self.traindataloader):

                    inputv, labelv = samples
                    self.model.zero_grad()

                    output = self.model(inputv, features=True)[:,:self.prototype_num*self.nclass+self.nclass]
                    loss = self.criterion(output, labelv)

                    _, preds = torch.max(output, 1)
                    train_correct.update(preds.eq(labelv).sum().item(), len(inputv))

                    loss.backward()
                    self.optimizer.step()

                with torch.no_grad():
                    acc_seen, auc_score_av, auc_score_softmax, oscr = self.val(iter)



                print("epoch:%d, train_acc:%.4f, acc:%.4f, av_roc:%.4f, softmax_roc:%.4f, best_softmax_roc:%.4f,                     oscr:%.4f" % (
                    epoch, train_correct.avg, acc_seen, auc_score_av, auc_score_softmax, self.best_roc, oscr))

                print("\n")

            if iter == self.incremental_num-1:
                print("all finished")
                exit()



            # construct and optimize unknown prototypes
            self.update_prototypes = Variable(torch.rand([self.nclass, self.prototype_per_incremental, self.semantics_dim]).to(self.device))
            self.update_prototypes.requires_grad = True
            self.prototype_optimizer = torch.optim.SGD([self.update_prototypes], lr=self.lr_prototype, momentum=0, weight_decay=0,
                                                  nesterov=False)
            train_scheduler = optim.lr_scheduler.MultiStepLR(self.prototype_optimizer, milestones=self.milestone_prototype, gamma=0.1)

            for k in range(self.epoch_prototype):
                loss = self.prototype_loss()
                self.prototype_optimizer.zero_grad()
                loss.backward()
                self.prototype_optimizer.step()
                train_scheduler.step()

            self.generate_anti_feature()

            self.prototype_num = self.prototype_num + self.prototype_per_incremental

    def prototype_loss(self):

        minmargin_loss = 0
        maxmargin_loss = 0
        diversity_loss = 0

        maxmargin_pairs_inverse = 1/ (self.update_prototypes.shape[0]*self.update_prototypes.shape[1] )
        minmargin_pairs_inverse = 1/ ( self.update_prototypes.shape[0]*self.update_prototypes.shape[0]*self.update_prototypes.shape[1])
        diversity_pairs_inverse = 1/ ((self.update_prototypes.shape[1]*(self.update_prototypes.shape[1]-1)/2*self.update_prototypes.shape[0]))

        # margin loss
        for k in range(len(self.known_class_semantics)):
            dis = torch.abs(self.update_prototypes - self.known_class_semantics[k])
            value, index = torch.sort(dis, dim=2)
            max_value = value[:, :, -self.minmargin:]
            min_value = value[:, :, :(dis.shape[2] - self.maxmargin)]


            minmargin_loss = torch.sum(((max_value - self.dissim) < 0) * (self.dissim - max_value)) + minmargin_loss
            maxmargin_loss = torch.sum(((min_value[k]) > self.sim) * (min_value[k])) + maxmargin_loss

        # diversity loss
        for i in range(self.update_prototypes.shape[0]):
            for j in range(self.update_prototypes.shape[1]):

                z_u = self.update_prototypes[i]*(self.update_prototypes[i]<=1)*(self.update_prototypes[i]>=0)
                z_k = self.update_prototypes[i][j]*(self.update_prototypes[i][j]<=1)*(self.update_prototypes[i][j]>=0)
                diversity_loss = 1/torch.norm(z_u/torch.norm(z_u)-z_k/torch.norm(z_k))+ diversity_loss

        maxmargin_loss = maxmargin_loss*maxmargin_pairs_inverse
        minmargin_loss = minmargin_loss*minmargin_pairs_inverse
        diversity_loss = diversity_loss * diversity_pairs_inverse


        loss = maxmargin_loss+minmargin_loss+diversity_loss*self.diversity_weight

        return loss

    def generate_anti_feature(self):

        num = self.samples_per_prototype

        prototype_copy = None
        syn_feature_all = None
        syn_label_all = None

        for k in range(len(self.update_prototypes)):
            if prototype_copy == None:
                prototype_copy = self.update_prototypes[k]
            else:
                prototype_copy = torch.cat([prototype_copy, self.update_prototypes[k]])

        for k in range(len(prototype_copy)):
            prototype_buff = torch.ones(num, len(prototype_copy[k])).to(self.device)
            prototype_buff = prototype_buff * prototype_copy[k]
            syn_label = (torch.ones(len(prototype_buff)) * (self.nclass+self.nclass*self.prototype_num + k)).to(self.device).int()
            syn_noise = torch.FloatTensor(len(prototype_buff), opt.nz).to(self.device)

            syn_noise.normal_(0, 1)
            with torch.no_grad():
                output = self.netG(Variable(syn_noise), Variable(prototype_buff))

            if syn_feature_all == None:
                syn_feature_all = output
                syn_label_all = syn_label
            else:
                syn_feature_all = torch.cat([syn_feature_all, output])
                syn_label_all = torch.cat([syn_label_all, syn_label])

        syn_feature_all = syn_feature_all.cpu()
        syn_label_all = syn_label_all.cpu()


        self.data.train_feature = torch.cat((self.data.train_feature.cpu(), syn_feature_all), 0)
        self.data.train_label = torch.cat((self.data.train_label.cpu(), syn_label_all), 0)

        self.data.train_feature = self.data.train_feature.cuda()
        self.data.train_label = self.data.train_label.cuda()

    def val(self, iter):

        known_scores_av = []
        known_scores_softmax = []
        known_labels = []
        known_preds = []
        unknown_preds = []
        test_correct = AverageMeter()

        for batch_idx, samples in enumerate(self.testdataloader):

            inputv, labelv = samples

            output = torch.nn.functional.softmax(
                self.model(inputv, features=True)[:,:self.nclass + self.nclass * (self.prototype_per_incremental*iter)],
                dim=1)[:, :self.nclass]

            av_output = self.model(inputv, features=True)[:,
                        :self.nclass]

            known_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            known_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            known_labels.extend([labelv[i].cpu().numpy() for i in range(len(inputv))])

            _, preds = torch.max(av_output, 1)

            known_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

            test_correct.update(preds.eq(labelv).sum().item(), len(inputv))



        unknown_scores_av = []
        unknown_scores_softmax = []

        for batch_idx, samples in enumerate(self.opendataloader):

            inputv, labelv = samples

            output = torch.nn.functional.softmax(
                self.model(inputv, features=True)[:,:self.nclass + self.nclass * (self.prototype_per_incremental*iter)],
                dim=1)[:, :self.nclass]

            av_output = self.model(inputv, features=True)[:,
                        :self.nclass]

            unknown_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            unknown_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            _, preds = torch.max(av_output, 1)

            unknown_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

        distances = sorted(known_scores_softmax)
        self.threshold_list = [distances[int((i / 100) * len(distances))] for i in range(0, 100, 1)]


        auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
        auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)
        oscr = compute_oscr(np.array(known_scores_softmax), np.array(unknown_scores_softmax), np.array(known_preds), np.array(known_labels))



        return test_correct.avg, auc_score_av, auc_score_softmax, oscr,






class FeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data[0].shape[0]


    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x, features=False):
        if features:
            return self.fc(x)
        o = self.logic(self.fc(x))
        return o


def compute_roc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def compute_oscr(x1, x2, pred, labels):

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w


    return OSCR

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



if __name__ == '__main__':

    opt = hyperparameters.set_hyperparameters(opt)

    obj = USPS(opt)
    print("finished!")
    exit()