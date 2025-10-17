#######################################   Description   #######################################################
####  Goal: This script aims to obtain a GAN model which maps Z space to X space.
####  Running: python3 cgan.py --dataset cifar10 --split 3
####  Requirement: CIFAR10 samples(deep feature and label pairs), CIFAR10 class-semantic embeddings. See details in ./data/dataloader.py
####  Output: gan_gxx.pth, xx denotes the training epoch.
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
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
import matplotlib.pyplot as plt

######################  import self-constructed code file
from models import model
from data import data_preprocess



######################  running command analysis
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--batchsize_cgan', type=int, default=None, help='batch size')
parser.add_argument('--resSize', type=int, default=None, help='dimension of deep features')
parser.add_argument('--attSize', type=int, default=None, help='dimension of semantic features')
parser.add_argument('--nz', type=int, default=None, help='dimension of the latent z vector in GAN')
parser.add_argument('--ngh', type=int, default=None, help='dimension of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=None, help='dimension of the hidden units in discriminator')
parser.add_argument('--epoch_cgan', type=int, default=None, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=None, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=None, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=None, help='weight of the classification loss')
parser.add_argument('--lr_cgan', type=float, default=None, help='learning rate to train GAN ')
parser.add_argument('--beta1', type=float, default=None, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
parser.add_argument('--check', type=bool, default=None, help='read checkpoint')
parser.add_argument('--split', type=int, default=None, help='dataset split id')
parser.add_argument('--device', type=int, default=0, help='gpu id')

opt  = parser.parse_args()

print(opt)

model_path = "./checkpoints"
if not os.path.exists(model_path):
    os.makedirs(model_path)


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.size())

    alpha = alpha.to(opt.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(opt.device)

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    ones = ones.to(opt.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

class CLASSIFIER:
    # train_Y is interger
    def __init__(self):

        global opt
        global data

        # self.train_X = data.train_feature
        # self.train_Y = data.train_label

        self.train_X = data.train_feature.to(opt.device)
        self.train_Y = data.train_label.to(opt.device)

        self.test_X = data.test_seen_feature
        self.test_seen_label = data.test_seen_label
        self.test_unseen_X = data.test_unseen_feature
        self.test_unseen_label = data.test_unseen_label

        self.batch_size = opt.batchsize_classifier
        # _batch_size = 2048
        self.epoch = opt.epoch_classifier
        self.nclass = data.seenclasses.size(0)
        self.input_dim =opt.resSize
        self.beta1 = opt.adam_beta1_classifier
        self.device = opt.device
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.lr = opt.lr_classifier
        self.beta1 = opt.adam_beta1_classifier

        self.criterion = nn.NLLLoss()



        print('    classifier params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))
        self.model = self.model.to(self.device)
        self.criterion.to(self.device)
        self.model.apply(model.weights_init)
        # setup optimizer
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9,
                                         weight_decay=5e-4,
                                         nesterov=True)

        self.trainset = FeatDataset(data=[self.train_X, self.train_Y])
        self.traindataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.testset = FeatDataset(data=[self.test_X, self.test_seen_label])
        self.testdataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.openset = FeatDataset(data=[self.test_unseen_X, self.test_unseen_label])
        self.opendataloader = DataLoader(self.openset, batch_size=self.batch_size, shuffle=False, num_workers=0)


        print("fitting...")

        # for epoch in range(100):
        for epoch in range(self.epoch):

            for batch_idx, samples in enumerate(self.traindataloader):


                inputv, labelv = samples
                inputv = inputv.to(self.device)
                labelv = labelv.to(self.device)
                self.model.zero_grad()
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                acc, auc_score_av, auc_score_softmax = self.val()
                print(("epoch:%d,    ACC:%f,     AVROC:%f,     ROC:%f") % (epoch, acc, auc_score_av, auc_score_softmax))


    # test_label is integer
    def val(self):

        known_scores_av = []
        known_scores_softmax = []

        accuracy = AverageMeter()

        for batch_idx, data in enumerate(self.testdataloader):

            inputv, labelv = data
            inputv = inputv.to(self.device)
            labelv = labelv.to(self.device)


            av_output = self.model(inputv, features=True)

            output = torch.nn.functional.softmax(av_output,
                                                 dim=1)

            _, preds = torch.max(output, 1)

            known_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            known_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            accuracy.update(torch.sum(preds == labelv).item(), len(inputv))


        unknown_scores_av = []
        unknown_scores_softmax = []

        for batch_idx, data in enumerate(self.opendataloader):

            inputv, labelv = data
            inputv = inputv.to(self.device)
            labelv = labelv.to(self.device)


            av_output = self.model(inputv, features=True)

            output = torch.nn.functional.softmax(av_output,
                                                 dim=1)

            _, preds = torch.max(output, 1)

            unknown_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            unknown_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])


        auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
        auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)
        return accuracy.avg, auc_score_av, auc_score_softmax


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


class FeatDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data[0].shape[0]


    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

class FeatDataset2(Dataset):
    def __init__(self, data):
        self.data = data
        self.current_set_len = data[0].shape[0]

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx], self.data[2][self.data[1][idx]]



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

def train_cgan(opt):

    netG = model.MLP_G(opt)
    if opt.check == True:
        SAVE_PARA_PATH = './checkpoints/' + "gan_g" + '.pth'
        netG.load_state_dict(torch.load(SAVE_PARA_PATH))

    netD = model.MLP_CRITIC(opt)
    if opt.check == True:
        SAVE_PARA_PATH = './checkpoints/' + "gan_d" + '.pth'
        netD.load_state_dict(torch.load(SAVE_PARA_PATH))

    cls_criterion = nn.NLLLoss()

    netD.to(opt.device)
    netG.to(opt.device)

    cls_criterion.to(opt.device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_cgan, betas=(opt.adam_beta1_cgan, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_cgan, betas=(opt.adam_beta1_cgan, 0.999))



    #  train a classifier for the following cgan training.
    pretrain_cls = CLASSIFIER()
    with torch.no_grad():
        acc, auc_score_av, auc_score_softmax = pretrain_cls.val()
    print("----------------------------")
    print(("ACC:%f,     AVROC:%f,     ROC:%f") % (acc, auc_score_av, auc_score_softmax))

    #  cgan training
    d_loss_record = []
    g_loss_record = []
    c_loss_record = []


    if opt.check == False:
        for epoch in range(opt.epoch_cgan):

            d_loss = AverageMeter()
            g_loss = AverageMeter()
            c_loss = AverageMeter()
            batch_count = 0

            for batch_idx in range(len(data.train_feature)//opt.batchsize_cgan):

                for kk in range(opt.critic_iter):

                    index = torch.randperm(len(data.train_feature))
                    inputv = Variable(data.train_feature[index[:opt.batchsize_cgan]]).to(opt.device)
                    labelv = Variable(data.train_label[index[:opt.batchsize_cgan]]).to(opt.device)
                    attv = Variable(data.train_att.cpu()[labelv.cpu()]).to(opt.device)

                    noise = torch.FloatTensor(len(inputv), opt.nz)
                    noise = noise.to(opt.device)

                    for p in netD.parameters():
                        p.requires_grad = True

                    netD.zero_grad()

                    criticD_real = netD(inputv, attv)
                    criticD_real = -criticD_real.mean()
                    criticD_real.backward()

                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    fake = netG(noisev, attv)
                    criticD_fake = netD(fake.detach(), attv)
                    criticD_fake = criticD_fake.mean()
                    criticD_fake.backward()

                    gradient_penalty = calc_gradient_penalty(netD, inputv, fake, attv)
                    gradient_penalty.backward()

                    d_loss.update(criticD_real - criticD_fake, len(inputv))
                    # D_cost = criticD_fake - criticD_real + gradient_penalty
                    optimizerD.step()

                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = False  # avoid computation

                netG.zero_grad()
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, attv)
                criticG_fake = netD(fake, attv)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                # classification loss
                c_errG = cls_criterion(pretrain_cls.model(fake), Variable(labelv))

                # errG = G_cost + opt.cls_weight * c_errG + opt.proto_param2 * loss2 + opt.proto_param1 * loss1
                errG = G_cost + opt.cls_weight * c_errG
                errG.backward()
                optimizerG.step()

                g_loss.update(G_cost, len(inputv))
                c_loss.update(c_errG, len(inputv))


            d_loss_record.append(d_loss.avg.item())
            g_loss_record.append(g_loss.avg.item())
            c_loss_record.append(c_loss.avg.item())
            check_loss(d_loss_record, g_loss_record, c_loss_record)

            print('EP[%d/%d]  d loss:%f, g loss:%f, c loss:%f ' % (
                epoch, opt.epoch_cgan, d_loss.avg, g_loss.avg, c_loss.avg))

            # SAVE_PARA_PATH = './checkpoints/' + "gan_d"+str(epoch) + '.pth'
            # torch.save(netD.state_dict(), SAVE_PARA_PATH)

            SAVE_PARA_PATH = './checkpoints/' + "gan_g"+str(epoch) + '.pth'
            torch.save(netG.state_dict(), SAVE_PARA_PATH)

def check_loss(d_loss_record, g_loss_record, c_loss_record):

    plt.plot(d_loss_record, marker='o', color="b",linewidth=0.5,markersize=1,label='d')
    plt.plot(g_loss_record, marker='o', color="r",linewidth=0.5,markersize=1,label='g')
    plt.plot(c_loss_record, marker='o', color="g", linewidth=0.5, markersize=1, label='c')

    plt.legend()
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('epoch')
    plt.ylabel("loss")
    # plt.xticks(range(0,20))
    # pyplot.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.grid()
    plt.savefig('./'+'gan_loss''.jpg',dpi = 900)
    plt.close()

def set_hyperparameters(opt):

    opt.check = False

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    opt.device = torch.device("cuda:"+str(int(opt.device)) if torch.cuda.is_available() else "cpu")

    if opt.split == None:
        print("no dataset split hyperparameter!")
        exit()



    if opt.dataset == "cifar10" or opt.dataset == "cifar+50":

        data = data_preprocess.DATA_PREPROCESS(opt)

        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 40
        opt.epoch_classifier = 10
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan = 0.00001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 32
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "mnist":

        data = data_preprocess.DATA_PREPROCESS(opt)

        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 6
        opt.epoch_classifier = 10
        opt.ngh = 3200
        opt.ndh = 3200
        opt.lr_cgan = 0.00001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 32
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 1600
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "svhn":

        data = data_preprocess.DATA_PREPROCESS(opt)

        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 40
        opt.epoch_cgan = 6
        opt.epoch_classifier = 10
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan = 0.00001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 32
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "CUB":

        opt.dataroot = "./data/xlsa17/data"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        data = util.DATA_LOADER(opt)


        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 41
        opt.epoch_classifier = 100
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan =  0.0001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 32
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "AwA":

        opt.dataroot = "./data/xlsa17/data"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        data = util.DATA_LOADER(opt)


        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 80
        opt.epoch_classifier = 50
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan = 0.00001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 64
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "aPaY":

        opt.dataroot = "./data/xlsa17/data"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        data = util.DATA_LOADER(opt)


        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 41
        opt.epoch_classifier = 50
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan = 0.0001
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 64
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    if opt.dataset == "SUN":

        opt.dataroot = "./data/xlsa17/data"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        data = util.DATA_LOADER(opt)


        opt.nz = 312
        opt.cls_weight = 0.01
        opt.epoch_cgan = 41
        opt.epoch_classifier = 50
        opt.ngh = 4096
        opt.ndh = 4096
        opt.lr_cgan = 0.0002
        opt.lr_classifier = 0.001
        opt.critic_iter = 5
        opt.batchsize_cgan = 64
        opt.batchsize_classifier = 64
        opt.attSize = data.train_att.shape[1]
        opt.resSize = 2048
        opt.lambda1 = 10
        opt.adam_beta1_cgan = 0.5
        opt.adam_beta1_classifier = 0.5

    return data



if __name__ == '__main__':

    data = set_hyperparameters(opt)

    train_cgan(opt)


    print("finished cgan training!")
    exit()