#######################################   Description   #######################################################
####  Goal: This script aims to obtain deep features given a dataset.
####  Running: python3 run.py --dataset cifar-10-10 --split_idx 3  --data_root /root/data
####  Requirement: datasets in /root/data,  word embeddings stored in /root/data/glove.6B.50d.txt (https://github.com/stanfordnlp/GloVe)
####  Output: XXX_train_unknown_set.pth, XXX_train_known_set.pth, XXX_test_unknown_set.pth, XXX_test_known_set.pth, XXX_word_embeddings.pth
#######################################   Description   #######################################################


######################  import standard library
import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data_prepare import get_class_splits, get_datasets
import torch.nn.functional as F
from schedulers import get_scheduler
from models import get_model
from tqdm import tqdm
import numpy as np
import evaluation
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import random

#####################  running command analysis
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--split_idx', type=int, default=None, help='dataset split id')
parser.add_argument('--device', type=int, default=0, help='gpu id')
parser.add_argument('--data_root', type=str, default="/root/data", help='dataset location')

args  = parser.parse_args()
print(args)



def get_default_hyperparameters(args):

    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """



    args.device = torch.device("cuda:"+str(args.device))

    args.loss = "Softmax"
    # args.loss = "CPNLoss"
    args.optim=None
    args.seed = 0


    hyperparameter_path = './paper_hyperparameters.csv'
    df = pd.read_csv(hyperparameter_path)


    # 根据数据集和损失函数，设置超参。
    df = df.loc[df['Loss'] == args.loss]
    hyperparams = df.loc[df['Dataset'] == args.dataset].values[0][2:]

    # -----------------
    # DATASET / LOSS specific hyperparams
    # -----------------
    args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = hyperparams

    if args.dataset in ('imagenet'):

        args.model = 'timm_resnet50_pretrained'
        args.resnet50_pretrain = 'places_moco'
        args.feat_dim = 2048

    else:

        args.model = 'classifier32'
        args.feat_dim = 128

    if args.loss == "Softmax":

        args.max_epoch = 240
        args.scheduler = 'step'
        args.num_restarts = 2
        args.weight_decay = 1e-4

        if args.dataset == "tinyimagenet":
            args.transform = 'pytorch-tinyimagenet'
        if args.dataset in ["cifar-10-10", "cifar-10-100"]:
            args.transform = 'pytorch-cifar'

        print("args.transform:", args.transform)


    return args

def get_optimizer(args, params_list):


    if args.optim is None:

        if options['dataset'] == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == 'sgd':

        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


    elif args.optim == 'adam':

        optimizer = torch.optim.Adam(params_list, lr=args.lr)

    else:

        raise NotImplementedError

    return optimizer




def main_worker(options, args):

    torch.manual_seed(options['seed'])


    # -----------------------------
    # DATALOADERS
    # -----------------------------
    trainloader = dataloaders['train']
    testloader = dataloaders['test_known']
    outloader = dataloaders['test_unknown']

    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))

    net = get_model(args)

    feat_dim = args.feat_dim

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
        }
    )


    # -----------------------------
    # GET LOSS
    # -----------------------------
    # Loss = importlib.import_module('methods.ARPL.loss.'+options['loss'])

    criterion = nn.CrossEntropyLoss()



    # -----------------------------
    # PREPARE EXPERIMENT
    # -----------------------------

    net = nn.DataParallel(net).cuda()
    criterion = criterion.cuda()

    # model_path = os.path.join(args.log_dir, 'arpl_models', options['dataset'])
    model_path = os.path.join("./results", options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]




    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list)




    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    scheduler = get_scheduler(optimizer, args)




    start_time = time.time()

    # -----------------------------
    # TRAIN
    # -----------------------------

    for epoch in range(options['max_epoch']):


        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        net.train()
        losses = AverageMeter()

        torch.cuda.empty_cache()

        loss_all = 0
        for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):


            data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                x, y = net(data, True)

                loss = criterion(y, labels)

                loss.backward()

                optimizer.step()

            losses.update(loss.item(), data.size(0))

            loss_all += losses.avg

        print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))


        print("==> Test", options['loss'])

        net.eval()
        correct, total = 0, 0

        torch.cuda.empty_cache()

        _pred_k, _pred_u, _labels = [], [], []

        with torch.no_grad():
            for data, labels, idx in tqdm(testloader):

                data, labels = data.cuda(), labels.cuda()

                with torch.set_grad_enabled(False):
                    x, y = net(data, True)
                    predictions = y.data.max(1)[1]
                    total += labels.size(0)
                    correct += (predictions == labels.data).sum()

                    if args.loss == "Softmax":
                        y = torch.nn.Softmax(dim=-1)(y)

                    _pred_k.append(y.data.cpu().numpy())
                    _labels.append(labels.data.cpu().numpy())

            for batch_idx, (data, labels, idx) in enumerate(tqdm(outloader)):

                data, labels = data.cuda(), labels.cuda()

                with torch.set_grad_enabled(False):

                    x, y = net(data, True)

                    if args.loss == "Softmax":
                        y = torch.nn.Softmax(dim=-1)(y)

                    _pred_u.append(y.data.cpu().numpy())

        # Accuracy
        acc = float(correct) * 100. / float(total)
        print('Acc: {:.5f}'.format(acc))

        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        _labels = np.concatenate(_labels, 0)

        # Out-of-Distribution detction evaluation
        x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
        results = evaluation.metric_ood(x1, x2)['Bas']

        # OSCR
        _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

        # Average precision
        ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                           list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

        results['ACC'] = acc
        results['OSCR'] = _oscr_socre * 100.
        results['AUPR'] = ap_score * 100


        print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch,
                                                                                          results['ACC'],
                                                                                          results['AUROC'],
                                                                                          results['OSCR']))



        if epoch == options['max_epoch'] - 1:
            weights = net.state_dict()
            file_name = options['dataset'] + '.csv'
            filename = '{}/{}_{}_split{}.pth'.format("./results", args.loss, file_name.split('.')[0], args.split_idx)
            torch.save(weights, filename)

        # -----------------------------
        # STEP SCHEDULER
        # ----------------------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def get_features(options, args):

    # -----------------------------
    # DATALOADERS
    # -----------------------------

    train_known_loader = dataloaders['train']
    train_unknown_loader = dataloaders['train_unknown']
    test_known_loader = dataloaders['test_known']
    test_unknown_loader = dataloaders['test_unknown']



    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))

    net = get_model(args)

    feat_dim = args.feat_dim

    file_name = options['dataset'] + '.csv'
    filename = '{}/{}_{}_split{}.pth'.format("./results", args.loss, file_name.split('.')[0], args.split_idx)



    if os.path.exists(filename):
        print("Loading from checkpoints {}".format(filename))

        if args.loss == "CPNLoss":
            net.load_state_dict(torch.load(filename))

        if args.loss == "Softmax":
            state_dict = net.strip_state_dict(torch.load(filename))
            net.load_state_dict(state_dict)
    else:
        print("no trained backbone!")
        exit()


    net.eval()
    net.to(args.device)

    with torch.no_grad():

        # -----------------------------
        # TEST
        # -----------------------------

        known_scores_av = []
        known_scores_softmax = []

        accuracy = AverageMeter()

        for batch_idx, data in enumerate(test_known_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

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

        for batch_idx, data in enumerate(test_unknown_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

            output = torch.nn.functional.softmax(av_output,
                                                 dim=1)

            _, preds = torch.max(output, 1)

            unknown_scores_av.extend(
                [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

            unknown_scores_softmax.extend(
                [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

        auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
        auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)
        print("ACC:", accuracy.avg, "AV_ROC:", auc_score_av, "SOFTMAX_ROC", auc_score_softmax)

        # -----------------------------
        # GET FEATURES
        # -----------------------------

        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(train_known_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

            if get_features == None:
                get_features = features
                get_labels = labelv
            else:
                get_features = torch.cat([get_features, features], dim=0)
                get_labels = torch.cat([get_labels, labelv], dim=0)

        print("number of training samples：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_train_known_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)

        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(test_known_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

            if get_features == None:
                get_features = features
                get_labels = labelv
            else:
                get_features = torch.cat([get_features, features], dim=0)
                get_labels = torch.cat([get_labels, labelv], dim=0)

        print("number of test samples：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_test_known_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)

        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(train_unknown_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

            if get_features == None:
                get_features = features
                get_labels = labelv
            else:
                get_features = torch.cat([get_features, features], dim=0)
                get_labels = torch.cat([get_labels, labelv], dim=0)

        # print("number of open set samples：", len(get_labels))
        # SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_train_unknown_set.pth"
        # torch.save([get_features, get_labels], SAVE_PARA_PATH)

        get_features2 = None
        get_labels2 = None

        for batch_idx, data in enumerate(test_unknown_loader):
            inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features, av_output = net(inputv, True)

            if get_features2 == None:
                get_features2 = features
                get_labels2 = labelv
            else:
                get_features2 = torch.cat([get_features2, features], dim=0)
                get_labels2 = torch.cat([get_labels2, labelv], dim=0)

        # print("number of online known samples：", len(get_labels2))
        # SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_test_unknown_set.pth"
        # torch.save([get_features2, get_labels2], SAVE_PARA_PATH)

        get_features3 = torch.cat([get_features,get_features2])
        get_labels3 = torch.cat([get_labels, get_labels2])
        print("number of open set samples：", len(get_labels3))
        SAVE_PARA_PATH = "./results/" + args.loss + "_" + args.dataset + "_split" + str(
            args.split_idx) + "_open_set.pth"
        torch.save([get_features3, get_labels3], SAVE_PARA_PATH)


        ############## get semantic embeddings
        fp = open("/root/data/glove.6B.50d.txt", "r")
        # fp = open("../data/glove.6B.300d.txt", "r")
        sample = fp.readlines()
        result_dict = {}

        progress_bar = tqdm(sample)
        for i, line in enumerate(progress_bar):
            # for line in sample:
            #     print(line)
            sample_ = line.split(' ')  # 按俩空格进行文件中每一行的切割
            # print(sample_)
            # print(sample_[0])
            # sample_=['33\tInvasive\textended\n']
            # sample_[0]=33	Invasive	extended
            result_dict[sample_[0]] = sample_[1:]


        if args.dataset == "cifar-10-10":
            word = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif args.dataset == "mnist":
            word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        elif args.dataset == "svhn":
            word = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        elif args.dataset == "cifar-10-100":
            word = ['apple', 'gouramis', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
                    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
                    'chimpanzee',
                    'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                    'elephant', 'flatfish',
                    'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
                    'leopard', 'lion', 'lizard',
                    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                    'orchid', 'otter',
                    'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                    'rabbit', 'raccoon',
                    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                    'snake', 'spider', 'squirrel',
                    'streetcar', 'sunflower', 'pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                    'train', 'trout', 'tulip',
                    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

        else:
            print("no word setting!!!")
            exit(0)


        word = [word[i] for i in args.known]
        print(word)
        word_embedding = None
        for k in range(len(word)):
            for j in range(len(result_dict[word[k]])):
                result_dict[word[k]][j] = float(result_dict[word[k]][j])
            # print(result_dict[word[k]])

        for k in range(len(word)):
            buff = torch.tensor(result_dict[word[k]])
            buff = torch.unsqueeze(buff, dim=0)
            if word_embedding == None:
                word_embedding = buff
            else:
                word_embedding = torch.cat([word_embedding, buff])
        # print(word_embedding)
        print(word_embedding.shape)

        SAVE_PARA_PATH = "./results/" + args.loss + "_" + args.dataset + "_split" + str(
            args.split_idx) + "_word_embeddings.pth"
        torch.save(word_embedding, SAVE_PARA_PATH)

    print("finished")
    exit()




class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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

def compute_roc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score


if __name__ == '__main__':




    exp_root = "./checkpoints"

    print('NOTE: Using default hyper-parameters...')
    args = get_default_hyperparameters(args)




    args.exp_root = exp_root
    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()


    for i in range(1):

        # ------------------------
        # INIT
        # ------------------------

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)

        img_size = args.image_size

        args.save_name = '{}_{}'.format(args.model, args.dataset)

        # ------------------------
        # SEED
        # ------------------------
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ------------------------
        # DATASETS
        # ------------------------

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                image_size=args.image_size, seed=args.seed,
                                args=args)



        # ------------------------
        # RANDAUG HYPERPARAM SWEEP
        # ------------------------
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=16)



        # ------------------------
        # SAVE PARAMS
        # ------------------------
        options = vars(args)
        options.update(
            {
                'item':     i,
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

        # ------------------------
        # TRAIN
        # ------------------------
        file_name = options['dataset'] + '.csv'
        filename = '{}/{}_{}_split{}.pth'.format("./results", args.loss, file_name.split('.')[0], args.split_idx)
        print(options)


        if os.path.exists(filename):
            get_features(options, args)
        else:
            res = main_worker(options, args)
            get_features(options, args)