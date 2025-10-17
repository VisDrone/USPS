import random
import torch
import torch.backends.cudnn as cudnn
from data import data_preprocess



def set_hyperparameters(opt):



    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    opt.device = torch.device("cuda:"+str(int(opt.gpu)) if torch.cuda.is_available() else "cpu")

    opt.check = False

    if opt.dataset == "cifar10" or opt.dataset == "cifar+50":

        opt.data = data_preprocess.DATA_PREPROCESS(opt)
        opt.nz = 312
        opt.ngh = 4096
        opt.ndh = 4096
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 2048
        opt.epoch_cgan = 39   # use the gan model which have trained 40 epoch
        opt.lr = 0.001


        opt.initialization_training_epoch = 10
        opt.adversarial_training_epoch = 30
        opt.adversarial_training_batchsize = 32
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 2
        opt.prototype_per_incremental = 50
        opt.samples_per_prototype = 40  # scale level

        opt.maxmargin = 20
        opt.minmargin = 1
        opt.dissim = 0.8
        opt.sim = 0.01
        opt.diversity_weight = 0.01


    if opt.dataset == "mnist":

        opt.data = data_preprocess.DATA_PREPROCESS(opt)
        opt.nz = 312
        opt.ngh = 3200
        opt.ndh = 3200
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 1600
        opt.epoch_cgan = 5  # sota
        opt.lr = 0.001

        opt.adversarial_training_batchsize = 32
        opt.initialization_training_epoch = 10
        opt.adversarial_training_epoch = 20
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 2
        opt.prototype_per_incremental = 10
        opt.samples_per_prototype = 40

        opt.maxmargin = 5
        opt.minmargin = 1
        opt.dissim = 0.8
        opt.sim = 0.01
        opt.diversity_weight = 0.01

    if opt.dataset == "CUB":

        opt.data = util.DATA_LOADER(opt)
        opt.nz = 312
        opt.ngh = 4096
        opt.ndh = 4096
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 2048
        opt.epoch_cgan = 40
        opt.lr = 0.001

        opt.adversarial_training_batchsize = 32
        opt.initialization_training_epoch = 50
        opt.adversarial_training_epoch = 40
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 8
        opt.prototype_per_incremental = 10
        opt.samples_per_prototype = 80

        opt.maxmargin = 10
        opt.minmargin = 1
        opt.dissim = 0.5
        opt.sim = 0.01
        opt.diversity_weight = 0.01


    if opt.dataset == "AwA":

        opt.dataroot = "../data/xlsa17"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        opt.data = util.DATA_LOADER(opt)
        opt.nz = 312
        opt.ngh = 4096
        opt.ndh = 4096
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 2048
        opt.epoch_cgan = 70
        opt.lr = 0.001

        opt.adversarial_training_batchsize = 32
        opt.initialization_training_epoch = 50
        opt.adversarial_training_epoch = 1
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 8
        opt.prototype_per_incremental = 10
        opt.samples_per_prototype = 400

        opt.maxmargin = 3
        opt.minmargin = 1
        opt.dissim = 0.5
        opt.sim = 0.01
        opt.diversity_weight = 0.01

    if opt.dataset == "aPaY":


        opt.dataroot = "../data/xlsa17"
        opt.image_embedding = 'res101'
        opt.class_embedding = 'att'
        opt.data = util.DATA_LOADER(opt)
        opt.nz = 312
        opt.ngh = 4096
        opt.ndh = 4096
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 2048
        opt.epoch_cgan = 40
        opt.lr = 0.001

        opt.adversarial_training_batchsize = 64
        opt.initialization_training_epoch = 50
        opt.adversarial_training_epoch = 10
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 8
        opt.prototype_per_incremental = 10
        opt.samples_per_prototype = 80

        opt.maxmargin = 5
        opt.minmargin = 1
        opt.dissim = 0.8
        opt.sim = 0.01
        opt.diversity_weight = 0.01



    if opt.dataset == "SUN":

        opt.data = util.DATA_LOADER(opt)
        opt.nz = 312
        opt.ngh = 4096
        opt.ndh = 4096
        opt.attSize = opt.data.train_att.shape[1]
        opt.resSize = 2048
        opt.epoch_cgan = 40
        opt.lr = 0.001

        opt.adversarial_training_batchsize = 64
        opt.initialization_training_epoch = 10
        opt.adversarial_training_epoch = 1
        opt.epoch_prototype = 100
        opt.milestone_prototype = [50, 70, 90]
        opt.lr_prototype = 10
        opt.incremental_num = 8
        opt.prototype_per_incremental = 5
        opt.samples_per_prototype = 40

        opt.maxmargin = 5
        opt.minmargin = 1
        opt.dissim = 0.8
        opt.sim = 0.01
        opt.diversity_weight = 0.01

    return opt

