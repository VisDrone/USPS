import torch
import argparse
from torch import nn
from functools import partial
from loss import CPNLoss




class classifier32(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super(self.__class__, self).__init__()

        # if feat_dim is None:
        #     feat_dim = 128
        torch.manual_seed(0)

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_feature=False):


        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)


        x = torch.flatten(x, 1)
        y = self.fc(x)

        if return_feature:
            return x, y
        else:
            return y

    def strip_state_dict(self, state_dict, strip_key='module.'):

        """
        Strip 'module' from start of state_dict keys
        Useful if model has been trained as DataParallel model
        """

        for k in list(state_dict.keys()):
            if k.startswith(strip_key):
                state_dict[k[len(strip_key):]] = state_dict[k]
                del state_dict[k]

        return state_dict


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classifier32CPNWrapper(torch.nn.Module):

    def __init__(self, base_model, loss_layer):

        super().__init__()

        self.base_model = base_model
        self.loss_layer = loss_layer

    def forward(self, imgs, return_feature=False):

        x, y = self.base_model(imgs, True)

        logits = self.loss_layer(x)

        if return_feature:
            return x, logits
        else:
            return logits

    def load_state_dict(self, state_dict):

        """
        Override method to take list of state dicts for loss layer and criterion
        """

        base_model_state_dict, loss_layer_state_dict = strip_state_dict(state_dict)

        self.base_model.load_state_dict(base_model_state_dict)
        self.loss_layer.load_state_dict(loss_layer_state_dict)

        self.base_model.eval()
        self.loss_layer.eval()


def get_model(args):

    if args.model == 'classifier32':

        try:
            feat_dim = args.feat_dim
            cs = args.cs
        except:
            feat_dim = None
            cs = False

        model = classifier32(num_classes=len(args.train_classes), feat_dim=feat_dim)

        if args.loss == 'CPNLoss':

            loss_layer = CPNLoss.CPNLoss(len(args.known), 1, args.feat_dim, "random")


            model = Classifier32CPNWrapper(base_model=model, loss_layer=loss_layer)
    else:

        raise NotImplementedError

    return model




def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    base_model_state_dict = {}
    loss_layer_state_dict = {}

    for k in list(state_dict.keys()):
        if k.startswith("base_model"):
            base_model_state_dict[k] = state_dict[k]
        else:
            loss_layer_state_dict[k] = state_dict[k]

    strip_key = "base_model."
    for k in list(base_model_state_dict.keys()):
        if k.startswith(strip_key):

            base_model_state_dict[k[len(strip_key):]] = base_model_state_dict[k]
            del base_model_state_dict[k]

    strip_key = "loss_layer."
    for k in list(loss_layer_state_dict.keys()):
        if k.startswith(strip_key):
            loss_layer_state_dict[k[len(strip_key):]] = loss_layer_state_dict[k]
            del loss_layer_state_dict[k]


    return base_model_state_dict, loss_layer_state_dict



if __name__ == '__main__':
    debug = True