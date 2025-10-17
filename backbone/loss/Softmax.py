import torch
import torch.nn as nn
import torch.nn.functional as F

# from methods.ARPL.loss.LabelSmoothing import smooth_cross_entropy_loss

# from methods.ARPL.loss.LabelSmoothing import smooth_cross_entropy_loss

class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()
        # self.temp = options['temp']
        self.temp = 1.0
        self.label_smoothing = options['label_smoothing']

    def forward(self, x, y, labels=None):

        # 'y' is logits of the classifier
        logits = y

        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss = F.cross_entropy(logits / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss

def smooth_cross_entropy_loss(logits, labels, smoothing, dim=-1):

    """
    :param logits: Predictions from model (before softmax) (B x C)
    :param labels: LongTensor of class indices (B,)
    :param smoothing: Float, how much label smoothing
    :param dim: Channel dimension
    :return:
    """

    # Convert labels to distributions
    labels = smooth_one_hot(true_labels=labels, smoothing=smoothing, classes=logits.size(dim))

    preds = logits.log_softmax(dim=dim)

    return torch.mean(torch.sum(-labels * preds, dim=dim))

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):

    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist
