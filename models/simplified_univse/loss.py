import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init


# Helper functions (from original vse++ github page: https://github.com/fartashf/vsepp)
def cosine_sim(im, s):
    """
    Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """
    Order embeddings similarity measure $max(0, s-im)$
    """
    y_m_x = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
             - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -y_m_x.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


# LOSS FUNCTIONS
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss of sentence or component embeddings and image embeddings
    (from original vse++ github page: https://github.com/fartashf/vsepp)
    """

    def __init__(self, margin=0, measure=False, max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s, unidirectional=False):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if unidirectional:
            return cost_s.sum()
        else:
            return cost_s.sum() + cost_im.sum()


class UniVSELoss(nn.Module):
    """
    UniVSE loss function, which uses three different types of losses to compute the
    loss value of a batch
    """
    def __init__(self):
        super(UniVSELoss, self).__init__()

        self.contrastive_loss = ContrastiveLoss()

    def forward(self, embeddings):
        """
        Computes loss function of UniVSE model given all computed embeddings
        :param embeddings: computed embeddings of a batch (default amount of 128 sentences)
        :return: two elements: a tensor with the loss value of the batch, and a list of five
        values, each one representing the loss value of all objects, attributes, relations,
        components and sentences, respectively.
        """
        l_sent = self.contrastive_loss(embeddings["img_emb"], embeddings["sent_emb"])
        return l_sent, float(l_sent.data.cpu().numpy())
