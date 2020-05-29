import numpy as np
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

    def __init__(self, margin=0, measure=False, max_violation=False):
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


class GlobalImageLevelLoss(nn.Module):
    """
    Unidirectional triplet loss that computes global image level alignment
    between relations and images
    """
    def __init__(self, margin=0.0):
        """
        Initializes loss function class
        :param margin: margin error for the triplet loss function, default 0.0
        """
        super(GlobalImageLevelLoss, self).__init__()
        self.margin = margin

    def forward(self, img_feats, positives, negatives):
        """
        Compute loss function of a mini-batch (following Eq. 1 of UniVSE
        paper)
        :param img_feats: tensor with image features of shape
        (batch_size, embedding_size)
        :param positives: list of tensors of shape (num_pos, 1024), each
        one of them containing num_pos object/attribute embeddings per
        sentence/caption (num_pos == amount of positive embeddings)
        :param negatives: list of tensors of shape (num_pos, num_neg, 1024),
        each one of them containing num_neg object/attribute embeddings for
        each positive instance in a sentence
        :return: global image level alignment loss value
        """
        # Positive Instances
        sim_list = []
        # For each sentence...
        for i, emb_stack in enumerate(positives):
            # Check if that sentence has relation embeddings
            if emb_stack is not None:
                # Compute similarity between relation embeddings and its
                # corresponding image feature embedding
                sims = emb_stack.mv(img_feats[i])  # (num_pos, 1024) * (1024) -> (num_pos)
                sim_list.append(sims)

        # Negative Instances
        max_sim_list = []
        # For each sentence...
        for i, emb_stack in enumerate(negatives):
            # Check if that sentence has relation embeddings
            if emb_stack is not None:
                # Compute similarity of each negative sample with its image
                # embedding and stack them to obtain tensor of shape (num_pos, num_neg)
                sim_stack = torch.stack(
                    # Stack tensors of shape: (num_neg, 1024) * (1024) -> (num_neg)
                    [emb_stack[j].mv(img_feats[i]) for j in range(emb_stack.size(0))]
                )
                # Get maximum similarity obtained with a negative sample for each positive instance
                max_sim, _ = torch.max(sim_stack, dim=1)  # (num_pos, num_neg) -> (num_pos)
                max_sim_list.append(max_sim)

        # Stack all values (in dimension 0), all tensors of shape (num_pos)
        all_emb_sim = torch.cat(sim_list, dim=0)
        max_neg_emb_sim = torch.cat(max_sim_list, dim=0)

        loss = torch.sum((self.margin + max_neg_emb_sim - all_emb_sim).clamp(min=0.0))

        return loss


class LocalRegionLevelLoss(nn.Module):
    """
    Triplet loss that computes local region level alignment between object/attributes
    and image regions
    """
    def __init__(self, margin=0.0):
        """
        Initialize loss function class
        :param margin: margin error for the triplet loss function, default 0.0
        """
        super(LocalRegionLevelLoss, self).__init__()
        self.margin = margin

    def forward(self, img_feats, positives, negatives=None):
        """
        Compute loss function of a mini-batch (following Eq. 2 and 3 of
        UniVSE paper)
        :param img_feats: tensor with region features of images of shape
        (batch_size, num_regions, embedding_size)
        :param positives: list of tensors of shape (num_pos, 1024), each
        one of them containing num_pos object/attribute embeddings per
        sentence/caption (num_pos == amount of positive embeddings)
        :param negatives: list of tensors of shape (num_pos, num_neg, 1024),
        each one of them containing num_neg object/attribute embeddings for
        each positive instance in a sentence
        :return: local region level alignment loss value
        """
        # Positive Instances
        m_list = []
        sim_list = []

        # For each sentence...
        for i, emb_stack in enumerate(positives):
            # Check if that sentence has object/attribute embeddings
            if emb_stack is not None:
                # Compute similarity between object/attribute embeddings and all regions of the image
                sims = emb_stack.mm(img_feats[i].t())  # (num_pos, 1024) * (1024, 49) -> (num_pos, 49)
                m = torch.softmax(sims, dim=1)  # (num_pos, 49) -> (num_pos, 49)
                sim_list.append(sims)
                m_list.append(m)

        # Negative Instances
        max_sim_list = []
        # For each sentence...
        for i, emb_stack in enumerate(negatives):
            # Check if that sentence has object/attribute embeddings
            if emb_stack is not None:
                # Compute similarity of each negative sample with all image regions
                # and stack them to obtain tensor of shape (num_pos, num_neg, 49)
                sim_stack = torch.stack(
                    # Stack tensors of shape: (num_neg, 1024) * (1024, 49) -> (num_neg, 49)
                    [emb_stack[j].mm(img_feats[i].t()) for j in range(emb_stack.size(0))]
                )
                # Get maximum similarity obtained with a negative sample for each positive instance
                max_sim, _ = torch.max(sim_stack, dim=1)  # (num_elem, num_neg, 49) -> (num_elem, 49)
                max_sim_list.append(max_sim)

        # Stack list of tensors (we don't need to separate instances per sentences anymore)
        all_emb_sim = torch.cat(sim_list, dim=0)
        all_emb_m = torch.cat(m_list, dim=0)
        max_neg_emb_sim = torch.cat(max_sim_list, dim=0)

        # Compute loss function
        loss = torch.sum(all_emb_m * (self.margin + max_neg_emb_sim - all_emb_sim).clamp(min=0.0))

        return loss


class UniVSELoss(nn.Module):
    """
    UniVSE loss function, which uses three different types of losses to compute the
    loss value of a batch
    """
    def __init__(self, n_o=0.5, n_a=0.5, n_r=0.0, n_c=0.5):
        super(UniVSELoss, self).__init__()

        self.contrastive_loss = ContrastiveLoss()
        self.global_loss = GlobalImageLevelLoss()
        self.local_loss = LocalRegionLevelLoss()

        self.n_o = n_o
        self.n_a = n_a
        self.n_r = n_r
        self.n_c = n_c

    def forward(self, embeddings):
        """
        Computes loss function of UniVSE model given all computed embeddings
        :param embeddings: computed embeddings of a batch (default amount of 128 sentences)
        :return: two elements: a tensor with the loss value of the batch, and a list of five
        values, each one representing the loss value of all objects, attributes, relations,
        components and sentences, respectively.
        """
        l_sent = self.contrastive_loss(embeddings["img_emb"], embeddings["sent_emb"])
        """ USE THIS RETURN IN CASE OF TRYING THE SIMPLIFIED VERSION OF UNIVSE (similar to VSE++)
        # return l_sent, float(l_sent.data.cpu().numpy())
        """
        # FIXME
        l_comp = self.contrastive_loss(embeddings["img_emb"], embeddings["comp_emb"])

        l_obj = self.local_loss(embeddings["img_feat_emb"], embeddings["obj_emb"], embeddings["neg_obj_emb"])

        l_attr_nouns = self.local_loss(embeddings["img_feat_emb"], embeddings["attr_emb"], embeddings["neg_attr_n_emb"])
        l_attr_attributes = self.local_loss(embeddings["img_feat_emb"], embeddings["attr_emb"],
                                            embeddings["neg_attr_a_emb"])

        l_attr = l_attr_nouns + l_attr_attributes

        # Choose at most one relation from each sentence randomly
        img_emb_samp = []
        rel_emb_samp = []
        for img, rel in zip(embeddings["img_emb"], embeddings["rel_emb"]):
            if rel is not None:
                img_emb_samp.append(img.view(1, -1))
                i = np.random.randint(rel.size(0), size=1)[0]
                rel_emb_samp.append(rel[i].view(1, -1))
        img_emb_samp = torch.cat(img_emb_samp, dim=0)
        rel_emb_samp = torch.cat(rel_emb_samp, dim=0)

        l_rel_others = self.contrastive_loss(img_emb_samp, rel_emb_samp, unidirectional=True)
        l_rel_mutation = self.global_loss(embeddings["img_emb"], embeddings["rel_emb"], embeddings["neg_rel_emb"])
        l_rel = l_rel_others + l_rel_mutation

        total_loss = l_sent + self.n_c * l_comp + self.n_r * l_rel + self.n_a * l_attr + self.n_o * l_obj
        other_loss = [float(elem.data.cpu().numpy()) for elem in [l_obj, l_attr, l_rel, l_comp, l_sent]]

        return total_loss, other_loss
        # FIXME """
