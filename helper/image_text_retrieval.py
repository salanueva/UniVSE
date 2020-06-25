"""
The 4 functions of this script have been copied and slightly modified from the vse++
original implementation: https://github.com/fartashf/vsepp
"""

import numpy as np
import os
import sys
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.getcwd())
from models.vsepp.loss import order_sim
from models.univse import model as univse
from models.univse.corpus import CocoCaptions
from models.simplified_univse import model as simp_univse
from models.simplified_univse.corpus import CocoCaptions as CocoCaptionsSimple


def encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # switch to evaluate mode
    model.val_start()

    count = 0

    # numpy array to keep all the embeddings
    img_embeddings = None
    cap_embeddings = None
    for i, (images, captions) in tqdm(enumerate(data_loader), desc="Compute embeddings for R@k", leave=False):

        # compute the embeddings
        images = images.to(device)
        captions = captions
        embeddings = model(images, captions)
        img_emb = embeddings["img_emb"]
        cap_emb = embeddings["sent_emb"]

        # initialize the numpy arrays given the size of the embeddings
        if img_embeddings is None:
            img_embeddings = np.zeros((len(data_loader), img_emb.size(1)))
            cap_embeddings = np.zeros((len(data_loader), cap_emb.size(1)))

        aux_count = count + cap_emb.size(0)

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embeddings[count:aux_count] = img_emb.data.cpu().numpy().copy()
        cap_embeddings[count:aux_count] = cap_emb.data.cpu().numpy().copy()

        count = aux_count

        del images, captions

    return img_embeddings, cap_embeddings


def evalrank(model_path, vocab_path, data_path, model_type='univse', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # construct model
    if model_type == "vse++":
        raise NotImplementedError
    elif model_type == "univse":
        model = univse.UniVSE.from_filename(vocab_path)
        model.load_model(model_path)
    elif model_type == "simp_univse":
        model = simp_univse.UniVSE.from_filename(vocab_path)
        model.load_model(model_path)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    print('Loading dataset')
    img_path, ann_path = data_path
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    if model_type.startswith("simp"):
        data_loader = CocoCaptionsSimple(img_path, ann_path, transform=transform,
                                         target_transform=None, transforms=None, split="test")
    else:
        data_loader = CocoCaptions(img_path, ann_path, transform=transform, target_transform=None, transforms=None)

    params = {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 6
    }
    data_loader = data.DataLoader(data_loader, **params)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure='cosine', return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, measure='cosine', return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure='cosine',
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure='cosine',
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)

    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().np()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)

    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr
