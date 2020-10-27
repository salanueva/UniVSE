from collections import OrderedDict
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm

sys.path.append(os.getcwd())
from models.vsepp.corpus import VocabularyEncoder
from models.vsepp.loss import ContrastiveLoss


def l2norm(x):
    """
    L2-normalize columns of x
    """
    norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x


def image_encoder(data_name, img_dim, embed_size, fine_tune=False,
                  cnn_type='resnet152', use_abs=False, no_img_norm=False):
    """
    A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_img_norm)
    else:
        img_enc = EncoderImageFull(
            embed_size, fine_tune, cnn_type, use_abs, no_img_norm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, fine_tune=False, cnn_type='resnet152',
                 use_abs=False, no_img_norm=False):
        """Load pretrained RESNET152 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_img_norm = no_img_norm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = fine_tune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(
                self.cnn.classifier._modules['6'].in_features,
                embed_size
            )
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_img_norm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_file, word_dim, embed_size, num_layers, use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        self.vocabulary = VocabularyEncoder(vocab_file)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class VSE(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, vocab_size, word_dim=300, embed_size=1024, lr=2e-3, num_layers=1, finetune=False,
                 cnn_type="resnet152"):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.grad_clip = 2.0
        self.img_enc = image_encoder('coco', 2048, embed_size,
                                     finetune, cnn_type,
                                     use_abs=False,
                                     no_img_norm=False)
        self.txt_enc = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_abs=False)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=0.2,
                                         measure='cosine',
                                         max_violation=True)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=lr)

    @classmethod
    def from_captions(cls, captions, input_size=400, hidden_size=1024, grad_clip=0.0, rnn_layers=1,
                      train_cnn=False):
        """
        Initializes the Unified Visual Semantic Embeddings model and creates vocabulary
        encoder from scratch given a list of sentences and a file with GloVe embeddings
        :param captions: list of sentences/captions
        :param glove_file: path of the file with GloVe embeddings
        :param input_size: length of the concatenation of basic and modif embeddings
        :param hidden_size: length of embeddings in UniVSE space
        :param grad_clip: gradient clipping value
        :param rnn_layers: number of layers of the neural combiner
        :param train_cnn: Image encoder's backbone is trainable if set to True
        """
        vocab_encoder = corpus.VocabularyEncoder(captions, glove_file)
        return cls(vocab_encoder, input_size, hidden_size, grad_clip, rnn_layers, train_cnn)

    @classmethod
    def from_filename(cls, vocab_file, input_size=400, hidden_size=1024, grad_clip=0.0, rnn_layers=1,
                      train_cnn=False):
        """
        Initializes the Unified Visual Semantic Embeddings model and creates vocabulary
        encoder from scratch given a list of sentences and a file with GloVe embeddings
        :param vocab_file: path of the pickle file with VocabularyEncoder object
        :param input_size: length of the concatenation of basic and modif embeddings
        :param hidden_size: length of embeddings in UniVSE space
        :param grad_clip: gradient clipping value
        :param rnn_layers: number of layers of the neural combiner
        :param train_cnn: Image encoder's backbone is trainable if set to True
        """
        vocab_encoder = corpus.VocabularyEncoder(None, None)
        vocab_encoder.load_corpus(vocab_file)
        return cls(vocab_encoder, input_size, hidden_size, grad_clip, rnn_layers, train_cnn)

    def load_model(self, model_file):
        """
        Load model weights
        :param model_file: file with weights of the model
        """
        with open(model_file, "rb") as in_f:
            model_data = pickle.load(in_f)

        self.img_enc.load_state_dict(model_data[0])
        self.txt_enc.load_state_dict(model_data[1])

    def save_model(self, model_file):
        """
        Save model weights
        :param model_file: string of output filename
        """
        model_data = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict()
        ]

        with open(model_file, "wb") as out_f:
            pickle.dump(model_data, out_f)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward(self, images, captions, lengths, volatile=False):
        """
        Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    # USE IT IN TRAINING
    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
