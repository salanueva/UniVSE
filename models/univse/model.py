import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision
import os
import sys

sys.path.append(os.getcwd())
from models.univse import loss
from models.univse import corpus


# HELPER FUNCTIONS (for unflattening purposes)
def unflatten_pos_embeddings(embeddings, num_per_caption):
    """
    Separate positive objects, attributes or relations per caption
    :param embeddings: tensor of computed embeddings, not separated by captions
    :param num_per_caption: information about how many objects, attributes or relations
    can be found on each caption
    :return: list of tensors, one tensor per caption
    """
    start, end = 0, 0
    unflattened_pos = []
    for num in num_per_caption:
        end += num
        if num == 0:
            unflattened_pos.append(None)
        else:
            unflattened_pos.append(embeddings[start:end])
        start = end
    return unflattened_pos


def unflatten_neg_embeddings(embeddings, num_pos_per_caption, num_neg_per_caption):
    """
    Separate negative objects, attributes or relations per caption and positive instances
    :param embeddings: tensor of computed embeddings, not separated by captions and positive instances
    :param num_pos_per_caption: information about how many objects, attributes or relations
    can be found on each captions
    :param num_neg_per_caption: information about how many negative instances
    can be found for each positive instance
    :return: list of lists of tensors, one tensor per negative instance
    """
    start, end = 0, 0
    start_b, end_b = 0, 0
    unflattened_neg = []
    for num in num_pos_per_caption:
        end_b += num
        current_neg = []
        for num_neg in num_neg_per_caption[start_b:end_b]:
            end += num_neg
            current_neg.append(embeddings[start:end])
            start = end
        start_b = end_b
        if len(current_neg) == 0:
            unflattened_neg.append(None)
        else:
            unflattened_neg.append(torch.cat(current_neg, dim=0))
    return unflattened_neg


# MODEL SUBCLASSES
class ObjectEncoder(nn.Module):
    """
    Combines basic semantic embeddings (GloVe) and modifier semantic embeddings
    (trainable) into its final representation. If needed, its output is fed into
    a neural combiner (in order to compute relation and sentence embeddings)
    """
    def __init__(self, input_size, output_size):
        """
        Initializes object encoder
        :param input_size: length of input embedddings (concatenation of basic and modif embeddings)
        :param output_size: length of input embedddings, same as UniVSE space
        """
        super(ObjectEncoder, self).__init__()
        self.forward_layer = nn.Linear(input_size, output_size)
        self.forward_tanh = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, embeddings):
        current = self.sigmoid(self.forward_layer(embeddings))  # (elem, 400) -> (elem, 1024)
        tanh_values = self.tanh(self.forward_tanh(embeddings))  # (elem, 400) -> (elem, 1024)
        output = f.normalize(current * tanh_values, dim=1, p=2)
        return output  # (elem, 1024)


# Class extracted from https://github.com/fartashf/vsepp and slightly modified
class NeuralCombiner(nn.Module):
    def __init__(self, word_dim, embed_size, num_layers, use_abs=False):
        super(NeuralCombiner, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # caption embedding
        self.rnn = nn.GRU(word_dim, self.embed_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        """
        Handles variable size captions
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1)
        if torch.cuda.is_available():
            I = I.cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = f.normalize(out, dim=1, p=2)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class CustomResNet152(nn.Module):
    """
    Image encoder that computes both its image embedding and its convolutional feature map
    """
    def __init__(self, dim=1024, train_resnet=False):
        """
        Initializes image encoder based on ResNet
        :param dim: length of the UniVSE space embeddings
        :param train_resnet: sets backbone's weights as trainable if true
        """
        super(CustomResNet152, self).__init__()
        self.dim = dim
        # Load pretrained resnet and delete its last two layers
        resnet = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete avg pool 2d + last fc layer from resnet
        self.resnet = nn.Sequential(*modules)
        # Add convolutional layer to project ResNet output into the UniVSE space
        self.conv = nn.Conv2d(2048, self.dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        for param in self.resnet.parameters():
            param.requires_grad = train_resnet

    def forward(self, x):
        """
        Forward pass of the image encoder
        :param x: tensor with images of a batch of shape (batch_size, 3, 224, 224)
        :return: image embeddings of shape (batch_size, self.dim) and feature
        embeddings of shape (batch_size, flattened_feature_map_size (49), self.dim)
        """

        # Extract features from backbone
        features = self.resnet(x)  # (bs, 3, 224, 224) -> (bs, 1024, 7, 7)
        features = self.conv(features).view(-1, self.dim, 49)  # (bs, 1024, 7, 7) -> (bs, 1024, 49)

        # Max k-pooling (k=10)
        index = features.topk(10, dim=2)[1].sort(dim=2)[0]
        k_pooled_imgs = features.gather(2, index)  # (bs, 1024, 49) -> (bs, 1024, 10)
        images = torch.mean(k_pooled_imgs, dim=2)  # (bs, 1024, 10) -> (bs, 1024)

        # Normalize vectors (and permute features to handle embeddings more easily)
        features = f.normalize(features, dim=1, p=2)
        features = features.permute((0, 2, 1))  # (bs, 1024, 49) -> (bs, 49, 1024)
        images = f.normalize(images, dim=1, p=2)

        return images, features


class UniVSE(nn.Module):
    """
    Entire UniVSE model
    """

    def __init__(self, vocab_encoder, input_size=400, hidden_size=1024, grad_clip=0.0, rnn_layers=1, train_cnn=False):
        """
        Initializes the Unified Visual Semantic Embeddings model
        :param vocab_encoder: object that inherits all functions from class VocabularyEncoder
        :param input_size: length of the concatenation of basic and modif embeddings
        :param hidden_size: length of embeddings in UniVSE space
        :param grad_clip: gradient clipping value
        :param rnn_layers: number of layers of the neural combiner
        :param train_cnn: Image encoder's backbone is trainable if set to True
        """
        super(UniVSE, self).__init__()

        # Embedding and layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        # Modules of the UniVSE model
        self.vocabulary_encoder = vocab_encoder
        self.object_encoder = ObjectEncoder(self.input_size, self.hidden_size)
        self.neural_combiner = NeuralCombiner(self.hidden_size, self.hidden_size, self.rnn_layers)
        self.image_encoder = CustomResNet152(train_resnet=train_cnn)

        # Loss function
        self.criterion = loss.UniVSELoss()

        # Trainable parameters
        params = list(self.vocabulary_encoder.modif.parameters())
        params += list(self.object_encoder.parameters())
        params += list(self.neural_combiner.parameters())
        params += list(self.image_encoder.conv.parameters())
        if train_cnn:
            params += list(self.image_encoder.resnet.parameters())
        self.params = params

        # Other hyper-parameters
        self.alpha = 0.75
        self.finetune_cnn = train_cnn
        self.grad_clip = grad_clip
        self.inference = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_captions(cls, captions, glove_file, input_size=400, hidden_size=1024, grad_clip=0.0, rnn_layers=1,
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

        self.object_encoder.load_state_dict(model_data[0])
        self.neural_combiner.load_state_dict(model_data[1])
        self.image_encoder.conv.load_state_dict(model_data[2])
        if self.finetune_cnn and len(model_data) > 3:
            self.image_encoder.resnet.load_state_dict(model_data[3])

    def save_model(self, model_file):
        """
        Save model weights
        :param model_file: string of output filename
        """
        model_data = [
            self.object_encoder.state_dict(),
            self.neural_combiner.state_dict(),
            self.image_encoder.conv.state_dict()
        ]
        if self.finetune_cnn:
            model_data.append(self.image_encoder.resnet.state_dict())

        with open(model_file, "wb") as out_f:
            pickle.dump(model_data, out_f)

    def to_device(self, device):
        """
        Set model to cpu or gpu
        :param device: torch.device() object
        """
        self.vocabulary_encoder = self.vocabulary_encoder.to(device)
        self.object_encoder = self.object_encoder.to(device)
        self.neural_combiner = self.neural_combiner.to(device)
        self.image_encoder = self.image_encoder.to(device)

    def train_start(self):
        """
        Switch to train mode
        """
        self.vocabulary_encoder.modif.train()
        self.object_encoder.train()
        self.neural_combiner.train()
        self.image_encoder.train()
        self.inference = False

    def val_start(self):
        """
        Switch to evaluate mode
        """
        self.vocabulary_encoder.modif.eval()
        self.object_encoder.eval()
        self.neural_combiner.eval()
        self.image_encoder.eval()
        self.inference = True

    def forward(self, images, captions):
        """
        Makes a forward pass given a batch of images and captions
        :param images: torch tensor of n images (n, 3, 224, 224)
        :param captions: list of n sentences
        :return: dictionary with all final embeddings
        """
        embeddings = {}

        # IMAGES
        images = images.to(self.device)
        embeddings["img_emb"], embeddings["img_feat_emb"] = self.image_encoder(images)

        # TEXT
        # Extract ids of tokens from each sentence, including negative samples
        components = self.vocabulary_encoder.get_components(captions)

        # Get embeddings from those ids with the VocabularyEncoder
        embeddings["sent_emb"] = self.vocabulary_encoder(components["words"]).to(self.device)

        embeddings["obj_emb"] = self.vocabulary_encoder(components["obj"]).to(self.device)
        embeddings["attr_emb"] = self.vocabulary_encoder(components["attr"]).to(self.device)
        embeddings["rel_emb"] = self.vocabulary_encoder(components["rel"]).to(self.device)

        embeddings["neg_obj_emb"] = self.vocabulary_encoder(components["neg_obj"]).to(self.device)
        embeddings["neg_attr_n_emb"] = self.vocabulary_encoder(components["neg_attr_n"]).to(self.device)
        embeddings["neg_attr_a_emb"] = self.vocabulary_encoder(components["neg_attr_a"]).to(self.device)
        embeddings["neg_rel_emb"] = self.vocabulary_encoder(components["neg_rel"]).to(self.device)

        # Use Object Encoder to compute their embeddings in the UniVSE space
        embeddings["sent_emb"] = self.object_encoder(embeddings["sent_emb"])

        embeddings["obj_emb"] = self.object_encoder(embeddings["obj_emb"])
        embeddings["attr_emb"] = self.object_encoder(embeddings["attr_emb"])
        embeddings["rel_emb"] = self.object_encoder(embeddings["rel_emb"])

        embeddings["neg_obj_emb"] = self.object_encoder(embeddings["neg_obj_emb"])
        embeddings["neg_attr_n_emb"] = self.object_encoder(embeddings["neg_attr_n_emb"])
        embeddings["neg_attr_a_emb"] = self.object_encoder(embeddings["neg_attr_a_emb"])
        embeddings["neg_rel_emb"] = self.object_encoder(embeddings["neg_rel_emb"])

        # Relations and captions must be processed more with the Neural Combiner (RNN)
        embeddings["rel_emb"] = self.neural_combiner(
            embeddings["rel_emb"], torch.tensor([3] * embeddings["rel_emb"].size(0))
        )
        embeddings["neg_rel_emb"] = self.neural_combiner(
            embeddings["neg_rel_emb"], torch.tensor([3] * embeddings["neg_rel_emb"].size(0))
        )

        # Add padding in order to process captions
        start, end = 0, 0
        padded_emb = torch.zeros(len(captions), components["max_words"], self.hidden_size).float().to(self.device)
        for i, num in enumerate(components["num_words"]):
            end += num
            padded_emb[i, :num, :] = embeddings["sent_emb"][start:end]
            start = end
        embeddings["sent_emb"] = self.neural_combiner(padded_emb, torch.tensor(components["num_words"]))

        # Compute Component Embedding aggregating object, attribute and relation embeddings
        start_o, start_a, start_r = 0, 0, 0
        end_o, end_a, end_r = 0, 0, 0
        aggregation = torch.zeros((len(components["num_words"]), self.hidden_size), dtype=torch.float).to(self.device)
        for i, (n_o, n_a, n_r) in enumerate(zip(components["num_obj"], components["num_attr"], components["num_rel"])):
            end_o += n_o
            end_a += n_a
            end_r += n_r
            aggregation[i] += embeddings["obj_emb"][start_o:end_o].view(-1, self.hidden_size).sum(dim=0).view(-1)
            aggregation[i] += embeddings["attr_emb"][start_a:end_a].view(-1, self.hidden_size).sum(dim=0).view(-1)
            aggregation[i] += embeddings["rel_emb"][start_r:end_r].view(-1, self.hidden_size).sum(dim=0).view(-1)
            start_o = end_o
            start_a = end_a
            start_r = end_r
        norm_aggregation = f.normalize(aggregation, dim=1, p=2)
        embeddings["comp_emb"] = norm_aggregation

        # Compute Caption Embedding using alpha (alpha is not trainable)
        embeddings["cap_emb"] = self.alpha * embeddings["sent_emb"] + (1 - self.alpha) * embeddings["comp_emb"]

        # Unflatten objects, attributes and relations (both positives and negatives), in order to ease loss computations
        embeddings["obj_emb"] = unflatten_pos_embeddings(embeddings["obj_emb"], components["num_obj"])
        embeddings["attr_emb"] = unflatten_pos_embeddings(embeddings["attr_emb"], components["num_attr"])
        embeddings["rel_emb"] = unflatten_pos_embeddings(embeddings["rel_emb"], components["num_obj"])

        embeddings["neg_obj_emb"] = unflatten_neg_embeddings(
            embeddings["neg_obj_emb"], components["num_obj"], components["num_neg_obj"]
        )
        embeddings["neg_attr_n_emb"] = unflatten_neg_embeddings(
            embeddings["neg_attr_n_emb"], components["num_neg_attr_n"], components["num_neg_attr_n"]
        )
        embeddings["neg_attr_a_emb"] = unflatten_neg_embeddings(
            embeddings["neg_attr_a_emb"], components["num_neg_attr_a"], components["num_neg_attr_a"]
        )
        embeddings["neg_rel_emb"] = unflatten_neg_embeddings(
            embeddings["neg_rel_emb"], components["num_rel"], components["num_neg_rel"]
        )

        return embeddings
