import torch
import torch.nn as nn
import torch.nn.init
import os
import sys

sys.path.append(os.getcwd())
from models.simplified_univse import model as simp_univse
from models.univse import model as univse


# Helper Function
def init_weights(m):
    """
    Initialize weights of layer m
    :param m: pytorch layer
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class UniVSE(nn.Module):
    """
    Combines basic semantic embeddings (GloVe) and modifier semantic embeddings
    (trainable) into its final representation. If needed, its output is fed into
    a neural combiner (in order to compute relation and sentence embeddings)
    """
    def __init__(self, vocab_file, graph_file=None, simple=False, dropout_prob=0.25):
        """
        Initializes UniVSE model for vSTS
        :param vocab_file: path of the pickle file with VocabularyEncoder object
        :param graph_file: pickle file containing graphs extracted from Scene Graph Parser
        :param simple: true if simplified univse is going to be used, false otherwise
        :param dropout_prob: dropout probability in regressor layer
        """
        super(UniVSE, self).__init__()

        if simple:
            self.univse_layer = simp_univse.UniVSE.from_filename(vocab_file=vocab_file)
        else:
            self.univse_layer = univse.UniVSE.from_filename(vocab_file=vocab_file)
            if graph_file is not None:
                self.univse_layer.vocabulary_encoder.add_graphs(graph_file)

        self.hidden_size = self.univse_layer.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size, 1)
        )
        self.regressor.apply(init_weights)

        self.criterion = torch.nn.MSELoss()
        self.simple = simple

        params = list(self.univse_layer.params)
        params += list(self.regressor.parameters())
        self.params = params

    def train_start(self):
        """
        Switch to train mode
        """
        self.univse_layer.train_start()
        self.regressor.train()

    def eval_start(self):
        """
        Switch to validation mode
        """
        self.univse_layer.val_start()
        self.regressor.eval()

    def forward(self, img_1, sent_1, img_2, sent_2):
        embeddings_1 = self.univse_layer(img_1, sent_1)
        embeddings_2 = self.univse_layer(img_1, sent_1)

        emb_img_1 = embeddings_1["img_emb"]
        emb_img_2 = embeddings_2["img_emb"]

        if self.simple:
            emb_sent_1 = embeddings_1["sent_emb"]
            emb_sent_2 = embeddings_2["sent_emb"]
        else:
            emb_sent_1 = embeddings_1["cap_emb"]
            emb_sent_2 = embeddings_2["cap_emb"]

        output = torch.cat((emb_img_1, emb_sent_1, emb_img_2, emb_sent_2), dim=1)
        logits = self.regressor(output)

        return logits.view(-1, 1)


class SiameseRegressor(nn.Module):
    """
    Embedding similarity estimator that implements a siamese architecture. It expects pre-calculated embeddings as input
    Idea based on the following paper:
        Mueller's et al., "Siamese Recurrent Architectures for Learning Sentence Similarity." (AAAI, 2019).
        https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12195/12023
    Differences:
        Implements different similarity functions that compares input embeddings:  subtraction, cosine, manhantan
    """
    def __init__(self, input_dim=1024, hidden_dim=100, comb_function='concat',
                 lr=0.01, dropout=0.0, wdecay=.0, momentum=.0, batchnorm=False):
        super(SiameseRegressor, self).__init__()

        self.comb_function = comb_function
        self.learning_rate = lr
        self.dropout = dropout
        self.weight_decay = wdecay
        self.momentum = momentum
        self.use_bn = batchnorm

        self.hidden_size = hidden_dim
        # Define layers
        if self.comb_function in ['subtract', 'multiply']:
            self.comb = nn.Linear(input_dim, self.hidden_size)
        else:  # self.comb_function in ['concat']:
            self.comb = nn.Linear(2 * input_dim, self.hidden_size)

        self.dropout = nn.Dropout(p=self.dropout)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1)

        self.criterion = torch.nn.MSELoss()

    def forward(self, x1, x2):

        if self.comb_function == 'subtract':
            inp = x1 - x2
        elif self.comb_function == 'multiply':
            inp = x1 * x2
        elif self.comb_function == 'concat':
            h_sub = torch.abs(x1 - x2)
            h_ang = x1 * x2
            inp = torch.cat((h_sub, h_ang), 1)

        x = self.comb(inp)
        x = self.dropout(x)
        if self.use_bn:
            x = self.bn(x)
        x = torch.sigmoid(x)
        logits = self.output(x)

        return logits.view(-1, 1)
