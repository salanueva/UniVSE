import torch
import torch.nn as nn
import torch.nn.init


class SiameseRegressor(nn.Module):
    """
    Embedding similarity estimator that implements a siamese architecture. It expects pre-calculated embeddings as input
    Idea based on the following paper:
        Mueller's et al., "Siamese Recurrent Architectures for Learning Sentence Similarity." (AAAI, 2019).
        https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12195/12023
    Differences:
        Implements different similarity functions that compares input embeddings:  subtraction, cosine, manhantan
    """
    def __init__(self, input_dim=1024, hidden_dim=100, comb_function='concat', dropout=0.0, batch_norm=False):
        super(SiameseRegressor, self).__init__()

        self.comb_function = comb_function
        self.dropout = dropout
        self.use_bn = batch_norm

        self.hidden_size = hidden_dim
        # Define layers
        if self.comb_function in ['subtract', 'multiply']:
            self.comb = nn.Linear(input_dim, self.hidden_size)
        elif self.comb_function in ['concat']:
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
        else:
            raise ValueError

        x = self.comb(inp)
        x = self.dropout(x)
        if self.use_bn:
            x = self.bn(x)
        x = torch.sigmoid(x)
        logits = self.output(x)

        return logits.view(-1, 1)


class MultimodalProjectedSiameseRegressor(nn.Module):
    """
    Embedding similarity estimator that implements a siamese architecture.
    """
    def __init__(self, input1_dim=1024, input2_dim=1024, hidden_input_dim=300, hidden_dim=100,
                 comb_function='concat', dropout=0.0, batch_norm=False):
        super(MultimodalProjectedSiameseRegressor, self).__init__()

        self.comb_function = comb_function
        self.dropout = dropout
        self.use_bn = batch_norm

        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.hidden_input_dim = hidden_input_dim
        self.hidden_size = hidden_dim

        # Define layers
        self.projector1 = nn.Linear(input1_dim, self.hidden_input_dim)
        self.bn_p1 = nn.BatchNorm1d(2*hidden_input_dim)
        self.projector2 = nn.Linear(input2_dim, self.hidden_input_dim)
        self.bn_p2 = nn.BatchNorm1d(2*hidden_input_dim)

        if self.comb_function in ['subtract', 'multiply']:
            self.fc1 = nn.Linear(2 * self.hidden_input_dim, self.hidden_size)
        elif self.comb_function in ['concat']:
            self.fc1 = nn.Linear(4 * self.hidden_input_dim, self.hidden_size)

        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1)

        self.criterion = torch.nn.MSELoss()

    def _project_input(self, x):
        """
        :param x: list of two pytorch tensor (input_dim1, input_dim2)
        :return: projected and concatenated tensor (2*hidden_input_dim)
        """
        x1 = self.projector1(x[0])
        x1 = torch.sigmoid(x1)
        x2 = self.projector2(x[1])
        x2 = torch.sigmoid(x2)
        x = torch.cat((x1, x2), 1)
        return x

    def forward(self, x1, x2):
        """
        :param x1: list of two tensors
        :param x2: list of two tensors
        :return:
        """
        # Project inputs
        x1 = self._project_input(x1)
        x2 = self._project_input(x2)
        if self.use_bn:
            x1 = self.bn_p1(x1)
            x2 = self.bn_p2(x2)

        if self.comb_function == 'subtract':
            inp = x1 - x2
        elif self.comb_function == 'multiply':
            inp = x1 * x2
        elif self.comb_function == 'concat':
            h_sub = torch.abs(x1 - x2)
            h_ang = x1 * x2
            inp = torch.cat((h_sub, h_ang), 1)
        else:
            raise ValueError

        x = self.fc1(inp)
        if self.use_bn:
            x = self.bn(x)
        x = torch.sigmoid(x)
        logits = self.output(x)

        return logits.view(-1, 1)
