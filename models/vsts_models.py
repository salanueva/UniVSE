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
