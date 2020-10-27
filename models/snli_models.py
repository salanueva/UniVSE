import torch
import torch.nn as nn
import torch.nn.init


class SnliClassifier(nn.Module):
    """
    SNLI Classifier. It expects pre-calculated embeddings as input
    """
    def __init__(self, input_dim=2048, hidden_dim=100, dropout=0.0, batch_norm=False):
        super(SnliClassifier, self).__init__()

        self.dropout_p = dropout
        self.use_bn = batch_norm

        self.hidden_size = hidden_dim
        # Define layers
        self.first = nn.Linear(input_dim, self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 3)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inp):

        x = self.comb(inp)
        x = self.dropout(x)
        if self.use_bn:
            x = self.bn(x)
        x = torch.sigmoid(x)
        logits = torch.nn.Softmax(self.output(x))

        return logits.view(-1, 1)
