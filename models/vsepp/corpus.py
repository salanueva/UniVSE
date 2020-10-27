import nltk
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.init
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())


class Corpus(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class VocabularyEncoder(nn.Module):

    def __init__(self, captions):
        """
        Creates vocabulary for the VSE or VSE++ model.
        :param captions: list of sentences that will be parsed to extract all possible tokens
        """
        super(VocabularyEncoder, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.corpus = Corpus()
        self.fill_corpus(captions)

        self.embedding = nn.Embedding(len(self.corpus), 300)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, word_ids):
        """
        Converts a list of token ids into a list of tensors with their respective embeddings
        :param word_ids: list of: ids, tuple ids or lists of ids. It depends on which embedding
        type we want to create: object, attribute or relation embeddings.
        :return: a list of tensors containing embeddings of the input ids
        """
        stack = self.embedding(word_ids).to(self.device)
        return stack

    def fill_corpus(self, captions):
        """
        It parses all captions that will be used, detect which tokens will be needed and define
        nouns, attributes and relations that will be used in generated negative cases.
        :param captions: list of sentences
        """

        # Parse all captions to list all possible tokens and nouns
        for cap in tqdm(captions, desc="Creating Corpus"):
            tokenized_cap = nltk.word_tokenize(cap.lower())
            for token in tokenized_cap:
                self.corpus.add_word(token)

        self.corpus.add_word("<unk>")
        self.corpus.add_word("<start>")
        self.corpus.add_word("<end>")
        self.corpus.add_word("<pad>")

    def load_corpus(self, corpus_file):
        """
        Loads vocabulary encoder from pickle file
        :param corpus_file: pickle file containing object of this class
        """
        with open(corpus_file, "rb") as in_f:
            corpus = pickle.load(in_f)

        self.corpus = corpus[0]
        self.embedding = corpus[1]

    def save_corpus(self, corpus_file):
        """
        Saves vocabulary encoder object in pickle file
        :param corpus_file: path for pickle file
        """
        self.cpu()
        corpus = [
            self.corpus,
            self.embedding
        ]
        self.to(self.device)
        with open(corpus_file, "wb") as out_f:
            pickle.dump(corpus, out_f)