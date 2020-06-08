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
from helper import sng_parser


class VocabularyEncoder(nn.Module):

    def __init__(self, captions=None, glove_file=None):
        """
        Creates vocabulary for the UniVSE model. If a token of the vocabulary doesn't appear
        within glove_path, a random embedding is generated.
        :param captions: list of sentences that will be parsed to extract all possible tokens
        :param glove_file: file from which we extract embeddings for all tokens
        """
        super(VocabularyEncoder, self).__init__()

        self.parser = sng_parser.Parser('spacy', model='en')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.corpus = []
        self.word_ids = {}

        self.basic = []
        self.modif = []

        if captions is not None and glove_file is not None:
            self.define_corpus_nouns_attributes_and_relations(captions)

            self.basic = self.load_glove_embeddings(glove_file)
            self.modif = nn.Embedding(len(self.corpus), 100)

    def forward(self, word_ids):
        """
        Converts a list of token ids into a list of tensors with their respective embeddings
        :param word_ids: list of: ids, tuple ids or lists of ids. It depends on which embedding
        type we want to create: object, attribute or relation embeddings.
        :return: a list of tensors containing embeddings of the input ids
        """
        stack = None
        if word_ids is None or not word_ids:
            return stack

        if isinstance(word_ids[0], int):
            stack = torch.stack([
                torch.cat((self.basic[idx].to(self.device), self.modif(torch.tensor(idx).to(self.device))))
                for idx in word_ids
            ])
        elif isinstance(word_ids[0], tuple):
            stack = torch.stack([
                torch.cat((self.basic[obj_id].to(self.device), self.modif(torch.tensor(attr_id).to(self.device))))
                for obj_id, attr_id in word_ids
            ])
        elif isinstance(word_ids[0], list):
            stack = torch.stack([
                torch.stack([
                    torch.cat((self.basic[idx].to(self.device), self.modif(torch.tensor(idx).to(self.device))))
                    for idx in ids
                ])
                for ids in word_ids
            ])
        else:
            print("WARNING: Unknown input type in vocabulary encoder.")

        return stack

    def define_corpus_nouns_attributes_and_relations(self, captions):
        """
        It parses all captions that will be used, detect which tokens will be needed and define
        nouns, attributes and relations that will be used in generated negative cases.
        :param captions: list of sentences
        """

        # Parse all captions to list all possible tokens and nouns
        for cap in tqdm(captions, desc="Creating Corpus"):
            tokenized_cap = nltk.word_tokenize(cap.lower())
            for token in tokenized_cap:
                if token not in self.corpus:
                    self.word_ids[token] = len(self.corpus)
                    self.corpus.append(token)

    def get_embeddings(self, captions):
        """
        Extracts all needed ids from a list of captions
        :param captions: list of sentences/captions
        :return: dictionary with all ids, both positive and negative samples
        """
        components = {}

        # Parse captions in order to get ids of word/tokens
        caption_words = [nltk.word_tokenize(cap.lower()) for cap in captions]
        components["words"] = [[self.word_ids[w] for w in words] for words in caption_words]

        return components

    def load_corpus(self, corpus_file):
        """
        Loads vocabulary encoder from pickle file
        :param corpus_file: pickle file containing object of this class
        """
        with open(corpus_file, "rb") as in_f:
            corpus = pickle.load(in_f)

        self.corpus = corpus[0]
        self.word_ids = corpus[1]

        self.basic = corpus[5]
        self.modif = corpus[6]

    def save_corpus(self, corpus_file):
        """
        Saves vocabulary encoder object in pickle file
        :param corpus_file: path for pickle file
        """
        self.cpu()
        corpus = [
            self.corpus,
            self.word_ids,
            None,
            None,
            None,
            self.basic,
            self.modif
        ]
        self.to(self.device)
        with open(corpus_file, "wb") as out_f:
            pickle.dump(corpus, out_f)

    def load_glove_embeddings(self, glove_file):
        """
        Load glove embeddings from given file
        :param glove_file: file containing words with their respective embeddings
        """
        glove = {}
        embedding = None
        # Read and create dictionary with glove embeddings
        with open(glove_file, 'r') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Load GloVe Embeddings"):
                split_line = line.split(' ')
                word = split_line[0]
                if word in self.corpus:
                    embedding = np.array([float(val) for val in split_line[1:]])
                    glove[word] = torch.from_numpy(embedding).float()

        weights_matrix = []
        words_found = 0

        # Link each token of our vocabulary with glove embeddings.
        # If a token hasn't got an embedding, create one randomly.
        for word in self.corpus:
            try:
                weights_matrix.append(glove[word])
                words_found += 1
            except KeyError:
                weights_matrix.append(torch.from_numpy(np.random.normal(scale=0.6, size=(len(embedding),))).float())

        print(f"{words_found}/{len(self.corpus)} words in GloVe ({words_found * 100 / len(self.corpus):.2f}%)")

        return weights_matrix
