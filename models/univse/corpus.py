import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.init
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from helper import sng_parser


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

        self.corpus = Corpus()

        self.basic = []
        self.modif = []

        self.neg_obj = []
        self.neg_attr = []
        self.neg_rel = []

        self.graphs = {}

        self.train_corpus_length = 0

        if captions is not None and glove_file is not None:
            self.define_corpus_nouns_attributes_and_relations(captions)
            self.basic = self.load_glove_embeddings(glove_file)
            self.modif = nn.Embedding(len(self.corpus), 100)
            self.modif.weight.data.uniform_(-0.1, 0.1)
            self.modif.weight.data[self.train_corpus_length:] = torch.zeros(
                (len(self.corpus) - self.train_corpus_length, 100)
            )

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
            stack_basic = torch.stack([
                self.basic[idx].to(self.device)
                for idx in word_ids
            ])
            stack_modif = self.modif(torch.tensor(word_ids).to(self.device))
            stack = torch.cat([stack_basic, stack_modif], dim=1)
        elif isinstance(word_ids[0], tuple):
            stack_basic = torch.stack([
                self.basic[idx].to(self.device)
                for idx, _ in word_ids
            ])
            modif_ids = [idx for _, idx in word_ids]
            stack_modif = self.modif(torch.tensor(modif_ids).to(self.device))
            stack = torch.cat([stack_basic, stack_modif], dim=1)
        elif isinstance(word_ids[0], list):
            stack_basic = torch.stack([
                torch.stack([
                    self.basic[idx].to(self.device)
                    for idx in ids
                ])
                for ids in word_ids
            ])
            stack_modif = self.modif(torch.tensor(word_ids).to(self.device))
            stack = torch.cat([stack_basic, stack_modif], dim=2)
        else:
            print("WARNING: Unknown input type in vocabulary encoder.")

        return stack

    def add_graphs(self, graph_file):
        """
        Loads file with a dictionary where keys are captions and their values are their
        corresponding scene graphs
        :param graph_file: pickle file containing graphs extracted from Scene Graph Parser
        """
        with open(graph_file, "rb") as in_f:
            if len(self.graphs.keys()) == 0:
                self.graphs = pickle.load(in_f)
            else:
                aux_graphs = pickle.load(in_f)
                for k, v in aux_graphs.items():
                    self.graphs[k] = v

    def define_corpus_nouns_attributes_and_relations(self, captions):
        """
        It parses all captions that will be used, detect which tokens will be needed and define
        nouns, attributes and relations that will be used in generated negative cases.
        :param captions: list of sentences
        """
        # In order to detect nouns, use the following as our dictionary of all possible nouns
        nouns = [x.name().split('.', 1)[0].lower() for x in wn.all_synsets('n')]
        found_nouns = {k: 0 for k in nouns}
        found_relations = {}

        # Parse all captions to list all possible tokens and nouns
        for cap in tqdm(captions, desc="Creating Corpus"):
            tokenized_cap = nltk.word_tokenize(cap.lower())
            for token in tokenized_cap:
                if token in found_nouns:
                    found_nouns[token] += 1
                self.corpus.add_word(token)

            if cap in self.graphs:
                graph = self.graphs[cap]
            else:
                graph = self.parser.parse(cap)

            _ = [
                self.corpus.add_word(nltk.word_tokenize(graph["entities"][i]['head'])[0].lower())
                for i in range(len(graph["entities"]))
            ]
            _ = [
                self.corpus.add_word(nltk.word_tokenize(graph["entities"][i]['modifiers'][j]['span'])[0].lower())
                for i in range(len(graph["entities"]))
                for j in range(len(graph["entities"][i]['modifiers']))
                if graph["entities"][i]['modifiers'][j]['dep'] == 'amod' or
                graph["entities"][i]['modifiers'][j]['dep'] == 'nummod'
            ]

            relations = [
                nltk.word_tokenize(graph['relations'][i]['relation'])[0].lower()
                for i in range(len(graph["relations"]))
            ]

            for rel in relations:
                if rel in found_relations:
                    found_relations[rel] += 1
                else:
                    found_relations[rel] = 1
                self.corpus.add_word(rel)

        # Used negative attributes are given in their paper
        neg_attr_words = ["white", "black", "red", "green", "brown", "yellow", "orange", "pink", "gray", "grey",
                          "purple", "young", "wooden", "old", "snowy", "grassy", "cloudy", "colorful", "sunny",
                          "beautiful", "bright", "sandy", "fresh", "morden", "cute", "dry", "dirty", "clean", "polar",
                          "crowded", "silver", "plastic", "concrete", "rocky", "wooded", "messy", "square"]

        # Add these negative attributes to token and noun list (orange can be a noun, for example)
        for token in neg_attr_words:
            if token in found_nouns:
                found_nouns[token] += 1
            self.corpus.add_word(token)

        # Create list of objects, attributes and relations for negative samples
        # Nouns are a special case, as only those that appear 100+ times are counted
        self.neg_obj = [self.corpus.word2idx[k] for k, v in found_nouns.items() if v >= 100]
        self.neg_attr = [self.corpus.word2idx[word] for word in neg_attr_words]
        self.neg_rel = [self.corpus.word2idx[rel] for rel in found_relations]

        self.corpus.add_word("<unk>")
        self.corpus.add_word("<start>")
        self.corpus.add_word("<end>")
        self.corpus.add_word("<pad>")

        self.train_corpus_length = len(self.corpus)

    def extract_components(self, captions):
        """
        Given a list of sentences, extract objects, attributes and relations that appear on them
        :param captions: list of sentences/captions
        :return: dictionary with extracted objects, attributes and relations
        """
        components = {}

        objects = []
        attributes = []
        relations = []

        num_obj = [0] * len(captions)
        num_attr = [0] * len(captions)
        num_rel = [0] * len(captions)

        # Parse each caption and append its objects, attributes and relations to the output lists
        for k, cap in enumerate(captions):

            if cap in self.graphs:
                graph = self.graphs[cap]
            else:
                graph = self.parser.parse(cap)

            cur_obj = graph['entities']

            o = [self.corpus.word2idx[nltk.word_tokenize(cur_obj[i]['head'])[0].lower()] for i in range(len(cur_obj))]
            a = [
                (
                    self.corpus.word2idx[nltk.word_tokenize(cur_obj[i]['head'])[0].lower()],
                    self.corpus.word2idx[nltk.word_tokenize(cur_obj[i]['modifiers'][j]['span'])[0].lower()]
                )
                for i in range(len(cur_obj))
                for j in range(len(cur_obj[i]['modifiers']))
                if cur_obj[i]['modifiers'][j]['dep'] == 'amod' or cur_obj[i]['modifiers'][j]['dep'] == 'nummod'
            ]
            r = [
                [
                    o[graph['relations'][i]['subject']],
                    self.corpus.word2idx[nltk.word_tokenize(graph['relations'][i]['relation'])[0].lower()],
                    o[graph['relations'][i]['object']]
                ]
                for i in range(len(graph['relations']))
            ]

            num_obj[k] = len(o)
            num_attr[k] = len(a)
            num_rel[k] = len(r)

            objects += o
            attributes += a
            relations += r

        components["obj"] = objects
        components["attr"] = attributes
        components["rel"] = relations

        components["num_obj"] = num_obj
        components["num_attr"] = num_attr
        components["num_rel"] = num_rel

        return components

    def get_components(self, captions):
        """
        Extracts all needed ids from a list of captions
        :param captions: list of sentences/captions
        :return: dictionary with all ids, both positive and negative samples
        """

        # Parse captions in order to get their objects, attributes and relations
        components = self.extract_components(captions)

        # Generate negative instances for objects, attributes and relations
        components = self.negative_instances(components)

        # Parse captions in order to get ids of word/tokens
        caption_words = [["<start>"] + nltk.word_tokenize(cap.lower()) + ["<end>"] for cap in captions]
        caption_word_ids = [[self.corpus.word2idx[w] for w in words] for words in caption_words]
        components["num_words"] = [len(words) for words in caption_words]
        components["max_words"] = max(components["num_words"])
        flatten_words = []
        for word_ids in caption_word_ids:
            flatten_words += word_ids
        components["words"] = flatten_words

        return components

    def load_corpus(self, corpus_file):
        """
        Loads vocabulary encoder from pickle file
        :param corpus_file: pickle file containing object of this class
        """
        with open(corpus_file, "rb") as in_f:
            corpus = pickle.load(in_f)

        self.corpus = corpus[0]

        self.basic = corpus[1]
        self.modif = corpus[2]

        self.neg_obj = corpus[3]
        self.neg_attr = corpus[4]
        self.neg_rel = corpus[5]

    def save_corpus(self, corpus_file):
        """
        Saves vocabulary encoder object in pickle file
        :param corpus_file: path for pickle file
        """
        corpus = [
            self.corpus,
            self.basic,
            self.modif.cpu(),
            self.neg_obj,
            self.neg_attr,
            self.neg_rel
        ]
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
                if word in self.corpus.word2idx:
                    embedding = np.array([float(val) for val in split_line[1:]])
                    glove[word] = torch.from_numpy(embedding).float()

        weights_matrix = []
        words_found = 0

        # Link each token of our vocabulary with glove embeddings.
        # If a token hasn't got an embedding, create one randomly.
        for word in self.corpus.word2idx.keys():
            try:
                weights_matrix.append(glove[word])
                words_found += 1
            except KeyError:
                weights_matrix.append(torch.zeros(len(embedding)).float())

        for word in glove.keys():
            if word not in self.corpus.word2idx.keys():
                self.corpus.add_word(word)
                weights_matrix.append(glove[word])

        print(f"{words_found}/{self.train_corpus_length} words in GloVe "
              f"({words_found * 100 / self.train_corpus_length:.2f}%)")

        return weights_matrix

    def negative_instances(self, components):
        """
        Given lists of objects, attributes and relations, generates their random negative samples.
        :param components: dictionary with lists of objects, attributes and relations
        :return: updated components dictionary with negative samples of objects, attributes and relations
        """

        num_obj = min([16, len(self.neg_obj) - 1])
        if num_obj < 16:
            print(f"ERROR: At least 17 negative objects are needed (we have {num_obj}).")
            exit(0)

        num_attr_n = 8
        num_attr_a = min([8, len(self.neg_attr) - 1])
        if num_attr_a < 8:
            print(f"ERROR: At least 9 negative attributes are needed (we have {num_attr_a}).")
            exit(0)

        num_rel = min([4, len(self.neg_rel) - 1])
        if num_rel < 4:
            print(f"ERROR: At least 5 negative relations are needed (we have {num_rel}).")
            exit(0)

        # Generate Negative Objects
        # 16 negative samples for each positive object
        neg_objects = []

        for i, obj in enumerate(components["obj"]):
            rand_obj = random.sample(self.neg_obj, num_obj)
            while obj in rand_obj:
                ind = rand_obj.index(obj)
                rand_obj[ind] = random.sample(self.neg_obj, 1)[0]
            neg_objects += rand_obj

        # Generate Negative Attributes
        # 16 negative samples for each positive object-attribute pair:
        neg_attributes_noun = []  # 8 changing its object
        neg_attributes_attr = []  # Another 8 changing its attribute

        for i, (obj, attr) in enumerate(components["attr"]):

            rand_obj = random.sample(self.neg_obj, 8)
            while obj in rand_obj:
                ind = rand_obj.index(obj)
                rand_obj[ind] = random.sample(self.neg_obj, 1)[0]

            rand_attr = random.sample(self.neg_attr, min([8, len(self.neg_attr)]))
            while attr in rand_attr:
                ind = rand_attr.index(attr)
                rand_attr[ind] = random.sample(self.neg_attr, 1)[0]

            neg_attributes_noun += [(r_o, attr) for r_o in rand_obj]
            neg_attributes_attr += [(obj, r_a) for r_a in rand_attr]

        # Generate Negative Relations
        # 8 negative relation for each positive relation
        # All types grouped in the following list
        # 2 changing its noun, 4 changing its relation and another 2 changing its object
        neg_relations = []

        for i, (sub, rel, obj) in enumerate(components["rel"]):
            rand_sub = random.sample(self.neg_obj, 2)
            while sub in rand_sub:
                ind = rand_sub.index(sub)
                rand_sub[ind] = random.sample(self.neg_obj, 1)[0]

            rand_rel = random.sample(self.neg_rel, 4)
            while rel in rand_rel:
                ind = rand_rel.index(rel)
                rand_rel[ind] = random.sample(self.neg_rel, 1)[0]

            rand_obj = random.sample(self.neg_obj, 2)
            while obj in rand_obj:
                ind = rand_obj.index(obj)
                rand_obj[ind] = random.sample(self.neg_obj, 1)[0]

            neg_rel = [[r_s, rel, obj] for r_s in rand_sub] + \
                      [[sub, r_r, obj] for r_r in rand_rel] + \
                      [[sub, rel, r_o] for r_o in rand_sub]
            neg_relations += neg_rel

        components["num_neg_obj"] = num_obj
        components["num_neg_attr_n"] = num_attr_n
        components["num_neg_attr_a"] = num_attr_a
        components["num_neg_rel"] = 2 + num_rel + 2

        components["neg_obj"] = neg_objects
        components["neg_attr_n"] = neg_attributes_noun
        components["neg_attr_a"] = neg_attributes_attr
        components["neg_rel"] = neg_relations

        return components
