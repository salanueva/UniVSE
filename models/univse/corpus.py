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
            stack = torch.cat([stack_basic, stack_modif], dim=1)
        else:
            print("WARNING: Unknown input type in vocabulary encoder.")

        return stack

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

            graph = self.parser.parse(cap)

            _ = [
                self.corpus.add_word(nltk.word_tokenize(graph["entities"][i]['head'])[0].lower())
                for i in range(len(graph["entities"]))
            ]
            _ = [
                self.corpus.add_word(nltk.word_tokenize(graph["entities"][i]['modifiers'][j]['span'])[0].lower())
                for i in range(len(graph["entities"]))
                for j in range(len(graph["entities"][i]['modifiers']))
                if graph["entities"][i]['modifiers'][j]['dep'] == 'amod'
                   or graph["entities"][i]['modifiers'][j]['dep'] == 'nummod'
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

        self.train_corpus_length = len(self.corpus)

        self.corpus.add_word("<unk>")
        self.corpus.add_word("<start>")
        self.corpus.add_word("<end>")
        self.corpus.add_word("<pad>")

    def extract_components(self, captions):
        """
        Given a list of sentences, extract objects, attributes and relations that appear on them
        :param captions: list of sentences/captions
        :return: three elements: list of objects, list of attributes and list of relations
        (each element in a list is grouped in sentences)
        """
        objects = []
        attributes = []
        relations = []

        # Parse each caption and append its objects, attributes and relations to the output lists
        for cap in captions:

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

            objects.append(o)
            attributes.append(a)
            relations.append(r)

        return objects, attributes, relations

    def get_components(self, captions):
        """
        Extracts all needed ids from a list of captions
        :param captions: list of sentences/captions
        :return: dictionary with all ids, both positive and negative samples
        """
        components = {}

        # Parse captions in order to get ids of word/tokens
        caption_words = [nltk.word_tokenize(cap.lower()) for cap in captions]
        components["words"] = [[self.corpus.word2idx[w] for w in words] for words in caption_words]

        # Parse captions in order to get their objects, attributes and relations
        components["obj"], components["attr"], components["rel"] = self.extract_components(captions)

        # Generate negative instances for objects, attributes and relations
        components["n_obj"], components["n_attr_n"], components["n_attr_a"], components["n_rel"] = \
            self.negative_instances(components["obj"], components["attr"], components["rel"])

        return components

    def load_corpus(self, corpus_file):
        """
        Loads vocabulary encoder from pickle file
        :param corpus_file: pickle file containing object of this class
        """
        with open(corpus_file, "rb") as in_f:
            corpus = pickle.load(in_f)

        if len(corpus) != 6:
            IOError("Vocab_file must have the right format.")

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
        self.cpu()
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
        self.to(self.device)

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

    def negative_instances(self, objects, attributes, relations):
        """
        Given lists of objects, attributes and relations, generates their random negative samples.
        :param objects: list of objects grouped by sentences
        :param attributes: list of objects grouped by attributes
        :param relations: list of objects grouped by relations
        :return: three lists containing negative samples of objects, attributes and relations
        """

        # Generate Negative Objects
        # 16 negative samples for each positive object
        neg_objects = []
        for caption in objects:
            neg_obj_in_cap = []
            for obj in caption:
                rand_obj = random.sample(self.neg_obj, min([16, len(self.neg_obj)]))
                while obj in rand_obj:
                    ind = rand_obj.index(obj)
                    rand_obj[ind] = random.sample(self.neg_obj, 1)[0]
                neg_obj_in_cap.append(rand_obj)
            neg_objects.append(neg_obj_in_cap)

        # Generate Negative Attributes
        # 16 negative samples for each positive object-attribute pair:
        neg_attributes_noun = []  # 8 changing its object
        neg_attributes_attr = []  # Another 8 changing its attribute

        for caption in attributes:
            neg_attr_n_in_cap = []
            neg_attr_a_in_cap = []

            for obj, attr in caption:
                rand_obj = random.sample(self.neg_obj, min([8, len(self.neg_obj)]))
                while obj in rand_obj:
                    ind = rand_obj.index(obj)
                    rand_obj[ind] = random.sample(self.neg_obj, 1)[0]
                neg_attr_n_in_cap.append([(r_o, attr) for r_o in rand_obj])

                rand_attr = random.sample(self.neg_attr, min([8, len(self.neg_attr)]))
                while attr in rand_attr:
                    ind = rand_attr.index(attr)
                    rand_attr[ind] = random.sample(self.neg_attr, 1)[0]
                neg_attr_a_in_cap.append([(obj, r_a) for r_a in rand_attr])

            neg_attributes_noun.append(neg_attr_n_in_cap)
            neg_attributes_attr.append(neg_attr_a_in_cap)

        # Generate Negative Relations
        # 8 negative relation for each positive relation
        # All types grouped in the following list
        # 2 changing its noun, 4 changing its relation and another 2 changing its object
        neg_relations = []

        for caption in relations:
            neg_rel_in_cap = []

            for sub, rel, obj in caption:
                rand_sub = random.sample(self.neg_obj, min([2, len(self.neg_obj)]))
                while sub in rand_sub:
                    ind = rand_sub.index(sub)
                    rand_sub[ind] = random.sample(self.neg_obj, 1)[0]

                rand_rel = random.sample(self.neg_rel, min([4, len(self.neg_rel)]))
                while rel in rand_rel:
                    ind = rand_rel.index(rel)
                    rand_rel[ind] = random.sample(self.neg_rel, 1)[0]

                rand_obj = random.sample(self.neg_obj, min([2, len(self.neg_obj)]))
                while obj in rand_obj:
                    ind = rand_obj.index(obj)
                    rand_obj[ind] = random.sample(self.neg_obj, 1)[0]

                neg_rel = [[r_s, rel, obj] for r_s in rand_sub] + \
                          [[sub, r_r, obj] for r_r in rand_rel] + \
                          [[sub, rel, r_o] for r_o in rand_sub]
                neg_rel_in_cap.append(neg_rel)

            neg_relations.append(neg_rel_in_cap)

        return neg_objects, neg_attributes_noun, neg_attributes_attr, neg_relations
