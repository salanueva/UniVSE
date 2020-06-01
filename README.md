# UniVSE (Work In Progress)

UniVSE implementation on Python3 using PyTorch. This implementation has been written following its original paper https://arxiv.org/abs/1904.05521v1. Check its first appendix for more details about the implementation... The code is runnable but doesn't work as intended, as there is a bug that affects the training phase of the model.

The code is divided in 3 different folders:
* Experiments: four runnable scripts.
  * create_corpus.py: Example of how to create your own corpus given a list of sentences and GloVe embeddings found here (https://nlp.stanford.edu/projects/glove/).
  * eval_retrieval.py: Evaluates a trained UniVSE model in image/text-retrieval. Computes recall values using k={1,5,10}.
  * eval_vSTS.py: Evaluates a trained UniVSE model in vSTS task (https://arxiv.org/abs/2004.01894).
  * train.py: Train UniVSE model and creates plots of both loss and recall values on each epoch.
* Helper: you can find some functions that load vSTS dataset, compute recall values on image/text-retrieval and parse sentences to extract objects, attributes and relations.
* Models: VSE++ and UniVSE models are implemented in this folder. Inside models/univse folder 3 files can be found:
  * corpus.py: implementation of the class that creates and stores embeddings of the vocabulary that will be used in UniVSE model.
  * loss.py: definition of multiple loss functions used during the training of the UniVSE model.
  * model.py: implementation of the UniVSE model.

Some functions have been taken from two different repositories:
* Scene Graph Parser: https://github.com/vacancy/SceneGraphParser
* Improving Visual-Semantic Embeddings with Hard Negatives: https://github.com/fartashf/vsepp (from paper: https://arxiv.org/abs/1707.05612)
