import argparse
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from data import vsts_loader as ld
from models.simplified_univse import corpus as simp_univse
from models.univse import corpus as univse

# IMPORTANT: As it is, this script creates vocabulary based on MS-Coco and vSTS datasets


def parse_args():

    parser = argparse.ArgumentParser(description='Create corpus for UniVSE model.')

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "restval", "all", "all+vsts"],
        default="train",
        help='COCO split used to create corpus.'
    )
    parser.add_argument(
        '--simple',
        default=False,
        action='store_true',
        help='Use it if you want to use a simplified version of UniVSE. False by default.'
    )
    parser.add_argument(
        "--filename",
        type=str,
        default='corpus_univse.pickle',
        help='Path for output files.'
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # FIXME: Change argument parser so that directories are given as input parameters of the script
    glove_dir = "/gscratch/users/asalaberria009/embeddings/glove.840B.300d.txt"
    graph_dir = "/gscratch/users/asalaberria009/models/univse/corpus/scene_graphs.pickle"
    train_ann = "/gscratch/users/asalaberria009/datasets/mscoco/annotations/captions_train2014.json"
    val_ann = "/gscratch/users/asalaberria009/datasets/mscoco/annotations/captions_val2014.json"

    sentences = []
    data_1 = COCO(train_ann)
    data_2 = COCO(val_ann)
    data_3 = {}
    data_4 = {}
    data_5 = {}

    if args.split == "restval":
        ids = {k: 0 for k in list(np.load('data/coco_restval_ids.npy'))}
        for k in list(data_2.anns.keys()):
            if k not in ids:
                del data_2.anns[k]
    else:
        if args.split == "all+vsts":
            data_3, data_4, data_5 = ld.download_and_load_vsts_dataset(
                v2=True, images=False, root_path='/gscratch/users/asalaberria009/datasets/vSTS_v2'
            )

    sentences = []
    for idx in tqdm(data_1.anns.keys(), desc="MSCOCO Train"):
        sentences.append(data_1.anns[idx]['caption'])

    if args.split != "train":
        for idx in tqdm(data_2.anns.keys(), desc="MSCOCO Val"):
            sentences.append(data_2.anns[idx]['caption'])

    if args.split == "all+vsts":
        sentences += list(data_3["sent_1"]) + list(data_3["sent_2"]) + \
            list(data_4["sent_1"]) + list(data_4["sent_2"]) + \
            list(data_5["sent_1"]) + list(data_5["sent_2"])

    if args.simple:
        voc_encoder = simp_univse.VocabularyEncoder(sentences, glove_dir)
    else:
        voc_encoder = univse.VocabularyEncoder(sentences, glove_dir, graph_dir)
        print(f"Num Objects: {len(voc_encoder.neg_obj)}")

    voc_encoder.save_corpus(os.path.join('/gscratch/users/asalaberria009/models/univse', args.filename))
