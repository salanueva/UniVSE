import argparse
import numpy as np
import pickle
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from helper import load_data as ld
from helper import sng_parser


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
        "--ann-path",
        type=str,
        help="Path of COCO dataset's annotation files."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default='',
        help='Path for output file.'
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_ann = os.path.join(args.ann_path, "captions_train2014.json")
    val_ann = os.path.join(args.ann_path, "captions_val2014.json")

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

    graph_dict = {}
    for sent in tqdm(sentences, "Parsing"):
        if sent not in graph_dict:
            graph_dict[sent] = sng_parser.parse(sent)

    with open(os.path.join(args.output_path, "scene_graphs.pickle"), "wb") as out_f:
        pickle.dump(graph_dict, out_f)
