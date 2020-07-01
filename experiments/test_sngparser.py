import argparse
import os
import pandas as pd
from pycocotools.coco import COCO
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from helper import sng_parser


def parse_args():

    parser = argparse.ArgumentParser(description='Pretrain a given VSE model with MS-COCO dataset')

    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help='Number of caption parsed from ann_file (all of them will be parsed by default).'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        help="Annotation file that follows format of COCO dataset's ann_files."
    )

    return parser.parse_args()


def main():

    args = parse_args()
    parser = sng_parser.Parser('spacy', model='en')

    ann_file = args.ann_file
    ann = COCO(ann_file)

    # SCENE_GRAPH_PARSER
    num_obj = []
    num_rel = []

    n_caption = len(list(ann.anns.keys()))
    if args.n <= 0 or args.n > n_caption:
        n = n_caption
    else:
        n = args.n

    for k in tqdm(list(ann.anns.keys())[:n], desc="Parsing"):
        graph = parser.parse(ann.anns[k]["caption"])
        num_obj.append(len(graph["entities"]))
        num_rel.append(len(graph["relations"]))

    df = pd.DataFrame(list(zip(num_obj, num_rel)), columns=['Objects', 'Relations'])

    print("SCENE GRAPH PARSER (Vacancy)")
    print(df.describe())
    print(f"\nCaptions with 0 objects: {num_obj.count(0) * 100 / len(num_obj):.3f}%")
    print(f"Captions with 0 relations: {num_rel.count(0) * 100 / len(num_rel):.3f}%\n")

    print("USAGE (second caption as example)")
    graph = parser.parse(ann.anns[list(ann.anns.keys())[1]]["caption"])
    print(sng_parser.tprint(graph))
    print(f"Object list: {[graph['entities'][i]['head'] for i in range(len(graph['entities']))]}")
    print(f"Relation list: {[graph['relations'][i] for i in range(len(graph['relations']))]}")


if __name__ == '__main__':
    main()
