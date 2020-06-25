import argparse
import os
import sys

sys.path.append(os.getcwd())
from helper import image_text_retrieval as itr


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate UniVSE on Image/Text retrieval.')

    parser.add_argument(
        "--model",
        type=str,
        choices=["vse++", "univse", "simp_univse"],
        default="univse",
        help='Name of the model that will be used. Choices are: "vse++" (not implemented yet) and "univse".'
    )
    parser.add_argument(
        "--img-path",
        type=str,
        help='Folder with MS-Coco images.'
    )
    parser.add_argument(
        "--ann-path",
        type=str,
        help='Path of file with MS-Coco captions.'
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help='Path of pre-trained model.'
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        help='Path of vocabulary with pre-trained embeddings.'
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    data_path = (args.img_path, args.ann_path)
    print("### 1K Test Split ###")
    itr.evalrank(args.model_path, args.vocab_path, data_path, args.model, True)
    print("\n### 5K Test Split ###")
    itr.evalrank(args.model_path, args.vocab_path, data_path, args.model, False)
