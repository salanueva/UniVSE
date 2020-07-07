import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
from helper import vsts_captions as vsts
from models.simplified_univse import model as simp_univse
from models.univse import model as univse


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute embeddings of vSTS dataset')

    parser.add_argument(
        "--model",
        type=str,
        choices=["vse++", "univse", "simp_univse"],
        default="univse",
        help='Name of the model you want to use to precompute embeddings. Choices are: "vse++" (not implemented yet),'
             '"simp_univse" and "univse".'
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["img", "sent"],
        default="sent",
        help='Name of the modality you want to precompute. Choices are: "img" or "sent".'
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help='Path where vSTS data is found.'
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help='Path where model is found.'
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help='Path where scene graphs of captions are found.'
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help='Path for output files.'
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        help='Path where pickle file with vocabulary encoder is found.'
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("A) Load data")
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

    all_data = vsts.VstsCaptions(root=args.data_path, transform=transform, split="all")

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "simp_univse":
        model = simp_univse.UniVSE(args.vocab_path)
        model.load_model(args.model_path)
    elif args.model == "univse":
        model = univse.UniVSE(args.vocab_path)
        model.load_model(args.model_path)
        model.vocabulary_encoder.add_graphs(args.graph_path)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.val_start()

    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 6
    }
    generator = data.DataLoader(all_data, **params)

    # Obtain embeddings
    embs1 = []
    embs2 = []
    sims = []
    for batch in tqdm(generator, total=len(generator)):

        img_1, sent_1, img_2, sent_2, sim = batch

        embeddings_1 = model(img_1, list(sent_1))
        embeddings_2 = model(img_2, list(sent_2))

        if args.modality == "img":
            current_emb_1 = embeddings_1["img_emb"]
            current_emb_2 = embeddings_2["img_emb"]
        else:
            if args.model == "univse":
                current_emb_1 = embeddings_1["cap_emb"]
                current_emb_2 = embeddings_2["cap_emb"]
            elif args.model == "simp_univse":
                current_emb_1 = embeddings_1["sent_emb"]
                current_emb_2 = embeddings_2["sent_emb"]
            else:
                raise ValueError

        embs1.append(current_emb_1)
        embs2.append(current_emb_2)
        sims.append(sim.data.numpy()[0])

    embs1 = np.asarray(embs1)
    embs2 = np.asarray(embs2)
    sims = np.asarray(sims)

    # Check emb sizes
    print(f"Emb. 1: {embs1.shape}")
    print(f"Emb. 2: {embs2.shape}")
    print(f"Sim: {sims.shape}")

    # Save embedding
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    np.save(args.output_path + '/' + args.model + '_' + args.modality + '_sent1.npy', embs1)
    np.save(args.output_path + '/' + args.model + '_' + args.modality + '_sent2.npy', embs2)
    np.save(args.output_path + '/sim.npy', sims)


if __name__ == '__main__':
    main()
