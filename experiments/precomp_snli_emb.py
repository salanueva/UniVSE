import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
from data import snli_ve_dataset as snli
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
        choices=["img", "sent", "cap", "sent+img", "cap+img"],
        default="sent",
        help='Name of the modality you want to precompute. Choices are: "img", "sent", "cap" "sent+img" or "cap+img".'
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help='Path where SNLI-VE data is found.'
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

    ann_path = os.path.join(args.data_path, "snli_ve")
    img_path = os.path.join(args.data_path, "flickr30k_images")
    all_data = snli.SnliVECaptions(data_root=ann_path, img_root=img_path, transform=transform, split="all")

    if "cap" in args.modality and args.model != "univse":
        if "img" in args.modality:
            args.modality = "sent+img"
        else:
            args.modality = "sent"
        print(f"WARNING: modality of embedding changed from 'cap' to 'sent', as {args.model} has only sentence "
              f"embeddings as output.")

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "simp_univse":
        model = simp_univse.UniVSE.from_filename(args.vocab_path)
        model.load_model(args.model_path)
    elif args.model == "univse":
        model = univse.UniVSE.from_filename(args.vocab_path)
        model.load_model(args.model_path)
        model.vocabulary_encoder.add_graphs(args.graph_path)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 6
    }
    generator = data.DataLoader(all_data, **params)

    # Obtain embeddings
    embs = []
    labels = []
    for batch in tqdm(generator, total=len(generator), desc="Instances"):

        img, hypothesis, label = batch

        embeddings = model(img, hypothesis)

        if args.modality == "img":
            current_emb = embeddings["img_emb"]
        elif args.modality == "cap":
            current_emb = embeddings["cap_emb"]
        elif args.modality == "sent":
            current_emb = embeddings["sent_emb"]
        elif args.modality == "cap+img":
            current_emb = torch.cat((embeddings["cap_emb"], embeddings["img_emb"]), dim=1)
        else:  # if args.modality == "sent+img":
            current_emb = torch.cat((embeddings["sent_emb"], embeddings["img_emb"]), dim=1)

        embs.append(current_emb.data.cpu().numpy()[0])
        labels.append(label.data.numpy()[0])

    embs = np.asarray(embs)
    labels = np.asarray(labels)

    # Check emb sizes
    print(f"Embeddings: {embs.shape}")
    print(f"Labels: {labels.shape}")

    # Save embedding
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    np.save(args.output_path + '/snli_' + args.model + '_' + args.modality + '_emb.npy', embs)
    np.save(args.output_path + '/snli_labels.npy', labels)


if __name__ == '__main__':
    main()
