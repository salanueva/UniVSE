import argparse
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
from models.simplified_univse import model as simp_univse
from models.univse import model as univse
from models.vsepp import model as vsepp
from models.univse.corpus import CocoCaptions
from helper import image_text_retrieval as itr, plotter


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain a given VSE model with MS-COCO dataset')

    parser.add_argument(
        "--model",
        type=str,
        choices=["vse++", "univse"],
        default="univse",
        help='Name of the model you want to fine-tune. Choices are: "vse++" (not implemented yet) and "univse".'
    )
    parser.add_argument(
        '--recall',
        default=False,
        action='store_true',
        help='Use it if you want to compute R@k values on each epoch and create a plot at the end of the execution.'
    )
    parser.add_argument(
        '--simple',
        default=False,
        action='store_true',
        help='Use it if you want to use a simplified version of UniVSE. False by default.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs for the pre-training process.'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help='Initial learning rate of the fine-tuning process.'
    )
    parser.add_argument(
        "--lr_update",
        type=int,
        default=15,
        help='Number of epochs to update the learning rate.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Number of image and sentence pairs per batch.'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=1024,
        help='Embedding sizes in hidden layers. It will be the size of the UniVSE/VSE++ space.'
    )

    parser.add_argument(
        "--train-img-path",
        type=str,
        help='Path where training images are stored.'
    )
    parser.add_argument(
        "--dev-img-path",
        type=str,
        help='Path where development images are stored.'
    )
    parser.add_argument(
        "--train-ann-file",
        type=str,
        help='File where captions from training images are stored (following MS-COCO data structure).'
    )
    parser.add_argument(
        "--dev-ann-file",
        type=str,
        help='File where captions from development images are stored (following MS-COCO data structure).'
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help='Path of the file with vocabulary and pre-trained embeddings.'
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help='Path for output files.'
    )

    return parser.parse_args()


def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')


def main():

    args = parse_args()
    csv.field_size_limit(sys.maxsize)

    print("A) Load data")
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    train_data = CocoCaptions(args.train_img_path, args.train_ann_file, transform=transform,
                              target_transform=None, transforms=None)
    dev_data = CocoCaptions(args.dev_img_path, args.dev_ann_file, transform=transform,
                            target_transform=None, transforms=None)

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "univse":
        if args.simple:
            model = simp_univse.UniVSE.from_filename(args.vocab_file)
        else:
            model = univse.UniVSE.from_filename(args.vocab_file)
        # Randomize modifier
        model.vocabulary_encoder.modif = torch.nn.Embedding(len(model.vocabulary_encoder.corpus), 100)
        model.vocabulary_encoder.modif.weight.data[model.vocabulary_encoder.train_corpus_length:] = torch.zeros(
            (len(model.vocabulary_encoder.corpus) - model.vocabulary_encoder.train_corpus_length, 100)
        )
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.params, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_update], gamma=0.1)

    print("C) Train model")
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 6
    }
    train_gen = data.DataLoader(train_data, **train_params)

    dev_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 6
    }
    dev_gen = data.DataLoader(dev_data, **dev_params)

    train_losses = []
    dev_losses = []

    ir_r1 = []
    ir_r5 = []
    ir_r10 = []

    tr_r1 = []
    tr_r5 = []
    tr_r10 = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_modif_emb = copy.deepcopy(model.vocabulary_encoder.modif)
    best_loss = 1e10

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch"):

        lr_scheduler.step(epoch - 1)
        if epoch > 2:
            model.criterion.n_r = 1.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                generator = train_gen
                model.train_start()  # Set model to training mode
            else:
                generator = dev_gen
                model.val_start()  # Set model to evaluate mode

            running_loss = 0.0
            idx = 0

            img_embeddings = np.zeros((len(dev_data), args.hidden_size))
            cap_embeddings = np.zeros((len(dev_data), args.hidden_size))
            count = 0

            # Iterate over data.
            t = tqdm(generator, desc="Batch", leave=False)
            for img, sent in t:

                sentences = list(sent)
                embeddings = model(img, sentences)
                total_loss, _ = model.criterion(embeddings)

                if phase == "dev" and args.recall:
                    aux_count = count + embeddings["sent_emb"].size(0)
                    img_embeddings[count:aux_count] = embeddings["img_emb"].data.cpu().numpy().copy()
                    cap_embeddings[count:aux_count] = embeddings["sent_emb"].data.cpu().numpy().copy()
                    count = aux_count

                if phase == "train":
                    optimizer.zero_grad()
                    total_loss.backward()
                    if model.grad_clip > 0:
                        clip_grad_norm_(model.params, model.grad_clip)
                    optimizer.step()

                total_loss = float(total_loss.data.cpu().numpy())
                t.set_description(f"Batch Loss: {total_loss:.6f}")
                running_loss += total_loss
                idx += 1

            running_loss /= idx

            if phase == "train":
                train_losses.append(running_loss)
            else:
                dev_losses.append(running_loss)

                if args.recall:
                    # Compute R@k values
                    rt = itr.i2t(img_embeddings, cap_embeddings, measure='cosine', return_ranks=False)
                    ri = itr.t2i(img_embeddings, cap_embeddings, measure='cosine', return_ranks=False)

                    ir_r1.extend([ri[0]])
                    ir_r5.extend([ri[1]])
                    ir_r10.extend([ri[2]])

                    tr_r1.extend([rt[0]])
                    tr_r5.extend([rt[1]])
                    tr_r10.extend([rt[2]])

            # deep copy the model
            if running_loss < best_loss:
                del best_modif_emb, best_model_wts
                best_loss = running_loss
                best_modif_emb = copy.deepcopy(model.vocabulary_encoder.modif)
                best_model_wts = copy.deepcopy(model.state_dict())

            # Save intermediate loss and recall plots after the second epoch
            if phase == "dev" and epoch > 1:
                fig = plotter.plot_loss_curve(range(1, epoch + 1), train_losses, dev_losses, yexp=True)
                plt.savefig(os.path.join(args.output_path, f"training_losses_{args.model}.png"))
                plt.close(fig)

                if args.recall:
                    fig = plotter.plot_recall_curve(range(1, epoch + 1), ir_r1, ir_r5, ir_r10, title="Image Retrieval")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_ir.png"))
                    plt.close(fig)

                    fig = plotter.plot_recall_curve(range(1, epoch + 1), tr_r1, tr_r5, tr_r10, title="Text Retrieval")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_tr.png"))
                    plt.close(fig)

    model.load_state_dict(best_model_wts)
    model.save_model(os.path.join(args.output_path, f"best_{args.model}.pth"))

    model.vocabulary_encoder.modif = best_modif_emb
    model.vocabulary_encoder.save_corpus(os.path.join(args.output_path, f"best_learned_corpus_{args.model}.pickle"))


if __name__ == '__main__':
    main()
