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
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils import data
import torchvision
from torchvision import transforms

sys.path.append(os.getcwd())
from models.univse import model as univse
from models.simplified_univse import model as simp_univse
from helper import image_text_retrieval as itr


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
        default=25,
        help='Number of epochs for the pre-training process.'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help='Initial learning rate of the fine-tuning process.'
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


def plot_loss_curve(par_values, train_scores, dev_scores, title="Loss Curve", xlab="Epoch", ylab="Loss", yexp=False):
    """
    Generate a simple plot of the test and training learning curve.
    :param par_values: list of checked values of the current parameter.
    :param train_scores : list of scores obtained in training set (same length as par_values).
    :param dev_scores : list of scores obtained in dev set (same length as par_values)
    :param title : title for the chart.
    :param xlab: name of horizontal axis
    :param ylab: name of vertical axis
    :param yexp : True for exponential vertical axis, False otherwise
    :return Defines minimum and maximum yvalues plotted.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.grid()
    plt.plot(par_values, train_scores, color="r", label="Training loss")
    plt.plot(par_values, dev_scores, color="g", label="Dev loss")

    if yexp:
        plt.yscale("log")

    plt.legend(loc="best")
    return plt


def plot_recall_curve(par_values, r1, r5, r10, title="R@k Curve", xlab="Epoch", ylab="R@k", yexp=False):
    """
    Generate a simple plot of the test and training learning curve.
    :param par_values: list of checked values of the current parameter.
    :param r1: recall@1 values for each epoch in dev set.
    :param r5: recall@5 values for each epoch in dev set.
    :param r10: recall@10 values for each epoch in dev set.
    :param title : title for the chart.
    :param xlab: name of horizontal axis
    :param ylab: name of vertical axis
    :param yexp : True for exponential vertical axis, False otherwise
    :return Defines minimum and maximum yvalues plotted.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.grid()
    plt.plot(par_values, r1, color="r", label="R@1")
    plt.plot(par_values, r5, color="g", label="R@5")
    plt.plot(par_values, r10, color="b", label="R@10")

    if yexp:
        plt.yscale("log")

    plt.legend(loc="best")
    return plt


def main():

    args = parse_args()
    csv.field_size_limit(sys.maxsize)

    print("A) Load data")
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    train_data = torchvision.datasets.CocoCaptions(args.train_img_path, args.train_ann_file, transform=transform,
                                                   target_transform=None, transforms=None)
    dev_data = torchvision.datasets.CocoCaptions(args.dev_img_path, args.dev_ann_file, transform=transform,
                                                 target_transform=None, transforms=None)

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "univse":
        if args.simple:
            print("Simple")
            model = simp_univse.UniVSE.from_filename(args.vocab_file)
        else:
            model = univse.UniVSE.from_filename(args.vocab_file)
        # Randomize modifier
        model.vocabulary_encoder.modif = torch.nn.Embedding(len(model.vocabulary_encoder.corpus), 100)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.params, lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)

    optimizer_late = optim.Adam(model.params, lr=1e-5)

    print("C) Train model")
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 6}

    train_gen = data.DataLoader(train_data, **params)
    dev_gen = data.DataLoader(dev_data, **params)

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

    optimizer_changed = False
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch"):
    
        if epoch > 2:
            model.criterion.n_r = 1.0
            if epoch >= 6 and not optimizer_changed:
                lr_scheduler.step(epoch - 5)
                if lr_scheduler.get_lr()[0] < 1e-5:
                    optimizer = optimizer_late
                    optimizer_changed = True

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

            # Iterate over data.
            t = tqdm(generator, desc="Batch", leave=False)
            for img, sent in t:

                sentences = [
                    sent[idx][enum]
                    for enum, idx in enumerate(list(np.random.randint(5, size=len(img))))
                ]
                embeddings = model(img, sentences)
                total_loss, _ = model.criterion(embeddings)

                if phase == "train":
                    optimizer.zero_grad()
                    total_loss.backward()
                    if model.grad_clip > 0:
                        clip_grad_norm(model.params, model.grad_clip)
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

                # TODO: In case of true, the code could be better optimized
                if args.recall:
                    # Compute R@k values
                    img_embs, cap_embs = itr.encode_data(model, dev_data)
                    rt = itr.i2t(img_embs, cap_embs, measure='cosine', return_ranks=False)
                    ri = itr.t2i(img_embs, cap_embs, measure='cosine', return_ranks=False)

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
                plot_loss_curve(range(1, epoch + 1), train_losses, dev_losses, yexp=True)
                plt.savefig(os.path.join(args.output_path, f"training_losses_{args.model}.png"))
                plt.close()

                if args.recall:
                    plot_recall_curve(range(1, epoch + 1), ir_r1, ir_r5, ir_r10, title="Image Retrieval")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_ir.png"))
                    plt.close()

                    plot_recall_curve(range(1, epoch + 1), tr_r1, tr_r5, tr_r10, title="Text Retrieval")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_tr.png"))
                    plt.close()

    model.load_state_dict(best_model_wts)
    model.save_model(os.path.join(args.output_path, f"best_{args.model}.pth"))

    model.vocabulary_encoder.modif = best_modif_emb
    model.vocabulary_encoder.save_corpus(os.path.join(args.output_path, f"best_learned_corpus_{args.model}.pickle"))

    plot_loss_curve(range(1, args.epochs + 1), train_losses, dev_losses, yexp=True)
    plt.savefig(os.path.join(args.output_path, f"training_losses_{args.model}.png"))
    plt.close()

    if args.recall:
        plot_recall_curve(range(1, args.epochs + 1), ir_r1, ir_r5, ir_r10, title="Image Retrieval")
        plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_ir.png"))
        plt.close()

        plot_recall_curve(range(1, args.epochs + 1), tr_r1, tr_r5, tr_r10, title="Text Retrieval")
        plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_tr.png"))
        plt.close()


if __name__ == '__main__':
    main()
