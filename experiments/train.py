import argparse
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
from tqdm import tqdm

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
from helper.coco_captions import CocoCaptions
from helper import image_text_retrieval as itr, plotter
from models.simplified_univse import model as simp_univse
from models.univse import model as univse
from models.vsepp import model as vsepp


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
        '--plot',
        default=False,
        action='store_true',
        help='Use it if you want to create plots of R@k values and loss values during training.'
    )
    parser.add_argument(
        '--simple',
        default=False,
        action='store_true',
        help='Use it if you want to use a simplified version of UniVSE. False by default.'
    )
    parser.add_argument(
        '--restval',
        default=False,
        action='store_true',
        help='Restval instances will be used for training.'
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
        "--graph-file",
        type=str,
        help='Path of the file with precomputed scene graphs of different captions.'
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

    if args.restval:
        train_data = CocoCaptions(
            (args.train_img_path, args.dev_img_path),
            (args.train_ann_file, args.dev_ann_file),
            transform=transform, target_transform=None, transforms=None, split="restval"
        )
    else:
        train_data = CocoCaptions(
            args.train_img_path,
            args.train_ann_file,
            transform=transform, target_transform=None, transforms=None, split="train"
        )

    dev_data = CocoCaptions(
        args.dev_img_path,
        args.dev_ann_file,
        transform=transform, target_transform=None, transforms=None, split="dev"
    )

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "univse":
        if args.simple:
            model = simp_univse.UniVSE.from_filename(args.vocab_file)
        else:
            model = univse.UniVSE.from_filename(args.vocab_file)
            model.vocabulary_encoder.add_graphs(args.graph_file)
        # Randomize modifier
        model.vocabulary_encoder.modif = torch.nn.Embedding(len(model.vocabulary_encoder.corpus), 100)
        model.vocabulary_encoder.modif.weight.data.uniform_(-0.1, 0.1)
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

    ir_r1_1k = []
    ir_r5_1k = []
    ir_r10_1k = []

    tr_r1_1k = []
    tr_r5_1k = []
    tr_r10_1k = []

    ir_r1_5k = []
    ir_r5_5k = []
    ir_r10_5k = []

    tr_r1_5k = []
    tr_r5_5k = []
    tr_r10_5k = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_modif_emb = copy.deepcopy(model.vocabulary_encoder.modif)
    best_rsum = 0

    t_epoch = tqdm(range(1, args.epochs + 1), desc="Epoch")
    for epoch in t_epoch:

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
            t_batch = tqdm(generator, desc="Batch", leave=False)
            for img, sent in t_batch:

                sentences = list(sent)
                embeddings = model(img, sentences)

                time_start = time.time()
                total_loss, _ = model.criterion(embeddings)
                model.times["loss"] += time.time() - time_start

                # ####### DEBUG ######## #
                if idx % 100 == 1:
                    with open("times.txt", "a+") as t_file:
                        t_file.write(f" # EPOCH {epoch}\t# BATCH {idx} #\n")
                        t_file.write(f"Image:  {model.times['image'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Input:  {model.times['input'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Vocab:  {model.times['vocab'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Object: {model.times['object'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Neural: {model.times['neural'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Compos: {model.times['comp'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Unflat: {model.times['unflatten'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"Loss:   {model.times['loss'] * 1000 / model.times['n']} ms\n")
                        t_file.write(f"\n")

                if phase == "dev":
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
                    lr_scheduler.step(epoch - 1)

                total_loss = float(total_loss.data.cpu().numpy())
                t_batch.set_description(f"Batch Loss: {total_loss:.6f}")
                running_loss += total_loss
                idx += 1

            running_loss /= idx

            if phase == "train":
                train_losses.append(running_loss)
            else:
                dev_losses.append(running_loss)

                # Compute R@k values for 1K Validation
                rt = itr.i2t(img_embeddings[:5000], cap_embeddings[:5000], measure='cosine', return_ranks=False)
                ri = itr.t2i(img_embeddings[:5000], cap_embeddings[:5000], measure='cosine', return_ranks=False)
                current_rsum_1k = ri[0] + ri[1] + ri[2] + rt[0] + rt[1] + rt[2]

                ir_r1_1k.extend([ri[0]])
                ir_r5_1k.extend([ri[1]])
                ir_r10_1k.extend([ri[2]])

                tr_r1_1k.extend([rt[0]])
                tr_r5_1k.extend([rt[1]])
                tr_r10_1k.extend([rt[2]])

                # Compute R@k values for 5K Validation
                rt = itr.i2t(img_embeddings, cap_embeddings, measure='cosine', return_ranks=False)
                ri = itr.t2i(img_embeddings, cap_embeddings, measure='cosine', return_ranks=False)

                current_rsum = ri[0] + ri[1] + ri[2] + rt[0] + rt[1] + rt[2]
                t_epoch.set_description(f"Epoch RSum: {current_rsum_1k:.1f} (1K) / {current_rsum:.1f} (5K)")

                ir_r1_5k.extend([ri[0]])
                ir_r5_5k.extend([ri[1]])
                ir_r10_5k.extend([ri[2]])

                tr_r1_5k.extend([rt[0]])
                tr_r5_5k.extend([rt[1]])
                tr_r10_5k.extend([rt[2]])

                # Deep copy the model if it's the best rsum
                if current_rsum > best_rsum:
                    del best_modif_emb, best_model_wts
                    best_rsum = current_rsum
                    best_modif_emb = copy.deepcopy(model.vocabulary_encoder.modif)
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Plot recall@k values
                if args.plot and epoch > 1:
                    fig = plotter.plot_recall_curve(range(1, epoch + 1), ir_r1_1k, ir_r5_1k, ir_r10_1k,
                                                    title="Image Retrieval (1K)")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_ir_1k.png"))
                    plt.close(fig)

                    fig = plotter.plot_recall_curve(range(1, epoch + 1), tr_r1_1k, tr_r5_1k, tr_r10_1k,
                                                    title="Text Retrieval (1K)")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_tr_1k.png"))
                    plt.close(fig)

                    fig = plotter.plot_recall_curve(range(1, epoch + 1), ir_r1_5k, ir_r5_5k, ir_r10_5k,
                                                    title="Image Retrieval (5K)")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_ir_5k.png"))
                    plt.close(fig)

                    fig = plotter.plot_recall_curve(range(1, epoch + 1), tr_r1_5k, tr_r5_5k, tr_r10_5k,
                                                    title="Text Retrieval (5K)")
                    plt.savefig(os.path.join(args.output_path, f"training_recalls_{args.model}_tr_5k.png"))
                    plt.close(fig)

            # Save intermediate loss and recall plots after the second epoch
            if args.plot and phase == "dev" and epoch > 1:
                fig = plotter.plot_loss_curve(range(1, epoch + 1), train_losses, dev_losses, yexp=True)
                plt.savefig(os.path.join(args.output_path, f"training_losses_{args.model}.png"))
                plt.close(fig)

    model.load_state_dict(best_model_wts)
    model.save_model(os.path.join(args.output_path, f"best_{args.model}.pth"))

    model.vocabulary_encoder.modif = best_modif_emb
    model.vocabulary_encoder.save_corpus(os.path.join(args.output_path, f"best_learned_corpus_{args.model}.pickle"))

    with open(os.path.join(args.output_path, "losses.pickle"), "wb") as f:
        losses = {"train": train_losses, "dev": dev_losses}
        pickle.dump(losses, f)

    with open(os.path.join(args.output_path, "recalls_at_k.pickle"), "wb") as f:
        recalls_at_k = {
            "ir_r1_1k": ir_r1_1k, "ir_r5_1k": ir_r5_1k, "ir_r10_1k": ir_r10_1k,
            "tr_r1_1k": tr_r1_1k, "tr_r5_1k": tr_r5_1k, "tr_r10_1k": tr_r10_1k,
            "ir_r1_5k": ir_r1_5k, "ir_r5_5k": ir_r5_5k, "ir_r10_5k": ir_r10_5k,
            "tr_r1_5k": tr_r1_5k, "tr_r5_5k": tr_r5_5k, "tr_r10_5k": tr_r10_5k
        }
        pickle.dump(recalls_at_k, f)


if __name__ == '__main__':
    main()
