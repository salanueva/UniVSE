import argparse
import copy
import matplotlib.pyplot as plt
import os
import pickle
import sys
from tqdm import tqdm

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torchvision import transforms

sys.path.append(os.getcwd())
from helper import plotter, vsts_captions as vsts
from models import vsts_models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain a given VSE model with MS-COCO dataset')

    parser.add_argument(
        "--model",
        type=str,
        choices=["vse++", "univse", "simp_univse"],
        default="univse",
        help='Name of the model you want to fine-tune. Choices are: "vse++" (not implemented yet),'
             '"simp_univse" and "univse".'
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["vsts"],
        default="vsts",
        help='Name of the task you want to fine-tune. For now, "vsts" is the only choice.'
    )
    parser.add_argument(
        '--plot',
        default=False,
        action='store_true',
        help='Use it if you want to create plots of R@k values and loss values during training.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs for the pre-training process.'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00002,
        help='Initial learning rate of the fine-tuning process.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Number of image and sentence pairs per batch.'
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

    train_data = vsts.VstsCaptions(root=args.data_path, transform=transform, split="train")
    dev_data = vsts.VstsCaptions(root=args.data_path, transform=transform, split="dev")

    print("B) Load model")
    if args.model == "vse++":
        raise NotImplementedError
    elif args.model == "simp_univse":
        model = models.UniVSE(args.vocab_path, simple=True)
        model.univse_layer.load_model(args.model_path)
    elif args.model == "univse":
        model = models.UniVSE(args.vocab_path, graph_file=args.graph_path)
        model.univse_layer.load_model(args.model_path)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.params, lr=args.lr)

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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_modif_emb = copy.deepcopy(model.univse_layer.vocabulary_encoder.modif)
    best_loss = 1e10

    t_epoch = tqdm(range(1, args.epochs + 1), desc="Epoch")
    for _ in t_epoch:

        # Each epoch has a training and development phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                generator = train_gen
                model.train_start()  # Set model to training mode
            else:
                generator = dev_gen
                model.eval_start()  # Set model to evaluate mode

            running_loss = 0.0
            idx = 0

            # Iterate over data.
            t_batch = tqdm(generator, desc="Batch", leave=False)
            for current_batch in t_batch:

                img_1, sent_1, img_2, sent_2, sim = current_batch

                logits = model(img_1, list(sent_1), img_2, list(sent_2))

                sim = sim.view(-1, 1).to(device)
                loss = model.criterion(logits, sim)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_loss = float(loss.data.cpu().numpy())
                t_batch.set_description(f"Batch Loss: {batch_loss:.6f}")
                running_loss += batch_loss
                idx += 1

            running_loss /= idx

            if phase == "train":
                train_losses.append(running_loss)
            else:
                dev_losses.append(running_loss)
                t_epoch.set_description(f"Epoch Loss: {train_losses[-1]:.3f} (train) / {dev_losses[-1]:.3f} (val)")

                # Deep copy the model if it's the best rsum
                if running_loss < best_loss:
                    del best_modif_emb, best_model_wts
                    best_loss = running_loss
                    best_modif_emb = copy.deepcopy(model.univse_layer.vocabulary_encoder.modif)
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(args.output_path, f"ft_model_{args.model}.pth"))

    model.univse_layer.vocabulary_encoder.modif = best_modif_emb
    model.univse_layer.vocabulary_encoder.save_corpus(os.path.join(args.output_path, f"ft_corpus_{args.model}.pickle"))

    # Save loss plot
    if args.plot:
        fig = plotter.plot_loss_curve(range(1, args.epochs + 1), train_losses, dev_losses, yexp=True)
        plt.savefig(os.path.join(args.output_path, f"training_losses_{args.model}.png"))
        plt.close(fig)

    with open(os.path.join(args.output_path, "losses.pickle"), "wb") as f:
        losses = {"train": train_losses, "dev": dev_losses}
        pickle.dump(losses, f)


if __name__ == '__main__':
    main()
