import argparse
import copy
import matplotlib.pyplot as plt
import os
import pickle
import sys
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
from torch import optim
from torch.utils import data

sys.path.append(os.getcwd())
from helper import plotter, vsts_captions as vsts
from models import vsts_models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune regressor for vSTS task using precomputed embeddings.')
    parser.add_argument(
        '--plot',
        default=False,
        action='store_true',
        help='Use it if you want to create plots of R@k values and loss values during training.'
    )
    parser.add_argument(
        '--eval',
        default=False,
        action='store_true',
        help='Use it if you want to evaluate the fine-tuned model on vSTS.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
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
        '--momentum',
        type=float,
        default=0.95,
        help='Momentum of SGD optimizer.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='Weight decay of SGD optimizer.'
    )

    parser.add_argument(
        "--sent1-emb-file",
        type=str,
        help='Numpy file with embeddings of sent_1 of each instance.'
    )
    parser.add_argument(
        "--sent2-emb-file",
        type=str,
        help='Numpy file with embeddings of sent_2 of each instance.'
    )
    parser.add_argument(
        "--sim-file",
        type=str,
        help='Numpy file with similarities of vSTS dataset.'
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help='Path for output files.'
    )

    return parser.parse_args()


def inference(data_gen, model, device):

    all_logits = []

    for batch in data_gen:

        emb_1, emb_2, _ = batch
        logits = model(emb_1.to(device), emb_2.to(device))
        all_logits.extend(logits.unsqueeze().data.cpu().tolist())

    return all_logits


def main():

    args = parse_args()

    print("A) Load data")
    train_data = vsts.VstsCaptionsPrecomp(
        file_1=args.sent1_emb_file, file_2=args.sent2_emb_file, file_sim=args.sim_file, split="train"
    )
    dev_data = vsts.VstsCaptionsPrecomp(
        file_1=args.sent1_emb_file, file_2=args.sent2_emb_file, file_sim=args.sim_file, split="dev"
    )

    print("B) Load model")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.SiameseRegressor()
    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    print("C) Train model")
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 6
    }
    train_gen = data.DataLoader(train_data, **train_params)

    eval_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 6
    }
    dev_gen = data.DataLoader(dev_data, **eval_params)

    train_losses = []
    dev_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    t_epoch = tqdm(range(1, args.epochs + 1), desc="Epoch")
    for _ in t_epoch:

        # Each epoch has a training and development phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                generator = train_gen
                model.train()  # Set model to training mode
            else:
                generator = dev_gen
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            idx = 0

            # Iterate over data.
            for current_batch in generator:

                emb_1, emb_2, sim = current_batch
                logits = model(emb_1.to(device), emb_2.to(device))

                sim = sim.view(-1, 1).to(device)
                loss = model.criterion(logits, sim)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_loss = float(loss.data.cpu().numpy())
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
                    del best_model_wts
                    best_loss = running_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(args.output_path, f"ft_model_lr{args.lr:.1E}.pth"))

    # Save loss plot
    if args.plot:
        fig = plotter.plot_loss_curve(range(1, args.epochs + 1), train_losses, dev_losses, yexp=True)
        plt.savefig(os.path.join(args.output_path, f"training_losses_{args.lr:.1E}.png"))
        plt.close(fig)

    with open(os.path.join(args.output_path, "losses.pickle"), "wb") as f:
        losses = {"train": train_losses, "dev": dev_losses}
        pickle.dump(losses, f)

    if args.eval:

        test_data = vsts.VstsCaptionsPrecomp(
            file_1=args.sent1_emb_file, file_2=args.sent2_emb_file, file_sim=args.sim_file, split="test"
        )
        train_gen = data.DataLoader(train_data, **eval_params)
        test_gen = data.DataLoader(test_data, **eval_params)

        print("D) Inference")
        model.eval()
        pred_sim_train = inference(train_gen, model, device)
        pred_sim_dev = inference(dev_gen, model, device)
        pred_sim_test = inference(test_gen, model, device)
        print(pred_sim_train)
        print(train_data.sim)

        print("E) Compute Pearson Correlations (between predicted similarities and ground truth)")
        pearson_values = [
            pearsonr(pred_sim_train, train_data.sim)[0],
            pearsonr(pred_sim_dev, dev_data.sim)[0],
            pearsonr(pred_sim_test, test_data.sim)[0]
        ]

        print(
            f"{args.model.upper()} (Cosine): {pearson_values[0]:.4f}\t{pearson_values[1]:.4f}\t{pearson_values[2]:.4f}"
        )


if __name__ == '__main__':
    main()
