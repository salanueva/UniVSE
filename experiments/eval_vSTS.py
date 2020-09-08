import argparse
import os
from PIL import Image
from scipy.stats import pearsonr
import sys
from torchvision import transforms
from tqdm import tqdm

import torch

sys.path.append(os.getcwd())
from helper import load_data as ld
from models.simplified_univse import model as simp_univse
from models.univse import model as univse


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VSE models in vSTS')

    parser.add_argument(
        "--model",
        type=str,
        choices=["vse++", "univse", "simp_univse"],
        default="univse",
        help='Name of the model you want to evaluate on vSTS. ' +
             'Choices are: "vse++", "univse" or "simp_univse".'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["vsts"],
        default="vsts",
        help='Name of the task/dataset you want to use. Choices are: "vsts" (for now).'
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help='Path of dataset that will be used.'
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help='Path of precomputed scene graphs.'
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


def inference(dataset, model, device, model_type="univse"):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    pred_similarities = [0.] * len(dataset)
    index = 0

    for img_1, sent_1, img_2, sent_2, sim in tqdm(dataset, total=len(dataset)):

        # Predict hidden states features for each layers
        with torch.no_grad():

            if model_type == "univse":
                output_1 = model(img_1, [sent_1])
                output_2 = model(img_2, [sent_2])
                emb_1 = model.alpha * output_1["sent_emb"] + (1 - model.alpha) * output_1["comp_emb"]
                emb_2 = model.alpha * output_2["sent_emb"] + (1 - model.alpha) * output_2["comp_emb"]
            elif model_type == "simp_univse":
                output_1 = model(img_1, [sent_1])
                output_2 = model(img_2, [sent_2])
                emb_1 = torch.cat([output_1["img_emb"], output_1["sent_emb"]], dim=1)
                emb_2 = torch.cat([output_2["img_emb"], output_2["sent_emb"]], dim=1)
                # emb_1 = output_1["sent_emb"]
                # emb_2 = output_2["sent_emb"]
            else:
                print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
                return

        pred_similarities[index] = cos(emb_1, emb_2).data.cpu().numpy()[0]
        index += 1

    return pred_similarities


def load_data(dataset, transform, data_path):

    img_1 = [
        transform(Image.open(os.path.join(data_path, "visual_sts.v2.0", elem)).convert('RGB')).view(1, 3, 224, 224)
        for elem in tqdm(dataset["img_1"])
    ]
    img_2 = [
        transform(Image.open(os.path.join(data_path, "visual_sts.v2.0", elem)).convert('RGB')).view(1, 3, 224, 224)
        for elem in tqdm(dataset["img_2"])
    ]
    sent_1 = list(dataset["sent_1"])
    sent_2 = list(dataset["sent_2"])
    sim = list(dataset["sim"])

    new_dataset = list(zip(img_1, sent_1, img_2, sent_2, sim))

    return new_dataset


def main():

    args = parse_args()

    print("A) Download and load data")

    if args.dataset == "stsb":
        train_set, dev_set, test_set = ld.download_and_load_stsb_dataset(root_path=args.data_path)
    elif args.dataset == "vsts":
        train_set, dev_set, test_set = ld.download_and_load_vsts_dataset(v2=True, images=True, root_path=args.data_path)
    else:
        print("ERROR: dataset input argument unknown.")  # You shouldn't be able to reach here!
        return

    print("B) Preprocess data")

    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

    train_data = load_data(train_set, transform, args.data_path)
    dev_data = load_data(dev_set, transform, args.data_path)
    test_data = load_data(test_set, transform, args.data_path)

    print("C) Load pre-trained model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == "univse":
        model = univse.UniVSE.from_filename(args.vocab_path)
        model.load_model(args.model_path)
        if args.graph_path is not None:
            model.vocabulary_encoder.add_graphs(args.graph_path)
    elif args.model == "simp_univse":
        model = simp_univse.UniVSE.from_filename(args.vocab_path)
        model.load_model(args.model_path)
    else:
        print("ERROR: model name unknown.")  # You shouldn't be able to reach here!
        return

    model.eval()
    model = model.to(device)

    print("D) Inference")

    pred_sim_train = inference(train_data, model, device, args.model)
    pred_sim_dev = inference(dev_data, model, device, args.model)
    pred_sim_test = inference(test_data, model, device, args.model)

    print("E) Compute Pearson Correlations (between predicted similarities and ground truth)")
    true_sim_train = list(train_set["sim"])
    true_sim_dev = list(dev_set["sim"])
    true_sim_test = list(test_set["sim"])

    pearson_values = [
        pearsonr(pred_sim_train, true_sim_train)[0],
        pearsonr(pred_sim_dev, true_sim_dev)[0],
        pearsonr(pred_sim_test, true_sim_test)[0]
    ]

    print(
        f"{args.model.upper()} (Cosine): {pearson_values[0]:.4f}\t{pearson_values[1]:.4f}\t{pearson_values[2]:.4f}"
    )


if __name__ == '__main__':
    main()
