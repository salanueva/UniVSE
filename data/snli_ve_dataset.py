import jsonlines
import numpy as np
import os
from PIL import Image
from torch.utils import data
import torchvision
from tqdm import tqdm


class SnliVECaptions(torchvision.datasets.vision.VisionDataset):
    """SNLI-VE <https://github.com/necla-ml/SNLI-VE>_ Dataset."""

    def __init__(self, data_root, img_root, split, transform=None, target_transform=None, transforms=None):
        """
        :param data_root: Root directory where images are downloaded to (can be a tuple of paths).
        :param img_root: Root directory where images are downloaded to (can be a tuple of paths).
        :param split: Split you want to load: train, dev, test or all.
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        :param transforms: A function/transform that takes input sample and its target as entry and returns a
        transformed version.
        """
        super(SnliVECaptions, self).__init__(data_root)

        snli_ve_files = {'dev': 'snli_ve_dev.jsonl',
                         'test': 'snli_ve_test.jsonl',
                         'train': 'snli_ve_train.jsonl'}

        try:
            filename = os.path.join(data_root, snli_ve_files[split])
            with jsonlines.open(filename) as jsonl_file:
                self.data = [line for line in tqdm(jsonl_file, desc=f"Load {split} split")]
        except KeyError:
            if split == "all":
                with jsonlines.open(os.path.join(data_root, snli_ve_files["train"])) as jsonl_file:
                    self.data = [line for line in tqdm(jsonl_file, desc=f"Load train split")]
                with jsonlines.open(os.path.join(data_root, snli_ve_files["dev"])) as jsonl_file:
                    self.data += [line for line in tqdm(jsonl_file, desc=f"Load dev split")]
                with jsonlines.open(os.path.join(data_root, snli_ve_files["test"])) as jsonl_file:
                    self.data += [line for line in tqdm(jsonl_file, desc=f"Load test split")]
            else:
                print(
                    "WARNING: Unexpected split for SNLI_VE dataset. "
                    "Choices are: 'train' (by default), 'dev', 'test' or 'all'."
                )
                filename = os.path.join(data_root, snli_ve_files["train"])
                with jsonlines.open(filename) as jsonl_file:
                    self.data = [line for line in tqdm(jsonl_file, desc=f"Load train split")]

        self.data_root = data_root
        self.img_root = img_root
        self.split = split
        self.length = len(self.data)

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (img, hypothesis, label).
        """
        img = str(self.data[index]["Flickr30K_ID"])
        hypothesis = str(self.data[index]["sentence2"])
        label = str(self.data[index]["gold_label"])
        label_id = 0

        img = Image.open(os.path.join(self.img_root, img + ".jpg")).convert('RGB')

        if self.transforms is not None:
            img, hypothesis = self.transforms(img, hypothesis)

        if label == "entailment":
            label_id = 0
        elif label == "neutral":
            label_id = 1
        elif label == "contradiction":
            label_id = 2
        else:
            print(f"ERROR: Unknown {label} label in index {index}.")
            exit(0)

        return img, hypothesis, label_id

    def __len__(self):
        return self.length


class SnliVECaptionsPrecomp(data.Dataset):

    def __init__(self, file_emb, file_label, split="all"):
        """
        Dataloader for precomputed embeddings
        :param file_emb: numpy file with precomputed image and hypothesis embeddings of each SNLI-VE dataset's instance.
        :param file_sim: numpy file with label ids of SNLI-VE, following the same indices as file_emb
        :param split: split of this dataloader.
        """
        super(SnliVECaptionsPrecomp, self).__init__()

        instance = np.load(file_emb)
        label = np.load(file_label)

        if split == "train":
            instance = instance[0:529527]
            label = label[0:529527]
        elif split == "dev":
            instance = instance[529527:547385]
            label = label[529527:547385]
        elif split == "test":
            instance = instance[547385:565286]
            label = label[547385:565286]
        elif split != "all":
            raise ValueError

        self.instance = instance
        self.label = label

        self.length = len(self.label)

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (embbeding, label).
        """
        embedding = self.instance[index]
        label = self.label[index]

        return embedding, label

    def __len__(self):
        return self.length


if __name__ == "__main__":

    path = "/home/ander/Documentos/Datuak/snli/"

    dataset = SnliVECaptions(
        data_root=os.path.join(path, "snli_ve"), img_root=os.path.join(path, "flickr30k_images"), split="all"
    )
    print(dataset)
