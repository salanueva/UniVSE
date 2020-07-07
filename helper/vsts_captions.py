import numpy as np
import os
from PIL import Image
from torch.utils import data
import torchvision

from helper import load_data as ld


class VstsCaptions(torchvision.datasets.vision.VisionDataset):
    """vSTS Captions <https://oierldl.github.io/vsts/>_ Dataset."""

    def __init__(self, root, transform=None, target_transform=None, transforms=None, split="train"):
        """
        :param root: Root directory where images are downloaded to (can be a tuple of paths).
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        :param transforms: A function/transform that takes input sample and its target as entry and returns a
        transformed version.
        :param split: Split you want to load: train, dev, test or all.
        """
        super(VstsCaptions, self).__init__(root, transforms, transform, target_transform)

        if split == "train":
            data, _, _ = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        elif split == "dev":
            _, data, _ = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        elif split == "test":
            _, _, data = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        elif split == "all":
            data_1, data_2, data_3 = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
            data = {
                "img_1": list(data_1["img_1"]) + list(data_2["img_1"]) + list(data_3["img_1"]),
                "img_2": list(data_1["img_2"]) + list(data_2["img_2"]) + list(data_3["img_2"]),
                "sent_1": list(data_1["sent_1"]) + list(data_2["sent_1"]) + list(data_3["sent_1"]),
                "sent_2": list(data_1["sent_2"]) + list(data_2["sent_2"]) + list(data_3["sent_2"]),
                "sim": list(data_1["sim"]) + list(data_2["sim"]) + list(data_3["sim"])
            }
        else:
            raise NotImplementedError("Unknown split.")

        self.root = root
        self.data = data
        self.length = len(self.data["sim"])

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (image_1, sent_1, image_2, sent_2, sim).
        """
        img_1 = self.data["img_1"][index]
        sent_1 = self.data["sent_1"][index]
        img_2 = self.data["img_1"][index]
        sent_2 = self.data["sent_2"][index]
        sim = self.data["sim"][index]

        img_1 = Image.open(os.path.join(self.root, "visual_sts.v2.0", img_1)).convert('RGB')
        img_2 = Image.open(os.path.join(self.root, "visual_sts.v2.0", img_2)).convert('RGB')

        if self.transforms is not None:
            img_1, sent_1 = self.transforms(img_1, sent_1)
            img_2, sent_2 = self.transforms(img_2, sent_2)

        return img_1, sent_1, img_2, sent_2, sim

    def __len__(self):
        return self.length


class VstsCaptionsPrecomp(data.Dataset):

    def __init__(self, file_1, file_2, file_sim, split="all"):
        """
        Dataloader for precomputed embeddings
        :param file_1: numpy file with precomputed embeddings of sent_1 of each vSTS dataset's instance.
        :param file_2: numpy file with precomputed embeddings of sent_2 of each vSTS dataset's instance.
        :param file_sim: numpy file with similarities of each vSTS dataset's instance.
        """
        super(VstsCaptionsPrecomp, self).__init__(file_1, file_2, file_sim, split)

        sent_1 = np.load(file_1)
        sent_2 = np.load(file_2)
        sim = np.load(file_sim)

        if split == "train":
            sent_1 = sent_1[0:1338]
            sent_2 = sent_2[0:1338]
        elif split == "dev":
            sent_1 = sent_1[1338:2007]
            sent_2 = sent_2[1338:2007]
        elif split == "test":
            sent_1 = sent_1[2007:2677]
            sent_2 = sent_2[2007:2677]
        elif split != "all":
            raise ValueError

        self.sent_1 = sent_1
        self.sent_2 = sent_2
        self.sim = sim

        self.length = len(self.sim)

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (emb_1, emb_2, sim).
        """
        emb_1 = self.sent_1[index]
        emb_2 = self.sent_2[index]
        sim = self.sim[index]

        return emb_1, emb_2, sim

    def __len__(self):
        return self.length
