from helper import load_data as ld
import os
from PIL import Image
import torchvision


class VstsCaptions(torchvision.datasets.vision.VisionDataset):
    """vSTS Captions <https://oierldl.github.io/vsts/>_ Dataset."""

    def __init__(self, root, transform=None, target_transform=None, transforms=None, split="train"):
        """
        :param root: Root directory where images are downloaded to (can be a tuple of paths).
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        :param transforms: A function/transform that takes input sample and its target as entry and returns a
        transformed version.
        :param split: Split you want to load: train, dev, test or restval (in restval option the training split is
        included as well).
        """
        super(VstsCaptions, self).__init__(root, transforms, transform, target_transform)

        if split == "train":
            data, _, _ = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        elif split == "dev":
            _, data, _ = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        elif split == "test":
            _, _, data = ld.download_and_load_vsts_dataset(images=True, v2=True, root_path=root)
        else:
            raise NotImplementedError("Unknown split.")

        self.root = root
        self.data = data
        self.length = len(self.data)

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

        img_1 = Image.open(os.path.join(self.root, img_1)).convert('RGB')
        img_2 = Image.open(os.path.join(self.root, img_2)).convert('RGB')

        if self.transforms is not None:
            img_1, sent_1 = self.transforms(img_1, sent_1)
            img_2, sent_2 = self.transforms(img_2, sent_2)

        return img_1, sent_1, img_2, sent_2, sim

    def __len__(self):
        return self.length
