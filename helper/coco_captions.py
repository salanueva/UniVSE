import numpy as np
import os
from PIL import Image
from pycocotools.coco import COCO
import torchvision


class CocoCaptions(torchvision.datasets.vision.VisionDataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset."""

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None, split="train"):
        """
        :param root: Root directory where images are downloaded to (can be a tuple of paths).
        :param ann_file: Path to json annotation file (can be a tuple of files).
        :param transform: A function/transform that takes in an PIL image and returns a transformed version.
        :param target_transform: A function/transform that takes in the target and transforms it.
        :param transforms: A function/transform that takes input sample and its target as entry and returns a
        transformed version.
        :param split: Split you want to load: train, dev, test or restval (in restval option the training split is
        included as well).
        """
        super(CocoCaptions, self).__init__(root, transforms, transform, target_transform)

        self.root = root

        if isinstance(ann_file, tuple) and split == "restval":
            self.coco = (COCO(ann_file[0]), COCO(ann_file[1]))
            self.ids = list(np.load('data/coco_train_ids.npy'))
            self.bp = len(self.ids)
            self.ids += list(np.load('data/coco_restval_ids.npy'))
        else:
            self.coco = COCO(ann_file)
            if split == "train":
                self.ids = list(np.load('data/coco_train_ids.npy'))
            elif split == "dev":
                self.ids = list(np.load('data/coco_dev_ids.npy'))
            elif split == "test":
                self.ids = list(np.load('data/coco_test_ids.npy'))
            else:
                raise ValueError
            self.bp = len(self.ids)

        self.length = len(self.ids)

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (image, target). target is a unique caption
        """
        if isinstance(self.coco, tuple):
            if index < self.bp:
                coco = self.coco[0]
                root = self.root[0]
            else:
                coco = self.coco[1]
                root = self.root[1]
        else:
            coco = self.coco
            root = self.root

        ann_id = self.ids[index]
        target = coco.anns[ann_id]['caption']

        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.length
