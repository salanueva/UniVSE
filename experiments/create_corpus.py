import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from helper import load_data as ld
from models.univse import corpus as univse

# IMPORTANT: As it is, this script creates vocabulary based on MS-Coco and vSTS datasets

if __name__ == "__main__":

    # FIXME: Add argument parser so that directories are given as input parameters of the script
    glove_dir = "/home/ander/Documentos/Datuak/glove/glove.840B.300d.txt"
    images_2014 = "/home/ander/Documentos/Datuak/mscoco/train2014"
    ann_file_2014 = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_train2014.json"
    images_2017 = "/home/ander/Documentos/Datuak/mscoco/val2017"
    ann_file_2017 = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_val2017.json"

    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

    cap_2014 = torchvision.datasets.CocoCaptions(images_2014, ann_file_2014, transform=transform, target_transform=None,
                                                 transforms=None)
    cap_2017 = torchvision.datasets.CocoCaptions(images_2017, ann_file_2017, transform=transform, target_transform=None,
                                                 transforms=None)
    cap_vsts_tr, cap_vsts_de, cap_vsts_te = ld.download_and_load_vsts_dataset(v2=True, images=False,
                                                                              root_path="/home/ander/Documentos/Datuak")

    sentences = []
    for _, sent in tqdm(cap_2014, desc="MSCOCO Train"):
        sentences += sent

    for _, sent in tqdm(cap_2017, desc="MSCOCO Test"):
        sentences += sent

    sentences += list(cap_vsts_tr["sent_1"]) + list(cap_vsts_tr["sent_2"]) + \
        list(cap_vsts_de["sent_1"]) + list(cap_vsts_de["sent_2"]) + \
        list(cap_vsts_te["sent_1"]) + list(cap_vsts_te["sent_2"])

    voc_encoder = univse.VocabularyEncoder(sentences, glove_dir)
    voc_encoder.save_corpus('corpus_coco_vsts.pickle')
