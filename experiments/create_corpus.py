from pycocotools.coco import COCO
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from helper import load_data as ld
from models.simplified_univse import corpus as univse

# IMPORTANT: As it is, this script creates vocabulary based on MS-Coco and vSTS datasets

if __name__ == "__main__":

    # FIXME: Add argument parser so that directories are given as input parameters of the script
    glove_dir = "/home/ander/Documentos/Datuak/glove/glove.840B.300d.txt"
    images_2014 = "/home/ander/Documentos/Datuak/mscoco/train2014"
    ann_file_2014 = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_train2014.json"
    images_2017 = "/home/ander/Documentos/Datuak/mscoco/val2014"
    ann_file_2017 = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_val2014.json"

    cap_2014 = COCO(ann_file_2014)
    cap_2017 = COCO(ann_file_2017)

    cap_vsts_tr, cap_vsts_de, cap_vsts_te = ld.download_and_load_vsts_dataset(v2=True, images=False,
                                                                              root_path="/home/ander/Documentos/Datuak")

    sentences = []
    for idx in tqdm(cap_2014.anns.keys(), desc="MSCOCO Train"):
        sentences.append(cap_2014.anns[idx]['caption'])

    for idx in tqdm(cap_2017.anns.keys(), desc="MSCOCO Val"):
        sentences.append(cap_2017.anns[idx]['caption'])

    sentences += list(cap_vsts_tr["sent_1"]) + list(cap_vsts_tr["sent_2"]) + \
        list(cap_vsts_de["sent_1"]) + list(cap_vsts_de["sent_2"]) + \
        list(cap_vsts_te["sent_1"]) + list(cap_vsts_te["sent_2"])

    voc_encoder = univse.VocabularyEncoder(sentences, glove_dir)
    voc_encoder.save_corpus('simple_corpus_coco_vsts.pickle')

