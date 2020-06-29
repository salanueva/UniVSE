from pycocotools.coco import COCO
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
# from helper import load_data as ld
from models.univse import corpus as univse

# IMPORTANT: As it is, this script creates vocabulary based on MS-Coco and vSTS datasets

if __name__ == "__main__":

    # FIXME: Add argument parser so that directories are given as input parameters of the script
    glove_dir = "/home/ander/Documentos/Datuak/glove/glove.840B.300d.txt"
    train_images = "/home/ander/Documentos/Datuak/mscoco/train2014"
    train_ann = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_train2014.json"
    val_images = "/home/ander/Documentos/Datuak/mscoco/val2014"
    val_ann = "/home/ander/Documentos/Datuak/mscoco/annotations/captions_val2014.json"

    cap_2014 = COCO(train_ann)
    # cap_2017 = COCO(val_ann)

    # cap_vsts_tr, cap_vsts_de, cap_vsts_te = ld.download_and_load_vsts_dataset(v2=True, images=False,
    #                                                                           root_path="/home/ander/Documentos/Datuak")

    sentences = []
    for idx in tqdm(cap_2014.anns.keys(), desc="MSCOCO Train"):
        sentences.append(cap_2014.anns[idx]['caption'])

    """
    for idx in tqdm(cap_2017.anns.keys(), desc="MSCOCO Val"):
        sentences.append(cap_2017.anns[idx]['caption'])

    sentences += list(cap_vsts_tr["sent_1"]) + list(cap_vsts_tr["sent_2"]) + \
        list(cap_vsts_de["sent_1"]) + list(cap_vsts_de["sent_2"]) + \
        list(cap_vsts_te["sent_1"]) + list(cap_vsts_te["sent_2"])
    """

    voc_encoder = univse.VocabularyEncoder(sentences, glove_dir)
    print(f"Num Objects: {len(voc_encoder.neg_obj)}")
    voc_encoder.save_corpus('corpus_univse.pickle')
