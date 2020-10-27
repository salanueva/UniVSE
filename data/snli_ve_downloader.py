#!/usr/bin/env python

"""
SNLI-VE Generator

Authors: Ning Xie, Farley Lai(farleylai@nec-labs.com)

# Copyright (C) 2020 NEC Laboratories America, Inc. ("NECLA").
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""

import os
import argparse
import jsonlines
import subprocess
from collections import defaultdict, OrderedDict
from pathlib import Path


def prepare_all_data(snli_root, snli_files):
    """
    This function will prepare the recourse towards generating SNLI-VE dataset

    :param snli_root: root for SNLI dataset
    :param snli_files: original SNLI files, which can be downloaded via
                       https://nlp.stanford.edu/projects/snli/snli_1.0.zip

    :return:
    all_data: a set of data containing all split of SNLI dataset
    image_index_dict: a dict, key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID
    """
    data_dict = {}
    for data_type, filename in snli_files.items():
        filepath = os.path.join(snli_root, filename)
        data_list = []
        with jsonlines.open(filepath) as jsonl_file:
            for line in jsonl_file:
                pair_id = line['pairID']
                gold_label = line['gold_label']
                # only consider Flickr30k (pairID.find('vg_') == -1) items whose gold_label != '-'
                if gold_label != '-' and pair_id.find('vg_') == -1:
                    image_id = pair_id[:pair_id.rfind('.jpg')]  # XXX Removed suffix: '.jpg'
                    # Add Flickr30kID to the dataset
                    line['Flickr30K_ID'] = image_id
                    line = OrderedDict(sorted(line.items()))
                    data_list.append(line)
        data_dict[data_type] = data_list

    # all_data contains all lines in the original jsonl file
    all_data = data_dict['train'] + data_dict['dev'] + data_dict['test']

    # image_index_dict = {image:[corresponding line index in data_all]}
    image_index_dict = defaultdict(list)
    for idx, line in enumerate(all_data):
        pair_id = line['pairID']
        image_id = pair_id[:pair_id.find('.jpg')]
        image_index_dict[image_id].append(idx)

    return all_data, image_index_dict


def _split_data_helper(image_list, image_index_dict):
    """
    This will generate a dict for a data split (train/dev/test).
    key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID

    :param image_list: a list of Flickr30k imageID for a data split (train/dev/test)
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()

    :return: a dict of format {ImageID: a lost of data indices} for a data split (train/dev/test)
    """
    ordered_dict = OrderedDict()
    for imageID in image_list:
        ordered_dict[imageID] = image_index_dict[imageID]
    return ordered_dict


def split_data(all_data, image_index_dict, split_root, split_files, snli_ve_root, snli_ve_files):
    """
    This function is to generate SNLI-VE dataset based on SNLI dataset and Flickr30k split.
    The files are saved to paths defined by `SNLI_VE_root` and `SNLI_VE_files`

    :param all_data: a set of data containing all split of SNLI dataset, generated via prepare_all_data()
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()
    :param split_root: root for Flickr30k split
    :param split_files: Flickr30k split list files
    :param snli_ve_root: root to save generated SNLI-VE dataset
    :param snli_ve_files: filenames of generated SNLI-VE dataset for each split (train/dev/test)
    """
    print('\n*** Generating data split using SNLI dataset and Flickr30k split files ***')
    with open(os.path.join(split_root, split_files['test'])) as f:
        content = f.readlines()
        test_list = [x.strip() for x in content]
    with open(os.path.join(split_root, split_files['train_val'])) as f:
        content = f.readlines()
        train_val_list = [x.strip() for x in content]
    train_list = train_val_list[:-1000]
    # select the last 1000 images for dev dataset
    dev_list = train_val_list[-1000:]

    train_index_dict = _split_data_helper(train_list, image_index_dict)
    dev_index_dict = _split_data_helper(dev_list, image_index_dict)
    test_index_dict = _split_data_helper(test_list, image_index_dict)

    all_index_dict = {'train': train_index_dict, 'dev': dev_index_dict, 'test': test_index_dict}
    # # Write jsonl files
    for data_type, data_index_dict in all_index_dict.items():
        print('Current processing data split : {}'.format(data_type))
        with jsonlines.open(os.path.join(snli_ve_root, snli_ve_files[data_type]), mode='w') as jsonl_writer:
            for _, index_list in data_index_dict.items():
                for idx in index_list:
                    jsonl_writer.write(all_data[idx])


def parse_args():
    parser = argparse.ArgumentParser(description='Download and generate SNLI 1.0 and SNLI-VE')
    parser.add_argument(
        "--download-path",
        type=str,
        help='Path for the dataset.'
    )

    return parser.parse_args()


def main():

    args = parse_args()
    if not os.path.exists(args.download_path):
        print(f"ERROR: {args.download_path} must exist.")
        exit(0)

    # subprocess.call(["sh", f"./download_snli.sh {args.download_path}"])

    # SNLI-VE generation resource: SNLI dataset
    root = Path(args.download_path)
    snli_root = root / 'snli_1.0'
    snli_files = {'dev': 'snli_1.0_dev.jsonl',
                  'test': 'snli_1.0_test.jsonl',
                  'train': 'snli_1.0_train.jsonl'}

    # SNLI-VE generation resource: Flickr30k file lists
    split_root = Path('./splits')
    split_files = {'test': 'flickr30k_test.lst',
                   'train_val': 'flickr30k_train_val.lst'}

    # SNLI-VE generation destination
    snli_ve_root = root / "snli_ve"
    snli_ve_files = {'dev': 'snli_ve_dev.jsonl',
                     'test': 'snli_ve_test.jsonl',
                     'train': 'snli_ve_train.jsonl'}

    print('*** SNLI-VE Generation Start! ***')
    all_data, image_index_dict = prepare_all_data(snli_root, snli_files)
    split_data(all_data, image_index_dict, split_root, split_files, snli_ve_root, snli_ve_files)
    print('*** SNLI-VE Generation Done! ***')


if __name__ == '__main__':
    main()
