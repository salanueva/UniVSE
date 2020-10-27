import pandas
import os
import shutil

import wget
import tarfile


def download_file(url, path):
    """Downloads file from the specified URL."""
    if not os.path.isdir(path):
        os.mkdir(path)
        filename = wget.download(url)
    else:
        filename = None
    return filename


def extract_file(fname, path):
    """Extracts .tar or .tar.gz files."""
    if not os.path.isdir(path):
        os.mkdir(path)
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path)
        tar.close()
    else:
        print("Error: file is not *.tar or *.tar.gz")


def delete_extracted_files(path):
    """Deletes all files that are inside data_aux."""
    shutil.rmtree(path)


def load_stsb_dataset(filename):
    """Loads a subset of the STS-B into a DataFrame. In particular both
        sentences and their human rated similarity score."""
    sent_pairs = []
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f:  # (sth, sth, sth, id, similarity_score, sent1, sent2)
            ts = line.strip().split("\t")
            if len(ts) >= 7:  # check if it is an instance
                sent_pairs.append((ts[5], ts[6], float(ts[4])))  # (sent_1, sent_2, similarity_score)
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def load_vsts_dataset(filename):
    """Loads a subset of the vSTS into a DataFrame. In particular both
        sentences and their human rated similarity score."""
    sent_pairs = []
    with open(filename, "r") as f:
        for line in f:  # (id, source, sent1, img_path1, sent2, img_path2, similarity_score)
            ts = line.strip().split("\t")
            if len(ts) == 7 and ts[0] != "id":  # check if it is an instance
                sent_pairs.append((ts[2], ts[4], float(ts[6])))  # (sent_1, sent_2, similarity_score)
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def load_vsts_dataset_with_images(filename):
    """Loads a subset of the vSTS into a DataFrame. In particular both
            sentences, both images and their human rated similarity score."""
    sent_pairs = []
    with open(filename, "r") as f:
        for line in f:  # (id, source, sent1, img_path1, sent2, img_path2, similarity_score)
            ts = line.strip().split("\t")
            # if len(ts) == 7 and ts[0] != "id" and ts[1] != "coco":  # check if it is an instance
            if len(ts) == 7 and ts[0] != "id":  # check if it is an instance
                sent_pairs.append((ts[2], ts[3], ts[4], ts[5], float(ts[6])))  # (sent_1, sent_2, similarity_score)
    return pandas.DataFrame(sent_pairs, columns=["sent_1", "img_1", "sent_2", "img_2", "sim"])


def load_stsb_no_img_dataset(folder):
    """Loads the whole STS-B no images dataset."""
    sts_train = load_stsb_dataset(os.path.join(folder, "sts-train.no-images.csv"))
    sts_dev = load_stsb_dataset(os.path.join(folder, "sts-dev.no-images.csv"))
    sts_test = load_stsb_dataset(os.path.join(folder, "sts-test.no-images.csv"))

    return sts_train, sts_dev, sts_test


def download_and_load_stsb_dataset(root_path="."):
    """Downloads and loads the whole STS-B dataset."""
    filename = download_file("http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz", root_path)
    extract_file(filename, root_path)

    sts_train = load_stsb_dataset(os.path.join(root_path, "stsbenchmark", "sts-train.csv"))
    sts_dev = load_stsb_dataset(os.path.join(root_path, "stsbenchmark", "sts-dev.csv"))
    sts_test = load_stsb_dataset(os.path.join(root_path, "stsbenchmark", "sts-test.csv"))
    # delete_extracted_files()

    if os.path.isfile(os.path.join(root_path, "Stsbenchmark.tar.gz")):
        os.remove("Stsbenchmark.tar.gz")

    return sts_train, sts_dev, sts_test


def download_and_load_vsts_dataset(images=False, v2=False, root_path="."):
    """Downloads and loads the whole vSTS dataset."""
    if not v2:

        filename = download_file("http://ixa2.si.ehu.eus/~jibloleo/visual_sts.tar.gz", root_path)

        if filename is not None:
            extract_file(filename, root_path)

        # 829 instances
        if not images:
            vsts_data = load_vsts_dataset(os.path.join(root_path, "visual_sts",
                                                       "visual_sts.all.nopar.tsv"))
        else:
            vsts_data = load_vsts_dataset_with_images(os.path.join(root_path, "visual_sts",
                                                                   "visual_sts.all.nopar.tsv"))

        # delete_extracted_files()
        if os.path.isfile(os.path.join("../helper", "visual_sts.tar.gz")):
            os.remove("visual_sts.tar.gz")

        return vsts_data

    else:

        filename = download_file("http://ixa2.si.ehu.eus/~jibloleo/visual_sts.v2.0.tar.gz", root_path)

        if filename is not None:
            extract_file(filename, root_path)

        if not images:
            vsts_train = load_vsts_dataset(os.path.join(root_path, "visual_sts.v2.0", "train-dev-test",
                                                        "visual_sts.v2.0.train.tsv"))
            vsts_dev = load_vsts_dataset(os.path.join(root_path, "visual_sts.v2.0", "train-dev-test",
                                                      "visual_sts.v2.0.dev.tsv"))
            vsts_test = load_vsts_dataset(os.path.join(root_path, "visual_sts.v2.0", "train-dev-test",
                                                       "visual_sts.v2.0.test.tsv"))

        else:
            vsts_train = load_vsts_dataset_with_images(os.path.join(root_path, "visual_sts.v2.0",
                                                                    "train-dev-test", "visual_sts.v2.0.train.tsv"))
            vsts_dev = load_vsts_dataset_with_images(os.path.join(root_path, "visual_sts.v2.0",
                                                                  "train-dev-test", "visual_sts.v2.0.dev.tsv"))
            vsts_test = load_vsts_dataset_with_images(os.path.join(root_path, "visual_sts.v2.0",
                                                                   "train-dev-test", "visual_sts.v2.0.test.tsv"))

        # delete_extracted_files()
        if os.path.isfile(os.path.join(root_path, "visual_sts.v2.0.tar.gz")):
            os.remove("visual_sts.v2.0.tar.gz")

        return vsts_train, vsts_dev, vsts_test
