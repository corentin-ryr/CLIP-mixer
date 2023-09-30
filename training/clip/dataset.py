import logging
import tarfile
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from braceexpand import braceexpand

from PIL import Image
from multiprocessing import Pool
import time
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient
from tqdm.contrib.concurrent import process_map
from io import BytesIO
import json

logging.basicConfig(level=logging.WARNING)


def _processPath(path, containerClient: ContainerClient):
    try:
        # Open the parquet file from the blob storage in the folder UI/2023-08-11_082922_UTC
        stream = BytesIO()
        containerClient.get_blob_client(f"UI/2023-08-11_082922_UTC{path[:-4] + '.parquet'}").download_blob(timeout=600).readinto(stream)
        pf = pd.read_parquet(stream, columns=["caption", "key"], filters=[("status", "==", "success")])
        pf["key"] = pd.to_numeric(pf["key"], downcast="integer")
    except Exception as e:
        print(f"Can't open parquet file UI/2023-08-11_082922_UTC{path[:-4] + '.parquet'} because {e}")
        raise e

    return pf


class STS(Dataset):
    def __init__(self, selectedSet) -> None:
        super().__init__()
        if selectedSet == "sick":
            self.stsDataset = load_dataset(selectedSet)["test"]
        else:
            self.stsDataset = load_dataset(selectedSet)["test"]
            self.stsDataset = self.stsDataset.rename_column("sentence1", "sentence_A")
            self.stsDataset = self.stsDataset.rename_column("sentence2", "sentence_B")
            self.stsDataset = self.stsDataset.rename_column("score", "relatedness_score")

        self.datasetName = selectedSet

    def __getitem__(self, index):
        sample = self.stsDataset[index]
        label = torch.tensor(float(sample["relatedness_score"]))

        data = sample["sentence_A"].lower().strip(), sample["sentence_B"].lower().strip(), label
        return data

    def __len__(self):
        return self.stsDataset.num_rows


class SST(Dataset):
    def __init__(self, selectedSplit) -> None:
        super().__init__()
        self.stsDataset = load_dataset("sst2")[selectedSplit]

        self.datasetName = "sst2"

    def __getitem__(self, index):
        sample = self.stsDataset[index]
        label = torch.tensor(float(sample["label"]))

        data = sample["sentence"].lower().strip(), label
        return data

    def __len__(self):
        return self.stsDataset.num_rows


class MNIST(Dataset):
    def __init__(self, selectedSplit, preprocess) -> None:
        super().__init__()
        self.stsDataset = load_dataset("mnist")[selectedSplit]

        self.datasetName = "mnist"
        self.preprocess = preprocess

    def __getitem__(self, index):
        sample = self.stsDataset[index]
        label = torch.tensor(float(sample["label"]))

        data = self.preprocess(sample["image"])
        return data, label

    def __len__(self):
        return self.stsDataset.num_rows


class LaionCoco(Dataset):
    def __init__(self, files, images_path, preprocess, verbose=False, limit: int = None) -> None:
        super().__init__()

        self.length = 0
        self.images_path = images_path
        self.preprocess = preprocess
        self.limit = limit

        blobService = BlobServiceClient.from_connection_string(json.load(open("azureCredentials.json"))["connectionStringSaveStorage"])

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-unzip")
        if not self.containerClient.exists():
            self.containerClient.create_container()

        # Use multiprocessing to open the parquet files in parallel
        self.containerClientParquet = blobService.get_container_client("azureml-blobstore-f6cf7981-35d0-479f-aa34-7f6fcca5d1a9")

        startTime = time.time()
        with Pool(32) as p:
            data = p.starmap(_processPath, [(path, self.containerClientParquet) for path in list(braceexpand(files))], chunksize=128)

        self.captionKey = pd.concat(data).reset_index(drop=True)
        self.length = self.captionKey.shape[0]
        print(self.captionKey.memory_usage())

        if verbose:
            print(f"Time taken to init the dataset: {time.time() - startTime}")

        self.cacheMap = {}

    def __getitem__(self, index):
        line = self.captionKey.iloc[index]
        caption, key = line["caption"], line["key"]

        key = str(key).zfill(9)

        stream = BytesIO()
        numberAttempts = 0
        while True:
            try:
                self.containerClient.download_blob(key[:5] + key).readinto(stream)
                break
            except Exception:
                numberAttempts += 1
                time.sleep(1)
                if numberAttempts > 10:
                    raise Exception(f"Impossible to download the image {key[:5] + key}")

        with Image.open(stream) as image:
            image = self.preprocess(image)

        return image, caption

    def __len__(self):
        return self.length if self.limit is None else self.limit


class UnzipDataset:
    def __init__(self, path, imagePath) -> None:
        super().__init__()
        self.path = path
        self.imagePath = imagePath

        blobService = BlobServiceClient.from_connection_string(json.load(open("azureCredentials.json"))["connStringDataset"])

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-unzip")
        if not self.containerClient.exists():
            self.containerClient.create_container()

    def unzipDataset(self, tarFiles):  # tarFiles is a string in the form of {0..9}.tar
        braceString = list(braceexpand(self.path + tarFiles))
        process_map(self._unzipTar, braceString, max_workers=64, chunksize=4)

    def _unzipTar(self, tarPath):
        try:
            with tarfile.open(tarPath, "r|") as tar:
                numberElements = len([tarInfo for tarInfo in tar.getmembers() if tarInfo.name.endswith(".jpg")])
                numberElementsStorage = len(list(self.containerClient.walk_blobs(tarPath[-9:-4])))

                if numberElements == numberElementsStorage:
                    logging.warning(f"Tar {tarPath} already unzipped.")
                    return

            with tarfile.open(tarPath, "r|") as tar:
                try:
                    for tarInfo in tar:
                        if tarInfo.name.endswith(".jpg"):
                            # Upload image to blob storage
                            self.containerClient.upload_blob(tarPath[-9:-4] + tarInfo.name[:-4], tar.extractfile(tarInfo).read(), overwrite=True)
                except Exception as e:
                    print(f"Impossible to unzip {tarPath} because {e}.")
        except Exception as e:
            print(f"Can't unzip shard {tarPath} because {e}")


if __name__ == "__main__":
    # dataset = LaionCoco("/mnt/laion-coco/", "{0..9}.tar", "/mnt/laion-coco-unzip/", None, verbose=True)
    # print(len(dataset))
    # print(dataset[0])

    # dataset = STS()
    # print(len(dataset))
    # print(dataset[0])

    dataset = UnzipDataset("/mnt/laion-coco/", "/mnt/laion-coco-unzip/")
    dataset.unzipDataset("{0..9}.tar")
