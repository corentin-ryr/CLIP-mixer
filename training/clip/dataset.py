import csv
import logging
import sys
import tarfile
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from braceexpand import braceexpand

from fastparquet import ParquetFile
from PIL import Image
from multiprocessing import Pool
import time
import pandas as pd
from azure.storage.blob import BlobServiceClient, ContainerClient
from tqdm.contrib.concurrent import process_map
from io import BytesIO

logging.basicConfig(level=logging.WARNING)



def _processPath(path, containerClient:ContainerClient):
    try:
        # Open the parquet file from the blob storage
        stream = BytesIO()
        containerClient.get_blob_client(f"UI/2023-08-11_082922_UTC/{path[:-4] + '.parquet'}").download_blob().readinto(stream)

        # pf = ParquetFile(path[:-4] + ".parquet")
        pf = pd.read_parquet(stream, columns=["caption", "key"], filters=[("status", "==", "success")])
    except Exception as e:
        print(f"Can't open parquet file {path} because {e}")
        raise e
    # pf:pd.DataFrame = pf.to_pandas(["caption", "key", "status"])
    # pf = pf[pf["status"] == "success"]

    # captionKey = list(zip(pf["caption"], pf["key"]))
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
    def __init__(self, data_path, files, images_path, preprocess, verbose=False) -> None:
        super().__init__()

        self.length = 0
        self.images_path = images_path
        self.preprocess = preprocess

        blobService = BlobServiceClient.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=machinelearnin8258572776;AccountKey=cGUVN9SjtlwfBjZ8Z5yl3DN/P+pXNlZwbs4AP4lT1JX781pGOfWU/GkUp7BwMD+YFpec3lXbZc5d+AStsmXLLw==;EndpointSuffix=core.windows.net"
        )

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-unzip")
        if not self.containerClient.exists():
            self.containerClient.create_container()

        # Use multiprocessing to open the parquet files in parellel
        # This is done because the parquet files are stored on a network drive and opening them sequentially is slow
        self.containerClientParquet = blobService.get_container_client("azureml-blobstore-f6cf7981-35d0-479f-aa34-7f6fcca5d1a9")

        startTime = time.time()
        with Pool(32) as p:
            data = p.starmap(_processPath, [(path, self.containerClientParquet) for path in list(braceexpand(data_path + files))], chunksize=128)
        
        # data = []
        # for path in braceexpand(data_path + files):
        #     data.append(_processPath(path))
        
        self.captionKey = pd.concat(data)

        self.length = self.captionKey.shape[0]

        print(self.captionKey.memory_usage())

        if verbose: print(f"Time taken to init the dataset: {time.time() - startTime}")

    def __getitem__(self, index):
        line = self.captionKey.iloc[index]
        caption, key = line["caption"], line["key"]

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
        return self.length


class UnzipDataset:
    def __init__(self, path, imagePath) -> None:
        super().__init__()
        self.path = path
        self.imagePath = imagePath

        blobService = BlobServiceClient.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=machinelearnin8258572776;AccountKey=cGUVN9SjtlwfBjZ8Z5yl3DN/P+pXNlZwbs4AP4lT1JX781pGOfWU/GkUp7BwMD+YFpec3lXbZc5d+AStsmXLLw==;EndpointSuffix=core.windows.net"
        )

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-unzip")
        if not self.containerClient.exists():
            self.containerClient.create_container()

    def unzipDataset(self, tarFiles):  # tarFiles is a string in the form of {0..9}.tar
        braceString = list(braceexpand(self.path + tarFiles))
        # self._unzipTar(braceString[0])
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
