import tarfile
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from braceexpand import braceexpand

from fastparquet import ParquetFile
from PIL import Image
import random
from multiprocessing import Pool
import time
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from tqdm.contrib.concurrent import process_map, thread_map

class STS(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.stsDataset = load_dataset("sick")["test"]

    def __getitem__(self, index):
        sample = self.stsDataset[index]
        label = torch.tensor(float(sample["relatedness_score"]))

        data = sample["sentence_A"].lower().strip(), sample["sentence_B"].lower().strip(), label
        return data

    def __len__(self):
        return self.stsDataset.num_rows
    


class LaionCoco(Dataset):
    def __init__(self, data_path, files, images_path, preprocess, verbose=False, seed=None) -> None:
        super().__init__()

        self.length = 0
        self.captionKeyShard = []
        self.images_path = images_path
        self.preprocess = preprocess

        # Use multiprocessing to open the parquet files in parellel
        # This is done because the parquet files are stored on a network drive and opening them sequentially is slow
        startTime = time.time()
        with Pool(32) as p:
            data = p.map(self._processPath, list(braceexpand(data_path + files)))
            
            for localLength, captionKeyShard in data:
                self.length += localLength
                self.captionKeyShard += captionKeyShard
        
        if verbose: print(f"Time taken to init the dataset: {time.time() - startTime}")

        # Shuffle the dataset
        random.Random(seed).shuffle(self.captionKeyShard)

    def _processPath(self, path):
        pf = ParquetFile(path[:-4] + ".parquet")
        pf:pd.DataFrame = pf.to_pandas(["caption", "key", "status"])
        pf = pf[pf["status"] == 'success']

        shard = path[-9:-4]
        captionKeyShard = []
        for ind in pf.index:
            captionKeyShard.append((pf["caption"][ind], pf["key"][ind], shard))
       
        localLength = len(pf)
        return localLength, captionKeyShard


    def __getitem__(self, index):
        caption, key, shard = self.captionKeyShard[index]
        
        image = Image.open(os.path.join(self.images_path, shard + key))
        image = self.preprocess(image)
        return image, caption
    
    def __len__(self):
        return self.length



class UnzipDataset():
    def __init__(self, path, imagePath) -> None:
        super().__init__()
        self.path = path
        self.imagePath = imagePath

        blobService = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=machinelearnin8258572776;AccountKey=cGUVN9SjtlwfBjZ8Z5yl3DN/P+pXNlZwbs4AP4lT1JX781pGOfWU/GkUp7BwMD+YFpec3lXbZc5d+AStsmXLLw==;EndpointSuffix=core.windows.net")

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-unzip")
        if not self.containerClient.exists():
            self.containerClient.create_container()


    def unzipDataset(self, tarFiles): # tarFiles is a string in the form of {0..9}.tar

        braceString = list(braceexpand(self.path + tarFiles))
        # self._unzipTar(braceString[0])
        process_map(self._unzipTar, braceString, max_workers=32, chunksize=8)


    def _unzipTar(self, tarPath):
        with tarfile.open(tarPath, "r|") as tar:
            # Extract all in the tmp directory
            for tarInfo in tar:
                if tarInfo.name.endswith(".jpg"):
                    # Upload image to blob storage
                    self.containerClient.upload_blob(tarPath[-9:-4] + tarInfo.name[:-4], tar.extractfile(tarInfo).read(), overwrite=True)

