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
    def __init__(self, data_path, files, preprocess, verbose=False, seed=None, writeToTmp=False) -> None:
        super().__init__()

        self.length = 0
        self.captionKeyShard = []
        self.data_path = data_path
        self.preprocess = preprocess

        self.writeToTmp = writeToTmp

        self.cacheImages = {}

        blobService = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=machinelearnin8258572776;AccountKey=cGUVN9SjtlwfBjZ8Z5yl3DN/P+pXNlZwbs4AP4lT1JX781pGOfWU/GkUp7BwMD+YFpec3lXbZc5d+AStsmXLLw==;EndpointSuffix=core.windows.net")

        # Check if container exists and create it otherwise
        self.containerClient = blobService.get_container_client("laion-coco-images")
        if not self.containerClient.exists():
            self.containerClient.create_container()


        # Use multiprocessing to open the parquet files in parellel
        # This is done because the parquet files are stored on a network drive and opening them sequentially is slow
        with Pool(16) as p:

            startTime = time.time()
            data = p.map(self._processPath, list(braceexpand(data_path + files)))
            print(f"Time taken to read parquet files: {time.time() - startTime}")
            
            startTime = time.time()
            for localLength, captionKeyShard in data:
                self.length += localLength
                self.captionKeyShard += captionKeyShard
            print(f"Time taken to create the list: {time.time() - startTime}")

        # Shuffle the dataset
        random.Random(seed).shuffle(self.captionKeyShard)

    def _processPath(self, path):
        pf = ParquetFile(path[:-4] + ".parquet")
        pf:pd.DataFrame = pf.to_pandas(["caption", "key", "status"])
        pf = pf[pf["status"] == 'success']

        localLength = len(pf)
        tempDict = pf[["key", "caption"]].set_index("key").to_dict()["caption"]

        captionKeyShard = []
        for key in tempDict:
            captionKeyShard.append((tempDict[key], key, path[-9:]))

        # Open the tar file
        if self.writeToTmp:
            with tarfile.open(os.path.join(path), "r") as tar:
                # Extract all in the tmp directory
                for tarInfo in tar.getmembers():
                    if tarInfo.name.endswith(".jpg"):
                        # self.cacheImages[path[-9:] + tarInfo.name[:-4]] = tar.extractfile(tarInfo.name)

                        # Upload image to blob storage
                        self.containerClient.upload_blob(path[-9:] + tarInfo.name[:-4], tar.extractfile(tarInfo.name))

        return localLength, captionKeyShard


    def __getitem__(self, index):
        # Get the key of the shard
        caption, key, shard = self.captionKeyShard[index]
        
        # Open the tar file and get the image
        image = Image.open(self.cacheImages[shard + key])


        # with tarfile.open(os.path.join(self.data_path, shard), "r") as tar:
        #     image = Image.open(tar.extractfile(key + ".jpg"))
        
        image = self.preprocess(image)

        return image, caption
    

    def __len__(self):
        return self.length
    


        

if __name__ == "__main__":
    dataset = STS()
    print(len(dataset))

    # for i in tqdm(range(100)):
    #     dataset[i]
    text1, text2, label = dataset[0]
    print(text1)
    print(text2)
    print(label)

