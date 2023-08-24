from torch.utils.data import Dataset
from datasets import load_dataset
import torch

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
        


if __name__ == "__main__":
    dataset = STS()
    print(len(dataset))

    # for i in tqdm(range(100)):
    #     dataset[i]
    text1, text2, label = dataset[0]
    print(text1)
    print(text2)
    print(label)

