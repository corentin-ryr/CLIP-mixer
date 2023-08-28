import json
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from accelerate import Accelerator


from clip import clip
from clip.model import CLIP
from clip.validation import ImageNetValidator, CosineSimValidator

from torchdata.datapipes.iter import FileOpener
from torchdata.dataloader2 import DataLoader2
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from braceexpand import braceexpand


def collate_fn(batch):
    print(batch)

    return batch

class Trainer:

    # Init takes a clip model and a dataset
    def __init__(self, model, preprocess, epochs, data_path):
        self.iterationPerEpoch = float("inf")
        self.epochs = epochs
        self.model = model
        maxlr = 5e-4
        batch_size = 4096

        self.preprocess = preprocess

        self.accelerator = Accelerator(step_scheduler_with_optimizer=True)
        if self.accelerator.is_local_main_process: self.writer = SummaryWriter(log_dir="outputs/runs") 

        self.optimizer = optim.Adam(self.model.parameters(), lr=maxlr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        braceString = data_path + "/{00000..00001}.tar" # "/{00000..16667}.tar"
        datasetLength = getDatasetSize(braceString)
        dp = FileOpener(list(braceexpand(braceString)), mode="b") 
        dp = dp.load_from_tar(length=datasetLength).webdataset()
        dp = dp.shuffle()
        if self.accelerator.num_processes > 1:
            dp = dp.sharding_filter()
            dp.apply_sharding(self.accelerator.num_processes, self.accelerator.process_index) 
        dp = dp.map(self.decode)
        dp = dp.batch(batch_size=batch_size, drop_last=True)
        self.numBatches = len(dp)
        self.trainLoader = DataLoader2(dp) 

        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, self.epochs * min(self.iterationPerEpoch, self.numBatches), max_lr=maxlr, min_lr=maxlr/100, warmup_steps=20) #CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        self.accelerator.free_memory()

        if self.accelerator.is_local_main_process: print(f"The dataset contains {datasetLength} pairs of images and texts.")

        if self.accelerator.is_local_main_process: self.imageNetValidator = ImageNetValidator(self, self.preprocess, self.accelerator.device, self.writer)
        if self.accelerator.is_local_main_process: self.cosineValidator = CosineSimValidator(self, self.accelerator.device, self.writer)

        if os.path.exists(os.path.join("outputs", "checkpoints")): 
            epoch, step = self.load_model()
            self.scheduler.step(epoch * self.numBatches + step)
            self.currentEpoch = epoch
            self.currentStep = step
        else:
            self.currentEpoch = 0
            self.currentStep = 0

        self.accelerator.wait_for_everyone()

    # Train function 
    def train(self):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        # add your own code to track the training progress.
        for epoch in range(self.epochs - self.currentEpoch):
            self.model.train()

            if self.currentStep > 0: dataloader = self.accelerator.skip_first_batches(self.trainLoader, self.currentStep)
            else: dataloader = self.trainLoader

            print(len(dataloader))

            for idx, batch in enumerate(tqdm(dataloader, disable=not self.accelerator.is_local_main_process, total=self.numBatches - self.currentStep, miniters=20, mininterval=30, desc=f"Epoch {epoch}")):                
                images, texts = list(zip(*batch))
                texts = [text[0] for text in texts]

                texts = clip.tokenize(texts, truncate=True).to(self.accelerator.device)
                images:torch.Tensor = torch.stack(images).squeeze().to(self.accelerator.device)

                image_features, text_features, logit_scale = self.model(images, texts)

                image_features_gathered = self.accelerator.gather(image_features.detach())
                text_features_gathered = self.accelerator.gather(text_features.detach())

                # cosine similarity as logits
                logits_per_image = logit_scale * image_features @ text_features_gathered.t()
                logits_per_text = logit_scale * text_features @ image_features_gathered.t()
                # print(logits_per_image.shape, logits_per_text.shape)
                # print(torch.min(logits_per_image), torch.max(logits_per_image))

                ground_truth = torch.arange(len(image_features), dtype=torch.long).to(self.accelerator.device) + self.accelerator.process_index * len(image_features)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

                self.accelerator.backward(total_loss)
                # Clamp logit scale to 100
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, max=100)
                else:
                    self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, max=100)

                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.scheduler.step()

                # print(f"Step took {time.time() - startTime} seconds")

                if idx > self.iterationPerEpoch: 
                    print("Reached iteration limit")
                    break

                global_step = epoch * self.numBatches + idx
                if self.accelerator.is_local_main_process:
                    self.writer.add_scalar("Learning rate", self.scheduler.get_lr()[0], global_step=global_step)
                    self.writer.add_scalar("Loss", total_loss.item(), global_step=global_step)

                if global_step % 100 == 99: self.save_model(currentEpoch = epoch, currentStep = idx)

            self.currentStep = 0
            self.validate(epoch)


    def validate(self, step):
        if self.accelerator.is_local_main_process:
            self.imageNetValidator.validate(step, self.accelerator.is_main_process)
            self.cosineValidator.validate(step, self.accelerator.is_main_process)

    def decode(self, x):
        image = x[".jpg"]
        text = x[".txt"].read().decode("utf-8")
        image = self.preprocess(Image.open(image))

        return image, text

    
    def save_model(self, currentEpoch, currentStep, savePath=None):
        savePath = savePath if savePath else os.path.join(os.path.join("outputs", "checkpoints"))
        path = os.path.join(savePath)
        
        self.accelerator.save_state(path)

        # self.accelerator.wait_for_everyone()
        # modelToSave = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
        # state_dict = modelToSave.state_dict()

        if self.accelerator.is_main_process: 
            # self.accelerator.save(state_dict, os.path.join(path, "model.plk"))
            json.dump({"epoch": currentEpoch, "step": currentStep}, open(os.path.join(path, "epoch.json"), "w"))

        self.accelerator.free_memory()

        

    def load_model(self):
        self.accelerator.load_state(os.path.join("outputs", "checkpoints"))
        self.accelerator.wait_for_everyone()

        data = json.load(open(os.path.join("outputs", "checkpoints", "epoch.json")))
        return data["epoch"], data["step"]
        

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, default="C:/Users/royc/Documents/DeduplicationSourceCode/Data/STS-b/laion-coco-images")
    return parser.parse_args()

    
def getDatasetSize(paths):
    length = 0
    for path in list(braceexpand(paths)):
        stats = json.load(open(path[:-4] + "_stats.json"))
        length += stats["successes"]

    return length


if __name__ == "__main__":

    # Get the argumants passed to the script
    args = parse_args()

    model = CLIP(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                 transformer_layers=12, transformer_width=512, transformer_heads=8, vocab_size=49408, context_length=77)
    preprocess = clip._transform(model.visual.input_resolution)

    trainer = Trainer(model, preprocess, epochs=32, data_path=args.data_path)

    trainer.validate(step=-1)

    trainer.train()

   

