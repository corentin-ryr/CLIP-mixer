import json
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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


class Trainer:

    # Init takes a clip model and a dataset
    def __init__(self, model, preprocess, epochs, data_path):
        self.iterationPerEpoch = float("inf")
        self.epochs = epochs
        self.model = model
        maxlr = 5e-4 / 20
        batch_size = 4096

        self.preprocess = preprocess

        self.accelerator = Accelerator(step_scheduler_with_optimizer=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=maxlr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        braceString = data_path + "/{00000..16667}.tar" #"/{00000..00006}.tar" #
        datasetLength = getDatasetSize(braceString, verbose=self.accelerator.is_local_main_process)
        dp = FileOpener(list(braceexpand(braceString)), mode="b") 
        dp = dp.load_from_tar(length=datasetLength).webdataset()
        dp = dp.shuffle()
        if self.accelerator.num_processes > 1:
            dp = dp.sharding_filter()
            dp.apply_sharding(self.accelerator.num_processes, self.accelerator.process_index) 
        dp = dp.map(self.decode)
        dp = dp.batch(batch_size=batch_size, drop_last=True)
        self.numBatches = len(dp)
        self.trainLoader = DataLoader(dp) 

        self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer, self.epochs * min(self.iterationPerEpoch, self.numBatches) * self.accelerator.num_processes, 
                                                       max_lr=maxlr, min_lr=maxlr/100, warmup_steps=2000 * self.accelerator.num_processes)
        
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        if self.accelerator.is_local_main_process:
            print(f"The dataset contains {datasetLength} pairs of images and texts.")
            self.writer = SummaryWriter(log_dir="outputs/runs")
            self.imageNetValidator = ImageNetValidator(self, self.preprocess, self.accelerator.device, self.writer)
            self.cosineValidator = CosineSimValidator(self, self.accelerator.device, self.writer)

        
        epoch, step = self.load_model()
        self.startEpoch = epoch
        self.currentStep = step

        if self.accelerator.is_local_main_process: print(f"Loaded model from epoch {epoch} and step {step}")


        self.accelerator.wait_for_everyone()

    # Train function 
    def train(self):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        # add your own code to track the training progress.
        for epoch in range(self.startEpoch, self.epochs):
            self.model.train()

            dataloader = self.accelerator.skip_first_batches(self.trainLoader, self.currentStep)
            for idx, batch in enumerate(tqdm(dataloader, disable=not self.accelerator.is_local_main_process, total=self.numBatches - self.currentStep, miniters=20, mininterval=30, desc=f"Epoch {epoch}")):                
                global_step = epoch * self.numBatches + idx
                
                images, texts = list(zip(*batch))
                texts = [text[0] for text in texts]

                texts = clip.tokenize(texts, truncate=True).to(self.accelerator.device)
                images:torch.Tensor = torch.stack(images).squeeze().to(self.accelerator.device)

                image_features, text_features, logit_scale = self.model(images, texts)

                image_features_gathered = self.accelerator.gather(image_features.detach())
                text_features_gathered = self.accelerator.gather(text_features.detach())

                # cosine similarity as logits
                logits_per_text = logit_scale * text_features @ image_features_gathered.t()
                logits_per_image = logit_scale * image_features @ text_features_gathered.t()

                ground_truth = torch.arange(len(image_features), dtype=torch.long).to(self.accelerator.device) + self.accelerator.process_index * len(image_features)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

                self.accelerator.backward(total_loss)

                # Clamp logit scale to 100
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, max=100)
                else:
                    self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, max=100)


                if self.accelerator.is_local_main_process and global_step % 100 == 99:
                    ax = plot_grad_flow(self.model.named_parameters())
                    self.writer.add_figure("Gradient flow", ax.figure, global_step=global_step)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if idx > self.iterationPerEpoch: 
                    print("Reached iteration limit")
                    break

                if self.accelerator.is_local_main_process:
                    self.writer.add_scalar("Learning rate", self.scheduler.get_lr()[0], global_step=global_step)
                    self.writer.add_scalar("Loss", total_loss.item(), global_step=global_step)

                if global_step % 100 == 99: 
                    self.save_model(currentEpoch = epoch, currentStep = idx)
                    self.validate(global_step)


            self.currentStep = 0
            # if epoch % 5 == 0: self.validate(epoch)

            if self.accelerator.is_local_main_process: self.writer.flush()


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
        path = savePath if savePath else os.path.join(os.path.join("outputs", "checkpoints"))        
        self.accelerator.save_state(path)

        # self.accelerator.wait_for_everyone()
        # modelToSave = self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
        # state_dict = modelToSave.state_dict()

        if self.accelerator.is_main_process: 
            # self.accelerator.save(state_dict, os.path.join(path, "model.plk"))
            json.dump({"epoch": currentEpoch, "step": currentStep}, open(os.path.join(path, "epoch.json"), "w"))

        

    def load_model(self):
        try:
            print(os.listdir("outputs"))
            print(os.listdir("outputs/checkpoints"))
            self.accelerator.load_state("outputs/checkpoints")
            data = json.load(open("outputs/checkpoints/epoch.json"))
        except Exception as e:
            print(f"Could not load model, starting from scratch because {e}")
            return 0, 0
        
        return data["epoch"], data["step"]
        

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, default="C:/Users/royc/Documents/DeduplicationSourceCode/Data/STS-b/laion-coco-images")
    parser.add_argument("--epochs", type=int, default=32)
    return parser.parse_args()

    
def getDatasetSize(paths, verbose=False):
    length = 0
    for path in tqdm(list(braceexpand(paths)), miniters=40, mininterval=40, desc="Computing dataset length", disable=not verbose):
        stats = json.load(open(path[:-4] + "_stats.json"))
        length += stats["successes"]

    return length

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().cpu().mean())
            max_grads.append(p.grad.abs().cpu().max())

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    print(max(max_grads), max(ave_grads))
    
    return ax


if __name__ == "__main__":


    # Get the argumants passed to the script
    args = parse_args()

    model = CLIP(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                 transformer_layers=12, transformer_width=512, transformer_heads=8, vocab_size=49408, context_length=77)
    preprocess = clip._transform(model.visual.input_resolution)

    trainer = Trainer(model, preprocess, epochs=args.epochs, data_path=args.data_path)

    trainer.validate(step=-1)

    trainer.train()

   

