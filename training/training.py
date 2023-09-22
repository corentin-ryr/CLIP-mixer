from distutils.util import strtobool
import json
from multiprocessing import Pool
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
from clip.validation import ImageNetValidator, CosineSimValidator, MNISTValidator, SST2Validator
from clip.dataset import LaionCoco

from torch.utils.data import DataLoader

from argparse import ArgumentParser
from tqdm import tqdm

from azure.storage.blob import BlobServiceClient, ContainerClient
from torchvision.transforms import Normalize


class Trainer:
    # Init takes a clip model and a dataset
    def __init__(self, model, preprocess, epochs:int, args):
        self.runName = args.run_name

        # Create a Azure container client
        blob_client = BlobServiceClient.from_connection_string(
            conn_str="DefaultEndpointsProtocol=https;AccountName=machinelearnin8258572776;AccountKey=cGUVN9SjtlwfBjZ8Z5yl3DN/P+pXNlZwbs4AP4lT1JX781pGOfWU/GkUp7BwMD+YFpec3lXbZc5d+AStsmXLLw==;EndpointSuffix=core.windows.net",
            max_block_size=128 * 1024 * 1024,
            max_single_put_size=128 * 1024 * 1024,
        )
        # Get the container or cretae it if it does not exist
        self.container_client = blob_client.get_container_client(self.runName)
        try:
            self.container_client.create_container()
        except:
            pass

        os.makedirs("outputs/checkpoints", exist_ok=True)

        self.iterationPerEpoch = float("inf")
        self.epochs = epochs
        self.model = model
        maxlr = 5e-5
        batch_size = 4096

        self.preprocess = preprocess

        self.timeBenchmark = args.verbose
        if self.timeBenchmark:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.timeList = []


        dataset = LaionCoco("/{00000..05000}.tar", args.image_path, preprocess=preprocess, verbose=True)

        self.trainLoader = DataLoader(
            dataset, shuffle=True, batch_size=batch_size, drop_last=True #, num_workers=0, prefetch_factor=1, timeout=1800
        )

        self.accelerator = Accelerator(step_scheduler_with_optimizer=True)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=maxlr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
        )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            self.epochs * min(self.iterationPerEpoch, len(self.trainLoader)),
            max_lr=maxlr,
            min_lr=maxlr / 100,
            warmup_steps=2000 * self.accelerator.num_processes,
        )

        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model, self.optimizer, self.scheduler, self.trainLoader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.trainLoader
        )
        self.numBatches = len(self.trainLoader)

        if self.accelerator.is_local_main_process:
            print(f"The dataset contains {len(dataset)} pairs of images and texts.")
            self.writer = SummaryWriter(log_dir="outputs/runs")
            self.imageNetValidator = ImageNetValidator(self, self.preprocess, self.accelerator.device, self.writer)
            self.cosineValidator = CosineSimValidator(self, self.accelerator.device, self.writer)
            self.mnistValidator = MNISTValidator(self, self.preprocess, self.accelerator.device, self.writer)
            self.sstValidator = SST2Validator(self, self.accelerator.device, self.writer)


        epoch, step = self.load_model()
        self.startEpoch = epoch
        self.currentStep = step

        if self.accelerator.is_local_main_process:
            print(f"Loaded model from epoch {epoch} and step {step}")

        self.accelerator.wait_for_everyone()

        self.normalizer = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


    # Train function
    def train(self):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        # add your own code to track the training progress.
        for epoch in range(self.startEpoch, self.epochs):
            # self.trainLoader.dataset.shuffle(epoch)

            self.model.train()
            showFirstText = True

            while self.currentStep < self.numBatches:
                dataloader = self.accelerator.skip_first_batches(self.trainLoader, self.currentStep + 1)
                for idx, batch in enumerate(
                    tqdm(
                        dataloader,
                        disable=not self.accelerator.is_local_main_process,
                        total=len(self.trainLoader),
                        miniters=20,
                        mininterval=30,
                        desc=f"Epoch {epoch}",
                        initial=self.currentStep + 1,
                    ),
                    start=self.currentStep + 1,
                ):
                    self.optimizer.zero_grad()
                    if self.timeBenchmark: self.start.record()

                    global_step = epoch * self.numBatches + idx

                    images, texts = batch
                    images = torch.tensor(images, device=self.accelerator.device, dtype=torch.float) / 256
                    images = self.normalizer(images)
                    if  self.accelerator.is_local_main_process and showFirstText: 
                        print(texts[0])
                        showFirstText = False

                    texts = clip.tokenize(texts, truncate=True).to(self.accelerator.device)

                    image_features, text_features, logit_scale = self.model(images, texts)

                    image_features_gathered = self.accelerator.gather(image_features.detach())
                    text_features_gathered = self.accelerator.gather(text_features.detach())

                    # cosine similarity as logits
                    logits_per_text = logit_scale * text_features @ image_features_gathered.t()
                    logits_per_image = logit_scale * image_features @ text_features_gathered.t()

                    ground_truth = torch.arange(len(image_features), dtype=torch.long).to(self.accelerator.device) + self.accelerator.process_index * len(
                        image_features
                    )
                    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

                    self.accelerator.backward(total_loss)

                    if self.timeBenchmark: self.end.record()

                    # Clamp logit scale to 100
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        self.model.module.logit_scale.data = torch.clamp(self.model.module.logit_scale.data, max=100)
                    else:
                        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, max=100)

                    self.optimizer.step()
                    self.scheduler.step()

                    if idx > self.iterationPerEpoch:
                        print("Reached iteration limit")
                        break

                    if self.accelerator.is_local_main_process:
                        self.writer.add_scalar("Learning rate", self.scheduler.get_lr()[0], global_step=global_step)
                        self.writer.add_scalar("Loss", total_loss.item(), global_step=global_step)

                    if self.timeBenchmark and idx > 50 and idx < 350:
                        torch.cuda.synchronize()
                        self.timeList.append(self.start.elapsed_time(self.end))
                    
                    self.currentStep = idx
                    if global_step % 400 == 399: break
                else:
                    break
                if self.timeBenchmark: print(f"Average step time: {sum(self.timeList) / len(self.timeList)}")


                self.save_model(currentEpoch=epoch, currentStep=idx)
                self.validate(global_step)
                self.accelerator.wait_for_everyone()
                showFirstText = True

            self.currentStep = 0

            if epoch == 3: break
        
        self.validate(global_step)


    def validate(self, step):
        if self.accelerator.is_local_main_process:
            self.imageNetValidator.validate(step, self.accelerator.is_local_main_process)
            self.cosineValidator.validate(step, self.accelerator.is_local_main_process)
            self.mnistValidator.validate(step, self.accelerator.is_local_main_process)
            self.sstValidator.validate(step, self.accelerator.is_local_main_process)

    def save_model(self, currentEpoch, currentStep, savePath=None):
        path = savePath if savePath else "outputs/checkpoints"
        self.accelerator.save_state(path)

        if self.accelerator.is_main_process:
            json.dump({"epoch": currentEpoch, "step": currentStep}, open(os.path.join(path, "epoch.json"), "w"))

            # Upload the outputs/checkpoints folder to Azure
            for file in os.listdir(path):
                _uploadBlob(path, file, self.container_client)

        self.accelerator.wait_for_everyone()

    def load_model(self):
        # Download the outputs/checkpoints folder from Azure
        if self.accelerator.is_local_main_process:
            for blob in self.container_client.list_blobs():
                with open(os.path.join("outputs/checkpoints", blob.name), "wb") as data:
                    self.container_client.get_blob_client(blob.name).download_blob().readinto(data)
        self.accelerator.wait_for_everyone()

        try:
            if self.accelerator.is_local_main_process:
                print(os.listdir("outputs/checkpoints"))
            self.accelerator.load_state("outputs/checkpoints")
            data = json.load(open("outputs/checkpoints/epoch.json"))
        except Exception as e:
            if self.accelerator.is_local_main_process:
                print(f"Could not load model, starting from scratch because {e}")
            return 0, 0

        return data["epoch"], data["step"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--verbose", type=lambda x:bool(strtobool(x)), default=False)
    return parser.parse_args()


def _uploadBlob(path, file, container_client: ContainerClient):
    print(f"Uploading {file}")
    with open(os.path.join(path, file), "rb") as data:
        data = data.read()
        blobClient = container_client.upload_blob(name=file, data=data, overwrite=True)
    del data, blobClient


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().cpu().mean())
            max_grads.append(p.grad.abs().cpu().max())

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    ax.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )

    print(max(max_grads), max(ave_grads))

    return ax


if __name__ == "__main__":
    # Get the argumants passed to the script
    args = parse_args()

    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        transformer_layers=12,
        transformer_width=512,
        transformer_heads=8,
        vocab_size=49408,
        context_length=77,
        useTransformer=False,
    )
    preprocess = clip._transform(model.visual.input_resolution)

    torch.manual_seed(0)

    trainer = Trainer(model, preprocess, epochs=args.epochs, args=args)

    trainer.train()

