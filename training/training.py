from distutils.util import strtobool
import json
import math
from multiprocessing import Pool
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
# from clip.scheduler import CosineAnnealingWarmupRestarts
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
    def __init__(self, model:nn.Module, preprocess, epochs:int, args):
        self.runName = args.run_name

        # Create a Azure container client
        blob_client = BlobServiceClient.from_connection_string(
            conn_str=json.load(open("azureCredentials.json"))["connStringSaveStorage"],
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
        maxlr = 5e-4
        batch_size =  32768

        self.preprocess = preprocess

        dataset = LaionCoco("/{00000..35000}.tar", args.image_path, preprocess=preprocess, verbose=True)

        self.trainLoader = DataLoader(
            dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=64, timeout=1800
        )

        self.accelerator = Accelerator(step_scheduler_with_optimizer=True, split_batches=True)

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        self.optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": 0.2},
            ],
            lr=maxlr, 
            betas=(0.9, 0.98), 
            eps=1e-6
        )

        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            self.epochs * min(self.iterationPerEpoch, len(self.trainLoader)),
            max_lr=maxlr,
            min_lr=maxlr / 100,
            warmup_steps=2000,
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
                dataloader = self.accelerator.skip_first_batches(self.trainLoader, self.currentStep)
                for idx, batch in enumerate(
                    tqdm(
                        dataloader,
                        disable=not self.accelerator.is_local_main_process,
                        total=len(self.trainLoader),
                        miniters=20,
                        mininterval=30,
                        desc=f"Epoch {epoch}",
                        initial=self.currentStep,
                    ),
                    start=self.currentStep,
                ):
                    self.optimizer.zero_grad()

                    global_step = epoch * self.numBatches + idx

                    images, texts = batch
                    images = self.normalizer(images / 255)
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


                    # Clamp logit scale to 100
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        with torch.no_grad():
                            self.accelerator.unwrap_model(model).logit_scale.data.clamp_(0, math.log(100))
                    else:
                        self.model.logit_scale.data = torch.clamp(self.model.logit_scale.data, max=100)

                    if self.accelerator.sync_gradients:
                        gradientNormValue = self.accelerator.clip_grad_norm_(self.model.parameters(), 20)
                        if self.accelerator.is_local_main_process: self.writer.add_scalar("Gradient norm", gradientNormValue, global_step=global_step)

                    self.optimizer.step()
                    self.scheduler.step()


                    if self.accelerator.is_local_main_process:
                        self.writer.add_scalar("Learning rate", self.scheduler.get_lr()[0], global_step=global_step)
                        self.writer.add_scalar("Loss", total_loss.item(), global_step=global_step)

                    if idx > self.iterationPerEpoch:
                        print("Reached iteration limit")
                        break

                    self.currentStep = idx + 1
                    if global_step % 400 == 399: break
                else:
                    break


                self.save_model(currentEpoch=epoch, currentStep=self.currentStep)
                self.validate(global_step)
                self.accelerator.wait_for_everyone()
                showFirstText = True

            self.currentStep = 0
        
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

