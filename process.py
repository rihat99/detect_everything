from transformers import Owlv2Processor, Owlv2ForObjectDetection
from clip_text_decoder.common import load_vision_backbone

import requests
from PIL import Image
import torch
from torchvision.datasets import CocoCaptions

import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import shutil
import os

from datasets import load_dataset
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

blip_model, processor_blip = load_vision_backbone("blip:base", device)

# model_owl.eval()
blip_model.eval()

split = "train"

# if split == "val":
#     dataset = CocoCaptions("coco/val2017/", annFile="coco/annotations/captions_val2017.json")
# elif split == "train":
#     dataset = CocoCaptions("coco/train2017/", annFile="coco/annotations/captions_train2017.json")

class MyLVISDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS", "default")
        
        self.img_names = {}

        for i in tqdm(range(len(self.dataset["train"]))):
            image_url = self.dataset["train"][i]['url']
            split = image_url.split("/")[-2]
            image_id = image_url.split("/")[-1]
            image_path = f"{self.data_path}/{split}/{image_id}"
            
            if image_path not in self.img_names:
                self.img_names[image_path] = []
            self.img_names[image_path].append(i)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        key = list(self.img_names.keys())[idx]

        captions = [self.dataset["train"][i]['caption'] for i in self.img_names[key]]
        image = Image.open(key).convert("RGB")

        return image, captions
    
dataset = MyLVISDataset("coco")


# filename = f"{split}_data.pkl"
save_path = f"cached_data/{split}_blip_extended_data"

shutil.rmtree(save_path, ignore_errors=True)

# create a folder
os.makedirs(save_path, exist_ok=True)


for i in tqdm(range(len(dataset))):
    img, captions = dataset[i]

    # with torch.no_grad():
    #     inputs = processor_blip(text=[[""]], images=img, return_tensors="pt").to(device)
    #     outputs = model_blip(**inputs)
    
    # all_image_embeddings = outputs.image_embeds.reshape(1, -1, 768)

    # obejectness = torch.sigmoid(outputs.objectness_logits) > 0.2
    # indices = torch.where(obejectness)[1]
    # image_embeddings = all_image_embeddings[:, indices, :]

    # image_embeddings = image_embeddings.cpu().numpy()

    # all_image_embeddings = all_image_embeddings.cpu().numpy()

    img_processed = processor_blip(img).unsqueeze(0).to(device)

    features = blip_model.extract_features({"image": img_processed}, mode="image")
    image_embeddings = features.image_embeds[:, 0].cpu().numpy()

    np.savez(f"{save_path}/{i}.npz", embedding=image_embeddings, captions=captions)


    


