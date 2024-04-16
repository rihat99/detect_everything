# from transformers import OpenAIGPTTokenizer
from transformers import GPT2Tokenizer, GPT2Config
# from transformers import Owlv2VisionModel, Owlv2ImageProcessor
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import torch
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate


from PIL import Image
import json
import numpy as np

from model import GPT2WithImageEmbeddings

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image encoder
# model_owl = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
# processor_owl = Owlv2ImageProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)


# freeze the model
for param in model_owl.parameters():
    param.requires_grad = False

# Load tokenizer and model
# tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config.from_pretrained("distilgpt2", add_cross_attention=True)
model_gpt = GPT2WithImageEmbeddings.from_pretrained("distilgpt2", config=config).to(device)

# Load COCO dataset

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    # transforms.ToTensor()
])

def collate_fn(batch):

    # Separate images and captions, and collect all captions
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    # Use default_collate to handle images (and other tensor data automatically)
    # convert images to numpy array
    images = [np.array(image) for image in images]
    images = default_collate(images)
    
    # Choose random caption from the list of captions for each image
    captions = [caption[np.random.randint(len(caption))] for caption in captions]

    # Tokenize the captions
    captions = tokenizer(captions, padding="max_length", return_tensors='pt')['input_ids']

    # Pad the captions to the maximum length caption in the batch
    # captions_padded = pad_sequence(captions, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {'images': images, 'captions': captions}

train_dataset = CocoCaptions("coco/train2017/", annFile="coco/annotations/captions_train2017.json", transform=transform)
val_dataset = CocoCaptions("coco/val2017/", annFile="coco/annotations/captions_val2017.json", transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


# Training loop (simplified)
optimizer = torch.optim.Adam(model_gpt.parameters(), lr=1e-6)

model_owl.eval()
model_gpt.train()

for epoch in range(10):  # Example for 1 epoch
    for step, batch in enumerate(train_dataloader):  # Assuming coco_dataloader is defined
        # inputs, labels = batch

        with torch.no_grad():
            input_images = processor_owl(text=[[""]], images=batch['images'], return_tensors="pt").to(device)
            # image_embeddings = model_owl(**input_images).last_hidden_state[:, 1:, :].contiguous()
            outputs_owl = model_owl(**input_images)
            all_image_embeddings = outputs_owl.image_embeds.reshape(1, -1, 768)

            obejectness = torch.sigmoid(outputs_owl.objectness_logits) > 0.15
            indices = torch.where(obejectness)[1]
            image_embeddings = all_image_embeddings[:, indices, :].contiguous()

        # inputs = tokenizer(inputs["captions"], return_tensors="pt", padding=True, truncation=True).to(device)
        captions = batch["captions"].to(device).contiguous()
        # print(captions.shape)
        # print(image_embeddings.shape)
        outputs = model_gpt(input_ids=captions, image_embeddings=image_embeddings, labels=captions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
            model_gpt.save_pretrained("weights")

# Save model
model_gpt.save_pretrained("weights")