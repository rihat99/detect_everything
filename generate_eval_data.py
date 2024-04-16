from transformers import GPT2Tokenizer
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch
from torch.utils.data import Dataset

# from clip_text_decoder.model import DecoderInferenceModel
from model import DecoderInferenceModel
from clip_text_decoder.common import load_vision_backbone
import spacy

from PIL import Image
import numpy as np
import cv2
import json

from lvis import LVIS
from tqdm import tqdm

class DetectEverything:
    def __init__(self, decoder_weights: str, device: str):
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl.to(self.device)
        self.model_owl.eval()

        self.model_gpt = DecoderInferenceModel.load(decoder_weights).to(device)

        self.nlp = spacy.load('en_core_web_md')

    def get_image_features(self, image: Image.Image, objectness_threshold=0.2):
        inputs = self.processor_owl(text=[[""]], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_owl(**inputs)

        image_embeddings = outputs.image_embeds.reshape(1, -1, 768)
        obejectness = torch.sigmoid(outputs.objectness_logits) > objectness_threshold
        indices = torch.where(obejectness)[1]

        boxes = outputs.pred_boxes[0][indices]
        image_embeddings = image_embeddings[:, indices, :]

        return image_embeddings, boxes
    
    
    def get_captions(self, image_embeddings):
        all_captions = []

        for i in range(image_embeddings.shape[1]):
            with torch.no_grad():
                # outputs = inference_model(x = image_embeddings[:, i, :].unsqueeze(1), max_len=100, beam_size=3, input_text="Objects list:")
                outputs = self.model_gpt(x = image_embeddings[:, i, :].unsqueeze(1), max_len=100, beam_size=3, input_text=None)

                all_captions.append(outputs)

        return all_captions
    
    
    def extract_object_names(self, sentence):
        # Process the sentence
        doc = self.nlp(sentence)
        
        # Words to exclude specifically
        exclude_words = {"various", "several", "multiple", "many", "few", "some", "similar" 
                        , "bottom", "lower", "right", "left", "top", "upper", "center"}

        # Container for object names
        object_names = []

        # Iterate over tokens in the sentence
        for token in doc:
            # Skip unwanted words based on POS tags, dependency relations, and specific excludes
            if (token.pos_ in ['DET', 'ADP', 'NUM', 'AUX', 'PART'] or 
                token.dep_ in ['aux', 'prep', 'det', 'nummod'] or 
                token.text.lower() in exclude_words):
                continue

            # Append nouns, proper nouns, adjectives (if modifying a noun), or participles
            if token.pos_ in ['NOUN', 'PROPN'] or (token.dep_ in ['amod', 'compound', 'nsubj'] and token.head.pos_ in ['NOUN', 'PROPN']):
                # Check for compound nouns or noun-adjective combinations
                if token.dep_ == 'compound' or (token.dep_ == 'amod' and token.head.dep_ != 'ROOT'):
                    # Ensure the adjective is not in the exclude list
                    if token.text.lower() not in exclude_words:
                        object_name = f"{token.text} {token.head.text}"
                        object_names.append(object_name)
                else:
                    object_names.append(token.text)
        
        # Remove duplicates and reassemble object names considering adjacency
        object_names = list(set(object_names))
        
        # if objecy name is single word, change it to lemmatized form
        for i in range(len(object_names)):
            if len(object_names[i].split()) == 1:
                object_names[i] = self.nlp(object_names[i])[0].lemma_

        return object_names

    def get_all_object_names(self, all_captions):
        all_object_names = []

        for caption in all_captions:
            object_names = self.extract_object_names(caption)
            all_object_names.extend(object_names)

        all_object_names = list(set(all_object_names))

        return all_object_names        

    def owl_detection(self, image, object_names, threshold=0.25):
        inputs = self.processor_owl(text=[object_names], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_owl(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        return boxes, scores, labels
    
    def __call__(self, image: Image.Image, objectness_threshold=0.2, detection_threshold=0.25):
        width, height = image.size
        image = image.resize((1280, 1280))

        # print("Getting image features")
        image_embeddings, boxes = self.get_image_features(image, objectness_threshold=objectness_threshold)

        # print("Getting captions")
        all_captions = self.get_captions(image_embeddings)
        # print(all_captions)

        # print("Extracting object names")
        object_names = self.get_all_object_names(all_captions)
        # print(object_names)

        # print("Detecting objects")
        boxes, scores, labels = self.owl_detection(image, object_names, threshold=detection_threshold)

        labels = [object_names[i] for i in labels]

        for i in range(len(boxes)):
            box = boxes[i]

            x1, y1, x2, y2 = box
            x1 = int(x1 * width / 1280)
            y1 = int(y1 * height / 1280)
            x2 = int(x2 * width / 1280)
            y2 = int(y2 * height / 1280)

            boxes[i] = [x1, y1, x2, y2]

        return boxes, scores, labels

class DetectBaselineGPT:
    def __init__(self, decoder_weights: str, device: str):
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl.to(self.device)
        self.model_owl.eval()

        self.model_gpt = DecoderInferenceModel.load(decoder_weights).to(device)

        self.blip_model, self.processor_blip = load_vision_backbone("blip:base", device)

        self.nlp = spacy.load('en_core_web_md')

    def get_image_features(self, image: Image.Image):
        img_processed = self.processor_blip(image).unsqueeze(0).to(device)

        features = self.blip_model.extract_features({"image": img_processed}, mode="image")
        embeds = features.image_embeds[:, 0]

        return embeds
    
    
    def get_captions(self, image_embeddings):
        outputs = self.model_gpt(x = image_embeddings.unsqueeze(1), max_len=100, beam_size=3, input_text=None)

        return outputs
    
    
    def extract_object_names(self, sentence):
        # Process the sentence
        doc = self.nlp(sentence)
        
        # Words to exclude specifically
        exclude_words = {"various", "several", "multiple", "many", "few", "some", "similar" 
                        , "bottom", "lower", "right", "left", "top", "upper", "center"}

        # Container for object names
        object_names = []

        # Iterate over tokens in the sentence
        for token in doc:
            # Skip unwanted words based on POS tags, dependency relations, and specific excludes
            if (token.pos_ in ['DET', 'ADP', 'NUM', 'AUX', 'PART'] or 
                token.dep_ in ['aux', 'prep', 'det', 'nummod'] or 
                token.text.lower() in exclude_words):
                continue

            # Append nouns, proper nouns, adjectives (if modifying a noun), or participles
            if token.pos_ in ['NOUN', 'PROPN'] or (token.dep_ in ['amod', 'compound', 'nsubj'] and token.head.pos_ in ['NOUN', 'PROPN']):
                # Check for compound nouns or noun-adjective combinations
                if token.dep_ == 'compound' or (token.dep_ == 'amod' and token.head.dep_ != 'ROOT'):
                    # Ensure the adjective is not in the exclude list
                    if token.text.lower() not in exclude_words:
                        object_name = f"{token.text} {token.head.text}"
                        object_names.append(object_name)
                else:
                    object_names.append(token.text)
        
        # Remove duplicates and reassemble object names considering adjacency
        object_names = list(set(object_names))
        
        # if objecy name is single word, change it to lemmatized form
        for i in range(len(object_names)):
            if len(object_names[i].split()) == 1:
                object_names[i] = self.nlp(object_names[i])[0].lemma_

        return object_names

    def get_all_object_names(self, caption):
        object_names = self.extract_object_names(caption)
        object_names = list(set(object_names))

        return object_names        

    def owl_detection(self, image, object_names, threshold=0.25):
        inputs = self.processor_owl(text=[object_names], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_owl(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        return boxes, scores, labels
    
    def __call__(self, image: Image.Image, detection_threshold=0.25):
        width, height = image.size
        image = image.resize((1280, 1280))

        # print("Getting image features")
        image_embeddings = self.get_image_features(image)

        # print("Getting captions")
        all_captions = self.get_captions(image_embeddings)
        # print(all_captions)

        # print("Extracting object names")
        object_names = self.get_all_object_names(all_captions)
        # print(object_names)

        # print("Detecting objects")
        boxes, scores, labels = self.owl_detection(image, object_names, threshold=detection_threshold)

        labels = [object_names[i] for i in labels]

        for i in range(len(boxes)):
            box = boxes[i]

            x1, y1, x2, y2 = box
            x1 = int(x1 * width / 1280)
            y1 = int(y1 * height / 1280)
            x2 = int(x2 * width / 1280)
            y2 = int(y2 * height / 1280)

            boxes[i] = [x1, y1, x2, y2]

        return boxes, scores, labels

class DetectBaselineBLIP:
    def __init__(self, device: str):
        self.device = device

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.processor_owl = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model_owl.to(self.device)
        self.model_owl.eval()

        model_id = "Salesforce/blip-image-captioning-base"
        self.processor_blip = BlipProcessor.from_pretrained(model_id)
        self.model_blip = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

        self.nlp = spacy.load('en_core_web_md')

    
    
    def get_captions(self, image):
        inputs = self.processor_blip(images=image, return_tensors="pt", text="").to(device)

        # Generate captions
        outputs = self.model_blip.generate(**inputs, max_length=512, num_beams=3, return_dict_in_generate=True)

        # Decode and print the caption
        caption = self.processor_blip.decode(outputs.sequences[0], skip_special_tokens=True)

        return caption
    
    
    def extract_object_names(self, sentence):
        # Process the sentence
        doc = self.nlp(sentence)
        
        # Words to exclude specifically
        exclude_words = {"various", "several", "multiple", "many", "few", "some", "similar" 
                        , "bottom", "lower", "right", "left", "top", "upper", "center"}

        # Container for object names
        object_names = []

        # Iterate over tokens in the sentence
        for token in doc:
            # Skip unwanted words based on POS tags, dependency relations, and specific excludes
            if (token.pos_ in ['DET', 'ADP', 'NUM', 'AUX', 'PART'] or 
                token.dep_ in ['aux', 'prep', 'det', 'nummod'] or 
                token.text.lower() in exclude_words):
                continue

            # Append nouns, proper nouns, adjectives (if modifying a noun), or participles
            if token.pos_ in ['NOUN', 'PROPN'] or (token.dep_ in ['amod', 'compound', 'nsubj'] and token.head.pos_ in ['NOUN', 'PROPN']):
                # Check for compound nouns or noun-adjective combinations
                if token.dep_ == 'compound' or (token.dep_ == 'amod' and token.head.dep_ != 'ROOT'):
                    # Ensure the adjective is not in the exclude list
                    if token.text.lower() not in exclude_words:
                        object_name = f"{token.text} {token.head.text}"
                        object_names.append(object_name)
                else:
                    object_names.append(token.text)
        
        # Remove duplicates and reassemble object names considering adjacency
        object_names = list(set(object_names))
        
        # if objecy name is single word, change it to lemmatized form
        for i in range(len(object_names)):
            if len(object_names[i].split()) == 1:
                object_names[i] = self.nlp(object_names[i])[0].lemma_

        return object_names

    def get_all_object_names(self, caption):
        object_names = self.extract_object_names(caption)
        object_names = list(set(object_names))

        return object_names        

    def owl_detection(self, image, object_names, threshold=0.25):
        inputs = self.processor_owl(text=[object_names], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_owl(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor_owl.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        return boxes, scores, labels
    
    def __call__(self, image: Image.Image, detection_threshold=0.25):
        width, height = image.size
        image = image.resize((1280, 1280))

        # print("Getting captions")
        all_captions = self.get_captions(image)
        # print(all_captions)

        # print("Extracting object names")
        object_names = self.get_all_object_names(all_captions)
        # print(object_names)

        # print("Detecting objects")
        boxes, scores, labels = self.owl_detection(image, object_names, threshold=detection_threshold)

        labels = [object_names[i] for i in labels]

        for i in range(len(boxes)):
            box = boxes[i]

            x1, y1, x2, y2 = box
            x1 = int(x1 * width / 1280)
            y1 = int(y1 * height / 1280)
            x2 = int(x2 * width / 1280)
            y2 = int(y2 * height / 1280)

            boxes[i] = [x1, y1, x2, y2]

        return boxes, scores, labels

class MyValidationLvisDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.lvis = LVIS(data_path + "/lvis_v1_val.json")
        
        self.img_names = []
        self.image_ids = []

        for i in range(len(self.lvis.dataset["images"])):
            image_url = self.lvis.dataset["images"][i]['coco_url']
            split = image_url.split("/")[-2]
            image_id = image_url.split("/")[-1]
            image_path = f"{self.data_path}/{split}/{image_id}"
            
            self.image_ids.append(self.lvis.dataset["images"][i]['id'])
            self.img_names.append(image_path)
            

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_path = self.img_names[idx]
        image = Image.open(image_path)
        # convert to RGB
        image = image.convert("RGB")

        image_id = self.image_ids[idx]
        anns = self.lvis.get_ann_ids(img_ids=[image_id])
        anns = self.lvis.load_anns(anns)

        return image, anns
    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# detector = DetectEverything("clip-text-decoder/lightning_logs/version_12/model.pt", device)
detector = DetectBaselineGPT("clip-text-decoder/lightning_logs/version_16/model.pt", device)
# detector = DetectBaselineBLIP(device)

results = []

dataset = MyValidationLvisDataset("coco")

for i in tqdm(range(1000)):
    image, anns = dataset[i]

    boxes, scores, labels = detector(image)

    results.append({
        "id": i,
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels,
    })


# save results in json file
with open("results/baseline_gpt_long_captions.json", "w") as f:
    json.dump(results, f)