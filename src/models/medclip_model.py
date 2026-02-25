# code credit: tsou
import torch
import torch.nn as nn
from medclip import MedCLIPModel, MedCLIPVisionModel
from medclip import MedCLIPProcessor
from PIL import Image

# prepare for the demo image and texts
# processor = MedCLIPProcessor()
# image = Image.open('../data/samples/00000003_000.png')
# inputs = processor(
#     text=["lungs remain severely hyperinflated with upper lobe emphysema", 
#         "opacity left costophrenic angle is new since prior exam ___ represent some loculated fluid cavitation unlikely"], 
#     images=image, 
#     return_tensors="pt", 
#     padding=True
#     )

# # pass to MedCLIP model
# model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
# model.from_pretrained()
# model.cuda()
# outputs = model(**inputs)
# print(outputs.keys())
# dict_keys(['img_embeds', 'text_embeds', 'logits', 'loss_value', 'logits_per_text'])

class MedCLIP():
    def __init__(self):
        self.processor = MedCLIPProcessor()
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModel) #changed from ViT to VisionModel for ResNet-50 backbone
        self.model.from_pretrained()
        self.model.eval() 
        self.model.cuda()

        # add hook store we can write into and read from
        self.hook_store = {}

        

    def get_embeddings(self,image_paths,labels): # loads images and text, gets the embedding
        #Load the images
        images = []
        for p in image_paths:
            images.append(Image.open(p))
        
        #Process the inputs and pass to the model
        inputs = self.processor(
            text=labels, 
            images=images, 
            return_tensors="pt", 
            padding=True
            )
        outputs = self.model(**inputs)
        embeddings = {
            'img_embeds':outputs['img_embeds'].tolist(),
            'text_embeds':outputs['text_embeds'].tolist()
        }

        #Return the emddings
        return embeddings

    def get_predictions(self,image_paths,labels): # the zero-shot classification ability of CLIP-based models
        #Load the images
        images = []
        for p in image_paths:
            images.append(Image.open(p))
        
        #Process the inputs and pass to the model
        inputs = self.processor(
            text=labels, 
            images=images, 
            return_tensors="pt", 
            padding=True
            )
        outputs = self.model(**inputs)
        logits = outputs['logits']
        softmax = logits.softmax(dim=1)
        #Return the emddings
        return softmax.tolist()
    
    def get_embeddings_and_predictions(self,image_paths,labels): # combines the above into one pass
        #Load the images
        images = []
        for p in image_paths:
            images.append(Image.open(p))
        
        #Process the inputs and pass to the model
        inputs = self.processor(
            text=labels, 
            images=images, 
            return_tensors="pt", 
            padding=True
            )
        outputs = self.model(**inputs)

        result = {
            'img_embeds':outputs['img_embeds'].tolist(),
            'text_embeds':outputs['text_embeds'].tolist(),
            'logits':outputs['logits'].tolist()
        }
        #Return the emddings
        return result