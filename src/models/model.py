from medclip import MedCLIPModel, MedCLIPVisionModel

model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()
print(model)