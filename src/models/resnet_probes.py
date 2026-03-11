# models/resnet_probes.py, frozen MedCLIP ResNet-50 backbone with  LinearProbe
# attached after each of the 16 bottleneck blocks via forward hooks,
# plus one final probe on the avgpool output.


import functools
import torch
import torch.nn as nn
from medclip import MedCLIPModel, MedCLIPVisionModel

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes=2): # Little classifier
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = self.pool(x).flatten(1)
        return self.fc(x)

class FrozenResNetWithProbes(nn.Module): # building the frozen Resnet-50
    def __init__(self, num_classes=2):
        super().__init__()

        # loading MedCLIPs resnet-50 with the pretrained weights
        original_load = torch.load
        torch.load = functools.partial(original_load, map_location="cpu")
        medclip = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        medclip.from_pretrained()
        torch.load = original_load
        backbone = medclip.vision_model.model

        for p in backbone.parameters(): # Freeze all parameters in resnet-50
            p.requires_grad_(False)

        # input preprocessing
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool) 
        
        # define all 4 stages as layers
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Extract channel sizes from each bottleneck block
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        layer_channels = [block.conv3.out_channels for layer in layers for block in layer]

        # Define probes that matches the size of their layers
        self.probes = nn.ModuleList([LinearProbe(c, num_classes) for c in layer_channels]) 

        # Final output probe
        self.final_probe = LinearProbe(layer_channels[-1], num_classes)

        self._activations: list = []
        self._hooks: list = []
        self._register_hooks()

    # forward Hook logic
    def _register_hooks(self):
        # make a list with the 4 stages, that each has layers, adding up to 16
        blocks = (list(self.layer1) + list(self.layer2) + list(self.layer3) + list(self.layer4)) 
        
        for block in blocks: # for each block attach a hook
            h = block.register_forward_hook(lambda m, inp, out: self._activations.append(out))
            self._hooks.append(h)

    # Removing the hooks to avoid compute resources
    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # forward pass
    def forward(self, x):
        self._activations.clear()

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)   # (batch size, 2048)

        # Match up the activations with each probe
        probe_logits = [probe(act) for probe, act in zip(self.probes, self._activations)]

        # final prediction
        final_logit = self.final_probe(x)

        return probe_logits, final_logit