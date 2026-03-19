# checkpointing.py: this script will help us, if the model stops running, we have a checkpoitn to continue training from

import torch

def checkpoint(optimiser, model, filename):
    torch.save({
        'optimiser': optimiser.state_dict(),
        'model': model.state_dict(),
    }, filename)
    
def resume(model, optimiser, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimiser'])