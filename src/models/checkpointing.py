# checkpointing.py: this script will help us, if the model stops running, we have a checkpoitn to continue training from

import torch

def checkpoint(model, optimiser, epoch, filename):
    torch.save({
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'epoch': epoch,
    }, filename)
    
def resume(model, optimiser, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimiser'])
    return checkpoint['epoch']