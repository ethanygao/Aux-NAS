import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# poly schedule
def poly(start, end, steps, total_steps, period, power):
    """
    Default goes from start to end
    """
    delta = end - start
    rate = float(steps) / total_steps
    if rate <= period[0]:
        return start
    elif rate >= period[1]:
        return end
    base = total_steps * period[0]
    ceil = total_steps * period[1]
    return end - delta * (1. - float(steps - base) / (ceil - base)) ** power


def get_schedulers(config, optimizers):
    if config['General']['scheduler'] == 'reduce_plateau':
        return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]
    elif config['General']['scheduler'] == 'poly':
        return [optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda step: (1 - float(step) / config['General']['steps']) ** 0.9,
                                            last_epoch=-1)
                for optimizer in optimizers]
    elif config['General']['scheduler'] == 'cosine':
        return [optim.lr_scheduler.CosineAnnealingLR(optimizer, config['General']['steps']) for optimizer in optimizers]
    else:
        raise NotImplementedError