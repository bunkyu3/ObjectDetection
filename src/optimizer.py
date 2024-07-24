import torch

def set_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 
                                lr=cfg.train.learning_rate, 
                                momentum=cfg.train.momentum, 
                                weight_decay=cfg.train.weight_decay)
    return optimizer
