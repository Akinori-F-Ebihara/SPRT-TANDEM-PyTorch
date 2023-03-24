import torch.optim as optim
from utils.misc import extract_params_from_config


def initialize_optimizer(model, config):
    """
    Args:
        learning_rates: A list of learning rates.
        decay_steps: A list of steps at which learnig rate decays.
        name_optimizer: A str.
        flag_wd: A boolean.
        weight_decay: A float.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(['LEARNING_RATE', 'LR_DECAY_STEPS', 'OPTIMIZER', 'WEIGHT_DECAY'])
    conf = extract_params_from_config(requirements, config)

    if conf.optimizer == "adam":
        optimizer = optim.AdamW(
                model.parameters(),
                weight_decay=conf.weight_decay, 
                lr=conf.learning_rate)
    elif conf.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            weight_decay=conf.weight_decay, 
            lr=conf.learning_rate)
    elif conf.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=conf.learning_rate, 
            alpha=0.9, 
            momentum=0.0)
    else:
        raise ValueError(f'Optimizer "{conf.optimizer}" is not implemented!')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    return optimizer, scheduler

