from typing import Tuple, Dict, Any
import torch
import torch.optim as optim
from loguru import logger
from utils.misc import extract_params_from_config


def initialize_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> Tuple[torch.nn.Module, optim.Optimizer]:
    """
    Initializes the optimizer for training the model, based on the configuration parameters.

    Args:
    - model: The model to be trained.
    - config: A dictionary containing model configuration parameters.

    Returns:
    - Tuple[Module, optim.Optimizer]: A tuple containing the initialized model and optimizer.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "LEARNING_RATE",
            "LR_DECAY_STEPS",
            "OPTIMIZER",
            "WEIGHT_DECAY",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    if "adam" in conf.optimizer.lower():
        base_optimizer = optim.AdamW
    elif conf.optimizer.lower() == "rmsprop":
        base_optimizer = optim.RMSprop
    else:
        raise ValueError(f'Optimizer "{conf.optimizer}" is not implemented!')

    optimizer = base_optimizer(
        model.parameters(),
        weight_decay=conf.weight_decay,
        lr=conf.learning_rate,
    )
    logger.info(f"Optimizer:\n{optimizer}")

    return model, optimizer
