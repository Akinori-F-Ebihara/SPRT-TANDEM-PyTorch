# import tensorboard # just for launching TensorBoard Session on VSCode
import pdb
from typing import Tuple, Dict, Any
from tqdm import tqdm
from termcolor import colored
from loguru import logger
import torch
import optuna
from config.config_definition import config as config_orig
from utils.misc import grab_gpu, parse_args
from utils.logging import create_log_folders, ContexualLogger
from utils.hyperparameter_tuning import run_optuna, suggest_parameters
from utils.checkpoint import initialize_objectives, finalize_objectives
from utils.training import prepare_for_training, run_one_epoch


def objective(
    trial: optuna.Trial, device: torch.device, config_orig: Dict[str, Any]
) -> Tuple[float, ...]:
    """
    Defines the objective function for Optuna to optimize.

    Args:
    - trial: An optuna Trial object.
    - device: The device on which to run the computation.
    - config_orig: A dictionary containing model configuration parameters.

    Returns:
    - Any: The value of the objective function.
    """
    # copy and tune for each trial
    config = config_orig.copy()

    # Suggest parameters if the experimental phase is "tuning"
    # Run params sanity check under other phases.
    # This function may modify config in-place.
    suggest_parameters(trial, config)

    # This function modifies config
    create_log_folders(config)

    # Outputs under this context will be logged into one file.
    # you shall not use @logger.catch other than the functions used under
    # this context because it may let the code running even under errors.
    # The log file will be investigated earch time the code exits from
    # the context. Error will be raised when error logs are found.
    with ContexualLogger(config, is_stop_at_error=False):
        model, optimizer, data_loaders, tb_writer = prepare_for_training(config, device)

        best = initialize_objectives(config)
        global_step = 0
        for epoch in tqdm(
            range(config["NUM_EPOCHS"]),
            mininterval=5,
            desc=colored("Epoch progress: ", "blue"),
            colour="blue",
        ):
            logger.info(f"Starting epoch #{epoch}.")
            best, global_step = run_one_epoch(
                trial,
                model,
                optimizer,
                device,
                data_loaders,
                tb_writer,
                config,
                best,
                global_step,
            )

    return finalize_objectives(best)


def main() -> None:
    """
    The main function of the program.

    Returns:
    - None
    """
    # use arguments to overwrite the config file
    parse_args(config_orig)
    # grab a GPU
    device = grab_gpu(config_orig)
    # start learning!
    run_optuna(objective, device, config_orig)

    return


if __name__ == "__main__":
    main()
