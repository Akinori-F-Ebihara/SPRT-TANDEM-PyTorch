import pdb
from typing import Tuple, Dict, Any
from tqdm import tqdm
from termcolor import colored
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import Trial
from config.config_definition import config as config_orig
from utils.misc import grab_gpu, fix_random_seed, parse_args, set_cpu_workers
from utils.logging import (
    create_log_folders,
    save_config_info,
    ContexualLogger,
    save_sdre_imgs,
    log_training_results,
    get_tb_writer,
)
from utils.hyperparameter_tuning import run_optuna, suggest_parameters, report_to_pruner
from utils.checkpoint import (
    update_and_save_result,
    initialize_objectives,
    finalize_objectives,
)
from datasets.data_processing import lmdb_dataloaders, move_data_to_device
from models.temporal_integrators import import_model, load_pretrained_states
from models.optimizers import initialize_optimizer
from models.losses import compute_loss_and_metrics
from utils.misc import PyTorchPhaseManager, convert_torch_to_numpy
from utils.performance_metrics import training_setup, accumulate_performance


@logger.catch
def prepare_for_training(
    trial: Trial, config: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, optim.Optimizer, Dict[str, data.DataLoader], SummaryWriter]:
    """
    Prepare the network and data for training.

    Args:
    - trial: An object used by Optuna to generate trial parameters. Unused if not using Optuna.
    - config: A dictionary containing various configuration settings.
    - device: The device to run the computation on, e.g. 'cpu' or 'cuda'.

    Returns:
    - model: The initialized network.
    - optimizer: The optimizer used to update the network parameters during training.
    - data_loaders: A dictionary containing the train, validation, and test data loaders.
    - tb_writer: A SummaryWriter object for logging training progress to TensorBoard.
    """
    # accurate matrix multiplication when working with 32-bit floating-point values
    torch.set_float32_matmul_precision("high")

    # optimize cuda computation if network structure is static, sacrificing reproducibility
    torch.backends.cudnn.benchmark = True

    # set number of CPU threads (workers)
    set_cpu_workers(config)

    # set random seeds (optional)
    fix_random_seed(config)

    # save config.py and config (dict) for reproducibility
    save_config_info(config)

    # setup Tensorboard writer
    tb_writer = get_tb_writer(config)

    # initialize the network or load a pretrained one
    # tb_writer is optional: provide if you want to have a model graph on TB
    model = import_model(config, device, tb_writer=None)

    # setup the optimizer
    model, optimizer = initialize_optimizer(model, config)

    # (optional) load pretrained state_dicts
    load_pretrained_states(model, optimizer, config)

    # load train, val, and test data
    data_loaders = lmdb_dataloaders(config)

    return model, optimizer, data_loaders, tb_writer


@logger.catch
def iterating_over_dataset(
    trial: Trial,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    data_loaders: Dict[str, data.DataLoader],
    tb_writer: SummaryWriter,
    config: Dict[str, Any],
    best: Tuple[float, ...],
    global_step: int,
    phase: str,
) -> Tuple[Dict[str, float], int, Tuple[float], Tuple[float, ...]]:
    """
    Runs the model for a given phase (training, validation, or test) and logs results.

    Args:
    - trial: An object used by Optuna to generate trial parameters. Unused if not using Optuna.
    - model: The PyTorch model to be run.
    - optimizer: The PyTorch optimizer used to update model parameters during training.
    - device: The device to run the computation on, e.g. 'cpu' or 'cuda'.
    - data_loaders: A dictionary containing train, validation, and test data loaders.
    - tb_writer: A TensorBoard SummaryWriter to log results.
    - config: A dictionary containing model configuration parameters.
    - best: A tuple of the current best performance metrics.
    - global_step: The current iteration number.
    - phase: The phase for which the model is being run. Can be 'train', 'validation', or 'test'.

    Returns:
    - performance_metrics: A dictionary containing the performance metrics.
    - global_step: The updated iteration number.
    - last_example_to_plot: An object for plotting results.
    - best: A tuple of the updated best performance metrics.
    """

    is_train, iter_num, performance_metrics, barcolor = training_setup(phase, config)

    for local_step, data in tqdm(
        enumerate(data_loaders[phase]),
        mininterval=2,
        total=iter_num,
        desc=colored(f"{phase} in progress...", barcolor),
        colour=barcolor,
        leave=False,
    ):
        x_batch, y_batch, gt_llrs_batch = move_data_to_device(data, device)

        # a periodic validation is inserted between training iterations.
        # This function is recursively called under is_train=True, with phase='eval'.
        if is_train and global_step % config["VALIDATION_STEP"] == 0:
            best, global_step = run_one_epoch(
                trial,
                model,
                optimizer,
                device,
                data_loaders,
                tb_writer,
                config,
                best,
                global_step=global_step,
                phase="val",
            )

        # Train phase: run preprocesses (model.train(), optimizer.zero_grad()) and postprocesses (optimizer.step(), loss.backward())
        # Eval phase: run preprocesses (model.eval()), enter torch.no_grad() mode, no postprocess
        with PyTorchPhaseManager(
            model,
            optimizer,
            phase=phase,
            loss_func=compute_loss_and_metrics,
            loss_args=(x_batch, y_batch, global_step),
            config=config,
        ) as p:
            monitored_values = p.call_loss()
            p.loss = monitored_values["losses"]["total_loss"]

        # Store performance metrics
        performance_metrics = accumulate_performance(
            performance_metrics,
            y_batch,
            gt_llrs_batch,
            monitored_values,
            phase_manager=p,
        )

        # log results periodically. Return immediately when the condition is not met
        # performance_metrics is reset at training loop to avoid running out of memory
        performance_metrics = log_training_results(
            tb_writer,
            model,
            local_step,
            global_step,
            iter_num,
            config,
            performance_metrics,
            phase,
        )

        # for figures
        if local_step == iter_num - 1 and "val" in phase:
            last_example_to_plot = convert_torch_to_numpy(
                [
                    y_batch,
                    gt_llrs_batch,
                    monitored_values["llrs"],
                    monitored_values["thresholds"],
                ]
            )
        else:
            last_example_to_plot = None

        # increment global step that traces training process. skip if eval phase
        global_step = global_step + 1 if "train" in phase else global_step

    return performance_metrics, global_step, last_example_to_plot, best


@logger.catch
def eval_postprocess(
    model: nn.Module,
    optimizer: optim.Optimizer,
    best: Tuple[float, ...],
    performance_metrics: Dict[str, float],
    last_example_to_plot: Tuple[float],
    config: Dict[str, Any],
    tb_writer: SummaryWriter,
    global_step: int,
) -> Tuple[float, ...]:
    """
    Updates the status and saves a checkpoint.

    Args:
    - model: The PyTorch model.
    - optimizer: The PyTorch optimizer.
    - best: A tuple of the current best performance metrics.
    - performance_metrics: A dictionary containing the performance metrics.
    - last_example_to_plot: An object for plotting results.
    - config: A dictionary containing model configuration parameters.
    - global_step: The current iteration number.

    Returns:
    - best: A tuple of the updated best performance metrics.
    """
    # update status and save a checkpoint
    best = update_and_save_result(
        model, optimizer, config, best, performance_metrics, global_step=global_step
    )

    # save trajectory figures if needed
    save_sdre_imgs(
        config, best, tb_writer, global_step, last_example_to_plot, performance_metrics
    )

    return best


def run_one_epoch(
    trial: optuna.Trial,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    data_loaders: Dict[str, data.DataLoader],
    tb_writer: SummaryWriter,
    config: Dict[str, Any],
    best: Tuple[float, ...],
    global_step: int,
    phase: str = "train",
) -> Tuple[Tuple[float, ...], int]:
    """
    Runs one epoch of the model.

    Args:
    - trial: An optuna Trial object.
    - model: The PyTorch model.
    - optimizer: The PyTorch optimizer.
    - device: The device on which to run the computation.
    - data_loaders: A dictionary containing the data loaders for the train, validation, and test sets.
    - tb_writer: A tensorboard SummaryWriter to log results.
    - config: A dictionary containing model configuration parameters.
    - best: A tuple of the current best performance metrics.
    - global_step: The current iteration number.
    - phase: The phase for which the model is being run. Can be 'train', 'validation', or 'test'.

    Returns:
    - best: A tuple of the updated best performance metrics.
    - global_step: The current iteration number.
    """
    (
        performance_metrics,
        global_step,
        last_example_to_plot,
        best,
    ) = iterating_over_dataset(
        trial,
        model,
        optimizer,
        device,
        data_loaders,
        tb_writer,
        config,
        best,
        global_step,
        phase,
    )

    # postprocesses at an eval phase
    if "val" in phase:
        best = eval_postprocess(
            model,
            optimizer,
            best,
            performance_metrics,
            last_example_to_plot,
            config,
            tb_writer,
            global_step,
        )

        # optuna pruner for early stopping
        report_to_pruner(trial, best, global_step, config)

    return best, global_step


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
        model, optimizer, data_loaders, tb_writer = prepare_for_training(
            trial, config, device
        )

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
