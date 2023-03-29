import sys, os, signal, pdb
import numpy as np
from tqdm import tqdm
from termcolor import colored
from loguru import logger
import torch
from config.config import config
from utils.misc import grab_gpu, fix_random_seed, parse_args
from utils.logging import create_log_folders, save_config_info, ContexualLogger,\
                          save_sdre_imgs, log_training_results, get_tb_writer
from utils.hyperparameter_tuning import run_optuna, suggest_parameters_optuna, report_to_pruner
from utils.checkpoint import update_and_save_result, \
                             initialize_objectives, finalize_objectives
from datasets.data_processing import lmdb_dataloaders, move_to_device
from models.temporal_integrators import import_model, load_pretrained_states
from models.optimizers import initialize_optimizer
from models.losses import compute_loss_and_metrics
from utils.misc import PyTorchPhaseManager, convert_torch_to_numpy
from utils.performance_metrics import training_setup, accumulate_performance


@logger.catch
def prepare_for_training(trial, config, device):
    '''
    '''
    # accurate matrix multiplication when working with 32-bit floating-point values
    torch.set_float32_matmul_precision('high')
    
    # optimize cuda computation if network structure is static, sacrificing reproducibility
    torch.backends.cudnn.benchmark = True

    # set random seeds (optional)
    fix_random_seed(config)

    # Suggest parameters if the experimental phase is "tuning"
    # This function may modify config
    is_losscoef_all_zero = suggest_parameters_optuna(trial, config)

    # save config.py and config (dict) for reproducibility
    save_config_info(config)

    # setup Tensorboard writer
    tb_writer = get_tb_writer(config)

    # initialize the network or load a pretrained one
    # tb_writer is optional: provide if you want to have a model graph on TB
    model = import_model(config, device, tb_writer=tb_writer)

    # setup the optimizer
    optimizer, schedular = initialize_optimizer(model, config)

    # (optional) load pretrained state_dicts
    load_pretrained_states(model, optimizer, config)

    # load train, val, and test data
    data_loaders = lmdb_dataloaders(config)

    return is_losscoef_all_zero, model, optimizer, data_loaders, tb_writer


@logger.catch
def iterating_over_dataset(trial, model, optimizer, device, data_loaders, tb_writer, config, best, global_step, phase):
    '''
    Runs the model for a given phase (training, validation, or test) and logs results.

    Args:
    - model (nn.Module): The model to be run.
    - optimizer (torch.optim): The optimizer used to update model parameters during training.
    - data_loaders (dict of torch.utils.data.DataLoader): Database dict with keys 'train', 'val', and 'test'.
    - tb_writer (torch.utils.tensorboard.SummaryWrite): Tensorboard file writer to log results.
    - config (dict): A dictionary containing model configuration parameters.
    - global_step (int): The current iteration number.
    - phase (str): The phase for which the model is being run. Can be 'train', 'validation', or 'test'.

    Returns:
    - performance_metrics
    - global_step
    - last_example_to_plot
    '''

    is_train, iter_num, performance_metrics, barcolor = training_setup(phase, config)

    for local_step, data in tqdm(enumerate(data_loaders[phase]),
                                               mininterval=2, total=iter_num,
                                               desc=colored(f'{phase} in progress...', barcolor),
                                               colour=barcolor, leave=False):
        if len(data) == 3:
            # if the dataset contains ground-truth log likelihood ratio
            x_batch, y_batch, gt_llrs_batch = data
            x_batch, y_batch, gt_llrs_batch = move_to_device(device, x_batch, y_batch, gt_llrs_batch)
        elif len(data) == 2:
            # data and label only: typical real-world data
            x_batch, y_batch = data
            x_batch, y_batch = move_to_device(device, x_batch, y_batch)
            
        else:
            raise ValueError('data tuple length is expected either to be '
                            f'3 (x, y, llr) or 2 (x, y) but got {len(data)=}!')

        # a periodic validation is inserted between training iterations.
        # This function is recursively called under is_train=True, with phase='eval'.
        if is_train and global_step % config['VALIDATION_STEP'] == 0:
            best, global_step = run_one_epoch(trial, model, optimizer, device, data_loaders,
                                          tb_writer, config, best, global_step=global_step, phase='val')
        
        # Train phase: run preprocesses (model.train(), optimizer.zero_grad()) and postprocesses (optimizer.step(), loss.backward())
        # Eval phase: run preprocesses (model.eval()), enter torch.no_grad() mode, no postprocess
        with PyTorchPhaseManager(model, optimizer, phase=phase) as p:
            p.loss, monitored_values = compute_loss_and_metrics(
                p.model, x_batch, y_batch, global_step, config=config)

        # Store performance metrics
        performance_metrics = accumulate_performance(
            performance_metrics, y_batch, gt_llrs_batch, monitored_values)

        # log results periodically. Return immediately when the condition is not met
        # performance_metrics is reset at training loop to avoid running out of memory
        performance_metrics = log_training_results(
            tb_writer, model, local_step, global_step, iter_num, config, performance_metrics, phase)
            
        # for figures
        last_example_to_plot = convert_torch_to_numpy(
                                    [y_batch, gt_llrs_batch, monitored_values['llrs']])

        # increment global step that traces training process. skip if eval phase
        global_step = global_step + 1 if 'train' in phase else global_step

    return performance_metrics, global_step, last_example_to_plot, best


@logger.catch
def eval_postprocess(model, optimizer, best, performance_metrics, last_example_to_plot, config, global_step):
    '''
    '''
    # update status and save a checkpoint
    best = update_and_save_result(model, optimizer, config, best, performance_metrics, global_step=global_step)

    # save trajectory figures if needed
    save_sdre_imgs(config, best, global_step, last_example_to_plot, performance_metrics)

    return best


def run_one_epoch(trial, model, optimizer, device, data_loaders,
              tb_writer, config, best, global_step, phase='train'):
    '''
    '''
    performance_metrics, global_step, last_example_to_plot, best =\
        iterating_over_dataset(trial, model, optimizer, device, data_loaders,
                      tb_writer, config, best, global_step, phase)

    # postprocesses at an eval phase
    if 'val' in phase:

        best = eval_postprocess(
            model, optimizer, best, performance_metrics, last_example_to_plot, config, global_step)

        # optuna pruner for early stopping
        report_to_pruner(trial, best, global_step, config)

    return best, global_step


def objective(trial, device, config):
    '''
    '''
    create_log_folders(config) # This function modifies config

    # Outputs under this context will be logged into one file.
    # you shall not use @logger.catch other than the functions used under
    # this context because it may let the code running even under errors.
    # The log file will be investigated earch time the code exits from
    # the context. Error will be raised when error logs are found.
    with ContexualLogger(config):

        is_losscoef_all_zero, model, optimizer, data_loaders,\
            tb_writer = prepare_for_training(trial, config, device)
        if is_losscoef_all_zero: return np.Inf # cannot train if losses are all=zero!

        best = initialize_objectives(config)
        global_step = 0
        for epoch in tqdm(range(config['NUM_EPOCHS']), mininterval=5,
                          desc=colored('Epoch progress: ', 'blue'), colour='blue'):

            logger.info(f'Starting epoch #{epoch}.')
            best, global_step = run_one_epoch(trial, model, optimizer, device, data_loaders,
                                          tb_writer, config, best, global_step)

    return finalize_objectives(best)


def main():

    # use arguments to overwrite the config file
    parse_args(config)
    # grab a GPU
    device = grab_gpu(config)
    # start learning!
    run_optuna(objective, device, config)

    return


if __name__ == '__main__':
    main()
