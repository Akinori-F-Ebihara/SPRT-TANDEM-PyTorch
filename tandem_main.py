import sys, os, pdb
import numpy as np
from tqdm import tqdm
from termcolor import colored
from loguru import logger
from config.config import config
from utils.misc import grab_gpu, fix_random_seed, parse_args, extract_params_from_config
from utils.logging import create_log_folders, save_config_info, ContexualLogger,\
                          save_sdre_imgs, log_training_results, get_tb_writer
from utils.hyperparameter_tuning import run_optuna, suggest_parameters_optuna, report_to_pruner
from utils.checkpoint import update_and_save_result, \
                             initialize_objectives, finalize_objectives
from datasets.data_processing import SequentialDensityRatioData, slice_data 
from models.temporal_integrators import import_model
from models.optimizers import initialize_optimizer
from models.losses import compute_loss_and_metrics
from utils.misc import PyTorchPhaseManager
from utils.performance_metrics import initialize_performance_metrics, get_train_flag_and_iter_range, validation_helper

def run_model(data, model, device, tb_writer, config, optimizer, global_step, best, phase):
    '''
    Runs the model for a given phase (training, validation, or test) and logs results.
   
    Args:
    - data (class instance): The data on which the model will be trained or evaluated.
    - model (nn.Module): The model to be run.
    - tb_writer (torch.utils.tensorboard.SummaryWrite): Tensorboard file writer to log results.
    - config (dict): A dictionary containing model configuration parameters.
    - optimizer (torch.optim): The optimizer used to update model parameters during training.
    - global_step (int): The current iteration number.
    - best (float): The current best performance metric.
    - phase (str): The phase for which the model is being run. Can be 'train', 'validation', or 'test'.
   
    Returns:
    - best (float): The updated performance metric if phase is 'validation' or 'test'
    '''
  
    requirements = set(['NUM_VAL', 'BATCH_SIZE', 'TIME_STEPS'])
    conf = extract_params_from_config(requirements, config)

    training, iter_range = get_train_flag_and_iter_range(phase, conf)

    performance_metrics = initialize_performance_metrics()
    for local_step in tqdm(range(iter_range), mininterval=2,
                    desc=colored(f'{phase} in progress...', 'cyan'), 
                    colour='cyan', disable=training, leave=False): 
        # Decode features and move to the device
        x_batch, y_batch, gt_llrs_batch = slice_data(data, device, phase, iter=global_step, 
                                                     batch_size=conf.batch_size)
        # Train phase: run preprocesses (model.train(), optimizer.zero_grad()) and postprocesses (optimizer.step(), loss.backward())
        # Eval phase: run preprocesses (model.eval()), enter torch.no_grad() mode, no postprocess
        with PyTorchPhaseManager(model, optimizer, phase=phase) as p:
            p.loss, monitored_values = compute_loss_and_metrics( 
                p.model, x_batch, y_batch, global_step, config=config)
            
        # Store performance metrics
        performance_metrics = validation_helper(performance_metrics, y_batch, gt_llrs_batch, conf,
                                                monitored_values, local_step, is_within_loop=True)
    # Summarize performance metrics
    performance_metrics = validation_helper(performance_metrics, y_batch, gt_llrs_batch, conf,
                                            monitored_values, local_step, is_within_loop=False)
    # log results
    log_training_results(tb_writer, model, global_step, config, performance_metrics, phase)
    if not training: # validation 
        # update status and save a checkpoint
        best = update_and_save_result(model, config, best, performance_metrics, global_step=global_step)       
        # save trajectory figures if needed
        save_sdre_imgs(config, best, global_step, y_batch, gt_llrs_batch, 
                        monitored_values, performance_metrics)
        return best


@logger.catch
def prepare_for_training(trial, config, device):

    fix_random_seed(config) # set random seeds (optional)

    # Suggest parameters if the experimental phase is "tuning"
    is_losscoef_all_zero = suggest_parameters_optuna(trial, config)  # This function may modify config

    save_config_info(config) # save config.py and config (dict) for reproducibility

    tb_writer = get_tb_writer(config)

    model = import_model(config, device, tb_writer) # initialize the network or load a pretrained one

    optimizer, schedular = initialize_optimizer(model, config) # setup the optimizer

    data = SequentialDensityRatioData(config) # load the data!

    return is_losscoef_all_zero, model, optimizer, tb_writer, data


@logger.catch
def run_training_loop(trial, config, model, device, optimizer, tb_writer, data):
    
    requirements = set(['NUM_ITER', 'VALIDATION_STEP'])
    conf = extract_params_from_config(requirements, config)

    # Initialization
    best = initialize_objectives(config)
    for global_step in tqdm(range(conf.num_iter), mininterval=2,
                desc=colored('learning in progress...', 'blue'), colour='blue'):     
        # validation 
        if global_step % conf.validation_step == 0:
            best = run_model(data, model, device, tb_writer, config, optimizer, 
                             global_step, best, phase='validation')
            # for optuna pruner
            report_to_pruner(trial, best, global_step, config)
        # training
        run_model(data, model, device, tb_writer, config, optimizer, 
                  global_step, best, phase='training')
    return best


def objective(trial, device, config):
    
    create_log_folders(config) # This function modifies config
    
    # Outputs under this context will be logged into one file.
    # you shall not use @logger.catch other than the functions used under
    # this context because it lets the code running even under errors.
    # The log file will be investigated earch time the code exits from
    # the context. Error will be raised when error logs are found.
    with ContexualLogger(config):
       
        is_losscoef_all_zero, model, optimizer, \
            tb_writer, data = prepare_for_training(trial, config, device)
        if is_losscoef_all_zero: return np.Inf # cannot train if losses are all=zero!
       
        best = run_training_loop(
            trial, config, model, device, optimizer, tb_writer, data)

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