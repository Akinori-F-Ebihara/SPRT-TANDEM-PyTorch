import os, glob, pdb
import numpy as np
from termcolor import colored
from loguru import logger
import torch
from utils.misc import extract_params_from_config, convert_torch_to_numpy

@logger.catch
def initialize_objectives(config):
    '''
    '''
    # check if necessary parameters are defined in the config file
    requirements = set(['OPTIMIZATION_CANDIDATES', 'OPTIMIZATION_TARGET'])
    conf = extract_params_from_config(requirements, config)
    if 'all' in conf.optimization_target.lower():
        num_candidates = len(conf.optimization_candidates)
        return tuple([np.Inf] * num_candidates) # multiobjectives: 
    else:
        return tuple([np.Inf])

def finalize_objectives(best):
    '''
    Args:
    -best (tuple): objective(s) to be minimized.
    '''
    num_objectives = len(best)
    if len(best) == 1:
        return best[0]
    elif len(best) > 1:
        best0, *remaining_bests = best
        return best0, *remaining_bests

def update_and_save_result(model, optimizer, config, best, performance_metrics, global_step):
    '''
    Update the best result according to the optimization target.
    
    Args:
    - config (dict): Configuration dictionary that contains the optimization target.
    - ckpt_manager: (TensorFlow checkpoint manager)
    - best (tuple): current best values of the objective
    - performance_metrics (dict): dictionary containing mean macro-average recall (macrec),
      AUSAT_optimization loss, and mean absolute error rate (mean_abs_error)
    - global_step (int): Current iteration number.
    Returns:
    - best (tuple): The updated best value.
    '''
    def concatenate_variables(variables: tuple, n_digits: int) -> str:
        '''
        '''
        def format_float(f, n_digits):
            return '{{:.{}f}}'.format(n_digits).format(
                        np.round(f * 10**n_digits) / 10**n_digits)
        variables = convert_torch_to_numpy(variables) # make double sure they're numpy
        variables = [format_float(v, n_digits) for v in variables]
        return '_'.join(variables)

    def save_checkpoint(model, optimizer, best, conf, global_step):
        # save a checkpoint
        model_path = \
            f'{conf.dir_ckptlogs}/'\
            f'ckpt_step{global_step}_target_'\
            f'{conf.optimization_target}{concatenate_variables(best, n_digits=4)}'\
            f'.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        }, model_path)
        logger.info(colored('Saved checkpoint', 'cyan') + f' at step {global_step}.')


    def keep_max_num_saved_models(conf: dict, max_to_keep: int = 1) -> None:
        '''
        This function ensures that PyTorch saves a maximum number of checkpoint files (.pth format) in a given directory.
        If there are more saved models than the specified limit, it will delete the oldest ones, based on their creation time.

        Args:
        - conf (dict): stores conf.ckptlogs to specify checkpoint log directory.
        - max_to_keep (int, optional): The maximum number of checkpoint files to save in the directory. Defaults to 1.
        '''
        # Get a list of all files in the directory
        models = glob.glob(f'{conf.dir_ckptlogs}/' + '*.pt')

        # Sort the files based on their creation time
        models = sorted(models, key=os.path.getmtime)
        if len(models) > max_to_keep:
            os.remove(models[0])
            logger.info(f'Removed the oldest model for {max_to_keep=}')
            
    # check if necessary parameters are defined in the config file
    requirements = set(['OPTIMIZATION_CANDIDATES', 'OPTIMIZATION_TARGET', 'DIR_CKPTLOGS', 'MAX_TO_KEEP'])
    conf = extract_params_from_config(requirements, config)

    # Optuna *minimize* the objective by default - give it an error, not accuracy or equivalent
    targets = {
        'ausat_loss': performance_metrics['losses']['AUSAT_optimization'],  # This is an AUSAT "loss", which should be minimized
        'macrec': 1 - performance_metrics['mean_macro_recall'], # MacRec should be larger... flip the sign
        'ausat_confmx': 1 - performance_metrics['ausat_from_confmx'], # AUSAT "curve" from confmx has accuracy as y-axis: flip it
        'mabs': performance_metrics['mean_abs_error']} # MABS should be smaller.
    if 'all' in conf.optimization_target.lower():
        target = tuple(targets.values())
    elif conf.optimization_target.lower() in targets.keys():
        target = tuple([targets.get(conf.optimization_target.lower())])
    else:
        raise ValueError('Unknown optimization target!')
 
    indices = [i for i, (x, y) in enumerate(zip(target, best)) if x < y] # smaller is better
    if global_step == 0:
        best = target
    elif indices: # update the best value!
        best = tuple([target[i] if i in indices else x for i, x in enumerate(best)])
        logger.info(colored('Best value updated!', 'cyan'))
        save_checkpoint(model, optimizer, best, conf, global_step)
        keep_max_num_saved_models(conf, conf.max_to_keep)
    return best

