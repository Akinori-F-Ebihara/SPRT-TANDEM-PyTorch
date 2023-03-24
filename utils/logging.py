import shutil, datetime, pdb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from loguru import logger
from utils.misc import extract_params_from_config, create_directories_and_log, \
                       format_common_folder_structure, compile_comment, convert_torch_to_numpy

def save_config_info(config):
    '''
    Save config.py and config (dict) for reproducibility.
    '''
    requirements = set(['CONFIG_PATH', 'DIR_CONFIGS', 'NOW', 'EXP_PHASE'])
    conf = extract_params_from_config(requirements, config)
    
    comment = compile_comment(config)

    shutil.copyfile(conf.config_path, conf.dir_configs + f"/config_{conf.now}_orig.py")  
    with open(conf.dir_configs + f'/config_{conf.now}_cooked.py', 'w') as f:
        f.write('# This is the actual config dict used for the training.\n'
                '# Simply import this file for reproducing the results.\n\n')
        f.write('config = {\n')
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'  "{key}": "{value}", \n')
            else:
                f.write(f'  "{key}": {value}, \n')
        f.write('}\n')
    with open(conf.dir_configs + f'/{comment}.txt', 'w') as f:
        f.write('')

def create_log_folders(config):
    '''
    Create log folders for earch trial:
    - checkpoint log folder 
    - tensorboard event log folder
    - stdout log folder
    - image log folder (optional: created when IS_SAVE_FIGURE=True)
    Note that an optuna database log folder is created for each subproject, not each trial.
    For definition of optuna database log folder, see utils.hyperparameter_tuning:run_optuna.
    '''
    
    def setup_checkpoint_log_folder(config, conf):     
        config['DIR_CKPTLOGS'] = f"{conf.root_ckptlogs}" + format_common_folder_structure(conf)
        config['DIR_CONFIGS'] = config['DIR_CKPTLOGS'] + "/configs"
        create_directories_and_log(config['DIR_CONFIGS'])
    
    def setup_tensorboard_log_folder(config, conf):
        config['DIR_TBLOGS'] = f"{conf.root_tblogs}" + format_common_folder_structure(conf)                                                  
        create_directories_and_log(config['DIR_TBLOGS'])

    def setup_stdout_log_folder(config, conf):
        # do not make individual folder because one trial contains one .log file anyways
        config['DIR_STDOUTLOGS'] = f"{conf.root_stdoutlogs}" #+ format_common_folder_structure(conf)[:-1] 
        config['STDOUTLOG_NAME'] = f"{config['DIR_STDOUTLOGS']}/{conf.now}.log"
        create_directories_and_log(config['DIR_STDOUTLOGS'])

    def setup_image_log_folder(config, conf):
        config['DIR_IMGLOGS'] = f"{conf.root_imglogs}" + format_common_folder_structure(conf)
        create_directories_and_log(config['DIR_IMGLOGS'])
    
    # get a timestamp for each trial
    config['NOW'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

    # check if necessary parameters are defined in the config file
    requirements = set(['ROOT_CKPTLOGS', 'ROOT_TBLOGS', 'ROOT_STDOUTLOGS', 'ROOT_IMGLOGS',
                        'EXP_PHASE', 'COMMENT', 'CONFIG_PATH', 'NOW',
                        'ORDER_SPRT', 'LLLR_VERSION', 'PARAM_LLR_LOSS', 'PARAM_AUSAT_LOSS',
                        'AUCLOSS_VERSION', 'AUCLOSS_BURNIN', 'PARAM_MULTIPLET_LOSS', 'IS_ADAPTIVE_LOSS'])
    conf = extract_params_from_config(requirements, config)

    setup_checkpoint_log_folder(config, conf)
    setup_tensorboard_log_folder(config, conf)
    setup_stdout_log_folder(config, conf)
    if config['IS_SAVE_FIGURE']:
        setup_image_log_folder(config, conf)


def setup_logger(config):

    def tqdm_compatible_logger(logger, logpath, 
            rotation='10 MB', enqueue=True, level='DEBUG'):
        '''
        loguru logger outputs to console and log file specified with logpath.
        lambda function is used to wrap tqdm.write(), which is required to
        output message onto the console without disrupting tqdm progress bar.
        
        Args:
        - logger: loguru logger object

        Return:
        - None
        '''
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""),
                enqueue=enqueue, level=level,
                colorize=True, backtrace=True, diagnose=True)
        logger.add(logpath, rotation=rotation, 
                enqueue=enqueue, level=level)

    requirements = set(['STDOUTLOG_NAME'])
    conf = extract_params_from_config(requirements, config)

    tqdm_compatible_logger(logger, conf.stdoutlog_name)

   
def investigate_log(config):
    '''
    Investigate .log file and count WARNING and ERROR sign.
    Raise ValueError when non-zero ERROR are found.
    '''
    requirements = set(['STDOUTLOG_NAME'])
    conf = extract_params_from_config(requirements, config)

    with open(conf.stdoutlog_name, 'r') as f:
        contents = f.read()
    num_warning = contents.count('WARNING')
    num_error = contents.count('ERROR')
    logger.info(f'{num_warning=}/{num_error=} in the .log file.')
    
    if num_error > 0:
        raise ValueError('Found ERROR! Check the .log file for debug!')
    elif num_warning > 0:
        logger.warning('Found WARNING! check the .log file for debug!')
        

class ContexualLogger:
    """
    A context manager that sets up a logger before executing a code block, and
    investigates the logger after the code block has completed.

    Usage:
    config = initialize_setups()
    with LoggingContextManager(config):
        training_my_neural_network(config)
    """

    def __init__(self, config):
        self.config = config

    def __enter__(self):
        """
        Sets up a logger before entering the with block.

        Returns:
        self: The current LoggingContextManager instance.
        """
        setup_logger(self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Investigates the logger after exiting the with block.

        Args:
        exc_type: The type of exception raised (if any).
        exc_value: The value of the exception raised (if any).
        traceback: The traceback object associated with the exception (if any).
        """
        investigate_log(self.config)


def get_tb_writer(config):
    '''
    Get a PyTorch-builtin TensorBoard writer.

    Args:
    - config (dict): a dictionary containing key "DIR_TBLOGS" and 
      its value specifying the path to tensorboard log file.
    
    Return:
    - SummaryWriter(): PyTorch SummaryWriter object for logging.
    '''
    # check if necessary parameters are defined in the config file
    requirements = set(['DIR_TBLOGS'])
    conf = extract_params_from_config(requirements, config)
    
    return SummaryWriter(log_dir=conf.dir_tblogs)


def log_training_results(tb_writer, model, global_step:int, config, performance_metrics, phase):
    '''
    '''
    def subsample_dict(performance_metrics, selected_keys):
        return {k: v for k, v in performance_metrics.items() if k in selected_keys}

    # check if necessary parameters are defined in the config file
    requirements = set(['VERBOSE', 'IS_ADAPTIVE_LOSS', 'PARAM_MULTIPLET_LOSS',
        'PARAM_LLR_LOSS', 'PARAM_AUSAT_LOSS', 'AUCLOSS_VERSION', 'WEIGHT_DECAY',
        'TRAIN_DISPLAY_STEP'])
    conf = extract_params_from_config(requirements, config)

    # log train results only periodically
    if 'train' in phase.lower() and global_step % conf.train_display_step != 0:
        return
        
    # log only a subset of performance metrics
    subset_metrics = subsample_dict(
        performance_metrics, 
        ['losses', 'mean_abs_error', 'mean_macro_recall', 'ausat_from_confmx'])

    # output to console and log
    if conf.verbose:
        logger.info('')
        logger.info('Global Step ={:7d}'.format(global_step))
        logger.info("{} mean(MacRec)_t: {}".format(
            phase, performance_metrics['mean_macro_recall']))
        logger.info("{} mean ABSerr: {}".format(
            phase, performance_metrics['mean_abs_error']))
        logger.info("{} ausat from confmx: {}".format(
            phase, performance_metrics['ausat_from_confmx']))
        if conf.is_adaptive_loss:
            subset_metrics['adaptive_loss_weight'] ={
                'MCE': model.adaptive_loss_weights[0],
                'LLLR': model.adaptive_loss_weights[1],
                'AUSAT': model.adaptive_loss_weights[2],
                'weight_decay': model.adaptive_loss_weights[3]}
            logger.info('{} MCE loss:{:7.5f} * {} * {:7.5f}'.format(
                phase, performance_metrics['losses']['multiplet_crossentropy (MCE)'], 
                str(conf.param_multiplet_loss), subset_metrics['adaptive_loss_weight']['MCE']))
            logger.info('{} LLLR :{:7.5f} * {} * {:7.5f}'.format(
                phase, performance_metrics['losses']['LLR_estimation (LLRE)'], 
                str(conf.param_llr_loss), subset_metrics['adaptive_loss_weight']['LLLR']))
            logger.info('{} AUSAT loss ver{}:{:7.5f} * {} * {:7.5f}'.format(
                phase, conf.aucloss_version, performance_metrics['losses']['AUSAT_optimization'], 
                str(conf.param_ausat_loss), subset_metrics['adaptive_loss_weight']['AUSAT']))
            logger.info('{} weight decay:{:7.5f} * {} * {:7.5f}'.format(
                phase, performance_metrics['losses']['weight_decay'], str(conf.weight_decay),
                subset_metrics['adaptive_loss_weight']['weight_decay']))
        else:
            logger.info('{} MCE loss:{:7.5f} * {}'.format(
                phase, performance_metrics['losses']['multiplet_crossentropy (MCE)'], 
                str(conf.param_multiplet_loss)))
            logger.info('{} LLLR :{:7.5f} * {}'.format(
                phase, performance_metrics['losses']['LLR_estimation (LLRE)'], 
                str(conf.param_llr_loss)))
            logger.info('{} AUSAT loss ver{}:{:7.5f} * {}'.format(
                phase, conf.aucloss_version, performance_metrics['losses']['AUSAT_optimization'], 
                str(conf.param_ausat_loss)))
            logger.info('{} weight decay:{:7.5f} * {}'.format(
                phase, performance_metrics['losses']['weight_decay'], str(conf.weight_decay)))
    # write to TensorBoard
    tblog_writer(tb_writer, global_step, phase=phase, 
        metrics=subset_metrics)


def tblog_writer(tb_writer, global_step, phase, metrics):
    '''
    Logs various values to TensorBoard.
    The keys in the kwargs dictionary will be used to create the names for the values in TensorBoard.
    
    Args:
    - tb_writer (torch.utils.tensorboard.SummaryWriter()): TensorBoard logger to use for logging the values.
    - global_step (int): Global step to use for logging the values.
    - phase (str): The phase of the model, used to prefix the names of the logged values.
    - kwargs (dict): Dictionary of values to log. The keys in the dictionary will be used to create the names for the values in TensorBoard.
    
    Returns:
    None

    Example:
    - a keyword argument loss={'LLLR': 0.2} at a training phase will be logged as:
        'training_loss/LLLR' with value 0.2.
    ''' 
    def write_to_tensorboard(metrics):
        for key, value in metrics.items():
            if isinstance(value, dict):
                write_to_tensorboard(value)
            else:
                name = f'{phase}/{key}'
                tb_writer.add_scalar(name, value, global_step)
    
    assert isinstance(metrics, dict), f'metrics is not a dictionary!'
    # use a recursive function to log values
    write_to_tensorboard(metrics)        


def add_prefix_to_keys(original_dict: dict, prefix: str) -> dict:
    '''
    Add a prefix to all the keys in a dictionary.
    Parameters:
        original_dict (dict): The original dictionary whose keys will be modified.
        prefix (str): The prefix to add to all the keys in the original dictionary.
    Returns:
        dict: A new dictionary with the prefix added to all the keys.
    '''
    assert isinstance(original_dict, dict), 'original_dict should be of type dict'
    assert isinstance(prefix, str), 'prefix should be of type str'
    return {prefix + key: value for key, value in original_dict.items()}


def plot_likelihood_ratio_matrix(llrm, gt_labels, num_trajectories=50):
    '''
    '''
    num_classes = llrm.shape[-1]
    colors = matplotlib.cm.tab10(range(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):

            plt.subplot(num_classes, num_classes, (j + 1) + num_classes * i)
            for k, c in zip(range(num_classes), colors):
                plt.plot(np.transpose(llrm[gt_labels==k, :, i, j][:num_trajectories]), color=c)

    # labels
    plt.subplot(num_classes, num_classes, 1)
    for k, c, in zip(range(num_classes), colors):
        plt.plot(np.transpose(llrm[gt_labels==k, :, 0, 0][:1]), color=c, label=f'class {k}')
    plt.legend()


def plot_example_trajectories(llrm, gt_labels, dice=[None, None],
                              max_traj_per_cls_pair=10, max_example_cls=10):
    '''
    Summarize a log-likelihood ratio matrix (llrm) into one panel.
    
    Args:
    - llrm (float): array of size (batch_size, time_steps, num_classes, num_classes)
      log likelihood ratio matrix, either ground-truth or estimated.
    - gt_labels (int): array of size (batch_size,). gt_labels \in [0, 1, ... num_classes - 1]
      ground truth data labels.
    - dice (int or None): array of size (max_traj_per_cls_pair,) if provided.
      For selecting/reproducing trajectories to be plotted.
      - dice[0]: indices for trajectories.
      - dice[1]: indices for classes.
    - max_traj_per_cls_pair (int): positive integer specifies the maximum number of trajectories
      per class pair. e.g., if num_classes = 3, there will be 3*2=6 class pairs and thus 
      6 * max_traj_per_cls_pair trajectories will be plotted in total. Note that a number of 
      trajectories will be smaller than max_traj_per_cls_pair if llrm smaller number of examples.
    - max_example_cls (int): positive integer specifies the maximum number of classes to be plotted.
      classes are randomly selected if max_example_cls > num_classes.
    
    Returns:
    - dice (int): the dice that used to select example trajectories. Use it to plot to select
      the corresponding trajectories of different llrms (e.g., ground-truth llrm and estimated llrm).
    
    Example:
    plt.figure(figsize=(16.2, 10))

    plt.subplot(1,2,1)
    plt.title('Ground-truth LLR trajectories')
    dice = plot_example_trajectories(gt_llrm, y)

    plt.subplot(1,2,2)
    plt.title('Estimated LLR trajectories')
    _ = plot_example_trajectories(estimated_llrm, y, dice=dice)
      
    '''
    
    # limit example 
    num_classes = llrm.shape[-1] 
    assert np.max(gt_labels) + 1 == num_classes
    
    assert max_example_cls > 1, ('max_example_cls must be greater than 1 to ensure likelihood ratio calculation'
                               ' (i.e., ratio is defined with cls > 1.)')
    
    colors = matplotlib.cm.tab10(range(num_classes))
    if all(d is None for d in dice):
        
        cls_samples = []
        for k in range(num_classes):
            cls_samples.append(len(llrm[gt_labels==k]))
        dice[0] = np.random.permutation(np.min(cls_samples) - 1)[:max_traj_per_cls_pair]
        
        # limit example classes if needed
        if num_classes > max_example_cls:
            dice[1] = np.random.permutation(num_classes)[:max_example_cls]
        else: 
            dice[1] = range(num_classes)
    else:
        assert len(dice[0]) == max_traj_per_cls_pair, 'length of dice[0] must be equal to max_traj_per_cls_pair!'
    
    for j in dice[1]:
        for k, c in zip(dice[1], colors):
            if k == j:
                continue
            plt.plot(np.transpose(llrm[gt_labels==k, :, k, j][dice[0]]), color=c, 
                    label=f'class {k} vs. {j} at y={k}')
    # labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_label_set = dict(zip(labels, handles))
    plt.legend(unique_label_set.values(), unique_label_set.keys(), fontsize=12)
    
    return dice


def save_sdre_imgs(config, target, global_step, y_batch, gt_llrs, \
                   monitored_values, performance_metrics):
    '''
    Save the results of sequential density ratio estimation (SDRE).
    '''
    # check if necessary parameters are defined in the config file
    requirements = set(['IS_SAVE_FIGURE', 'IS_MULT_SAT', 'OPTIMIZATION_TARGET', 'DIR_IMGLOGS',
    'AUCLOSS_VERSION', 'PARAM_MULTIPLET_LOSS', 'PARAM_LLR_LOSS', 'PARAM_AUSAT_LOSS', 'NUM_CLASSES'])
    conf = extract_params_from_config(requirements, config)

    if conf.is_save_figure == False:
        return

    [y_batch, gt_llrs, monitored_values, performance_metrics] = convert_torch_to_numpy([y_batch, gt_llrs, monitored_values, performance_metrics])
    aesthetic_num_traj = np.max([1, 20 // conf.num_classes])
    llrs = monitored_values['llrs']

    plt.rcParams["font.size"] = 25
    fig, ax = plt.subplots(figsize=(30,15))
    plt.subplot(2,3,1)
    fig.patch.set_facecolor('white')
    plt.title('Ground-truth LLR trajectories')
    dice = plot_example_trajectories(gt_llrs, y_batch, max_traj_per_cls_pair=aesthetic_num_traj)
    plt.subplot(2,3,2)
    fig.patch.set_facecolor('white')
    plt.title('Estimated LLR trajectories')
    _ = plot_example_trajectories(llrs, y_batch, dice, max_traj_per_cls_pair=aesthetic_num_traj)    
    # plt.subplot(2,3,3)
    # fig.patch.set_facecolor('white')
    
    # plt.subplot(2,3,4)
    
    plt.subplot(2,3,5)
    fig.patch.set_facecolor('white')
    plt.scatter(performance_metrics['hitting_time'], performance_metrics['acc_eta'])
    plt.title('SAT curve from the AUSAT loss')
    plt.subplot(2,3,6)
    fig.patch.set_facecolor('white')
    plt.scatter(performance_metrics['hitting_time'], performance_metrics['sns_conf'])
    plt.title('SAT curve from the confusion matrix')
    
    targval = str(np.array([t for t in target]))

    plt.savefig('{}/AUCvers{}_FlagMult{}_Wmla{}{}{}_target{}{}_{}.jpg'.format(
                conf.dir_imglogs, conf.aucloss_version, conf.is_mult_sat,
                conf.param_multiplet_loss, conf.param_llr_loss, conf.param_ausat_loss,
                conf.optimization_target, targval,
                int(global_step)))
    plt.close('all')
    logger.info('Figures saved.')


