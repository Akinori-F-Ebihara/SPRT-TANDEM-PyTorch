import os
import time
import pdb
import tracemalloc
import numpy as np
from termcolor import colored
from typing import Callable, Tuple, Set, Dict
import argparse
from loguru import logger
import torch
import torch.nn.functional as F


class PyTorchPhaseManager:
    """
    A context manager for managing the training and evaluation phases of a PyTorch model.

    Args:
    - phase (str): The phase to manage ('train' or 'val').

    Methods:
    - __init__(self, phase)
    - __enter__(self)
    - __exit__(self, *args)
    - training_preprocess(self)
    - training_postprocess(self)
    - evaluation_preprocess(self)
    - evaluation_postprocess(self)

    Required variables:
    - model
    - optimizer
    - loss

    Remark:
    - assumption for the loss function is that it takes arguments in order of:
      (model, ..., config).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim,
        loss_func: Callable,
        loss_args: Tuple,
        phase: str,
        config: dict,
    ):
        """
        Initializes a PyTorchPhaseManager object with the specified phase.

        Args:
        - phase (str): The phase to manage ('train' or 'val').
        """

        self.config = config
        self.max_norm = config["MAX_NORM"]
        self.model = model
        self.optimizer = optimizer
        self.phase = phase
        self.loss = None
        self.loss_func = loss_func
        self.loss_args = loss_args
        self.scaler = torch.cuda.amp.GradScaler()

    def __enter__(self):
        """
        Enters the context for the current phase and performs any necessary preprocessing.

        Returns:
        - self: The PyTorchPhaseManager object.
        """

        if "train" in self.phase:
            self.mode = torch.enable_grad()
            self.training_preprocess()
        elif "val" in self.phase:
            self.mode = torch.no_grad()
            self.evaluation_preprocess()
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self.mode.__enter__()

        return self

    def __exit__(self, *args):
        """
        Exits the context for the current phase and performs any necessary postprocessing.

        Args:
        - *args: The arguments passed to __exit__.
        """
        if "train" in self.phase:
            self.training_postprocess()
            self.grad_norm = self.monitor_gradient()
        elif "val" in self.phase:
            self.evaluation_postprocess()
            self.grad_norm = torch.tensor(0.0)
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self.mode.__exit__(*args)

    def training_preprocess(self):
        """
        Sets the model to training mode and clears the gradients of the optimizer.
        """

        self.model.train()

        self.optimizer.zero_grad()

    def training_postprocess(self):
        """
        Computes the gradients of the loss with respect to the model parameters using loss.backward()
        and performs the optimization step using optimizer.step().
        """
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def evaluation_preprocess(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.eval()

    def evaluation_postprocess(self):
        """
        Performs no post-processing steps, as gradients are not computed during evaluation.
        """
        pass

    def monitor_gradient(self):
        """ """
        grad_raw = [
            torch.norm(p.grad.detach())
            for p in self.model.parameters()
            if p.grad is not None
        ]

        if not grad_raw:
            grad_norm = torch.tensor(0.0)
        else:
            grad_norm = torch.norm(torch.stack(grad_raw), 2)

        return grad_norm

    def call_loss(self):
        """
        compute loss function and evaluation metrics.

        Return:
        - self.loss
        - metrics (dict)
        """
        monitored_values = self.loss_func(
            self.model, *self.loss_args, config=self.config
        )
        self.loss = monitored_values["losses"]["total_loss"]

        return monitored_values


class StopWatch:
    """
    A context manager for measureing the elapsed time of a code block.

    Usage:
        with StopWatch() as time:
            <code block>
        print('elapsed time (sec):', time.elapsed)
    """

    def __init__(self, unit="hours", label=""):
        """
        Args:
        - unit (str): specifies time unit \in {seconds, minutes, hours}
        - label (str): added to log message
        """
        if "sec" in unit:
            self.denom = 1.0
            self.unit = "seconds"
        elif "min" in unit:
            self.denom = 60.0
            self.unit = "minutes"
        elif "hour" in unit:
            self.denom = 3600.0
            self.unit = "hours"
        self.label = label

    def __enter__(self):
        self.start_time = time.time()
        tracemalloc.start()

        return self

    def __exit__(self, *args):
        _end_time = time.time()
        self.elapsed = (_end_time - self.start_time) / self.denom

        _end_memory, _end_memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.memory_usage = _end_memory / (1024**2)  # convert to megabytes

        logger.info(f"elapsed time: {self.elapsed: 6f} {self.unit}. ({self.label})")
        logger.info(f"Memory usage: {self.memory_usage:.2f} megabytes")


class ErrorHandler:
    """
    A context manager that catches specified exceptions and invokes the pdb debugger.
    Can be useful for debugging purpose.

    Attributes:
    exception (Exception): The type of exception to catch. Defaults to Exception.

    Example:
        with ErrorHandler():
            some_computation_to_monitor()

    """

    def __init__(self, exception=Exception):
        self.exception = exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Invokes pdb.set_trace() if the caught exception matches the specified exception type.

        Args:
        exc_type (type): The type of the caught exception.
        exc_value (Exception): The caught exception.
        traceback (traceback): The traceback object for the caught exception.

        Returns:
        bool: True if the exception was caught and handled, False otherwise.
        """
        if exc_type is not None and issubclass(exc_type, self.exception):
            pdb.set_trace()
            return True
        return False


def convert_torch_to_numpy(data):
    """
    Recursively convert all values in a hierarchical python dictionary, a PyTorch tensor, a numpy array,
    or a list of dictionaries, tensors, or numpy arrays to numpy arrays. If a value is a PyTorch tensor
    on cpu or cuda device, it will be converted to a numpy array.

    Args:
        data (dict, tensor, array, or list): A hierarchical python dictionary, a PyTorch tensor,
        a numpy array, or a list of dictionaries, tensors, or numpy arrays whose values will be converted to numpy arrays.

    Returns:
        dict, array, list: A copy of the input dictionary, tensor, array or list with all values converted to numpy arrays.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = convert_torch_to_numpy(value)
        return result
    elif isinstance(data, torch.Tensor):
        # handle data in traini or eval phase
        data = data.detach() if data.requires_grad else data

        # handle data on GPU or CPU
        if data.is_cuda:
            return data.cpu().numpy()
        else:
            return data.numpy()
    elif isinstance(data, list):
        result = []
        for item in data:
            result.append(convert_torch_to_numpy(item))
        return result
    else:  # numpy array or else
        return data


def create_directories_and_log(dict_path):
    """
    A simple helper function create directories and output log message.
    """
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)
        logger.info(
            colored("A new log directory was created: ", "yellow") + f"{dict_path}"
        )


def format_common_folder_structure(conf):
    """ """
    if conf.comment:
        return f"/{conf.comment}_{conf.model_backbone}_{conf.now}"
    else:
        return f"/{conf.model_backbone}_{conf.now}"


class ConfigSubset:
    """
    A class to store the required key-value pairs as instance variables.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key.lower(), value)


def extract_params_from_config(
    requirements: Set[str], config: Dict[str, any]
) -> ConfigSubset:
    """
    Extract necessary (hyper)parameters for each function. The extracted
    parameters are stored as a class instance variables for easy access.
    Keys of the necessary parameters are defined in the set "requirements."
    The keys are converted to lowercase and used for instance variable names.

    Use this class to avoid accidentally overwrite the original config function -
    remember that python function gets a pointer to the original config dict,
    not a copy of it (i.e., config can be modified within a function).

    Args:
    - requirements (set): required keys for a given function.
    - config (dict): the original dictionary containing all necessary parameters.

    Returns:
    - sub_conf (ConfigSubset): required variables stored as instance variables of a class.
    """

    # assert that all the required keys exist in the config file.
    missing_keys = requirements.difference(config.keys())
    assert not missing_keys, f"Missing necessary parameters: {', '.join(missing_keys)}"

    sub_conf = ConfigSubset(**{key: config[key] for key in requirements})
    return sub_conf


def float_formatter(variable, digits=3):
    """
    Format float variable, or string-represented float, with
    specified effective digits.

    Args:
    - variable: str, float, or else

    Returns:
    - string f'{variable:.5}' if variable is float or
      string-represented float. Return variable unchanged if else
    """

    def formatted_float(var, digits):
        # return f'{var:.%d}' % digits
        return f"{var:.{digits}}"

    if isinstance(variable, float):
        return formatted_float(variable, digits)
    elif isinstance(variable, str):
        try:
            float_variable = float(variable)
            return formatted_float(float_variable, digits)
        except:
            return variable
    else:
        return variable


def grab_gpu(config, device=None):
    # check if necessary parameters are defined in the config file
    requirements = set(["GPU"])
    conf = extract_params_from_config(requirements, config)
    # restrict visible device(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.gpu)
    # Get cpu or gpu device for training.
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    return torch.device(device)


def make_directories(path):
    if not os.path.exists(path):
        logger.info("Path '{}' does not exist.".format(path))
        logger.warning("Make directory: " + path)
        os.makedirs(path)


def fix_random_seed(config):
    # check if necessary parameters are defined in the config file
    requirements = set(["IS_SEED", "SEED"])
    conf = extract_params_from_config(requirements, config)

    if conf.is_seed:
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)
        logger.info(
            "Numpy and Pytorch's random seeds are fixed: seed=" + str(conf.seed)
        )

    else:
        logger.info("Random seed is not fixed.")


def set_cpu_workers(config):
    """
    Set the number of CPU threads (workers).
    This number is set equal to the number of num_workers in PyTorch DataLoader.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(["NUM_WORKERS"])
    conf = extract_params_from_config(requirements, config)
    torch.set_num_threads(np.max([1, conf.num_workers]))


def compile_comment(config):
    """
    Name log file(s) based on selected parameters.
    """
    requirements = set(
        [
            "PRUNER_NAME",
            "ORDER_SPRT",
            "LLLR_VERSION",
            "PARAM_LLR_LOSS",
            "PARAM_MULTIPLET_LOSS",
            "IS_ADAPTIVE_LOSS",
            "EXP_PHASE",
            "SUBPROJECT_NAME_PREFIX",
            "SUBPROJECT_NAME_SUFFIX",
            "COMMENT",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    str_pruner = f"_{conf.pruner_name}pruner"
    str_main = (
        f"order{conf.order_sprt}_lllr{conf.lllr_version}{float_formatter(conf.param_llr_loss)}"
        f"_mult{float_formatter(conf.param_multiplet_loss)}_adap{conf.is_adaptive_loss}"
    )
    name = f"{conf.comment}_" + str_main
    if "tuning" in conf.exp_phase.lower():
        name += str_pruner
    return name


def compile_subproject_name(config):
    """
    define the subproject name, which is used to create a folder for training logs.
    """
    assert type(config) == dict
    requirements = set(
        [
            "DATA_SEPARATION",
            "OPTIMIZATION_TARGET",
            "SUBPROJECT_NAME_PREFIX",
            "SUBPROJECT_NAME_SUFFIX",
            "EXP_PHASE",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    str_main = (
        conf.subproject_name_prefix
        + f'offset{conf.data_separation.replace(".", "_")}_optim{conf.optimization_target}'
        + conf.subproject_name_suffix
        + "_"
        + conf.exp_phase
    )

    return str_main


def compile_directory_paths(config):
    common_path = f"{config['LOG_PATH']}{config['SUBPROJECT_NAME']}/"
    config["ROOT_TBLOGS"] = common_path + f"{config['TB_DIRNAME']}"
    config["ROOT_DBLOGS"] = common_path + f"{config['DB_DIRNAME']}"
    config["ROOT_CKPTLOGS"] = common_path + f"{config['CKPT_DIRNAME']}"
    config["ROOT_STDOUTLOGS"] = common_path + f"{config['STDOUT_DIRNAME']}"
    config["PATH_RESUME"] = (
        common_path + f"{config['CKPT_DIRNAME']}/{config['SUBPROJECT_TO_RESUME']}"
    )


def parse_args(config: dict) -> None:
    """
    Parse the command-line arguments for the training script.

    Args:
    - config (dict): a dictionary containing default hyperparameters.

    Return:
    - None: no need to return because this function receives a pointer to config dict.
    """

    def fix_dependent_parameters(config: dict) -> None:
        """
        Some config parameters are dependent on other parameters.
        Fix them based on the updated config file.
        """
        config["NUM_EPOCHS"] = (
            config["NUM_ITER"] * config["BATCH_SIZE"] // config["NUM_TRAIN"]
        )

    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "GPU",
            "NUM_ITER",
            "EXP_PHASE",
            "OPTIMIZATION_TARGET",
            "EXP_PHASE",
            "MODEL_BACKBONE",
            "SUBPROJECT_NAME_PREFIX",
            "DATA_SEPARATION",
            "NUM_TRIALS",
            "IS_LOAD_ONTO_MEMORY",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    parser = argparse.ArgumentParser(
        description="Sequential Density Ratio Estimation by the SPRT-TANDEM "
    )
    parser.add_argument("-g", "--gpu", type=int, default=conf.gpu, help="#gpu")
    parser.add_argument(
        "-d",
        "--data_separation",
        type=str,
        default=conf.data_separation,
        help="separation of Gaussian dataset.",
    )
    parser.add_argument(
        "-t", "--num_trials", type=int, default=conf.num_trials, help="#trials"
    )
    parser.add_argument(
        "-i", "--num_iter", type=int, default=conf.num_iter, help="#iterations"
    )
    parser.add_argument(
        "-e",
        "--exp_phase",
        type=str,
        default=conf.exp_phase,
        help='phase of an experiment, "try," "tuning," or "stat"',
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=conf.model_backbone,
        help='model backbone, "LSTM", or "Transformer"',
    )
    parser.add_argument(
        "-o",
        "--optimize",
        type=str,
        default=conf.optimization_target,
        help='optimization target: "MABS", "MacRec", "ausat_confmx", or "ALL"',
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=conf.subproject_name_prefix,
        help="subproject name",
    )
    parser.add_argument(
        "--flip_memory_loading",
        action="store_true",
        help="set a boolean flag indicating whether to load onto memory",
    )

    args = parser.parse_args()

    # overwrite config dict with the provided command-line arguments
    config["GPU"] = args.gpu
    config["DATA_SEPARATION"] = args.data_separation
    config["NUM_TRIALS"] = args.num_trials
    config["NUM_ITER"] = args.num_iter
    config["EXP_PHASE"] = args.exp_phase
    config["MODEL_BACKBONE"] = args.model
    config["OPTIMIZATION_TARGET"] = args.optimize
    config["SUBPROJECT_NAME_PREFIX"] = args.name
    config["SUBPROJECT_NAME"] = compile_subproject_name(config)
    config["IS_LOAD_ONTO_MEMORY"] = (
        not config["IS_LOAD_ONTO_MEMORY"]
        if args.flip_memory_loading
        else config["IS_LOAD_ONTO_MEMORY"]
    )
    fix_dependent_parameters(config)
    compile_directory_paths(config)


# Functions for graph plotting
def restrict_classes(llrs, labels, list_classes):
    """
    Args:
        list_classes: A list of integers.
    Remark:
        (batch, duration, num classes, num classes)
        -> (< batch, duration, num classes, num classes)
    """
    if list_classes == []:
        return llrs, labels

    assert torch.min(labels).item() <= min(list_classes)
    assert max(list_classes) <= torch.max(labels).item()

    ls_idx = []
    for itr_cls in list_classes:
        ls_idx.append(torch.squeeze(torch.where(labels == itr_cls)))
    idx = torch.cat(ls_idx, dim=0)
    idx, _ = torch.sort(idx)

    llrs_rest = torch.gather(
        llrs, 0, idx.unsqueeze(1).expand(-1, llrs.shape[1], -1, -1)
    )
    lbls_rest = torch.gather(
        labels, 0, idx.unsqueeze(1).expand(-1, labels.shape[1], -1, -1)
    )

    return llrs_rest, lbls_rest


def extract_positive_row(llrs, labels):
    """Extract y_i-th rows of LLR matrices.
    Args:
        llrs: (batch, duraiton, num classes, num classes)
        labels: (batch,)
    Returns:
        llrs_posrow: (batch, duration, num classes)
    """
    llrs_shape = llrs.shape
    duration = llrs_shape[1]
    num_classes = llrs_shape[2]

    labels_oh = F.one_hot(labels, num_classes=num_classes)
    # (batch, num cls)
    labels_oh = labels_oh.reshape(-1, 1, num_classes, 1)
    labels_oh = labels_oh.repeat(1, duration, 1, 1)
    # (batch, duration, num cls, 1)
    llrs_pos = llrs * labels_oh
    # (batch, duration, num cls, num cls)
    llrs_posrow = torch.sum(llrs_pos, dim=2)
    # (batch, duration, num cls): = LLR_{:, :, y_i, :}

    return llrs_posrow


def add_max_to_diag(llrs):
    """
    Args:
        llrs: (batch, duration, num classes, num classes)
    Returns:
        llrs_maxdiag: (batch, duration, num classes, num classes),
            max(|llrs|) is added to diag of llrs.
    """
    num_classes = llrs.shape[2]

    llrs_abs = torch.abs(llrs)
    llrs_max = torch.max(llrs_abs)
    # max |LLRs|
    tmp = torch.eye(num_classes) * llrs_max
    tmp = tmp.unsqueeze(0).unsqueeze(0)
    tmp = tmp.to(llrs.device)
    llrs_maxdiag = llrs + tmp

    return llrs_maxdiag
