import os, pdb
from prettytable import PrettyTable
from typing import Dict, Any, List
from termcolor import colored
import logging
from loguru import logger
import optuna
from optuna.trial import TrialState
from utils.misc import extract_params_from_config, StopWatch, create_directories_and_log

class LoggerProxy(logging.Handler):
    '''
    A logging handler that redirects logger output to the Loguru logger.
    This class overrides the `emit` method of the `logging.Handler` class to redirect the logger
    output to the Loguru logger, which is highly configurable and provides more advanced logging
    features than the standard Python logging library.
    Attributes:
    - None.
    Methods:
    - emit: Overrides the `emit` method of the `logging.Handler` class to redirect logger
        messages to the Loguru logger.
    Remark on the depth argument:
    - By default, the loguru logger uses a depth of 0, which means that it will show
      information about the calling function (i.e., the function that called the logger).
      Specifying a depth=n>0 will show information about the function that called the calling
      function, and so on, up to a depth of n. This can be useful for debugging complex code
      where multiple layers of function calls are involved.
    - CAUTION: Choosing too large of a depth value could lead to an error if the call stack
      isn’t deep enough to support the specified depth.
    '''
    def emit(self, record):
        logger.opt(
            depth=6, exception=record.exc_info).log(
            record.levelname, # INFO, WARNING, ERROR, etc.
            record.getMessage() # The actual message to log
            )

def redirect_logging_to_loguru():
    '''
    Redirects logging-logger output to the Loguru-logger using a LoggerProxy instance.
    '''
    def unlock_optuna_logger():
        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()
    
    unlock_optuna_logger()
    logging.root.addHandler(LoggerProxy())


def float_formatter(variable):
    
    def formatted_float(var):
        return f'{var:.5}'

    if isinstance(variable, float):
        return formatted_float(variable)
    elif isinstance(variable, str):
        stripped_variable = variable.replace(".", "").replace("-", "").replace("e", "").replace("E", "").replace("+", "")
        if stripped_variable.isdigit():
            float_variable = float(variable)
            return formatted_float(float_variable)
        else:
            return variable
    else:
        return variable


class AvoidForbiddenParamsSampler(optuna.samplers.BaseSampler):
    '''
    Avoid a certain combination of hyperparameters specified with 'self.forbidden_hyperparams', 
    such as 'loss weights all-zero.' The default sampling method is 'TPEsampler', but when the 
    forbidden combination of hyperparameters is generated, the method is switched to 
    'RandomSampler' to encourage it to select a different parameter.

    Parameters:
    -----------
    forbidden_hyperparams : dict
        A dictionary of forbidden hyperparameters with their corresponding values.

    Methods:
    --------
    -sample_relative(study, trial, search_space):
        Returns a relative sampled point using the TPE sampler.

    -sample_independent(study, trial, param_name, param_distribution):
        Samples an independent point using either the TPE or the Random sampler depending on the 
        current hyperparameters set and the forbidden hyperparameters set.

    -infer_relative_search_space(study, trial):
        Infers a relative search space using the TPE sampler.

    -should_use_random_sampler(hyperparams):
        Checks if the current hyperparameters set violates the forbidden hyperparameters set, 
        and if so, switches to the Random sampler.

    '''
    def __init__(self, forbidden_hyperparams: Dict[str, Dict[str, Any]]):
        
        # Forbidden combinations specified in a config file
        self.forbidden_hyperparams = forbidden_hyperparams

        # Default sampler
        self.tpe_sampler = optuna.samplers.TPESampler()

        # Random sampler 
        self.random_sampler = optuna.samplers.RandomSampler()

    def sample_relative(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial,
                         search_space: Dict[str, Any]) -> Dict[str, Any]:
        return self.tpe_sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial,
                           param_name: str, param_distribution: optuna.distributions.BaseDistribution) -> Any:
        
        current_param_set = trial.params

        # Try TPEsampler first
        param_value = self.tpe_sampler.sample_independent(study, trial, param_name, param_distribution)
    
        current_param_set[param_name] = param_value
        
        # Check if one of the forbidden combination is generated
        should_use_random_sampler = self.should_use_random_sampler(current_param_set)

        if should_use_random_sampler:

            while True:
                # Encourage selecting different value 
                param_value = self.random_sampler.sample_independent(study, trial, param_name, param_distribution)
                current_param_set[param_name] = param_value

                if not self.should_use_random_sampler(current_param_set):
                    # Log and exit loop when an acceptable value is selected
                    break
            return param_value
        else:
            return param_value
    
    def infer_relative_search_space(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
        return self.tpe_sampler.infer_relative_search_space(study, trial)
    
    def should_use_random_sampler(self, hyperparams: Dict[str, Any]) -> bool:
        for label, forbid in self.forbidden_hyperparams.items():
            if all(hyperparams.get(key) == value for key, value in forbid.items()):
                logger.info(colored(f'Forbidden combination of parameter set {label} is selected with TPEsampler. '
                                     'Switching to RandomSampler to circumvent...', 'yellow'))
                return True
        return False


def run_optuna(objective, device, config):
    '''
    Load or create optuna study and run optimization steps.
    Also sets pruner for early-stopping.

    Args:
    - objective(function): a function that train and evaluate a model.
    - config(dict): a dictionary containing hyperparameters.
    '''
    def setup_pruner(pruner_name, conf):
        '''
        '''
        logger.info(colored(f'Setting up a {pruner_name} pruner...', 'yellow'))
        if 'median' in pruner_name.lower():
            return optuna.pruners.MedianPruner(n_startup_trials=conf.pruner_startup_trials, 
            n_warmup_steps=conf.pruner_warmup_steps, interval_steps=conf.pruner_interval_steps
            )
        elif 'hyperband' in pruner_name.lower():
            return optuna.pruners.HyperbandPruner()
        elif 'percentile' in pruner_name.lower():
            # 60 percentile pruner is more permissive than median
            return optuna.pruners.PercentilePruner(60.0, n_startup_trials=conf.pruner_startup_trials, 
            n_warmup_steps=conf.pruner_warmup_steps, interval_steps=conf.pruner_interval_steps
            )
        elif 'threshold' in pruner_name.lower():
            return optuna.pruners.ThresholdPruner(
                n_warmup_steps=conf.pruner_warmup_steps, interval_steps=conf.pruner_interval_steps
            )
        elif 'successivehalving' in pruner_name.lower():
            return optuna.pruners.SuccessiveHalvingPruner()
        elif 'none' in pruner_name.lower():
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f'pruner {pruner_name} is not implemented yet!')
    
    def start_optimization(objective, conf):
        '''
        Load or create study, and start optimization.
        Do not use a particular parameter set.
        '''
        num_objectives = len(conf.optimization_candidates) if 'all' in conf.optimization_target.lower() else 1
        pruner = setup_pruner(conf.pruner_name, conf) \
            if 'tuning' in conf.exp_phase and num_objectives == 1 else setup_pruner('none', conf)

        study_name = 'optuna'  
        storage_name = "sqlite:///" + conf.root_dblogs + "/" + study_name + ".db"
        create_directories_and_log(conf.root_dblogs)
        
        study = optuna.create_study(
            study_name=study_name, storage=storage_name, 
            load_if_exists=True, pruner=pruner,
            directions=['minimize'] * num_objectives,
            sampler=AvoidForbiddenParamsSampler(conf.forbidden_param_sets)
            )
        
        # wrap the objective inside a lambda and call objective inside it
        # cf.) kaggle.com/general/261870
        study.optimize(
            lambda trial: objective(trial, device, config), 
            n_trials=conf.num_trials)
        return study

    def reproduce_trial(objective, conf):
        '''
        Start training with the parameter set of the best trial, or that of a specified trial.
        '''
        num_objectives = len(conf.optimization_candidates) if 'all' in conf.optimization_target.lower() else 1
        
        assert "tuning" not in conf.exp_phase, \
                        colored('Using the best param is prohibited under tuning phase.', 'red')
        
        assert num_objectives == 1, colored('''run with best parameter under multi-objective 
                optimization is needed to be implemented...!''', 'red')
        
        # Set paths
        study_name = 'optuna'
        path_db = conf.root_dblogs + '/' + study_name + '.db'
        storage_name = 'sqlite:///' + path_db
        if not os.path.exists(path_db):
            raise ValueError(colored(
               f'{path_db} not found. Hyperparameters could not be restored', 'red'))
        
        # Load parameters and start optimization
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        for cnt in range(conf.num_trials):
            trial = study.best_trial if 'best' in str(conf.reproduce_trial).lower() \
                                     else study.trials[conf.reproduce_trial]
            # report to the console
            logger.info(colored(f'reproducing the trial,\n', 'yellow') + f'{trial},from:\n{path_db}')
            pdb.set_trace()
            paramtable = ConsoleTable('Loaded parameters', ['Parameter name', 'Value'])  
            for key, value in trial.params.items():
                key = key.replace('list_', '') if key.startswith('list_') else key
                paramtable.add_entry([key, float_formatter(value)])
            paramtable.print()
            # run!
            objective(trial, device, config)
        return study
    
    def trial_statistics(study, conf):
        '''
        Summarize the optimization results.
        '''
        def print_value_and_params(trial, conf, num_objectives):
            if num_objectives == 1:
                paramtable = ConsoleTable('Objective', [conf.optimization_target])  
                paramtable.add_entry([trial.value])
            else:
                paramtable = ConsoleTable('Objectives', conf.optimization_candidates)  
                paramtable.add_entry(trial.values)
            paramtable.print()
            if 'tuning' in conf.exp_phase :
                paramtable = ConsoleTable('Best params', ['Parameter name', 'Value'])  
                for key, value in trial.params.items():
                    key = key.replace('list_', '') if key.startswith('list_') else key
                    paramtable.add_entry([key, float_formatter(value)])
                paramtable.print()
        
        def print_importance(importances: dict, conf):
            '''
            '''
            if not 'tuning' in conf.exp_phase:
                return
            response = input('Would you like to print a parameter importance table? (yes/[no]) ')
            if 'yes' in response.lower() or 'y' in response.lower():
                paramtable = ConsoleTable('Importance', ['Parameter name', 'Percentage'])  
                for key, value in importances.items():
                    key = key.replace('list_', '') if key.startswith('list_') else key
                    paramtable.add_entry([key, float_formatter(value * 100)])
                paramtable.print()

        num_objectives = len(conf.optimization_candidates) if 'all' in conf.optimization_target.lower() else 1

        pruned_trials = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
        completed_trials = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
        
        valid_trials = pruned_trials + completed_trials
        if valid_trials > 0:
            percent_pruned_trials = 100 * pruned_trials / valid_trials
            logger.info(f'{pruned_trials} out of {valid_trials} trials are pruned so far\
                    ({percent_pruned_trials:.2f}%, {conf.pruner_name} pruner). ')
        else:
            logger.error('No valid trials found!')

        logger.info('Best trial(s):')
        if num_objectives == 1:

            logger.info(f'Optimization target: {conf.optimization_target}')
            trial = study.best_trial
            print_value_and_params(trial, conf, num_objectives)
            if completed_trials > 1:
                print_importance(optuna.importance.get_param_importances(study), conf)

        else: # multi-objective

            trials = study.best_trials
            for i, trial in enumerate(trials):
                logger.info('Trial #', i)
                print_value_and_params(trial, conf, num_objectives)
            for i in range(num_objectives):
                if completed_trials > 1:
                    logger.info(f'Optimization target #{i}: {conf.optimization_candidates[i]}')
                    print_importance(
                        optuna.importance.get_param_importances(study, target=lambda t: t.values[i]), conf)

    ########### main ###########
    # check if necessary parameters are defined in the config file
    requirements = set(['ROOT_DBLOGS', 'EXP_PHASE', 'FORBIDDEN_PARAM_SETS',
                        'NUM_TRIALS', 'PRUNER_NAME', 'PRUNER_STARTUP_TRIALS', 
                        'PRUNER_WARMUP_STEPS', 'PRUNER_INTERVAL_STEPS',
                        'OPTIMIZATION_CANDIDATES', 'OPTIMIZATION_TARGET',
                        'REPRODUCE_TRIAL'])
    conf = extract_params_from_config(requirements, config)
    
    redirect_logging_to_loguru() # from logging logger to loguru logger

    # supported exp_phase:  try, tuning, stat
    assert (conf.exp_phase == "tuning") or (conf.exp_phase == "stat")\
        or (conf.exp_phase == "try")
    
    with StopWatch():
        # start optimization
        study = reproduce_trial(objective, conf) if conf.reproduce_trial \
        else start_optimization(objective, conf)

    # summarize the results at the end                            
    trial_statistics(study, conf) 
    logger.success(colored('Done and dusted!', 'yellow')) # well done!
    return 


def format_model_name(model_backbone: str) -> str:
        '''
        '''
        if any(char in model_backbone.lower() \
            for char in ['lstm', 'long short-term memory', 'b2bsqrt_tandem']):
            return 'LSTM'
        elif any(char in model_backbone.lower() \
                for char in ['transformer', 'tfmr', 'tandemformer']):
            return 'TRANSFORMER'
        elif any(char in model_backbone.lower() \
                for char in ['s4', 'sfour', 's_four', 'ssss']):
            return 'S4'
        else:
            raise ValueError(f'{model_backbone} is not supported!')
        

class ConsoleTable:
    
    def __init__(self, title, fields):
        '''
        Initializes the ConsoleTable instance with 
        the given title and fields.

        Args:
        - title (str): The title of the table.
        - fields (list[str]): The field names of the table.
        '''
        self.table = PrettyTable(fields)
        self.table.title = title

    def add_entry(self, entry):
        '''
        Adds an entry to the table, which can either be 
        a single row or a list of rows.
        
        Args:
        - entry (list[tuple[Any]] or tuple[Any]): 
            The entry to be added to the table.
        '''
        if type(entry[0]) == list:
            # add multiple rows to the table.
            for row in entry:
                self.table.add_row(row)
        else: # add a single row to the table.
            self.table.add_row(entry)
    
    def print(self):
        '''
        Prints the formatted table to the console.
        Also sort by the first entry (e.g., 'Parameter name').
        '''
        self.table.sortby = self.table.field_names[0]
        logger.info('\n' + str(self.table))


def sample_parameter(trial, paramname: str, paramspace:dict) -> Any:
    '''
    Suggest one parameter from the provided serch space.
    Key values starting "LIST_" containing parameter search space information.
    
    Args:
    -paramname (str): a name of parameter search space.
    -paramspace (dict): a python dictionary contains search space info.
        PARAM_SPACE: "float", "int", or "categorical".
        - if float: use suggest_float to suggest a float of range [LOW, HIGH], separated by STEP.
        if LOG=True, a float is sampled from logspace but you shall set STEP=None.
        - if int: use suggest_int to suggest an int of range [LOW, HIGH], separated by STEP.
        STEP should be divisor of the range, otherwise HIGH will be automatically modified. 
        if LOG=True, an int is sampled from logspace but you shall set STEP=None.
        - if categorical: use suggest_categorical to select one category from CATEGORY_SET.
        Note that if the parameter is continuous (e.g., 1, 2, 3, ..., or 1.0, 0.1, 0.001, ...),
        it is adviseable to use float or int space because suggest_categorical treats
        each category independently.

    Returns:
    - suggested_parameter: A suggested parameter, either category, float, or int.
    '''
    assert type(paramspace) == dict, 'LIST_ must be a dictionary!'

    if 'int' in paramspace['PARAM_SPACE']:
        suggested_parameter = trial.suggest_int(paramname, 
                                            low=paramspace['LOW'], high=paramspace['HIGH'], 
                                            step=paramspace['STEP'], log=paramspace['LOG'])
    elif 'float' in paramspace['PARAM_SPACE']:
        suggested_parameter = trial.suggest_float(paramname, 
                                            low=paramspace['LOW'], high=paramspace['HIGH'], 
                                            step=paramspace['STEP'], log=paramspace['LOG'])
    elif 'cat' in paramspace['PARAM_SPACE']:
        suggested_parameter = trial.suggest_categorical(paramname, paramspace['CATEGORY_SET'])
    else:
        raise ValueError('Unknown parameter space! Currently "float",'
                            '"int", or "categorical" are supprted.')
    return suggested_parameter


def check_forbidden_parameters(config: dict) -> None:
    '''
    Investigate whether the configuration file contains a forbidden combination of 
    parameters defined in `config['FORBIDDEN_PARAM_SETS']`.
    '''
    for label, paramset in config['FORBIDDEN_PARAM_SETS'].items():
        # Note that the forbidden parameter names are prefixed with 'LIST_' to indicate the Optuna search space. 
        # Thus, the parameter name without the prefix is used as a config key (`param_name[len('LIST_'):]`).
        if all(config[param_name[len('LIST_'):]] == param_value for param_name, param_value in paramset.items()):
            raise ValueError(f'Forbidden parameter combination "{label}" is found!'
                            f'Please select values other than {paramset.values()}.')
        

def set_config_param(key: str, config: Dict[str, Any], trial: optuna.Trial, paramtable: ConsoleTable) -> str:
    '''
    Set the value of a parameter in the config dictionary. See config_definition.py for detail.
    
    Args:
        key (str): The name of the parameter to be set.
        config (Dict[str, float]): The configuration dictionary where the parameter value will be set.
        trial (Trial): The Optuna trial object that will be used to suggest a new parameter value.
        paramtable (ConsoleTable): An instance of the ConsoleTable class to log the new parameter value.
        
    Returns:
        str: The new key for the parameter value in the config dictionary.    
    '''
    param_space = config[key]
    
    newkey = key.upper().replace('LIST_', '')
    
    if config.get(newkey) is not None:
        # suggest and assign the value in-place
        config[newkey] = sample_parameter(trial, key, param_space)
    else:
        raise ValueError(f"{newkey} not found in config!")
    
    # add to the table
    paramtable.add_entry([newkey, float_formatter(config[newkey])])
    
    return newkey


def setup_model_parameter_space(config: Dict[str, Any], model_key: str, model_zoo: List[str]) -> None:
    '''
    Setup the parameter space for a specific model in the config dictionary.

    Args:
        config (Dict[str, Any]): The config dictionary to be modified.
        model_key (str): The key in the config dictionary representing the name of the model.
        model_zoo (List[str]): A list of available model names.

    Returns:
        None
    '''
    # modify the rest of the parameter space
    model_name = config[model_key]
    config.update({key: value for key, value in config[model_name.upper()].items()}) 
    for key in model_zoo:
        if key.upper() in config:
            del config[key.upper()]


def conditional_suggestion(config: Dict[str, Any], trial: optuna.Trial, paramtable: ConsoleTable) -> None:
    '''
    Conditionally suggest hyperparameters using Optuna based on the configuration dictionary.

    Args:
        config (Dict[str, float]): The configuration dictionary containing the hyperparameters to be optimized.
        trial (Trial): The Optuna trial object to be used for hyperparameter optimization.
        paramtable (ConsoleTable): An instance of the ConsoleTable class to log the new parameter values.

    Returns:
        None
    '''
    space_is_pe = 'LIST_IS_POSITIONAL_ENCODING'
    space_is_grad = 'LIST_IS_TRAINABLE_ENCODING'
    space_model = 'LIST_MODEL_BACKBONE'
    model_zoo = config[space_model]['CATEGORY_SET']
    
    # Sample positional encoding flag
    if space_is_pe in config:
        newkey = set_config_param(space_is_pe, config, trial, paramtable)
        del config[space_is_pe]
        
        # if positional encoding is used, choose trainable or not
        if config[newkey] and space_is_grad in config:
            _ = set_config_param(space_is_grad, config, trial, paramtable)
        # delete trainable flag space for both pe=True and pe=False case
        del config[space_is_grad]
            
    # Sample model backbone
    if space_model in config:
        newkey = set_config_param(space_model, config, trial, paramtable)
        del config[space_model]
    
    # modify config structure according to the suggested model.
    setup_model_parameter_space(config, newkey, model_zoo)


def suggest_parameters(trial: optuna.Trial, config: Dict[str, Any]) -> None:
    '''
    Suggest hyperparameters. Note that the keys for hyperparameter lists in config must start with a prefix "list_," 

    Args:
    - trial: A trial object for optuna optimization.
    - config: A dictionary of hyperparameters.
    
    Returns:
    - config: An updated dictionary with suggested hyperparameters.
    
    Raise:
    - ValueError if keys in kwargs and config do not match.
    '''

    # check if necessary parameters are defined in the config file
    requirements = set(['EXP_PHASE', 'FORBIDDEN_PARAM_SETS'])
    conf = extract_params_from_config(requirements, config)

    # run sanity check and return if not sampling is necessary
    if not 'tuning' in conf.exp_phase:
        check_forbidden_parameters(config)
        return
    
    logger.info(colored('Starting a tuning phase.', 'yellow'))
    
    # move parameters that are relevant to current model une level up in the hierarchy
    
    # create a table for readable output
    paramtable = ConsoleTable('Hyperparameteres suggested by Optuna',
                             ['Parameter name', 'Suggested value'])   
    
    # conditional optimization to avoid unnecessary hyperparameters
    conditional_suggestion(config, trial, paramtable)
    
    # suggest the rest of the hyperparameters
    for key, value in config.items():
        if key.upper().startswith('LIST_'):
            _ = set_config_param(key, config, trial, paramtable)

    # print a table of suggested params
    paramtable.print()

    return


def report_to_pruner(trial, best, iter, config):
    '''
    Report the intermediate "best" variable to decide
    whether to prune the training (early-stopping).
    
    Args:
    - trial (optuna trial)
    - best (tuple): performance metrics to be minimized.
    - iter (int): the current training step.
    - config (dict): a custom dictionary storeing parameters.
    
    Returns:
    - None
    '''
    # check if necessary parameters are defined in the config file
    requirements = set(['OPTIMIZATION_TARGET'])
    conf = extract_params_from_config(requirements, config)
    
    if 'all' in conf.optimization_target.lower():
        return # multi-objective pruner is not supported
    else:
        # report intermediate objective value.
        trial.report(*best, iter)
        # handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

