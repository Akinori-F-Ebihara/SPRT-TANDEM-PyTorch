import os, pdb
from prettytable import PrettyTable
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
        try:
            float_variable = float(variable)
            return formatted_float(float_variable)
        except:
            return variable
    else:
        return variable

        
def run_optuna(objective, device, config):
    '''
    Load or create optuna study and run optimization steps.
    Also sets pruner for early-stopping.

    Args:
    - objective(function): a function that train and evaluate a model.
    - config(dict): a dictionary containing hyperparameters.
    '''
    def setup_pruner(pruner_name, conf):
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
            directions=['minimize'] * num_objectives)
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
        path_db = conf.root_dblogs + "/" + study_name + ".db"
        storage_name = "sqlite:///" + path_db
        if not os.path.exists(path_db):
            raise ValueError(colored(
               f"{path_db} not found. Hyperparameters could not be restored", 'red'))
        # Load parameters and start optimization
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        for cnt in range(conf.num_trials):
            trial = study.best_trial if 'best' in str(conf.reproduce_trial).lower() \
                                     else study.trials[conf.reproduce_trial]
            # report to the console
            logger.info(colored(
                f'reproducing the trial ({conf.reproduce_trial})...iter#{cnt}', 'yellow'))
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
            if not 'tuning' in conf.exp_phase:
                return
            response = input('Would you like to print a parameter importance table? (yes/[no]) ')
            if 'yes' in response.lower():
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
    requirements = set(['ROOT_DBLOGS', 'EXP_PHASE', 
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


def suggest_parameters_optuna(trial, config):
    """ Suggest hyperparameters. Note that the keys for hyperparameter lists in config must start with a prefix "list_," 
    Args:
    - trial: A trial object for optuna optimization.
    - config: A dictionary of hyperparameters.
    Returns:
    - config: An updated dictionary with suggested hyperparameters.
    Raise:
    - ValueError if keys in kwargs and config do not match
    """
    def suggest_parameter(paramname, paramspace):
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

    def get_model_name(model_backbone):
        if any(char in config['MODEL_BACKBONE'].lower() \
            for char in ['lstm', 'long short-term memory', 'b2bsqrt_tandem']):
            return 'LSTM'
        elif any(char in config['MODEL_BACKBONE'].lower() \
                for char in ['transformer', 'tfmr', 'tandemformer']):
            return 'TRANSFORMER'

    ### main ###
    # check if necessary parameters are defined in the config file
    requirements = set(['EXP_PHASE', 'MODEL_BACKBONE'])
    conf = extract_params_from_config(requirements, config)

    # this function is only relevant under tuning phase
    if not 'tuning' in conf.exp_phase:
        return
    # move parameters that are relevant to current model une level up in the hierarchy
    logger.info(colored('Starting a tuning phase.', 'yellow'))
    model_name = get_model_name(conf.model_backbone)
    config.update({key: value for key, value in config[model_name].items()})

    # create a table for readable output
    paramtable = ConsoleTable('Hyperparameteres suggested by Optuna',
                             ['Parameter name', 'Suggested value'])   
    # hyperparam list starts with a prefix 'LIST_'
    for key, value in config.items():
        if key.upper().startswith('LIST_'):
            newkey = key.upper().replace('LIST_', '')
            if config.get(newkey) is not None:
                config[newkey] = suggest_parameter(key, value) # choose a hyperparameter
                paramtable.add_entry([newkey, float_formatter(config[newkey])])
            else: 
                raise ValueError(f"{newkey} not found in config!")
                # otherwise config size changes within the loop, ends up in error.
    # print a table of suggested params
    paramtable.print()

    is_losscoef_all_zero = True if (config['PARAM_LLR_LOSS'] == 0) and \
            (config['PARAM_MULTIPLET_LOSS'] == 0) and \
            (config['PARAM_AUSAT_LOSS'] == 0) else False
    
    return is_losscoef_all_zero
        



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
        '''
        logger.info('\n' + str(self.table))
    

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

