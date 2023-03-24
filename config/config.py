# Assuption: this file is placed inside a directory named "config", under the project root directory.

from utils.misc import compile_subproject_name, compile_directory_paths, sort_config

##########################USER MODIFIABLE PARAMS######################################

# Base info
CONFIG_PATH = __file__
LOG_PATH = CONFIG_PATH[:CONFIG_PATH.find('config')] + 'logs/'

DATA_SEPARATION = '2.0'
DATA_PATH = '/home/afe/Dropbox/PYTHON/data/SDRE_data/'
NUM_CLASSES = 3 # 2
NUM_TRAIN = 25000 # 19000
NUM_VAL = 4000 # # 990 
NUM_TEST = 1000 # 10 
NUM_TOTAL_DATA = NUM_TRAIN + NUM_VAL + NUM_TEST
DATA_SUFFIX = f'_{NUM_TOTAL_DATA}_{NUM_CLASSES}class_offset{DATA_SEPARATION}'
TB_DIRNAME = 'TensorBoard_events'
DB_DIRNAME = 'Optuna_databases'
CKPT_DIRNAME = 'checkpoints'
IMG_DIRNAME = 'image_logs'
STDOUT_DIRNAME = 'stdout_logs'

SUBPROJECT_NAME_PREFIX = 'PTdev_' 
SUBPROJECT_NAME_SUFFIX = '' 
COMMENT = '' # for the log file name
EXP_PHASE = 'tuning' # 'try', 'tuning', or 'stat'
MODEL_BACKBONE = 'Transformer'
ORDER_SPRT = 1
PARAM_MULTIPLET_LOSS = 1.0
LLLR_VERSION = 'Eplus'
PARAM_LLR_LOSS = 1.0
AUCLOSS_VERSION = 'Anorm'
AUCLOSS_BURNIN = 1000
IS_ADAPTIVE_LOSS = False
PARAM_AUSAT_LOSS = 0.0
OPTIMIZATION_CANDIDATES = ('ausat_loss', 'macrec', 'mabs', 'ausat_confmx', ) # do not change the order
OPTIMIZATION_TARGET= "ausat_confmx" 
IS_RESUME = False
REPRODUCE_TRIAL = False # 'best', trial index (int) or False
SUBPROJECT_TO_RESUME = '_20230323_040731087ckpt_step1500_target_ausat_confmx0.0211.pth'

# of frequent use
GPU = 0
NUM_TRIALS = 1000
NUM_ITER = 7501
TRAIN_DISPLAY_STEP = 500
VALIDATION_STEP = 500
PRUNER_NAME = 'percentile' # hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
PRUNER_STARTUP_TRIALS = 100
PRUNER_WARMUP_STEPS = 1000
PRUNER_INTERVAL_STEPS = 500

##############################USER MODIFIABLE PARAMS END#######################################

info_for_subproject_name = {
    'DATA_SEPARATION': DATA_SEPARATION,
    'OPTIMIZATION_TARGET': OPTIMIZATION_TARGET,
    'MODEL_BACKBONE': MODEL_BACKBONE,
    'PRUNER_NAME': PRUNER_NAME,
    'ORDER_SPRT': ORDER_SPRT,
    'LLLR_VERSION': LLLR_VERSION,
    'PARAM_LLR_LOSS': PARAM_LLR_LOSS,
    'AUCLOSS_VERSION': AUCLOSS_VERSION,
    'PARAM_AUSAT_LOSS': PARAM_AUSAT_LOSS,
    'AUCLOSS_BURNIN': AUCLOSS_BURNIN,
    'PARAM_MULTIPLET_LOSS': PARAM_MULTIPLET_LOSS,
    'IS_ADAPTIVE_LOSS': IS_ADAPTIVE_LOSS,
    'EXP_PHASE': EXP_PHASE,
    'SUBPROJECT_NAME_PREFIX': SUBPROJECT_NAME_PREFIX,
    'SUBPROJECT_NAME_SUFFIX': SUBPROJECT_NAME_SUFFIX
}

SUBPROJECT_NAME = compile_subproject_name(info_for_subproject_name)

subconfig = {
    'LOG_PATH' : LOG_PATH,
    'TB_DIRNAME' : TB_DIRNAME,
    'DB_DIRNAME' : DB_DIRNAME,
    'CKPT_DIRNAME' : CKPT_DIRNAME,
    'IMG_DIRNAME' : IMG_DIRNAME,
    'STDOUT_DIRNAME' : STDOUT_DIRNAME,
    'SUBPROJECT_NAME': SUBPROJECT_NAME,
    'SUBPROJECT_TO_RESUME': SUBPROJECT_TO_RESUME,
}

compile_directory_paths(subconfig)

config = {

    'VERBOSE': True,
    'CONFIG_PATH': CONFIG_PATH,
    'IS_SEED': False,
    'SEED': 7,
    'GPU': GPU,

    # for logging
    'NAME_DATASET': 'Multivariate_Gaussian',
    'COMMENT': COMMENT, 
    'IS_SAVE_FIGURE': True,
    'DATA_SEPARATION': DATA_SEPARATION,
    'SUBPROJECT_NAME_PREFIX': SUBPROJECT_NAME_PREFIX,
    'SUBPROJECT_NAME_SUFFIX': SUBPROJECT_NAME_SUFFIX,
    
    # SDRE datasets
    'X_PATH': f'{DATA_PATH}x_batch{DATA_SUFFIX}.npy',
    'Y_PATH': f'{DATA_PATH}y_batch{DATA_SUFFIX}.npy',
    'GT_LLR_PATH': f'{DATA_PATH}gt_llrms{DATA_SUFFIX}.npy',
    'NUM_TRAIN': NUM_TRAIN, # total 100000
    'NUM_VAL': NUM_VAL,
    'NUM_TEST': NUM_TEST,
    'NUM_CLASSES': NUM_CLASSES,
    'FEAT_DIM': 128,

    # SPRT-TANDEM parameters
    'ORDER_SPRT': ORDER_SPRT,
    'TIME_STEPS': 50,
    'LLLR_VERSION': LLLR_VERSION, # LSEL
    'OBLIVIOUS': False, # Miyagawa & Ebihara, ICML2021
    'PARAM_MULTIPLET_LOSS': PARAM_MULTIPLET_LOSS,
    'PARAM_LLR_LOSS': PARAM_LLR_LOSS,

    # Network specs
    'MODEL_BACKBONE': MODEL_BACKBONE, # LSTM (B2Bsqrt-TANDEM) or Transformer (TANDEMformer)
    'ACTIVATION_OUTPUT': 'B2Bsqrt',
    'ACTIVATION_FC': 'relu',
    'ALPHA': 1.0, # for B2Bsqrt activation function
    'IS_NORMALIZE': False, # whether to use Normalization or not
    'IS_POSITIONAL_ENCODING': True,
    'IS_TRAINABLE_ENCODING': True,
    # For LSTM backbone
    'ACTIVATION_INPUT': 'sigmoid',
    'WIDTH_LSTM': 64,
    # For TRANSFORMER backbone
    'NUM_BLOCKS': 2, # transformer block
    'NUM_HEADS': 2,
    'DROPOUT': 0.1,
    'FF_DIM': 64,
    'MLP_UNITS': 64,

    # torch.compile (pytorch > 2.0)
    'IS_COMPILE': False,
    'MODE': 'default', #'reduce-overhead', 'default', or 'max-autotune' 

    # SAT curve
    'AUCLOSS_VERSION': AUCLOSS_VERSION,
    'IS_MULT_SAT': True,
    'NUM_THRESH': 1000,
    'SPARSITY': 'logspace', # "linspace", "logspace", "unirandom", or "lograndom".
    'BETA': 1.0,
    'PARAM_AUSAT_LOSS': PARAM_AUSAT_LOSS,
    'AUCLOSS_BURNIN': AUCLOSS_BURNIN,
    'IS_ADAPTIVE_LOSS': IS_ADAPTIVE_LOSS,

    # Training parameters
    'OPTIMIZATION_TARGET': OPTIMIZATION_TARGET,
    'OPTIMIZATION_CANDIDATES': OPTIMIZATION_CANDIDATES,
    'PRUNER_STARTUP_TRIALS': PRUNER_STARTUP_TRIALS,
    'PRUNER_WARMUP_STEPS': PRUNER_WARMUP_STEPS,
    'PRUNER_INTERVAL_STEPS':PRUNER_INTERVAL_STEPS, 
    'PRUNER_NAME': PRUNER_NAME, # hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
    'MAX_TO_KEEP': 1,
    'EXP_PHASE': EXP_PHASE,
    'REPRODUCE_TRIAL': REPRODUCE_TRIAL,
    'NUM_TRIALS': NUM_TRIALS,
    'BATCH_SIZE': 100,
    'LEARNING_RATE': 1e-4,
    'LR_DECAY_STEPS':  [100000000,],
    'WEIGHT_DECAY': 0.0,
    'OPTIMIZER': 'adam',
    'NUM_ITER': NUM_ITER, #7500
    'TRAIN_DISPLAY_STEP': TRAIN_DISPLAY_STEP,
    'VALIDATION_STEP': VALIDATION_STEP,
    'IS_RESUME': IS_RESUME,
    
    ####### Optuna hyperparameter search space ####### 
    
    # Key values starting "LIST_" containing parameter search space information.
    # PARAM_SPACE: "float", "int", or "categorical".
    # - float: use suggest_float to suggest a float of range [LOW, HIGH], separated by STEP.
    #   if LOG=True, a float is sampled from logspace but you shall set STEP=None.
    # - int: use suggest_int to suggest an int of range [LOW, HIGH], separated by STEP.
    #   STEP should be divisor of the range, otherwise HIGH will be automatically modified. 
    #   if LOG=True, an int is sampled from logspace but you shall set STEP=None.
    # - categorical: use suggest_categorical to select one category from CATEGORY_SET.
    #   Note that if the parameter is continuous (e.g., 1, 2, 3, ..., or 1.0, 0.1, 0.001, ...),
    #   it is adviseable to use float or int space because suggest_categorical treats
    #   each category independently.
    
    # common to all backbones
    'LIST_WEIGHT_DECAY': {
        'PARAM_SPACE': 'float',
        'LOW': 0.0,
        'HIGH': 0.001,
        'STEP': 0.0001,
        'LOG': False, # log is preferable but it doesn't allow LOW==0.0
    },
    'LIST_LEARNING_RATE': {
        'PARAM_SPACE': 'float',
        'LOW': 0.00001,
        'HIGH': 0.01,
        'STEP': None,
        'LOG': True,
    },
    'LIST_LLLR_VERSION': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': ['A', 'B', 'C', 'D', 'E', 'Eplus']
    },
    'LIST_AUCLOSS_VERSION': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': ['A', 'B', 'C', 'D', 'Anorm', 'Bnorm', 'Cnorm', 'Dnorm']
    },
    'LIST_ACTIVATION_FC': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': ['B2Bsqrt', 'B2BsqrtV2', 'B2Bcbrt', 'tanh', 'relu', 'gelu']
    },
    'LIST_PARAM_MULTIPLET_LOSS': {
        'PARAM_SPACE': 'float',
        'LOW': 0.0,
        'HIGH': 1.0,
        'STEP': 0.1,
        'LOG': False,
    },
    'LIST_PARAM_LLR_LOSS': {
        'PARAM_SPACE': 'float',
        'LOW': 0.0,
        'HIGH': 1.0,
        'STEP': 0.1,
        'LOG': False,
    },
    'LIST_PARAM_AUSAT_LOSS': {
        'PARAM_SPACE': 'float',
        'LOW': 0.0,
        'HIGH': 1.0,
        'STEP': 0.1,
        'LOG': False,
    },
    'LIST_AUCLOSS_BURNIN': {
        'PARAM_SPACE': 'int',
        'LOW': 0,
        'HIGH': NUM_ITER,
        'STEP': 1000,
        'LOG': False,
    },
    'LIST_IS_ADAPTIVE_LOSS': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': [True, False]
    },
    'LIST_ORDER_SPRT': {
        'PARAM_SPACE': 'int',
        'LOW': 0,
        'HIGH': 10,
        'STEP': 1,
        'LOG': False,
    },
    'LIST_ALPHA': {
        'PARAM_SPACE': 'float',
        'LOW': 0.8,
        'HIGH': 1.2,
        'STEP': 0.1,
        'LOG': False,
    },
    'LIST_IS_POSITIONAL_ENCODING': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': [True, False]
    },
    'LIST_IS_TRAINABLE_ENCODING': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': [True, False]
    },
    'LIST_OPTIMIZER': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': ['rmsprop', 'adam']
    },
    'LIST_IS_NORMALIZE': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': [True, False]
    },
    'LIST_IS_MULT_SAT': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': [True, False]
    },
    'LIST_NUM_THRESH': {
        'PARAM_SPACE': 'int',
        'LOW': 500,
        'HIGH': 1500,
        'STEP': 500,
        'LOG': False,
    },
  
    'LIST_SPARSITY': {
        'PARAM_SPACE': 'categorical',
        'CATEGORY_SET': ['linspace', 'logspace', 'unirandom', 'lograndom'],
    },
    # hyperparams specific to each backbone
    # B2Bsqrt-TANDEM
    'LSTM': {
        'LIST_ACTIVATION_OUTPUT': {
            'PARAM_SPACE': 'categorical',
            'CATEGORY_SET': ['B2Bsqrt', 'B2BsqrtV2', 'B2Bcbrt', 'tanh', 'sigmoid']
        },
        'LIST_ACTIVATION_INPUT': {
            'PARAM_SPACE': 'categorical',
            'CATEGORY_SET': ['tanh', 'sigmoid']
            },
        'LIST_WIDTH_LSTM': {
            'PARAM_SPACE': 'int',
            'LOW': 16,
            'HIGH': 128,
            'STEP': 1,
            'LOG': True,
            },
    },
    # TANDEMformer
    'TRANSFORMER':{
        'LIST_NUM_BLOCKS': {
            'PARAM_SPACE': 'int',
            'LOW': 1,
            'HIGH': 2,
            'STEP': 1,
            'LOG': False,
        },
        'LIST_NUM_HEADS': { # num_heads need to be a divisor of feat_dim (=embed_dim)
            'PARAM_SPACE': 'int',
            'LOW': 2,
            'HIGH': 4,
            'STEP': 2,
            'LOG': False,
        },
        'LIST_DROPOUT': {
        'PARAM_SPACE': 'float',
        'LOW': 0.0,
        'HIGH': 0.5,
        'STEP': 0.1,
        'LOG': False,
    },
        'LIST_FF_DIM': {
            'PARAM_SPACE': 'int',
            'LOW': 32,
            'HIGH': 64,
            'STEP': 32,
            'LOG': False,
        },
        'LIST_MLP_UNITS': {
            'PARAM_SPACE': 'int',
            'LOW': 32,
            'HIGH': 64,
            'STEP': 32,
            'LOG': False,
        },
    },
}

config.update(subconfig)
config = sort_config(config)



