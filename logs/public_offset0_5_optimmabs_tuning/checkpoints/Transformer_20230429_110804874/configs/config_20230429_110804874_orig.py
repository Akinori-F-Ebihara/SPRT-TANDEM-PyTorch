# Assuption: this file is placed inside a directory named "config", under the project root directory.

from utils.misc import compile_subproject_name, compile_directory_paths

########################## USER EDITABLE BLOCK STARTS ######################################

# Base info
CONFIG_PATH = __file__
LOG_PATH = CONFIG_PATH[: CONFIG_PATH.find("config")] + "logs/"

# Data info
NUM_CLASSES = 3  # 2
DATA_SEPARATION = "0.5"
DATA_PATH = "/home/afe/Dropbox/PYTHON/data/SDRE_data/"
DATA_FOLDER = f"SequentialGaussian_{NUM_CLASSES}class_offset{DATA_SEPARATION}"
NUM_TRAIN = 60000  # 300000  # 25000 # 19000
NUM_VAL = 6000  # 30000  # 4000 # # 990
NUM_TEST = 6000  # 30000  # 1000 # 10
TRAIN_DATA = f"{DATA_PATH}{DATA_FOLDER}/train_{NUM_TRAIN}"
VAL_DATA = f"{DATA_PATH}{DATA_FOLDER}/val_{NUM_VAL}"
TEST_DATA = f"{DATA_PATH}{DATA_FOLDER}/test_{NUM_TEST}"
DATA_NAMES = ("data", "label", "llr")
IS_SHUFFLE = True  # whether to shuffle training data. It may cause significant overhead
TB_DIRNAME = "TensorBoard_events"
DB_DIRNAME = "Optuna_databases"
CKPT_DIRNAME = "checkpoints"
STDOUT_DIRNAME = "stdout_logs"

SUBPROJECT_NAME_PREFIX = "public_"
SUBPROJECT_NAME_SUFFIX = ""
COMMENT = ""  # for the log file name
OPTIMIZATION_CANDIDATES = (
    "macrec",
    "mabs",
    "ausat_confmx",
)
OPTIMIZATION_TARGET = "ausat_confmx"
EXP_PHASE = "try"  # 'try', 'tuning', or 'stat'

# if EXP_PHASE=='tuning', these params will be overwritten
# with a new parameter sampled by Optuna.
MODEL_BACKBONE = "LSTM"
# non-negative value and must be less than time_steps of the data.
ORDER_SPRT = 1
PARAM_MULTIPLET_LOSS = 0.6
LLLR_VERSION = "LSEL"  # LLLR or LSEL
PARAM_LLR_LOSS = 0.3
IS_ADAPTIVE_LOSS = False

# Reproduce a particular trial or the best trial from specified subproject.
REPRODUCE_TRIAL = False  # 'best', trial index (int) or False
# Load pretrained weight from .pt file specified below.
IS_RESUME = False
SUBPROJECT_TO_RESUME = "_20230330_214519224/ckpt_step3500_target_ausat_confmx0.0011.pt"

# of frequent use
GPU = 0
BATCH_SIZE = 150
NUM_TRIALS = 1000
NUM_EPOCHS = 10
NUM_ITER = NUM_EPOCHS * NUM_TRAIN // BATCH_SIZE  # e.g., 20 * 25000 // 100 = 5000
TRAIN_DISPLAY_STEP = 250
VALIDATION_STEP = 250
# hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
PRUNER_NAME = "percentile"
PRUNER_STARTUP_TRIALS = 100
PRUNER_WARMUP_STEPS = 2000
PRUNER_INTERVAL_STEPS = 500

############################## USER EDITABLE BLOCK ENDS #######################################

info_for_subproject_name = {
    "DATA_SEPARATION": DATA_SEPARATION,
    "OPTIMIZATION_TARGET": OPTIMIZATION_TARGET,
    "MODEL_BACKBONE": MODEL_BACKBONE,
    "PRUNER_NAME": PRUNER_NAME,
    "ORDER_SPRT": ORDER_SPRT,
    "LLLR_VERSION": LLLR_VERSION,
    "PARAM_LLR_LOSS": PARAM_LLR_LOSS,
    "PARAM_MULTIPLET_LOSS": PARAM_MULTIPLET_LOSS,
    "IS_ADAPTIVE_LOSS": IS_ADAPTIVE_LOSS,
    "EXP_PHASE": EXP_PHASE,
    "SUBPROJECT_NAME_PREFIX": SUBPROJECT_NAME_PREFIX,
    "SUBPROJECT_NAME_SUFFIX": SUBPROJECT_NAME_SUFFIX,
}

SUBPROJECT_NAME = compile_subproject_name(info_for_subproject_name)

subconfig = {
    "LOG_PATH": LOG_PATH,
    "TB_DIRNAME": TB_DIRNAME,
    "DB_DIRNAME": DB_DIRNAME,
    "CKPT_DIRNAME": CKPT_DIRNAME,
    "STDOUT_DIRNAME": STDOUT_DIRNAME,
    "SUBPROJECT_NAME": SUBPROJECT_NAME,
    "SUBPROJECT_TO_RESUME": SUBPROJECT_TO_RESUME,
}

compile_directory_paths(subconfig)

config = {
    "VERBOSE": True,
    "CONFIG_PATH": CONFIG_PATH,
    "IS_SEED": False,
    "SEED": 7,
    "GPU": GPU,
    "MAX_NORM": 50000,  # used for gradient clipping
    # for logging
    "NAME_DATASET": "Multivariate_Gaussian",
    "COMMENT": COMMENT,
    "IS_SAVE_FIGURE": True,
    "DATA_SEPARATION": DATA_SEPARATION,
    "SUBPROJECT_NAME_PREFIX": SUBPROJECT_NAME_PREFIX,
    "SUBPROJECT_NAME_SUFFIX": SUBPROJECT_NAME_SUFFIX,
    # SDRE datasets
    "TRAIN_DATA": TRAIN_DATA,
    "VAL_DATA": VAL_DATA,
    "TEST_DATA": TEST_DATA,
    "DATA_NAMES": DATA_NAMES,
    "IS_SHUFFLE": IS_SHUFFLE,
    "NUM_TRAIN": NUM_TRAIN,
    "NUM_VAL": NUM_VAL,
    "NUM_TEST": NUM_TEST,
    "NUM_CLASSES": NUM_CLASSES,
    "FEAT_DIM": 128,
    # SPRT-TANDEM parameters
    "ORDER_SPRT": ORDER_SPRT,
    "TIME_STEPS": 50,
    "LLLR_VERSION": LLLR_VERSION,  # LSEL
    "OBLIVIOUS": False,  # Miyagawa & Ebihara, ICML2021
    "PARAM_MULTIPLET_LOSS": PARAM_MULTIPLET_LOSS,
    "PARAM_LLR_LOSS": PARAM_LLR_LOSS,
    # Network specs
    # LSTM (B2Bsqrt-TANDEM) or Transformer (TANDEMformer)
    "MODEL_BACKBONE": MODEL_BACKBONE,
    "ACTIVATION_OUTPUT": "B2Bsqrt",
    "ACTIVATION_FC": "relu",
    "ALPHA": 1.0,  # for B2Bsqrt activation function
    "IS_NORMALIZE": False,  # whether to use Normalization or not
    "IS_POSITIONAL_ENCODING": True,
    "IS_TRAINABLE_ENCODING": True,
    # For LSTM backbone
    "ACTIVATION_INPUT": "sigmoid",
    "WIDTH_LSTM": 64,
    # For TRANSFORMER backbone
    "NUM_BLOCKS": 2,  # transformer block
    "NUM_HEADS": 2,
    "DROPOUT": 0.1,
    "FF_DIM": 64,
    "MLP_UNITS": 64,
    # for performance
    "IS_COMPILE": False,  # torch.compile (pytorch > 2.0)
    "MODE": "reduce-overhead",  # 'reduce-overhead', 'default', or 'max-autotune'
    "NUM_WORKERS": 0,  # num_workers argument for pytorch dataloader
    # whether to load dataset onto memory at initialization.
    "IS_LOAD_ONTO_MEMORY": True,
    # SAT curve
    "NUM_THRESH": 1000,
    # "linspace", "logspace", "unirandom", or "lograndom".
    "SPARSITY": "logspace",
    "BETA": 1.0,
    "IS_ADAPTIVE_LOSS": IS_ADAPTIVE_LOSS,
    # Training parameters
    "OPTIMIZATION_TARGET": OPTIMIZATION_TARGET,
    "OPTIMIZATION_CANDIDATES": OPTIMIZATION_CANDIDATES,
    "PRUNER_STARTUP_TRIALS": PRUNER_STARTUP_TRIALS,
    "PRUNER_WARMUP_STEPS": PRUNER_WARMUP_STEPS,
    "PRUNER_INTERVAL_STEPS": PRUNER_INTERVAL_STEPS,
    # hyperband, median, percentile, etc... set to 'none' if no pruner is needed.
    "PRUNER_NAME": PRUNER_NAME,
    "MAX_TO_KEEP": 1,
    "EXP_PHASE": EXP_PHASE,
    "REPRODUCE_TRIAL": REPRODUCE_TRIAL,
    "NUM_TRIALS": NUM_TRIALS,
    "BATCH_SIZE": BATCH_SIZE,
    "LEARNING_RATE": 1e-4,
    "LR_DECAY_STEPS": [
        100000000,
    ],
    "WEIGHT_DECAY": 0.0,
    "OPTIMIZER": "adam",
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_ITER": NUM_ITER,  # 7500
    "TRAIN_DISPLAY_STEP": TRAIN_DISPLAY_STEP,
    "VALIDATION_STEP": VALIDATION_STEP,
    "IS_RESUME": IS_RESUME,
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
    "LIST_MODEL_BACKBONE": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["LSTM", "Transformer"],  # 'LSTM', 'Transformer'
    },
    # common to all backbones
    "LIST_WEIGHT_DECAY": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 0.001,
        "STEP": 0.0001,
        "LOG": False,  # log is preferable but it doesn't allow LOW==0.0
    },
    "LIST_LEARNING_RATE": {
        "PARAM_SPACE": "float",
        "LOW": 0.000001,
        "HIGH": 0.001,
        "STEP": None,
        "LOG": True,
    },
    "LIST_LLLR_VERSION": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["LLLR", "LSEL"],
    },
    "LIST_ACTIVATION_FC": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["B2Bsqrt", "B2BsqrtV2", "B2Bcbrt", "tanh", "relu", "gelu"],
    },
    "LIST_PARAM_MULTIPLET_LOSS": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 1.0,
        "STEP": 0.1,
        "LOG": False,
    },
    "LIST_PARAM_LLR_LOSS": {
        "PARAM_SPACE": "float",
        "LOW": 0.0,
        "HIGH": 1.0,
        "STEP": 0.1,
        "LOG": False,
    },
    "LIST_IS_ADAPTIVE_LOSS": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": [True, False],
    },
    "LIST_ORDER_SPRT": {
        "PARAM_SPACE": "int",
        "LOW": 0,
        "HIGH": 5,  # 10
        "STEP": 1,
        "LOG": False,
    },
    "LIST_IS_POSITIONAL_ENCODING": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": [True, False],
    },
    "LIST_IS_TRAINABLE_ENCODING": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": [True, False],
    },
    "LIST_OPTIMIZER": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["rmsprop", "adam"],
    },
    "LIST_IS_NORMALIZE": {"PARAM_SPACE": "categorical", "CATEGORY_SET": [True, False]},
    "LIST_NUM_THRESH": {
        "PARAM_SPACE": "int",
        "LOW": 500,
        "HIGH": 1500,
        "STEP": 500,
        "LOG": False,
    },
    "LIST_SPARSITY": {
        "PARAM_SPACE": "categorical",
        "CATEGORY_SET": ["linspace", "logspace", "unirandom", "lograndom"],
    },
    # hyperparams specific to each backbone
    # B2Bsqrt-TANDEM
    "LSTM": {
        "LIST_ACTIVATION_OUTPUT": {
            "PARAM_SPACE": "categorical",
            "CATEGORY_SET": ["B2Bsqrt", "B2BsqrtV2", "B2Bcbrt", "tanh", "sigmoid"],
        },
        "LIST_ACTIVATION_INPUT": {
            "PARAM_SPACE": "categorical",
            "CATEGORY_SET": ["tanh", "sigmoid"],
        },
        "LIST_WIDTH_LSTM": {
            "PARAM_SPACE": "int",
            "LOW": 16,
            "HIGH": 128,
            "STEP": 1,
            "LOG": True,
        },
    },
    # TANDEMformer
    "TRANSFORMER": {
        "LIST_NUM_BLOCKS": {
            "PARAM_SPACE": "int",
            "LOW": 1,
            "HIGH": 2,
            "STEP": 1,
            "LOG": False,
        },
        "LIST_NUM_HEADS": {  # num_heads need to be a divisor of feat_dim (=embed_dim)
            "PARAM_SPACE": "int",
            "LOW": 2,
            "HIGH": 4,
            "STEP": 2,
            "LOG": False,
        },
        "LIST_DROPOUT": {
            "PARAM_SPACE": "float",
            "LOW": 0.0,
            "HIGH": 0.5,
            "STEP": 0.1,
            "LOG": False,
        },
        "LIST_FF_DIM": {
            "PARAM_SPACE": "int",
            "LOW": 32,
            "HIGH": 64,
            "STEP": 32,
            "LOG": False,
        },
        "LIST_MLP_UNITS": {
            "PARAM_SPACE": "int",
            "LOW": 32,
            "HIGH": 64,
            "STEP": 32,
            "LOG": False,
        },
    },
    # Some combination of hyperparameters are incompatible or ineffective.
    "FORBIDDEN_PARAM_SETS": {
        "loss_all_zero": {
            "LIST_PARAM_MULTIPLET_LOSS": 0.0,
            "LIST_PARAM_LLR_LOSS": 0.0,
        },
    },
}

config.update(subconfig)
