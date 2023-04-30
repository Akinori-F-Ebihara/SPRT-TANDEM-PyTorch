# This is the actual config dict used for the training.
# Simply import this file for reproducing the results.

config = {
  "VERBOSE": True, 
  "CONFIG_PATH": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/config/config_definition.py", 
  "IS_SEED": False, 
  "SEED": 7, 
  "GPU": 0, 
  "MAX_NORM": 50000, 
  "NAME_DATASET": "Multivariate_Gaussian", 
  "COMMENT": "", 
  "IS_SAVE_FIGURE": True, 
  "DATA_SEPARATION": "0.5", 
  "SUBPROJECT_NAME_PREFIX": "public_", 
  "SUBPROJECT_NAME_SUFFIX": "", 
  "TRAIN_DATA": "/home/afe/Dropbox/PYTHON/data/SDRE_data/SequentialGaussian_3class_offset0.5/train_60000", 
  "VAL_DATA": "/home/afe/Dropbox/PYTHON/data/SDRE_data/SequentialGaussian_3class_offset0.5/val_6000", 
  "TEST_DATA": "/home/afe/Dropbox/PYTHON/data/SDRE_data/SequentialGaussian_3class_offset0.5/test_6000", 
  "DATA_NAMES": ('data', 'label', 'llr'), 
  "IS_SHUFFLE": True, 
  "NUM_TRAIN": 60000, 
  "NUM_VAL": 6000, 
  "NUM_TEST": 6000, 
  "NUM_CLASSES": 3, 
  "FEAT_DIM": 128, 
  "ORDER_SPRT": 5, 
  "TIME_STEPS": 50, 
  "LLLR_VERSION": "LSEL", 
  "OBLIVIOUS": False, 
  "PARAM_MULTIPLET_LOSS": 0.6000000000000001, 
  "PARAM_LLR_LOSS": 0.1, 
  "MODEL_BACKBONE": "Transformer", 
  "ACTIVATION_OUTPUT": "B2Bsqrt", 
  "ACTIVATION_FC": "gelu", 
  "ALPHA": 1.0, 
  "IS_NORMALIZE": False, 
  "IS_POSITIONAL_ENCODING": False, 
  "IS_TRAINABLE_ENCODING": True, 
  "ACTIVATION_INPUT": "sigmoid", 
  "WIDTH_LSTM": 64, 
  "NUM_BLOCKS": 2, 
  "NUM_HEADS": 4, 
  "DROPOUT": 0.1, 
  "FF_DIM": 64, 
  "MLP_UNITS": 64, 
  "IS_COMPILE": False, 
  "MODE": "reduce-overhead", 
  "NUM_WORKERS": 0, 
  "IS_LOAD_ONTO_MEMORY": True, 
  "NUM_THRESH": 1000, 
  "SPARSITY": "unirandom", 
  "BETA": 1.0, 
  "IS_ADAPTIVE_LOSS": True, 
  "OPTIMIZATION_TARGET": "mabs", 
  "OPTIMIZATION_CANDIDATES": ('macrec', 'mabs', 'ausat_confmx'), 
  "PRUNER_STARTUP_TRIALS": 100, 
  "PRUNER_WARMUP_STEPS": 2000, 
  "PRUNER_INTERVAL_STEPS": 500, 
  "PRUNER_NAME": "percentile", 
  "MAX_TO_KEEP": 1, 
  "EXP_PHASE": "tuning", 
  "REPRODUCE_TRIAL": False, 
  "NUM_TRIALS": 100, 
  "BATCH_SIZE": 150, 
  "LEARNING_RATE": 0.00024687210886585305, 
  "LR_DECAY_STEPS": [100000000], 
  "WEIGHT_DECAY": 0.0008, 
  "OPTIMIZER": "rmsprop", 
  "NUM_EPOCHS": 10, 
  "NUM_ITER": 4000, 
  "TRAIN_DISPLAY_STEP": 250, 
  "VALIDATION_STEP": 250, 
  "IS_RESUME": False, 
  "LIST_WEIGHT_DECAY": {'PARAM_SPACE': 'float', 'LOW': 0.0, 'HIGH': 0.001, 'STEP': 0.0001, 'LOG': False}, 
  "LIST_LEARNING_RATE": {'PARAM_SPACE': 'float', 'LOW': 1e-06, 'HIGH': 0.001, 'STEP': None, 'LOG': True}, 
  "LIST_LLLR_VERSION": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': ['LLLR', 'LSEL']}, 
  "LIST_ACTIVATION_FC": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': ['B2Bsqrt', 'B2BsqrtV2', 'B2Bcbrt', 'tanh', 'relu', 'gelu']}, 
  "LIST_PARAM_MULTIPLET_LOSS": {'PARAM_SPACE': 'float', 'LOW': 0.0, 'HIGH': 1.0, 'STEP': 0.1, 'LOG': False}, 
  "LIST_PARAM_LLR_LOSS": {'PARAM_SPACE': 'float', 'LOW': 0.0, 'HIGH': 1.0, 'STEP': 0.1, 'LOG': False}, 
  "LIST_IS_ADAPTIVE_LOSS": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': [True, False]}, 
  "LIST_ORDER_SPRT": {'PARAM_SPACE': 'int', 'LOW': 0, 'HIGH': 5, 'STEP': 1, 'LOG': False}, 
  "LIST_OPTIMIZER": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': ['rmsprop', 'adam']}, 
  "LIST_IS_NORMALIZE": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': [True, False]}, 
  "LIST_NUM_THRESH": {'PARAM_SPACE': 'int', 'LOW': 500, 'HIGH': 1500, 'STEP': 500, 'LOG': False}, 
  "LIST_SPARSITY": {'PARAM_SPACE': 'categorical', 'CATEGORY_SET': ['linspace', 'logspace', 'unirandom', 'lograndom']}, 
  "FORBIDDEN_PARAM_SETS": {'loss_all_zero': {'LIST_PARAM_MULTIPLET_LOSS': 0.0, 'LIST_PARAM_LLR_LOSS': 0.0}}, 
  "LOG_PATH": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/", 
  "TB_DIRNAME": "TensorBoard_events", 
  "DB_DIRNAME": "Optuna_databases", 
  "CKPT_DIRNAME": "checkpoints", 
  "STDOUT_DIRNAME": "stdout_logs", 
  "SUBPROJECT_NAME": "public_offset0_5_optimmabs_tuning", 
  "SUBPROJECT_TO_RESUME": "_20230330_214519224/ckpt_step3500_target_ausat_confmx0.0011.pt", 
  "ROOT_TBLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/TensorBoard_events", 
  "ROOT_DBLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/Optuna_databases", 
  "ROOT_CKPTLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/checkpoints", 
  "ROOT_STDOUTLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/stdout_logs", 
  "PATH_RESUME": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/checkpoints/_20230330_214519224/ckpt_step3500_target_ausat_confmx0.0011.pt", 
  "LIST_NUM_BLOCKS": {'PARAM_SPACE': 'int', 'LOW': 1, 'HIGH': 2, 'STEP': 1, 'LOG': False}, 
  "LIST_NUM_HEADS": {'PARAM_SPACE': 'int', 'LOW': 2, 'HIGH': 4, 'STEP': 2, 'LOG': False}, 
  "LIST_DROPOUT": {'PARAM_SPACE': 'float', 'LOW': 0.0, 'HIGH': 0.5, 'STEP': 0.1, 'LOG': False}, 
  "LIST_FF_DIM": {'PARAM_SPACE': 'int', 'LOW': 32, 'HIGH': 64, 'STEP': 32, 'LOG': False}, 
  "LIST_MLP_UNITS": {'PARAM_SPACE': 'int', 'LOW': 32, 'HIGH': 64, 'STEP': 32, 'LOG': False}, 
  "NOW": "20230429_105648064", 
  "DIR_CKPTLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/checkpoints/Transformer_20230429_105648064", 
  "DIR_CONFIGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/checkpoints/Transformer_20230429_105648064/configs", 
  "DIR_TBLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/TensorBoard_events/Transformer_20230429_105648064", 
  "DIR_STDOUTLOGS": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/stdout_logs", 
  "STDOUTLOG_NAME": "/home/afe/Dropbox/GitHub/SPRT-TANDEM-PyTorch/logs/public_offset0_5_optimmabs_tuning/stdout_logs//Transformer_20230429_105648064.log", 
}
