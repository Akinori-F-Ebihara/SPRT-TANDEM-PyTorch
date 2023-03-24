import numpy as np
from loguru import logger
import torch
from scipy.stats import multivariate_normal
from utils.misc import extract_params_from_config


class SequentialDensityRatioData:
    '''
    load data for sequential density ratio estimation (SDRE).
    '''
    def __init__(self, config):
        # check if necessary parameters are defined in the config file
        requirements = set(['X_PATH', 'Y_PATH', 'GT_LLR_PATH',
                            'NUM_TRAIN', 'NUM_VAL', 'NUM_TEST'])
        conf = extract_params_from_config(requirements, config)
        
        logger.info('Loading SDRE data. This may take a while...')
        x_pool = np.load(conf.x_path).astype('float32')
        y_pool = np.load(conf.y_path).astype('int32')
        gt_llr = np.load(conf.gt_llr_path).astype('float32')
        assert len(x_pool) == len(y_pool) == len(gt_llr) == \
               conf.num_train + conf.num_val + conf.num_test

        self.x_train = x_pool[:conf.num_train]
        self.x_val = x_pool[conf.num_train:conf.num_train + conf.num_val]
        self.x_test = x_pool[conf.num_train + conf.num_val:]
        self.y_train = y_pool[:conf.num_train]
        self.y_val = y_pool[conf.num_train:conf.num_train + conf.num_val]
        self.y_test = y_pool[conf.num_train + conf.num_val:]
        self.gt_llr_train = gt_llr[:conf.num_train]
        self.gt_llr_val = gt_llr[conf.num_train:conf.num_train + conf.num_val]
        self.gt_llr_test = gt_llr[conf.num_train + conf.num_val:]


def numpy_to_torch(*args):
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        *args: One or more inputs to be converted.

    Returns:
        A tuple containing the converted inputs.
    """
    converted_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = torch.from_numpy(arg)
        converted_args.append(arg)
    return tuple(converted_args)


def move_to_device(device, *args):
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        device: 
        *args: One or more inputs to be moved to the device.

    Returns:
        A tuple containing the inputs moved to the device..
    """
    moved_args = []
    for arg in args:
        arg = arg.to(device)        
        moved_args.append(arg)
    return tuple(moved_args)



def slice_data(data, device, phase, iter, batch_size):
    '''
    Get sequential density ratio estimation (SDRE) data depending on the phase.

    Args:
    - data (class): SDRE data.
    - phase (string): training, validation, or test.
    - iter (int): training step that is needed to determine data location
    - batch_size (int): batch size for the to-be-returned data.

    Returns:
    - x_batch (np.array): SDRE data of shape (batch_size, time_steps, feat_dim)
    - y_batch (np.array): label of shape (batch_size).
    - gt_llrs_batch (np.array): ground-truth log-likelihood ratio matrix of size 
                                (batch_size, time_steps, num_classes, num_clases)
    '''
    if 'train' in phase.lower():
        x = data.x_train
        y = data.y_train
        ground_truth_llr = data.gt_llr_train
    elif 'val' in phase.lower():
        x = data.x_val
        y = data.y_val
        ground_truth_llr = data.gt_llr_val
    elif 'test' in phase.lower():
        x = data.x_test
        y = data.y_test
        ground_truth_llr = data.gt_llr_test
    else:
        raise ValueError('Unknown phase!')
    
    assert x.shape[0] == y.shape[0], \
    'Shape of x and y do not match!'
    datalen = y.shape[0]

    start_idx = np.mod(iter * batch_size, datalen)
    end_idx = np.mod((iter +1) * batch_size, datalen)

    if end_idx > start_idx:
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        gt_llrs_batch = ground_truth_llr[start_idx:end_idx]
    else:
        x_batch = np.concatenate((x[start_idx:], x[:end_idx]), axis=0)
        y_batch = np.concatenate((y[start_idx:], y[:end_idx]), axis=0)
        gt_llrs_batch = np.concatenate((ground_truth_llr[start_idx:], ground_truth_llr[:end_idx]), axis=0)

    # convert to PyTorch Tensors
    x_batch, y_batch, gt_llrs_batch = numpy_to_torch(x_batch, y_batch, gt_llrs_batch)
    y_batch = y_batch.long()
    # move onto GPU
    x_batch, y_batch, gt_llrs_batch = move_to_device(device, x_batch, y_batch, gt_llrs_batch)
    
    return x_batch, y_batch, gt_llrs_batch


def initialize_multivariate_gaussian(conf):
    meanvecs = np.zeros((conf.num_classes, conf.feat_dim))
    covmat = np.eye(conf.feat_dim)
    pdfs = []
    for cls_i in range(conf.num_classes):
        meanvecs[cls_i, cls_i] = conf.density_offset
        pdfs.append(multivariate_normal(meanvecs[cls_i], covmat))
    return meanvecs, covmat, pdfs


def compute_log_likelihood_matrix(x, pdfs, conf):
        
    likelihood = np.zeros((conf.batch_size, conf.num_classes))
    for cls_i in range(conf.num_classes):
        likelihood[:, cls_i] = np.log(pdfs[cls_i].pdf(x))

    llrm = np.zeros((conf.batch_size, conf.num_classes, conf.num_classes))
    for cls_i in range(conf.num_classes):
        for cls_j in range(conf.num_classes):
            # diagonal is zero by definition
            llrm[:, cls_i, cls_j] = likelihood[:, cls_i] - likelihood[:, cls_j]

    return llrm


def generate_likelihood_ratio_matrix(conf):
    '''
    Generate sequential multivariate Gaussian likelihood ratio matrix for 
    sequential density ratio estimation (SDRE).
    
    Args:
    - config (dict): contains following parameters,
        FEAT_DIM
        DENSITY_OFFSET
        BATCH_SIZE
        NUM_ITER
        TIME_STAMP
        NUM_CLASSES
    
    Returns:
    - x_iter_pool: numpy array of size (BATCH_SIZE * NUM_ITER, TIME_STAMP, FEAT_DIM)
                    SDRE data.
    - y_iter_pool: numpy array of size (BATCH_SIZE * NUM_ITER). 
                    Represents ground-truth labels
    - llrm_iter_pool:  numpy array of size (BATCH_SIZE * NUM_ITER, TIME_STAMP, NUM_CLASSES, NUM_CLASSES).
                    Ground-truth LLR matrix.
    '''
    
    meanvecs, covmat, pdfs = initialize_multivariate_gaussian(conf)

    x_iter_pool = []
    y_iter_pool = []
    llrm_iter_pool = []
    for iter_i in range(conf.num_iter):
        logger.info(f'Starting {iter_i=} / {conf.num_iter - 1}')
        x_cls_pool = []
        y_cls_pool = []
        llrm_cls_pool = []
        for cls_i in range(conf.num_classes):
            y = cls_i * np.ones((conf.batch_size))

            x_time_pool = []
            llrm_time_pool = []
            for t_i in range(conf.time_steps):

                x = np.random.multivariate_normal(meanvecs[cls_i], covmat, conf.batch_size).astype('float32')
                llrm = compute_log_likelihood_matrix(x, pdfs, conf)

                x_time_pool.append(x)
                llrm_time_pool.append(llrm)

            x_cls = np.stack(x_time_pool, axis=1) # reshape into (BATCH_SIZE, TIME_STEPS, FEAT_DIM)
            llrm_cls = np.stack(llrm_time_pool, axis=1) # reshape into (BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
            assert x_cls.shape == (conf.batch_size, conf.time_steps, conf.feat_dim)
            assert y.shape == (conf.batch_size,) # size y: (BATCH_SIZE)
            assert llrm_cls.shape == (conf.batch_size, conf.time_steps, conf.num_classes, conf.num_classes)
            x_cls_pool.append(x_cls) 
            y_cls_pool.append(y) 
            llrm_cls_pool.append(llrm_cls) 

        x_iter = np.concatenate(x_cls_pool, axis=0) # reshape into (NUM_CLASSES * BATCH_SIZE, TIME_STEPS, FEAT_DIM)
        y_iter = np.concatenate(y_cls_pool, axis=0) # reshape into (NUM_CLASSES * BATCH_SIZE)
        llrm_iter = np.concatenate(llrm_cls_pool, axis=0) # reshape into (NUM_CLASSES * BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
        assert x_iter.shape == (conf.num_classes * conf.batch_size, conf.time_steps, conf.feat_dim) 
        assert y_iter.shape == (conf.num_classes * conf.batch_size,)
        assert llrm_iter.shape == (conf.num_classes * conf.batch_size, conf.time_steps, conf.num_classes, conf.num_classes)
        x_iter_pool.append(x_iter) 
        y_iter_pool.append(y_iter) 
        llrm_iter_pool.append(llrm_iter) 

    x_iter_pool = np.concatenate(x_iter_pool, axis=0) # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE, TIME_STEPS, FEAT_DIM)
    y_iter_pool = np.concatenate(y_iter_pool, axis=0) # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE)
    llrm_iter_pool = np.concatenate(llrm_iter_pool, axis=0) # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
    assert x_iter_pool.shape == (conf.num_iter * conf.num_classes * conf.batch_size, conf.time_steps, conf.feat_dim)
    assert y_iter_pool.shape == (conf.num_iter * conf.num_classes * conf.batch_size,)
    assert llrm_iter_pool.shape == (conf.num_iter * conf.num_classes * conf.batch_size, conf.time_steps, conf.num_classes, conf.num_classes)
    
    logger.success("successfully generated SDRE data!")

    # accumulate evidence
    llrm_iter_pool = np.cumsum(llrm_iter_pool, axis=1)
    
    if conf.is_shuffle:
        logger.info('shuffling the data...')
        total_data = conf.num_iter * conf.num_classes * conf.batch_size
        dice = np.random.permutation(total_data)

        x_iter_pool = x_iter_pool[dice]
        y_iter_pool = y_iter_pool[dice]
        llrm_iter_pool = llrm_iter_pool[dice]
    
    logger.success("done and dusted!")
    return x_iter_pool, y_iter_pool, llrm_iter_pool


def decode_feat():
    pass


def reshape_for_featext(x, y, feat_dim):
    """(batch, duration) to (batch * duration,)"""
    x_shape = x.shape
    batch_size = x_shape[0]
    duration = x_shape[1]

    # disentangle
    x = x.reshape(-1, feat_dim[0], feat_dim[1], feat_dim[2])

    y = y.repeat(duration)
    y = y.reshape(duration, batch_size)
    y = y.transpose(0, 1) 
    y = y.reshape(-1,)

    return x, y


def sequential_slice(x, y, order_sprt):
    """Slice, copy, and concat a batch to make a time-sliced, augumented batch
    Effective batch size will be batch * (duration - order_sprt)).
    e.g., nosaic MNIST and 2nd-order SPRT: 
        effective batch size is (20-2)=18 times larger than the original batch size.
    Args:
        x: A Tensor with shape (batch, duration, feature dimension).
        y: A Tensor with shape (batch).
        order_sprt: An int. The order of SPRT.
    Returns:
        x_slice: A Tensor with shape (batch*(duration-order_sprt), order_sprt+1, feat dim).
        y_slice: A Tensor with shape (batch*(duration-order_sprt),).
    Remark:
        y_slice may be a confusing name, because we copy and concatenate original y to obtain y_slice.
    """
    x, y = numpy_to_torch(x, y)

    duration = x.shape[1]

    if duration < order_sprt + 1:
        raise ValueError("order_sprt must be <= duration - 1. Now order_sprt={}, duration={} .".format(order_sprt, duration))

    for i in range(duration - order_sprt):
        if i == 0:
            x_slice = x[:, i:i+order_sprt+1, :]
            y_slice = y
        else:
            x_slice = torch.cat([x_slice, x[:, i:i+order_sprt+1, :]],0)
            y_slice = torch.cat([y_slice, y], 0)

    return x_slice, y_slice


def sequential_concat(x_slice, y_slice, duration):
    """Opposite operation of sequential_slice. 
    x_slice's shape will change 
    from (batch * (duration - order_sprt), order_sprt + 1, feat dim )
    to  (batch, (duration - order_sprt), order_sprt + 1, feat dim).
    y changes accordingly.
    Args:
        x_slice: A Tensor with shape (batch * (duration - order_sprt), order_sprt + 1, feat dim). This is the output of models.backbones_lstm.LSTMModel.__call__(inputs, training). 
        y_slice: A Tensor with shape (batch*(duration - order_sprt),).
        duration: An int. 20 for nosaic MNIST.
    Returns:
        x_cocnat: A Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, feat dim).
        y_concat: A Tensor with shape (batch).
    Remark:
        y_concat may be a confusing name, because we slice the input argument y_slice to get y_concat.
    """
    x_shape = x_slice.shape
    order_sprt = int(x_shape[1] - 1)
    batch_size = int(x_shape[0] / (duration - order_sprt))
    feat_dim = x_shape[-1]

    # Cancat time-sliced, augumented batch
    x_concat = x_slice.reshape(duration - order_sprt, batch_size, order_sprt + 1, feat_dim)
    x_concat = x_concat.permute(1, 0, 2, 3) # (batch, duration - order_sprt, order_sprt + 1, feat_dim)
    y_concat = y_slice[:batch_size]

    return x_concat, y_concat