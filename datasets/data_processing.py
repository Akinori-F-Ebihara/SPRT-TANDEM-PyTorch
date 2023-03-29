import pdb
import pickle
import torch
import lmdb
from termcolor import colored
from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import multivariate_normal
from utils.misc import extract_params_from_config
from typing import Callable, Tuple


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

    
class LMDBDataset(torch.utils.data.Dataset):
    '''
    A PyTorch dataset for reading data and labels from an LMDB database.

    Attributes:
    - lmdb_path (str): The path to the LMDB database.
    - device (torch.device): The device to move the data and labels to.
    - env (lmdb.Environment): The LMDB environment object.
    - data_size (int): The number of data points in the dataset.

    Methods:
    - __len__: Returns the number of data points in the dataset.
    - __getitem__: Returns the data and label for a given index.
    
    '''
    def __init__(self, lmdb_path: str, names: Tuple[str]):
        '''
        Required function for PyTorch Dataset class.
        Initializes a new LMDBDataset object.
        
        Args:
        -lmdb_path (str): The path to the LMDB database.
        -names (tuple of str): list of to-be-retrieved data. e.g., ('data', 'label')
        
        '''
        self.lmdb_path = lmdb_path
        self.names = names

        # Open the LMDB database
        self.env = lmdb.open(lmdb_path, readonly=True)

        # Open a read transaction
        # storing data like this enables faster readout, but may cause crash due to the heavy memory load.
        # suspect here if you encounter an error by using a new dataset
        self.data = []
        with self.env.begin() as txn:
            # Get the total number of data
            self.data_size = txn.stat()['entries'] // len(names)
            
            for i in range(self.data_size):
                item = {}
                for name in self.names:
                    item_bytes = txn.get(f'{i:08}_{name}'.encode('ascii'))
                    item[name] = pickle.loads(item_bytes)
                self.data.append(item)
                
    def __len__(self) -> int:
        '''
        Required function for PyTorch Dataset class.
        Returns the number of data points in the dataset.
        '''
        return self.data_size
    
    def __getitem__(self, index: int) -> tuple:
        '''
        Required function for PyTorch Dataset class.
        Returns the data and label for a given index.

        Args:
            index: The index of the data point.

        Returns:
            A tuple containing the data and label.
        '''''
        item = self.data[index]

        # If you avoid creating self.data above to save memory, use the following code here.
        # In which case do not forget to define self.data_size under __init__
        # with self.env.begin(buffers=True) as txn:
        #     item = {}
        #     for name in self.names:
        #         item_bytes = txn.get(f'{index:08}_{name}'.encode('ascii'))
        #         item[name] = pickle.loads(item_bytes)
        
        # Convert the data and label to PyTorch tensors
        _tensors = []
        for name in self.names:
            if 'label' in name:
                _tensors.append(torch.tensor(item[name]).to(torch.int64))
            else:
                _tensors.append(torch.from_numpy(item[name]).to(torch.float32))
        
        return tuple(_tensors)
    


def write_lmdb(lmdb_path: str, data: Tuple[np.ndarray], 
               names: Tuple[str], map_size: int = int(1e12)) -> None:
    '''
    Writes the data and labels to an LMDB database.

    Args:
    - lmdb_path (str): The path to the LMDB database.
    - data (tuple of numpy.ndarray): A tuple of numpy arrays containing the data to be saved.
    - name (tuple of str): A tuple containing the names of the arrays to be saved.
    - map_size (int, optional): The maximum size of the LMDB database in bytes. 
                                 Default is 1 terabyte.
    Returns:
    - None
    '''
    assert len(data) == len(names), 'Number of data and name list must match.'
    
    # Open a new LMDB database
    env = lmdb.open(lmdb_path, map_size=map_size)

    # Get the number of data points    
    data_number = data[0].shape[0]
    for data_array in data:
        assert data_array.shape[0] == data_number,\
            f'Total {data_array.shape[0]=} and {data_number=} does not match!'

    # Open a write transaction
    with env.begin(write=True) as txn:
        for i in tqdm(range(data_number)):
            # Write each data array to the database
            for j, data_array in enumerate(data):
                data_bytes = pickle.dumps(data_array[i])
                name = names[j]
                txn.put('{:08}_{}'.format(i, name).encode('ascii'), data_bytes)

    # Close the database
    env.close()


def lmdb_dataloaders(config: dict) -> dict:
    '''
    '''
    # check if necessary parameters are defined in the config file
    requirements = set(['TRAIN_DATA', 'VAL_DATA', 'TEST_DATA', 'DATA_NAMES', 
                        'BATCH_SIZE', 'IS_SHUFFLE', 'NUM_WORKERS'])
    conf = extract_params_from_config(requirements, config)
    
    train_dataset = LMDBDataset(conf.train_data, names=conf.data_names)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=conf.batch_size, pin_memory=True, num_workers=conf.num_workers, shuffle=conf.is_shuffle)
    
    val_dataset = LMDBDataset(conf.val_data, names=conf.data_names)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=conf.batch_size, pin_memory=True, num_workers=conf.num_workers, shuffle=False)

    test_dataset = LMDBDataset(conf.test_data, names=conf.data_names)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=conf.batch_size, pin_memory=True, num_workers=conf.num_workers, shuffle=False)

    data_loaders = {'train': train_dataloader,
                    'val': val_dataloader,
                    'test': test_dataloader}
    if conf.num_workers > 0:
        logger.info(colored(f'{conf.num_workers=}. ', 'yellow') + \
                    'non-zero value may cause unexpected memory-related error.')
    return data_loaders


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

