import pdb
import pickle
import torch
import lmdb
from termcolor import colored
from tqdm import tqdm
from loguru import logger
import numpy as np
from scipy.stats import multivariate_normal
from utils.misc import extract_params_from_config, ConfigSubset
from typing import Callable, Tuple, List, Iterator, Any, Union


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
        if len(args) == 1:
            return arg
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


def move_data_to_device(
    data: Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], ...],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """
    Move the input data to the specified device based on the length of the input data tuple.

    Args:
        data: Tuple containing either (x_batch, y_batch, gt_llrs_batch) or (x_batch, y_batch).
        device: The device to move the data to.

    Returns:
        A tuple with data moved to the specified device.

    Raises:
        ValueError: If the length of the input data tuple is not 2 or 3.
    """
    if len(data) == 3:
        # if the dataset contains ground-truth log likelihood ratio
        x_batch, y_batch, gt_llrs_batch = data
        x_batch, y_batch, gt_llrs_batch = move_to_device(
            device, x_batch, y_batch, gt_llrs_batch
        )
        return x_batch, y_batch, gt_llrs_batch
    elif len(data) == 2:
        # data and label only: typical real-world data
        x_batch, y_batch = data
        x_batch, y_batch = move_to_device(device, x_batch, y_batch)
        return x_batch, y_batch
    else:
        raise ValueError(
            "data tuple length is expected either to be "
            f"3 (x, y, llr) or 2 (x, y) but got {len(data)=}!"
        )


class LMDBDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for reading data and labels from an LMDB database.

    Attributes:
    - lmdb_path (str): The path to the LMDB database.
    - env (lmdb.Environment): The LMDB environment object.
    - data_size (int): The number of data points in the dataset.

    Methods:
    - __len__: Returns the number of data points in the dataset.
    - __getitem__: Returns the data and label for a given index.

    """

    def __init__(self, lmdb_path: str, names: Tuple[str], is_load_onto_memory=True):
        """
        Required function for PyTorch Dataset class.
        Initializes a new LMDBDataset object.

        Args:
        -lmdb_path (str): The path to the LMDB database.
        -names (Tuple[str]): list of to-be-retrieved data. e.g., ('data', 'label')

        """
        super().__init__()
        self.lmdb_path = lmdb_path
        self.names = names
        self.is_load_onto_memory = is_load_onto_memory

        # Open a read transaction
        with lmdb.open(lmdb_path, readonly=True) as env:
            with env.begin() as txn:
                # Get the total number of data
                self.data_size = txn.stat()["entries"] // len(names)
                if is_load_onto_memory:
                    self.data = []
                    for i in range(self.data_size):
                        item = {}
                        for name in self.names:
                            item_bytes = txn.get(f"{i:08}_{name}".encode("ascii"))
                            item[name] = pickle.loads(item_bytes)
                        self.data.append(item)

    def __len__(self) -> int:
        """
        Required function for PyTorch Dataset class.
        Returns the number of data points in the dataset.
        """
        return self.data_size

    def __getitem__(self, index: int) -> tuple:
        """
        Required function for PyTorch Dataset class.
        Returns the data and label for a given index.

        Args:
            index: The index of the data point.

        Returns:
            A tuple containing the data and label.
        """

        if self.is_load_onto_memory:
            item = self.data[index]
        else:
            # Open the LMDB database for each worker
            with lmdb.open(self.lmdb_path, readonly=True) as env:
                with env.begin(buffers=True) as txn:
                    item = {}
                    for name in self.names:
                        item_bytes = txn.get(f"{index:08}_{name}".encode("ascii"))
                        item[name] = pickle.loads(item_bytes)
        # Convert the data and label to PyTorch tensors
        _tensors = []
        for name in self.names:
            if "label" in name:
                _tensors.append(torch.tensor(item[name]).to(torch.int64))
            else:
                _tensors.append(torch.tensor(item[name]).to(torch.float32))

        return tuple(_tensors)


class LMDBIterableDataset(torch.utils.data.IterableDataset):
    """
    A custom IterableDataset for reading data from an LMDB database.

    Args:
        lmdb_path (str): Path to the LMDB database.
        names (Tuple[str]): A tuple of names representing different data entries.
    """

    def __init__(self, lmdb_path: str, names: Tuple[str]):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.names = names

    def __iter__(self) -> Iterator[Tuple[Any]]:
        """
        Iterator that yields data samples from the LMDB database.

        Returns:
            Iterator[Tuple[Any]]: An iterator of tuples containing PyTorch tensors.
        """

        # Open the LMDB database for each worker
        env = lmdb.open(self.lmdb_path, readonly=True)

        # Define an iterator that yields data samples
        index = 0
        while True:
            with env.begin(buffers=True) as txn:
                item = {}
                for name in self.names:
                    item_bytes = txn.get(f"{index:08}_{name}".encode("ascii"))

                    # If the item_bytes is None, break the loop
                    if item_bytes is None:
                        break

                    item[name] = pickle.loads(item_bytes)
                else:
                    # Convert the data and label to PyTorch tensors
                    _tensors = []
                    for name in self.names:
                        if "label" in name:
                            _tensors.append(torch.tensor(item[name]).to(torch.int64))
                        else:
                            _tensors.append(torch.tensor(item[name]).to(torch.float32))

                    # Yield the data before incrementing the index
                    yield tuple(_tensors)

                    # Increment index and continue the loop
                    index += 1
                    continue

                # If the loop is broken (no more data), break the outer loop
                break

        # Close the LMDB database for each worker
        env.close()


def write_lmdb(
    lmdb_path: str,
    data: Tuple[np.ndarray],
    names: Tuple[str],
    map_size: int = int(1e12),
) -> None:
    """
    Writes the data and labels to an LMDB database.

    Args:
    - lmdb_path (str): The path to the LMDB database.
    - data (tuple of numpy.ndarray): A tuple of numpy arrays containing the data to be saved.
    - name (tuple of str): A tuple containing the names of the arrays to be saved.
    - map_size (int, optional): The maximum size of the LMDB database in bytes.
                                 Default is 1 terabyte.
    Returns:
    - None
    """
    assert len(data) == len(names), "Number of data and name list must match."

    # Open a new LMDB database
    env = lmdb.open(lmdb_path, map_size=map_size)

    # Get the number of data points
    data_number = data[0].shape[0]
    for data_array in data:
        assert (
            data_array.shape[0] == data_number
        ), f"Total {data_array.shape[0]=} and {data_number=} does not match!"

    # Open a write transaction
    with env.begin(write=True) as txn:
        for i in tqdm(range(data_number)):
            # Write each data array to the database
            for j, data_array in enumerate(data):
                data_bytes = pickle.dumps(data_array[i])
                name = names[j]
                txn.put("{:08}_{}".format(i, name).encode("ascii"), data_bytes)

    # Close the database
    env.close()


def lmdb_dataloaders(config: dict, load_test=False) -> dict:
    """ """
    # check if necessary parameters are defined in the config file
    requirements = set(
        [
            "TRAIN_DATA",
            "VAL_DATA",
            "TEST_DATA",
            "DATA_NAMES",
            "BATCH_SIZE",
            "IS_SHUFFLE",
            "NUM_WORKERS",
            "IS_LOAD_ONTO_MEMORY",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    logger.info("loading data... ")
    if conf.is_load_onto_memory:
        logger.info(
            f"If this process takes long, "
            "consider setting is_load_onto_memory=False or use LMDBIterableDataset."
        )

    train_dataset = LMDBDataset(
        conf.train_data,
        names=conf.data_names,
        is_load_onto_memory=conf.is_load_onto_memory,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        pin_memory=True,
        num_workers=conf.num_workers,
        shuffle=conf.is_shuffle,
    )

    val_dataset = LMDBDataset(
        conf.val_data,
        names=conf.data_names,
        is_load_onto_memory=conf.is_load_onto_memory,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=conf.batch_size,
        pin_memory=True,
        num_workers=conf.num_workers,
        shuffle=False,
    )

    if load_test:
        test_dataset = LMDBDataset(
            conf.test_data,
            names=conf.data_names,
            is_load_onto_memory=conf.is_load_onto_memory,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=conf.batch_size,
            pin_memory=True,
            num_workers=conf.num_workers,
            shuffle=False,
        )

        data_loaders = {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
        }
    else:
        data_loaders = {"train": train_dataloader, "val": val_dataloader}

    return data_loaders


def initialize_multivariate_gaussian(
    conf: ConfigSubset,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Initialize a multivariate Gaussian distribution for each class.

    Args:
    - conf (ConfigSubset): an instance of the ConfigSubset class containing the following keys:
        - num_classes (int): the number of classes.
        - feat_dim (int): the feature dimension.
        - density_offset (float): the density offset used to initialize the mean vectors.

    Returns:
    - meanvecs (ndarray): an array of shape (num_classes, feat_dim) containing the mean vectors for each class.
    - covmat (ndarray): the covariance matrix, which is a diagonal matrix of shape (feat_dim, feat_dim).
    - pdfs (list): a list of multivariate normal distributions, one for each class.
    """

    meanvecs = np.zeros((conf.num_classes, conf.feat_dim))
    covmat = np.eye(conf.feat_dim)
    pdfs = []
    for cls_i in range(conf.num_classes):
        meanvecs[cls_i, cls_i] = conf.density_offset
        pdfs.append(multivariate_normal(meanvecs[cls_i], covmat))
    return meanvecs, covmat, pdfs


def compute_log_likelihood_ratio_matrix(
    x: np.ndarray, pdfs: List, conf: ConfigSubset
) -> np.ndarray:
    """
    Compute the log-likelihood ratio matrix for each sample in x.

    Args:
    - x (ndarray): an array of shape (batch_size, feat_dim) containing the feature vectors.
    - pdfs (list): a list of multivariate normal distributions, one for each class.
    - conf (ConfigSubset): an instance of the ConfigSubset class containing the following keys:
        - num_classes (int): the number of classes.
        - batch_size (int): the number of samples.
        - feat_dim (int): the feature dimension.

    Returns:
    - llrm (ndarray): an array of shape (batch_size, num_classes, num_classes) containing the log-likelihood ratio
      matrix for each sample in x.
    """

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
    """
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
    """

    meanvecs, covmat, pdfs = initialize_multivariate_gaussian(conf)

    x_iter_pool = []
    y_iter_pool = []
    llrm_iter_pool = []
    for iter_i in range(conf.num_iter):
        logger.info(f"Starting {iter_i=} / {conf.num_iter - 1}")
        x_cls_pool = []
        y_cls_pool = []
        llrm_cls_pool = []
        for cls_i in range(conf.num_classes):
            y = cls_i * np.ones((conf.batch_size))

            x_time_pool = []
            llrm_time_pool = []
            for t_i in range(conf.time_steps):
                x = np.random.multivariate_normal(
                    meanvecs[cls_i], covmat, conf.batch_size
                ).astype("float32")
                llrm = compute_log_likelihood_ratio_matrix(x, pdfs, conf)

                x_time_pool.append(x)
                llrm_time_pool.append(llrm)

            # reshape into (BATCH_SIZE, TIME_STEPS, FEAT_DIM)
            x_cls = np.stack(x_time_pool, axis=1)
            # reshape into (BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
            llrm_cls = np.stack(llrm_time_pool, axis=1)
            assert x_cls.shape == (conf.batch_size, conf.time_steps, conf.feat_dim)
            assert y.shape == (conf.batch_size,)  # size y: (BATCH_SIZE)
            assert llrm_cls.shape == (
                conf.batch_size,
                conf.time_steps,
                conf.num_classes,
                conf.num_classes,
            )
            x_cls_pool.append(x_cls)
            y_cls_pool.append(y)
            llrm_cls_pool.append(llrm_cls)

        # reshape into (NUM_CLASSES * BATCH_SIZE, TIME_STEPS, FEAT_DIM)
        x_iter = np.concatenate(x_cls_pool, axis=0)
        # reshape into (NUM_CLASSES * BATCH_SIZE)
        y_iter = np.concatenate(y_cls_pool, axis=0)
        # reshape into (NUM_CLASSES * BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
        llrm_iter = np.concatenate(llrm_cls_pool, axis=0)
        assert x_iter.shape == (
            conf.num_classes * conf.batch_size,
            conf.time_steps,
            conf.feat_dim,
        )
        assert y_iter.shape == (conf.num_classes * conf.batch_size,)
        assert llrm_iter.shape == (
            conf.num_classes * conf.batch_size,
            conf.time_steps,
            conf.num_classes,
            conf.num_classes,
        )
        x_iter_pool.append(x_iter)
        y_iter_pool.append(y_iter)
        llrm_iter_pool.append(llrm_iter)

    # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE, TIME_STEPS, FEAT_DIM)
    x_iter_pool = np.concatenate(x_iter_pool, axis=0)
    # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE)
    y_iter_pool = np.concatenate(y_iter_pool, axis=0)
    # reshape into (NUM_ITER * NUM_CLASSES * BATCH_SIZE, TIME_STEPS, NUM_CLASSES, NUM_CLASSES)
    llrm_iter_pool = np.concatenate(llrm_iter_pool, axis=0)
    assert x_iter_pool.shape == (
        conf.num_iter * conf.num_classes * conf.batch_size,
        conf.time_steps,
        conf.feat_dim,
    )
    assert y_iter_pool.shape == (conf.num_iter * conf.num_classes * conf.batch_size,)
    assert llrm_iter_pool.shape == (
        conf.num_iter * conf.num_classes * conf.batch_size,
        conf.time_steps,
        conf.num_classes,
        conf.num_classes,
    )

    logger.success("successfully generated SDRE data!")

    # accumulate evidence
    llrm_iter_pool = np.cumsum(llrm_iter_pool, axis=1)

    if conf.is_shuffle:
        logger.info("shuffling the data...")
        total_data = conf.num_iter * conf.num_classes * conf.batch_size
        dice = np.random.permutation(total_data)

        x_iter_pool = x_iter_pool[dice]
        y_iter_pool = y_iter_pool[dice]
        llrm_iter_pool = llrm_iter_pool[dice]

    logger.success("done and dusted!")
    return x_iter_pool, y_iter_pool, llrm_iter_pool


def sequential_slice(x, y, order_sprt):
    """Slice, copy, and concat a batch to make a time-sliced, augumented batch
    Effective batch size will be batch * (time_steps - order_sprt)).
    e.g., nosaic MNIST and 2nd-order SPRT:
        effective batch size is (20-2)=18 times larger than the original batch size.
    Args:
        x: A Tensor with shape (batch, time_steps, feature dimension).
        y: A Tensor with shape (batch).
        order_sprt: An int. The order of SPRT.
    Returns:
        x_slice: A Tensor with shape (batch*(time_steps-order_sprt), order_sprt+1, feat dim).
        y_slice: A Tensor with shape (batch*(time_steps-order_sprt),).
    Remark:
        y_slice may be a confusing name, because we copy and concatenate original y to obtain y_slice.
    """
    x, y = numpy_to_torch(x, y)

    time_steps = x.shape[1]

    if time_steps < order_sprt + 1:
        raise ValueError(
            "order_sprt must be <= time_steps - 1. Now order_sprt={}, time_steps={} .".format(
                order_sprt, time_steps
            )
        )

    for i in range(time_steps - order_sprt):
        if i == 0:
            x_slice = x[:, i : i + order_sprt + 1, :]
            y_slice = y
        else:
            x_slice = torch.cat([x_slice, x[:, i : i + order_sprt + 1, :]], 0)
            y_slice = torch.cat([y_slice, y], 0)

    return x_slice, y_slice


def sequential_slice(x, y, order_sprt):
    """Slice, copy, and concat a batch to make a time-sliced, augumented batch
    Effective batch size will be batch * (time_steps - order_sprt)).
    e.g., nosaic MNIST and 2nd-order SPRT:
        effective batch size is (20-2)=18 times larger than the original batch size.
    Args:
        x: A Tensor with shape (batch, time_steps, feature dimension).
        y: A Tensor with shape (batch).
        order_sprt: An int. The order of SPRT.
    Returns:
        x_slice: A Tensor with shape (batch*(time_steps-order_sprt), order_sprt+1, feat dim).
        y_slice: A Tensor with shape (batch*(time_steps-order_sprt),).
    Remark:
        y_slice may be a confusing name, because we copy and concatenate original y to obtain y_slice.
    """
    x, y = numpy_to_torch(x, y)

    time_steps = x.shape[1]

    if time_steps < order_sprt + 1:
        raise ValueError(
            "order_sprt must be <= time_steps - 1. Now order_sprt={}, time_steps={} .".format(
                order_sprt, time_steps
            )
        )

    for i in range(time_steps - order_sprt):
        if i == 0:
            x_slice = x[:, i : i + order_sprt + 1, :]
            y_slice = y
        else:
            x_slice = torch.cat([x_slice, x[:, i : i + order_sprt + 1, :]], 0)
            y_slice = torch.cat([y_slice, y], 0)

    return x_slice, y_slice


def sequential_slice_data(inputs, order_sprt):
    inputs = numpy_to_torch(inputs)

    time_steps = inputs.shape[1]

    if time_steps < order_sprt + 1:
        raise ValueError(
            "order_sprt must be <= time_steps - 1."
            f" Now {order_sprt=}, {time_steps=}."
        )

    for i in range(time_steps - order_sprt):
        if i == 0:
            x_slice = inputs[:, i : i + order_sprt + 1, :]
        else:
            x_slice = torch.cat([x_slice, inputs[:, i : i + order_sprt + 1, :]], 0)

    return x_slice


def sequential_slice_labels(labels, time_steps, order_sprt):
    labels = numpy_to_torch(labels)

    if time_steps < order_sprt + 1:
        raise ValueError(
            "order_sprt must be <= time_steps - 1."
            f" Now {order_sprt=}, {time_steps=}."
        )

    for i in range(time_steps - order_sprt):
        if i == 0:
            y_slice = labels
        else:
            y_slice = torch.cat([y_slice, labels], 0)

    return y_slice


def sequential_concat_logits(
    logits_slice: torch.Tensor, time_steps: int
) -> torch.Tensor:
    """
    Opposite operation of sequential_slice.
    logits_slice's shape will change
    from (batch * (time_steps - order_sprt), order_sprt + 1, feat dim )
    to  (batch, (time_steps - order_sprt), order_sprt + 1, feat dim).
    Args:
        logits_slice: A Tensor with shape (batch * (time_steps - order_sprt), order_sprt + 1, feat dim). This is the output of models.backbones_lstm.LSTMModel.__call__(inputs, training).
        time_steps: An int. 20 for nosaic MNIST.
        y_slice: A Tensor with shape (batch*(time_steps - order_sprt),). Default: None.
    Returns:
        A Tensor with shape (batch, (time_steps - order_sprt), order_sprt + 1, feat dim).
        If y_slice is not None, also returns a Tensor with shape (batch).
    """

    order_sprt = int(logits_slice.shape[1] - 1)
    batch_size = int(logits_slice.shape[0] / (time_steps - order_sprt))
    feat_dim = logits_slice.shape[-1]

    batch_size = logits_slice.shape[0] // (time_steps - logits_slice.shape[1] + 1)
    feat_dim = logits_slice.shape[-1]

    x_concat = logits_slice.reshape(
        time_steps - order_sprt, batch_size, order_sprt + 1, feat_dim
    )
    # (batch, duration - order_sprt, order_sprt + 1, feat_dim)
    x_concat = x_concat.permute(1, 0, 2, 3)

    return x_concat


def sequential_concat_labels(
    labels_slice: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """ """
    y_concat = labels_slice[:batch_size]
    return y_concat


def reshape_for_featext(x, y, feat_dim):
    """(batch, time_steps) to (batch * time_steps,)"""
    x_shape = x.shape
    batch_size = x_shape[0]
    time_steps = x_shape[1]

    # disentangle
    x = x.reshape(-1, feat_dim[0], feat_dim[1], feat_dim[2])

    y = y.repeat(time_steps)
    y = y.reshape(time_steps, batch_size)
    y = y.transpose(0, 1)
    y = y.reshape(
        -1,
    )

    return x, y
