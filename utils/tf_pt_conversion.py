import numpy as np


def assert_tf_pt_results(tf_tensor, pt_tensor, rtol=1e-05, atol=1e-08):
    '''
    Asserts that the TensorFlow and PyTorch tensors or arrays are equal.

    Args:
    - tf_tensor (tensor-like): A TensorFlow tensor or numpy array.
    - pt_tensor (tensor-like): A PyTorch tensor or numpy array.

    Returns:
    - None

    Raises:
    - ValueError: If the TensorFlow and PyTorch tensors or arrays are not equal.

    Example Usage:

    tf_mat = tf.constant([[1, 2], [3, 4]])
    pt_mat = torch.tensor([[1, 2], [3, 4]])

    assert_tf_pt_results(tf_mat, pt_mat)  # PT and TF results match!
    '''
    def assert_results(tf_mat, pt_mat):

        if np.allclose(tf_mat, pt_mat, rtol, atol):
            print(f'PT and TF results match!\n{pt_mat}')
        else:
            raise ValueError('PT and TF results are different!!!')

    if isinstance(tf_tensor, tuple) and isinstance(pt_tensor, tuple):
        if len(tf_tensor) != len(pt_tensor):
            raise ValueError("Two tuples have different lengths.")
        for tf_mat, pt_mat in zip(tf_tensor, pt_tensor):
            assert_tf_pt_results(tf_mat, pt_mat)
    else:
        assert_results(tf_tensor, pt_tensor)
