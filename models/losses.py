import pdb
import numpy as np
import torch
import torch.nn.functional as F
from datasets.data_processing import sequential_slice, sequential_concat
from utils.performance_metrics import calc_llrs, calc_oblivious_llrs, thresh_sanity_check,\
                                        seqconfmx_to_macro_ave_sns, thresh_truncated_MSPRT, threshold_generator, \
                                        multiply_diff_mht
                                        
from utils.misc import extract_params_from_config, ErrorHandler


def multiplet_crossentropy(logits_slice, labels_slice):
    """Multiplet loss for density estimation of time-series data.
    Args:
        model: A model.backbones_lstm.LSTMModel object. 
        logits_slice: An logit Tensor with shape 
            ((effective) batch size, order of SPRT + 1, num classes). 
            This is the output from LSTMModel.call(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,)
    Returns:
        xent: A scalar Tensor. Sum of multiplet losses.
    """
    # Calc multiplet losses     
    logits = logits_slice.permute(1, 0, 2)
    logits = logits.reshape(-1, logits.shape[2])
    labels = labels_slice.repeat(logits_slice.shape[1])
    xent = F.cross_entropy(logits, labels)

        #A scalar averaged in a batch and a sliding window

    return xent


def compute_llrs_and_estimation_loss(logits_concat, labels_concat, oblivious, version):
    """LLLR for early multi-classification of time series.
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (time_steps - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A label Tensor with shape (batch size,). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        oblivious: A bool, whether to use TANDEMwO or not (= TANDEM).
        version: "A", "B", "C", "D", or "E".
    Return:
        llr_loss: A scalar Tensor that represents the log-likelihoood ratio loss.
    Remark:
        - version A: original (use all LLRs)
        - version B: simple LLLR (extract the positive class' raw)
        - version C: logistic LLLR (logistic loss instead of sigmoid)
        - version D: simple logistic LLLR
        - margin is generated uniformly-randomly.
    """
    
    shapes = logits_concat.shape
    order_sprt = shapes[2] - 1
    time_steps = shapes[1] + order_sprt
    num_classes = shapes[3]
    
    labels_oh = F.one_hot(labels_concat, num_classes).to(torch.float32)
    labels_oh = labels_oh.reshape(-1, 1, num_classes, 1)
        # (batch, 1, num cls, 1)

    if oblivious: # Miyagawa and Ebihara, ICML 2021
        llrs = calc_oblivious_llrs(logits_concat)         
    else:
        llrs = calc_llrs(logits_concat) 
            # (batch, time_steps, num cls, num cls)
    
    llrs_ugnt = llrs # llrs.clone().detach()?
    llrs_orig = llrs_ugnt.clone()

    if version == "A":
        lllr = torch.abs(labels_oh - torch.sigmoid(llrs))
            # (batch, time_steps, num cls, num cls)
        lllr = 0.5 * (num_classes / (num_classes - 1.)) * torch.mean(lllr)

    elif version == "B":
        llrs = llrs * labels_oh
            # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
            # (batch, time_steps, num cls)
        lllr = torch.abs(1. - torch.sigmoid(llrs))
        lllr = (num_classes/ (num_classes - 1)) * torch.mean(lllr)

    elif version == "C":
        labels_oh = labels_oh.repeat(1, time_steps, 1, num_classes)
            # (batch, time_steps, num cls, num cls)
        lllr = F.binary_cross_entropy_with_logits(llrs, labels_oh)
            # scalar value (averaged)
        lllr = 0.5 * (num_classes/ (num_classes - 1)) * lllr        

    elif version == "D":
        llrs = llrs * labels_oh
            # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
            # (batch, time_steps, num cls)
        llrs = llrs.reshape(-1, num_classes)
        tmp = torch.ones_like(llrs).to(torch.float32)
        lllr = F.binary_cross_entropy_with_logits(llrs, tmp)
            # scalar value (averaged))
        lllr = (num_classes/ (num_classes - 1)) * lllr

    elif version == "E":
        llrs = llrs * labels_oh
            # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
            # (batch, time_steps, num cls)
        llrs = llrs.reshape(-1, num_classes)
            # (batch * time_steps, num cls)
        minllr = torch.min(llrs, dim=1, keepdims=True)[0]
        llrs = llrs - minllr
            # (batch * time_steps, num cls)
        lllr = torch.sum(torch.exp(-llrs), dim=1)
            # (batch * time_steps, )
        lllr = torch.mean(- minllr + torch.log(lllr + 1e-12))
            # scalar
    elif version == "Eplus":
        llrs = llrs * labels_oh
            # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
            # (batch, time_steps, num cls)
        llrs = llrs.reshape(-1, num_classes)
                # (batch * time_steps, num cls)
        minllr = torch.min(llrs, dim=1, keepdim=True)[0].detach()
        llrs = llrs - minllr
            # (batch * time_steps, num cls)
        lllr = torch.sum(torch.exp(-llrs), dim=1)
            # (batch * time_steps, )
        lllr = torch.mean(- minllr + torch.log(lllr + 1e-12))
            # scalar
    else:
        raise ValueError(
            "version={} must be either of 'A', 'B', 'C', 'D', 'E' or 'Eplus'.".\
                format(version))
    
    return lllr, llrs_orig # scalar


###############################################################################
#                           AUC-SATC Functions                                 #
###############################################################################
# Sub-functions for version_X (X=A,B,C,D,E) functions
######################################################
def extract_shapes(llrs, thresh):
    llrs_shape = llrs.shape
    batch_size = llrs_shape[0]
    time_steps = llrs_shape[1]
    num_classes = llrs_shape[2]
    num_thresh = thresh.shape[0]

    return batch_size, time_steps, num_classes, num_thresh

def normalize_LLR_thresh_scale(llrs, thresh):
    # Change LLRs and threshold scales
    # Otherwise margins_up and margins_low may contain lots of zeros.
    _max = torch.max(torch.abs(llrs)) + 1e-12
    llrs_norm = llrs / _max # Comment20221201: Should be stop_grad(llrs/_max)?
    thresh_norm = thresh / _max # Oops!! Should be stop_grad(thresh/_max)!!
    #****************************** Maybe modification needed *******************************************#

    return llrs_norm, thresh_norm

def restore_lost_significance(llrs):
    # Restore lost significance with epsilon
    tri = torch.ones_like(llrs)
    triu = torch.triu(tri) # Upper triangular part.
    tril = torch.tril(tri) # Lower triangular part.
    llrs_restore = llrs - 1e-10 * (triu - tril) 
        # (batch, time_steps, num cls, num cls)
        # To avoid double hit due to the values exactly equal to 0
        # in scores or when doing truncation, LLRs of the last frame.

    return llrs_restore

def calc_mask_pos_without_thresh(labels_concat, num_classes, time_steps):
    # For version E
    labels_oh = F.one_hot(labels_concat, num_classes).to(torch.float32)
        # (batch, num cls)
    mask = labels_oh.reshape(-1, 1, num_classes, 1)
    mask = mask.repeat(1, time_steps, 1, num_classes)
        # (batch, time_steps, num cls, num cls)

    return mask
    
def calc_scores_and_hittimes(llrs, thresh, num_thresh, time_steps, batch_size, num_classes):
    # Calc scores (= LLR - thresh) and hittimes
    def _calc_scores(llrs, thresh):
        # Calc scores and scores_full    
        llr_mtx = torch.unsqueeze(llrs, dim=0)
            # (1, batch, time_steps, num cls, num cls)
            # to admit the num-thresh axis.
        thresh_mtx = torch.unsqueeze(thresh, dim=1)
            # (num thresh, 1, time_steps, num cls, num cls)
            # to admit the batch axis.
        scores_full = llr_mtx - thresh_mtx
            # (num thresh, batch, time_steps, num cls, num cls)
        scores = torch.min(scores_full, dim=-1)[0]
            # (num thresh, batch, time_steps, num cls)
            # Values are non-positive.

        return scores, scores_full

    def _calc_truncated_predictions(scores, llrs, thresh, 
        num_thresh, batch_size, num_classes):
        # Calc all predictions and waits
        preds_all = torch.sign(scores) + 1
            # 0:wait, 1:hit (one-hot vector)
            # (num thresh, batch, time_steps, num cls)            
        preds_last = torch.sign(
            torch.min(llrs[:, -1:, :, :], dim=-1)[0]
            ) + 1
            # 0: wait, 1: hit (one-hot vector)
            # (batch, 1, num cls) 
        preds_last = torch.unsqueeze(preds_last, dim=0) 
            # (1, batch, 1, num cls)
        preds_last = preds_last.repeat(num_thresh, 1, 1, 1)
            # (num thresh, batch, 1, num cls)
        preds_all_trunc = torch.cat([preds_all[:, :, :-1, :], preds_last], dim=2)

            # (num thresh, batch, time_steps - 1, num cls)
            # + (num thresh, batch, 1, num cls)
            # = (num thresh, batch, time_steps, num cls) 
            # Now, preds_all_trunc[i, j, t, :] for fixed i and j is
            # a one-hot vector for t = time_steps - 1 
            # and
            # filled with 0 or a one-hot vector for t != time_steps - 1.

        return preds_all_trunc

    def _calc_mask(time_steps, num_thresh, batch_size, num_classes):
        # For hittimes
        mask = torch.tensor([i+1 for i in range(time_steps)][::-1], dtype=torch.float32)
        mask = mask.repeat(num_thresh * batch_size * num_classes)
        mask = mask.reshape(num_thresh, batch_size, num_classes, time_steps)
        mask = mask.permute(0, 1, 3, 2)
            # (num thresh, batch, time_steps , num cls)

        return mask

    def _calc_hittimes(preds_all_trunc, mask):
        masked = preds_all_trunc * mask 
            # (num thresh, batch, time_steps, num cls)
        hitidx = torch.max(masked, dim=2)[0]
            # (num thresh, batch, num cls)
        hittimes = time_steps - torch.max(hitidx, dim=2)[0] + 1
            # (num thresh, batch)

        assert torch.max(hittimes)  <= time_steps, f'hittimes must be \in {1, time_steps}, while got {torch.max(hittimes)=}!'
        return hitidx, hittimes

    _device = llrs.device

    # Calc scores and scores_full
    scores, scores_full = _calc_scores(llrs, thresh)
        # Score means LLR - thresh.
        # scores:      (num thresh, batch, time_steps, num cls)
        # scores_full: (num thresh, batch, time_steps, num cls, num cls)

    # Calc all predictions with truncation
    preds_all_trunc = _calc_truncated_predictions(scores, llrs, 
        thresh, num_thresh, batch_size, num_classes)
        # (num thresh, batch, time_steps, num cls) 
        # Now, preds_all_trunc[i, j, t, :] for fixed i and j is
        # a one-hot vector for t = time_steps - 1 
        # and
        # filled with 0 or a one-hot vector for t != time_steps - 1.
    
    # Calc mean hitting times
    mask = _calc_mask(time_steps, num_thresh, batch_size, num_classes).to(_device)
        # (num thresh, batch, time_steps , num cls)
    hitidx, hittimes = _calc_hittimes(preds_all_trunc, mask)
        # (num thresh, batch)

    return scores_full, hitidx, hittimes


def reduce_min_column(llrs, scores_pos, num_classes):
     # Calc min_{k != y_i} LLR[:, :, :, k]
    _device = llrs.device

    _max = torch.max(llrs)
        # This is <= 1, because of llr /= llrmax above.
        # But re-calc here to avoid problems
        # due to potential changes in the future.
    with ErrorHandler():
        _tmp = torch.diag(torch.ones(num_classes).to(_device) * _max)
        _tmp = _tmp.reshape(1, 1, 1, num_classes, num_classes)        
            # (1, 1, 1, num cls, num cls)
    scores_posmin = scores_pos + _tmp
        # (num thresh, batch, time_steps, num cls, num cls)
    scores_posmin = torch.min(scores_posmin, dim=4)[0]
        # (num thresh, batch, time_steps, num cls)
        # Collapses columns

    return scores_posmin


def calc_mask_pos(labels_concat, num_classes, time_steps, num_thresh):
    # For extracting positive-class's LLRs
    labels_oh = F.one_hot(labels_concat, num_classes).float()
    mask_pos = labels_oh.reshape(1, -1, 1, num_classes, 1)
    mask_pos = mask_pos.repeat(num_thresh, 1, time_steps, 1, num_classes)
        # (num thresh, batch, time_steps, num cls, num cls)

    return mask_pos


def calc_mask_ht(hittimes, time_steps, num_thresh, batch_size, num_classes):
    # For extracting LLRs at hitting times

    mask_ht = hittimes - 1
        # (num thresh, batch)
        # value \in {0, 1, ..., time_steps - 1}
    mask_ht = F.one_hot(mask_ht.to(torch.int64), num_classes=time_steps).float()
        # (num thresh, batch, time_steps)
    mask_ht = mask_ht.reshape(num_thresh, batch_size, time_steps, 1, 1)
    mask_ht = mask_ht.repeat(1, 1, 1, num_classes, num_classes)
        # (num thresh, batch, time_steps, num cls, num cls)

    return mask_ht


# Functions for approximated ACCs (WIP)
############################################
def version_A(scores_full, mask_pos, llrs, hittimes, beta, is_absnorm=True):
    """
    Args:
        scores_full: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        mask_pos: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        labels_concat: A binary label Tensor 
            with shape (batch size,) 
            with value = 0 or 1. This is the output from
            sequential_concat(logit_slice, labels_slice).
        llrs: A Tensor with shape 
            (batch, time_steps, num classes, num classes).
        hittimes: (num thresh, batch)
        thresh: A Tensor with shape 
            (num_thresh, time_steps, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
        beta: A positive float for the beta-sigmoid function.
        version: A string, either of "A", "B", "C", "D", or "E".
    """
    assert scores_full.shape[-1] == scores_full.shape[-2]
    num_classes = scores_full.shape[-1]
    time_steps = scores_full.shape[2]
    num_thresh = scores_full.shape[0]
    batch_size = scores_full.shape[1]
    
    mask_ht = calc_mask_ht(hittimes, time_steps, num_thresh, batch_size, num_classes)
        # (num thresh, batch, time_steps, num cls, num cls)
    mask = mask_pos * mask_ht
    scores_posht = scores_full * mask
        # (num thresh, batch, time_steps, num cls, num cls)
        # Note that all the frames, except for the hitting
        # time, of each sequence are filled with 0.
        # Note that all the rows, except for the y_{i}-th row, 
        # of each LLR matrix are filled with 0.
    # Calc min_{k != y_i} LLR[:, :, :, k]
    #######################################
    scores_poshtmin = reduce_min_column(llrs, scores_posht, num_classes)
        # (num thresh, batch, time_steps, num cls)

    acc_eta = torch.sum(scores_poshtmin, dim=[2, 3])
        # (num thresh, time_steps)
        # Collapses rows of LLR matrix and frames.
        # Note that all the frames, except for the hitting
        # time, of each sequence are filled with 0.
        # Note that all the rows, except for the y_{i}-th one, 
        # of each LLR matrix are filled with 0.   
    
    if is_absnorm:
        acc_eta = (2 * torch.sigmoid(beta * acc_eta) - 1)
        acc_eta = torch.mean(acc_eta, dim=1) / torch.mean(torch.abs(acc_eta), dim=1) 
    else: # TM original
        acc_eta = torch.mean(
            torch.sigmoid(beta * acc_eta),
            dim=1)
        # (num thresh,)
        # Mean w.r.t. i (datapt).             

    return acc_eta


def version_B(scores_full, mask_pos, llrs, beta, is_absnorm=True):
    """
    Args:
        scores_full: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        mask_pos: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        labels_concat: A binary label Tensor 
            with shape (batch size,) 
            with value = 0 or 1. This is the output from
            sequential_concat(logit_slice, labels_slice).
        llrs: A Tensor with shape 
            (batch, time_steps, num classes, num classes).
        thresh: A Tensor with shape 
            (num_thresh, time_steps, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
        beta: A positive float for the beta-sigmoid function.
        version: A string, either of "A", "B", "C", "D", or "E".
    """
    assert scores_full.shape[-1] == scores_full.shape[-2]
    num_classes = scores_full.shape[-1]
    
    scores_pos = mask_pos * scores_full
        # (num thresh, batch, time_steps, num cls, num cls)
        # Only positive-class's LLRs are non-zero, i.e.,
        # all the rows, except for the y_{i}-th row, 
        # are filled with 0.
    # Calc min_{k != y_i} LLR[:, :, :, k]
    #######################################
    scores_posmin = reduce_min_column(llrs, scores_pos, num_classes)
        # (num thresh, batch, time_steps, num cls)
        
    acc_eta = torch.sum(scores_posmin, dim=3)
        # (num thresh, batch, time_steps)
        # Collapses rows of LLR matrix.
        # Note that all the rows except for the y_{i}-th row 
        # are filled with 0.
    if is_absnorm:
        acc_eta = (2 * torch.sigmoid(beta * acc_eta) - 1)
        acc_eta = torch.mean(acc_eta, dim=[1, 2]) / torch.mean(torch.abs(acc_eta), dim=[1, 2]) 
    else:
        acc_eta = torch.mean(
            torch.sigmoid(beta * acc_eta),
            dim=[1, 2])
        # (num thresh,)
        # Mean w.r.t. i (batch) and t (time_steps).

    return acc_eta


def version_C(scores_full, mask_pos, hittimes, beta, is_absnorm=True):
    """
    Args:
        scores_full: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        mask_pos: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        labels_concat: A binary label Tensor 
            with shape (batch size,) 
            with value = 0 or 1. This is the output from
            sequential_concat(logit_slice, labels_slice).
        thresh: A Tensor with shape 
            (num_thresh, time_steps, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
        beta: A positive float for the beta-sigmoid function.
        version: A string, either of "A", "B", "C", "D", or "E".
    """
    assert scores_full.shape[-1] == scores_full.shape[-2]
    num_classes = scores_full.shape[-1]
    time_steps = scores_full.shape[2]
    num_thresh = scores_full.shape[0]
    batch_size = scores_full.shape[1]
    
    mask_ht = calc_mask_ht(hittimes, time_steps, num_thresh, batch_size, num_classes)
        # (num thresh, batch, time_steps, num cls, num cls)
    mask = mask_pos * mask_ht
    scores_posht = scores_full * mask
        # (num thresh, batch, time_steps, num cls, num cls)
        # Note that all the frames, except for the hitting
        # time, of each sequence are filled with 0.
        # Note that all the rows, except for the y_{i}-th row, 
        # of each LLR matrix are filled with 0.
    acc_eta = torch.sum(scores_posht, dim=[2, 3])
        # (num thresh, batch, num cls(column))
        # Collapses rows of LLR matrix and frames.
        # Note that all the frames, except for the hitting
        # time, of each sequence are filled with 0.
        # Note that all the rows, except for the y_{i}-th row, 
        # of each LLR matrix are filled with 0.
    if is_absnorm:
        acc_eta =  (num_classes / (num_classes - 1.)) * (2 * torch.sigmoid(beta * acc_eta) - 1)
        acc_eta = torch.mean(acc_eta, dim=[1, 2]) / torch.mean(torch.abs(acc_eta), dim=[1, 2]) 
    else:
        acc_eta = torch.mean(
            (num_classes / (num_classes - 1.)) * torch.sigmoid(beta * acc_eta),
            dim=[1, 2])
        # (num thresh,)
        # Mean w.r.t. i (datapt) and k (column).
        
    return acc_eta

def version_D(scores_full, mask_pos, beta, is_absnorm=True):
    """
    Args:
        scores_full: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        mask_pos: A Tensor with shape 
            (num thresh, batch, time_steps, num cls, num cls)
        labels_concat: A binary label Tensor 
            with shape (batch size,) 
            with value = 0 or 1. This is the output from
            sequential_concat(logit_slice, labels_slice).
        thresh: A Tensor with shape 
            (num_thresh, time_steps, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
        beta: A positive float for the beta-sigmoid function.
        version: A string, either of "A", "B", "C", "D", or "E".
    """
    assert scores_full.shape[-1] == scores_full.shape[-2]
    num_classes = scores_full.shape[-1]

    scores_pos = mask_pos * scores_full
        # (num thresh, batch, time_steps, num cls, num cls)
        # Only positive-class's LLRs are non-zero, i.e.,
        # all the rows, except for the y_{i}-th row, 
        # are filled with 0.
    acc_eta = torch.sum(scores_pos, dim=3)
        # Shape=(num thresh, batch, time_steps, num cls(column))
        # Collapse rows of the LLR matrices.
        # Note that all the rows except for the y_{i}-th row
        # are filled with 0.
    if is_absnorm:
        acc_eta = (2 * torch.sigmoid(beta * acc_eta) - 1)
        acc_eta = torch.mean(acc_eta, dim=[1, 2, 3]) / torch.mean(torch.abs(acc_eta), dim=[1, 2, 3]) 
    else:
        acc_eta = torch.mean(
            (num_classes / (num_classes - 1.)) * torch.sigmoid(beta * acc_eta),
            dim=[1, 2, 3])
        # (num thresh, num cls)
        # Mean w.r.t. i (batch), t (time_steps), and k (column).

    return acc_eta


def version_Z(labels_concat, hitidx):
    '''
    Calc sat based on confusion matrix.
    For visualization purpose.

    Args:
    - labels_concat: 
    - hitidx: index of hitting time. Size: (NUM_THRESH, BATCH_SIZE, NUM_CLASSES)

    Return:
    - 
    '''
    num_thresh, _, num_classes = hitidx.shape

    preds = torch.squeeze(torch.topk(hitidx, k=1).indices)
        # (num thresh, batch,)
    preds = F.one_hot(preds, num_classes=num_classes)
        # (num thresh, batch, num cls)

    labels_oh = F.one_hot(labels_concat, num_classes=num_classes)
        # (batch, num cls)
    labels_oh = torch.unsqueeze(labels_oh, dim=0)
    labels_oh = labels_oh.repeat(num_thresh, 1, 1)
        # (num thresh, batch, num cls)
    preds = torch.unsqueeze(preds, dim=-2)
    labels_oh = torch.unsqueeze(labels_oh, dim=-1)
    confmx = torch.sum(labels_oh * preds, dim=1, dtype=torch.float32)
    macro_ave_sns = seqconfmx_to_macro_ave_sns(confmx)
    
    return macro_ave_sns
    


def calc_ausat_loss(version, is_mult_sat, num_thresh, sparsity,
    llrs, labels_concat, beta):

    thresh = threshold_generator(llrs, num_thresh, sparsity).detach()
    thresh = thresh_truncated_MSPRT(thresh).detach() # set the last thresh to zero (truncated (M)SPRT)
    thresh_sanity_check(thresh)

    acc_eta, hittimes, sns_from_confmx = calc_soft_auc(
        llrs, labels_concat, thresh, beta, version)
    
    # Whether to multiply diff mht by acc_eta
    ausat = multiply_diff_mht(acc_eta, hittimes) if is_mult_sat\
       else torch.mean(acc_eta) #  (num thresh,) -> scalar

    aucsat_loss = 1. - ausat
    mht = torch.mean(hittimes, dim=1)
    
    return aucsat_loss, mht, acc_eta, sns_from_confmx, thresh



def calc_soft_auc(llrs, labels_concat, thresh, beta, version):
    """ Calculate multiclass AUC-SAT
    Args:
        llrs: A Tensor with shape 
            (batch, time_steps, num classes, num classes).
        labels_concat: A binary label Tensor 
            with shape (batch size,) 
            with value = 0 or 1. This is the output from
            sequential_concat(logit_slice, labels_slice).
        thresh: A Tensor with shape 
            (num_thresh, time_steps, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
        beta: A positive float for the beta-sigmoid function.
        version: A string, either of "A", "B", "C", "D", or "E".
    Returns:
        aucsat: A scalar Tensor, the normalized AUC-SAT \in [0, 1]. 
    Remark:
        - Currently time_steps = 1 is not supported.
        - A: hittime, eta, min
        - B:       t, eta, min
        - C: hittime, eta, all
        - D:       t, eta, all
        See lines from ~455.
    """
    def calc_sat_from_confmx(labels_concat, hitidx):
        '''
        Calc non-differentiable confusion matrix.
        For visualization purpose.

        Args:
        - labels_concat: 
        - hitidx: index of hitting time. Size: (NUM_THRESH, BATCH_SIZE, NUM_CLASSES)

        Return:
        - 
        '''
        preds = torch.argmax(hitidx, dim=2)
            # (num thresh, batch,)
        preds = F.one_hot(preds, num_classes=num_classes)
            # (num thresh, batch, num cls)

        labels_oh = F.one_hot(labels_concat, num_classes=num_classes)
            # (batch, num cls)
        labels_oh = torch.unsqueeze(labels_oh, dim=0)
        labels_oh = labels_oh.repeat(num_thresh, 1, 1)
            # (num thresh, batch, num cls)
        preds = torch.unsqueeze(preds, dim=-2)
        labels_oh = torch.unsqueeze(labels_oh, dim=-1)
        confmx = torch.sum(labels_oh * preds, dim=1, dtype=torch.int32)
        ls_sns = seqconfmx_to_macro_ave_sns(confmx)

        return ls_sns

    assert beta > 0, "beta must be positive. Got {} ".format(beta)

    batch_size, time_steps, num_classes, num_thresh = extract_shapes(llrs, thresh)
    # Change LLRs and threshold scales
    llrs, thresh = normalize_LLR_thresh_scale(llrs, thresh)
        # Otherwise margins_up and margins_low may contain lots of zeros.
        # No shape changes.
        # llrs: (batch, time_steps, num classes, num classes)
        # thresh: (num_thresh, time_steps, num classes, num classes)
        # Comment 20221201: May cause optimization instability because of the fluctuating LLR scales? 
        #                   Maybe stop_grad (somewhere) needed?
            
    # Restore lost significance with epsilon
    llrs = restore_lost_significance(llrs)
        # (batch, time_steps, num cls, num cls)
        # To avoid double hit due to the values exactly equal to 0
        # in scores or when doing truncation (LLRs of the last frame).

    # Calc scores (:= LLR - thresh) and hittimes
    ##########################
    scores_full, hitidx, hittimes = calc_scores_and_hittimes(
        llrs, thresh, num_thresh, time_steps, batch_size, num_classes)
        # scores_full: (num thresh, batch, time_steps, num cls, num cls)
        # hitidx: (num thresh, batch, num cls)
        # hittimes: (num thresh, batch)
    
    mask_pos = calc_mask_pos(labels_concat, num_classes, time_steps, num_thresh)
        # (num thresh, batch, time_steps, num cls, num cls)

    # Version B and D: No need for hitting times
    # and no min func for the last axis of LLR
    #############################################
    if version == 'A':
        acc_eta = version_A(scores_full, mask_pos, llrs, hittimes, beta, is_absnorm=False)
    elif version == 'Anorm':
        acc_eta = version_A(scores_full, mask_pos, llrs, hittimes, beta, is_absnorm=True)
    elif version == 'B':
        acc_eta = version_B(scores_full, mask_pos, llrs, beta, is_absnorm=False)
    elif version == 'Bnorm':
        acc_eta = version_B(scores_full, mask_pos, llrs, beta, is_absnorm=True)
    elif version == 'C': 
        acc_eta = version_C(scores_full, mask_pos, hittimes, beta, is_absnorm=False)
    elif version == 'Cnorm': 
        acc_eta = version_C(scores_full, mask_pos, hittimes, beta, is_absnorm=True)
    elif version == 'D': 
        acc_eta = version_D(scores_full, mask_pos, beta, is_absnorm=False)
    elif version == 'Dnorm':
        acc_eta = version_D(scores_full, mask_pos, beta, is_absnorm=True)
    elif version == 'Z':
        acc_eta = torch.squeeze(version_Z(labels_concat, hitidx))
    else:
        raise ValueError('Found unknown AUSATloss version!')

    sns_from_confmx = calc_sat_from_confmx(labels_concat, hitidx)
    
    return acc_eta, hittimes, sns_from_confmx


def calc_weight_decay(model):
        wd_reg = 0.
        for name, param in model.named_parameters():
            if 'bias' not in name:  # Exclude bias terms from weight decay
                wd_reg += torch.norm(param, p=2)**2
        return wd_reg

###############################################################################
#                          Gradient Calculators                               #
###############################################################################


def compute_loss_and_metrics(model, x, y, global_step, config):
    """Calculate loss and gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        time_steps: An int. Num of frames in a sequence.
        num_thresh: An int. Num of thresholds for AUC-SAT.
        beta: A positive float for beta-sigmoid in AUC-SAT.
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_aucsat_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad. 
                The weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
            aucsat_loss: A scalar Tensor.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (time_steps - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        - All the losses below will be calculated if not calc_grad 
          for TensorBoard logs.
            total_loss 
            multiplet_loss
            llr_loss 
            aucsat_loss
            wd_reg
    """
    # check if necessary parameters are defined in the config file
    requirements = set(['ORDER_SPRT', 'TIME_STEPS', 'OBLIVIOUS', 'AUCLOSS_VERSION', 'AUCLOSS_BURNIN', 'NUM_THRESH',
        'SPARSITY', 'BETA', 'IS_MULT_SAT', 'IS_ADAPTIVE_LOSS',
        'PARAM_MULTIPLET_LOSS', 'PARAM_LLR_LOSS', 'PARAM_AUSAT_LOSS', 'LLLR_VERSION', 'WEIGHT_DECAY'])
    conf = extract_params_from_config(requirements, config)

    _device = x.device
    x_slice, y_slice = sequential_slice(x, y, conf.order_sprt)

    def adaptive_weight_loss(losses, constant_weights):
        assert len(losses) == model.adaptive_loss_weights.shape[0], 'Number of losses and weights must match!'
        assert len(constant_weights) == model.adaptive_loss_weights.shape[0], 'Number of constant weights and adaptive weights must match!'
        total_loss = 0.
        for i, (loss, const) in enumerate(zip(losses, constant_weights)):
            weight = torch.finfo(torch.float32).eps + model.adaptive_loss_weights[i] ** 2 
            total_loss += 0.5 * loss * const / weight + torch.log(weight)
        return total_loss
        
    logits_slice = model(x_slice)

    # logits_slice: (bs*(time-order_sprt), order_sprt+1, nb_cls)
    logits_concat, labels_concat = sequential_concat(
        logits_slice, y_slice, conf.time_steps)
    # logits_concat: (bs, time-order_sprt, order_sprt+1, nb_cls)
    # labels_concat: (bs)

    # multiplet_loss
    mce = multiplet_crossentropy(logits_slice, y_slice)
    
    lllr, llrs = compute_llrs_and_estimation_loss(
            logits_concat, labels_concat, conf.oblivious, conf.lllr_version) 
    
    # AUC loss
    aucsat_loss, mht, acc_eta_sat, sns_from_confmx, thresholds = calc_ausat_loss(
            conf.aucloss_version, conf.is_mult_sat, conf.num_thresh, conf.sparsity, 
            llrs, labels_concat, conf.beta)
    
    aucsat_loss = torch.tensor(0.).to(_device) if global_step < conf.aucloss_burnin else aucsat_loss 

    wd_reg = calc_weight_decay(model)
            
    if conf.is_adaptive_loss:
        total_loss = adaptive_weight_loss([mce, lllr, aucsat_loss, wd_reg],
        [conf.param_multiplet_loss, conf.param_llr_loss, conf.param_ausat_loss, conf.weight_decay])
    else:
        total_loss = conf.param_multiplet_loss * mce + conf.param_llr_loss * lllr + \
                        conf.param_ausat_loss * aucsat_loss + conf.weight_decay * wd_reg  

    losses = {'total_loss': total_loss, 'multiplet_crossentropy (MCE)': mce,
                'LLR_estimation (LLRE)': lllr, 'AUSAT_optimization': aucsat_loss,
                'weight_decay': wd_reg}
    monitored_values = {'losses': losses,
                        'logits_concat': logits_concat, 'thresholds': thresholds,
                        'llrs': llrs, 
                        'mht': mht, 'acc_eta_sat': acc_eta_sat, 
                        'sns_from_confmx': sns_from_confmx}
    return total_loss, monitored_values




