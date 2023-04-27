import pdb
import numpy as np
import optuna
from loguru import logger
import torch
import torch.nn.functional as F
from datasets.data_processing import sequential_slice_labels
from utils.performance_metrics import (
    calc_llrs,
    calc_oblivious_llrs,
    seqconfmx_to_macro_ave_sns,
    multiply_diff_mht,
)
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

    # A scalar averaged in a batch and a sliding window

    return xent


def compute_llrs(logits_concat: torch.Tensor, oblivious: bool = False) -> torch.Tensor:
    """
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (time_steps - order_sprt), order_sprt + 1, num classes).
            This is the output from
            datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        oblivious: A bool, whether to use the oblivion formula or not (= TANDEM formula).

    Returns:
        llrs: A Tensor with shape (batch size, time_steps, num cls, num cls).
    """

    if oblivious:
        # Miyagawa and Ebihara, ICML 2021
        llrs = calc_oblivious_llrs(logits_concat)
    else:
        # compute with TANDEM formula: Ebihara+, ICLR 2021
        llrs = calc_llrs(logits_concat)

    return llrs  # (batch_size, time_steps, num cls, num cls)


def compute_loss_for_llr(
    llrs: torch.Tensor, labels_concat: torch.Tensor, version: str
) -> torch.Tensor:
    """
    Compute the loss for log-likelihood ratio estimation (LLLR).
    Args:
        llrs: A Tensor with shape (batch size, time_steps, num cls, num cls).
        labels_concat: A Tensor with shape (batch size,).
        version: A string, which version of the loss to compute.

    Returns:
        loss: A scalar Tensor.
    """
    shapes = llrs.shape
    assert shapes[-1] == shapes[-2]
    time_steps = shapes[1]
    num_classes = shapes[-1]

    labels_oh = F.one_hot(labels_concat, num_classes).to(torch.float32)
    labels_oh = labels_oh.reshape(-1, 1, num_classes, 1)
    # (batch_size, 1, num cls, 1)

    if version == "A":  # LLLR, Ebihara+, ICLR 2021
        lllr = torch.abs(labels_oh - torch.sigmoid(llrs))
        # (batch, time_steps, num cls, num cls)
        lllr = 0.5 * (num_classes / (num_classes - 1.0)) * torch.mean(lllr)

    elif version == "B":
        llrs = llrs * labels_oh
        # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
        # (batch, time_steps, num cls)
        lllr = torch.abs(1.0 - torch.sigmoid(llrs))
        lllr = (num_classes / (num_classes - 1)) * torch.mean(lllr)

    elif version == "C":
        labels_oh = labels_oh.repeat(1, time_steps, 1, num_classes)
        # (batch, time_steps, num cls, num cls)
        lllr = F.binary_cross_entropy_with_logits(llrs, labels_oh)
        # scalar value (averaged)
        lllr = 0.5 * (num_classes / (num_classes - 1)) * lllr

    elif version == "D":
        llrs = llrs * labels_oh
        # (batch, time_steps, num cls, num cls)
        llrs = torch.sum(llrs, dim=2)
        # (batch, time_steps, num cls)
        llrs = llrs.reshape(-1, num_classes)
        tmp = torch.ones_like(llrs).to(torch.float32)
        lllr = F.binary_cross_entropy_with_logits(llrs, tmp)
        # scalar value (averaged))
        lllr = (num_classes / (num_classes - 1)) * lllr

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
        lllr = torch.mean(-minllr + torch.log(lllr + 1e-12))
        # scalar
    elif version == "Eplus":  # LSEL, Miyagawa and Ebihara, ICML 2021
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
        lllr = torch.mean(-minllr + torch.log(lllr + 1e-12))
        # scalar
    else:
        raise ValueError(
            "version={} must be either of 'A', 'B', 'C', 'D', 'E' or 'Eplus'.".format(
                version
            )
        )

    return lllr


def extract_shapes(llrs, scores_full):
    llrs_shape = llrs.shape
    batch_size = llrs_shape[0]
    time_steps = llrs_shape[1]
    num_classes = llrs_shape[2]
    num_thresh = scores_full.shape[0]

    return batch_size, time_steps, num_classes, num_thresh


def restore_lost_significance(llrs):
    # Restore lost significance with epsilon
    tri = torch.ones_like(llrs)
    triu = torch.triu(tri)  # Upper triangular part.
    tril = torch.tril(tri)  # Lower triangular part.
    llrs_restore = llrs - 1e-10 * (triu - tril)
    # (batch, time_steps, num cls, num cls)
    # To avoid double hit due to the values exactly equal to 0
    # in scores or when doing truncation, LLRs of the last frame.

    return llrs_restore


def calc_scores(llrs: torch.Tensor, thresh: torch.Tensor) -> torch.Tensor:
    """
    Calculate scores (= LLR - thresh).

    Args:
        llrs: (batch_size, time_steps, num cls, num cls)
        thresh: (num_thresh, batch_size, time_steps, num cls, num cls)
    Returns:
        scores_full: (num_thresh, batch_size, time_steps, num cls, num cls)
    """
    assert len(thresh.shape) == 5, "thresh must be 5-dim tensor."

    llr_mtx = torch.unsqueeze(llrs, dim=0)
    scores_full = llr_mtx - thresh

    return scores_full


def calc_hittimes(llrs, scores_full, num_thresh, time_steps, batch_size, num_classes):
    """ """

    def _calc_truncated_predictions(scores_full, llrs, num_thresh):
        """ """
        scores = torch.min(scores_full, dim=-1)[
            0
        ]  # (num thresh, batch, time_steps, num cls)
        # Calc all predictions and waits
        preds_all = torch.sign(scores) + 1
        # 0:wait, 1:hit (one-hot vector)
        # (num thresh, batch, time_steps, num cls)
        preds_last = torch.sign(torch.min(llrs[:, -1:, :, :], dim=-1)[0]) + 1
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
        mask = torch.tensor(
            [i + 1 for i in range(time_steps)][::-1], dtype=torch.float32
        )
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

        assert (
            torch.max(hittimes) <= time_steps
        ), f"hittimes must be \in {1, time_steps}, while got {torch.max(hittimes)=}!"
        return hitidx, hittimes

    _device = llrs.device

    # Calc scores and scores_full
    # scores, scores_full = _calc_scores(llrs, thresh)
    # Score means LLR - thresh.
    # scores:      (num thresh, batch, time_steps, num cls)
    # scores_full: (num thresh, batch, time_steps, num cls, num cls)

    # Calc all predictions with truncation
    preds_all_trunc = _calc_truncated_predictions(scores_full, llrs, num_thresh)
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

    return hitidx, hittimes


def calc_ausat(llrs, scores_full, labels_concat):
    """ """

    batch_size, time_steps, num_classes, num_thresh = extract_shapes(llrs, scores_full)

    hitidx, hittimes = calc_hittimes(
        llrs, scores_full, num_thresh, time_steps, batch_size, num_classes
    )
    mht = torch.mean(hittimes, dim=1)
    sns_from_confmx = calc_sat_from_confmx(
        labels_concat, hitidx, num_classes, num_thresh
    )

    return mht, sns_from_confmx


def calc_sat_from_confmx(labels_concat, hitidx, num_classes, num_thresh):
    """
    Calc non-differentiable confusion matrix.
    For visualization purpose.

    Args:
    - labels_concat:
    - hitidx: index of hitting time. Size: (NUM_THRESH, BATCH_SIZE, NUM_CLASSES)

    Return:
    -
    """
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


def calc_weight_decay(model):
    """ """
    wd_reg = 0.0
    for name, param in model.named_parameters():
        if "bias" not in name:  # Exclude bias terms from weight decay
            wd_reg += torch.norm(param, p=2) ** 2
    return wd_reg


def adaptive_weight_loss(model, losses, constant_weights):
    """ """
    assert (
        len(losses) == model.adaptive_loss_weights.shape[0]
    ), "Number of losses and weights must match!"
    assert (
        len(constant_weights) == model.adaptive_loss_weights.shape[0]
    ), "Number of constant weights and adaptive weights must match!"
    total_loss = 0.0
    for i, (loss, const) in enumerate(zip(losses, constant_weights)):
        weight = torch.finfo(torch.float32).eps + model.adaptive_loss_weights[i] ** 2
        total_loss += 0.5 * loss * const / weight + torch.log(weight)
    return total_loss


def compute_loss_and_metrics(model, x, labels, global_step, config):
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
    requirements = set(
        [
            "ORDER_SPRT",
            "TIME_STEPS",
            "OBLIVIOUS",
            "NUM_THRESH",
            "SPARSITY",
            "BETA",
            "IS_ADAPTIVE_LOSS",
            "MAX_NORM",
            "BATCH_SIZE",
            "ORDER_SPRT",
            "PARAM_MULTIPLET_LOSS",
            "PARAM_LLR_LOSS",
            "LLLR_VERSION",
            "WEIGHT_DECAY",
        ]
    )
    conf = extract_params_from_config(requirements, config)

    y_slice = sequential_slice_labels(labels, conf.time_steps, conf.order_sprt)

    model_forward = model
    model_module = model

    llrs, logits_slice, thresholds, scores_full = model_forward(x)

    lllr = compute_loss_for_llr(llrs, labels, conf.lllr_version)

    # multiplet_loss
    mce = multiplet_crossentropy(logits_slice, y_slice)

    # AUC
    mht, sns_from_confmx = calc_ausat(llrs, scores_full, labels)

    # L2 weight decay regularization
    wd_reg = calc_weight_decay(model_module)

    if conf.is_adaptive_loss:
        # adaptively change weights with trainable parameters
        total_loss = adaptive_weight_loss(
            model_module,
            [mce, lllr, wd_reg],
            [
                conf.param_multiplet_loss,
                conf.param_llr_loss,
                conf.weight_decay,
            ],
        )

    else:
        # use constant weights predefined in config file
        total_loss = (
            conf.param_multiplet_loss * mce
            + conf.param_llr_loss * lllr
            + conf.weight_decay * wd_reg
        )

    # store loss values
    losses = {
        "total_loss": total_loss,
        "multiplet_crossentropy (MCE)": mce,
        "LLR_estimation (LLRE)": lllr,
        "weight_decay": wd_reg,
    }

    # store performance metrics and training status with the losses
    monitored_values = {
        "losses": losses,
        "thresholds": thresholds,
        "llrs": llrs,
        "mht": mht,
        "sns_from_confmx": sns_from_confmx,
    }

    return monitored_values
