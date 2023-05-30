import pdb
import warnings
import torch
from torchmetrics import ConfusionMatrix
import numpy as np
from utils.misc import extract_params_from_config, convert_torch_to_numpy, ErrorHandler


def confusion_matrix(preds, labels, num_classes):
    """
    Calculate confusion matrix.
    """
    task = (
        "binary" if num_classes == 2 else "multiclass"
    )  # ConfusionMatrix also supports multilabel
    confmat = ConfusionMatrix(task=task, num_classes=num_classes)
    return confmat(preds, labels)


# Functions: logit to confusion matrix
def logits_to_confmx(logits, labels):
    """Calculate the confusion matrix from logits.
    Args:
        logits: A logit Tensor with shape (batch, num classes).
        labels: A non-one-hot label Tensor with shape (batch,).
    Returns:
        confmx: A Tensor with shape (num classes, num classes).
    """
    logits_shape = logits.shape  # (batch, num classes)
    num_classes = logits_shape[-1]

    # First order_sprt+1 frames
    preds = torch.argmax(logits, dim=-1, keepdim=False).to(torch.int32)
    confmx = confusion_matrix(preds, labels, num_classes)

    return confmx


# Functions related to differentials of mean hitting times
######################################################
def multiply_diff_mht(acc_eta, hittimes):
    """Multiplies acc_eta by Delta t (ver x hor = area).
    # Args
    acc_eta: ...
    hittingtimes: Shape=(num thresh, batch)
    time_steps: An int.
    """
    time_steps = hittimes.shape[0]
    assert time_steps == acc_eta.shape[0], "Length of time_steps mismatched!"

    _device = acc_eta.device
    # Calc differentials of mean hitting times
    mht = torch.mean(hittimes, dim=1) if len(hittimes.shape) > 1 else hittimes
    # from (num thresh, batch)
    # to (num thresh,)
    _mht = torch.cat(
        [mht[1:], torch.tensor(
            [time_steps], dtype=torch.float32, device=_device)],
        dim=0,
    )
    diff_mht = _mht - mht

    aucsat = torch.sum(diff_mht * torch.squeeze(acc_eta))
    aucsat_norm = aucsat / time_steps  # normalize

    return aucsat_norm


def initialize_performance_metrics():
    """ """
    return {
        "losses": [],
        "mean_abs_error": [],
        "sns_conf": [],
        "seqconfmx_llr": 0,
        "hitting_time": [],
        "ausat_from_confmx": None,
        "grad_norm": [],
    }


def training_setup(phase: str, config):
    """
    Returns:
    - is_train (bool): train flag to decide if trainable weights are updated.
    - iter_num (int): an index of the last full-size batch.
    - performance_metrics (dict): dictionary of model performance metrics
                                  initialized with zeros or empty list.
    - barcolor (str): specifies tqdm bar color for each phase.
    """
    # check if necessary parameters are defined in the config file
    requirements = set(["NUM_TRAIN", "NUM_VAL", "NUM_TEST", "BATCH_SIZE"])
    conf = extract_params_from_config(requirements, config)

    # suppress warnings at gradient clipping
    warnings.filterwarnings(
        "ignore",
        message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    is_train = True if "train" in phase.lower() else False
    barcolor = "cyan" if "train" in phase.lower() else "yellow"

    if "train" in phase.lower():
        iter_num = np.ceil(conf.num_train / conf.batch_size).astype("int")
    elif "val" in phase.lower():
        iter_num = np.ceil(conf.num_val / conf.batch_size).astype("int")
    elif "test" in phase.lower():
        iter_num = np.ceil(conf.num_test / conf.batch_size).astype("int")
    else:
        raise ValueError("Unknown phase!")
    performance_metrics = initialize_performance_metrics()

    return is_train, iter_num, performance_metrics, barcolor


def accumulate_performance(
    performance_metrics, y_batch, gt_llrs_batch, monitored_values, phase_manager
):
    """ """

    performance_metrics["losses"].append(monitored_values["losses"])
    performance_metrics["mean_abs_error"].append(
        calc_llr_abserr(monitored_values["llrs"], gt_llrs_batch)
    )  # (batch_size, time_steps)
    performance_metrics["seqconfmx_llr"] += llr_sequential_confmx(
        monitored_values["llrs"], y_batch
    )
    performance_metrics["hitting_time"].append(monitored_values["mht"])
    performance_metrics["sns_conf"].append(monitored_values["sns_from_confmx"])
    performance_metrics["grad_norm"].append(phase_manager.grad_norm)

    return performance_metrics


def summarize_performance(performance_metrics):
    """ """

    def calc_macrec(confmx):
        """
        Args:
        - confmx: sequential confusion matrix
        Return:
        - macrec: macro-averaged recall
        """
        return torch.mean(seqconfmx_to_macro_ave_sns(confmx))

    # average
    performance_metrics["losses"] = average_dicts(
        performance_metrics["losses"])
    performance_metrics["mean_macro_recall"] = calc_macrec(
        performance_metrics["seqconfmx_llr"]
    )
    performance_metrics["mean_abs_error"] = torch.mean(
        torch.cat(performance_metrics["mean_abs_error"], dim=0)
    )
    performance_metrics["hitting_time"] = torch.mean(
        torch.stack(performance_metrics["hitting_time"]), dim=0
    )  # each entry has a size of (500)
    performance_metrics["sns_conf"] = torch.mean(
        torch.stack(performance_metrics["sns_conf"]), dim=0
    )  # each entry has a size of (500, 1)
    performance_metrics["ausat_from_confmx"] = multiply_diff_mht(
        performance_metrics["sns_conf"], performance_metrics["hitting_time"]
    )
    performance_metrics["grad_norm"] = torch.mean(
        torch.stack(performance_metrics["grad_norm"])
    )
    # to numpy
    performance_metrics = convert_torch_to_numpy(performance_metrics)

    return performance_metrics


def multiplet_sequential_confmx(logits_concat, labels_concat):
    """Calculate the confusion matrix for each frame from logits. Lite.
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (time_steps - order_sprt), order_sprt + 1, num_classes).
            This is the output from
            datasets.data_processing.sequential_concat(logit_slice, y_slice).
        labels_concat: A non-one-hot label Tensor with shape (batch,).
            This is the output from
            datasets.data_processing.sequential_conclogit_slice, y_slice).
    Return:
        seqconfmx_mult: A Tensor with shape
        (time_steps, num classes, num classes). This is the series of
        confusion matrices computed from multiplets.
    Remark:
        e.g., order_sprt = 5, time_steps = 20:
            confusion matrix for frame001 is given by the 001let of frame001
            confusion matrix for frame002 is given by the 002let of frame002
            ...
            confusion matrix for frame005 is given by the 004let of frame004
            confusion matrix for frame005 is given by the 005let of frame005
            confusion matrix for frame006 is given by the 005let of frame006 computed from frame002-006
            confusion matrix for frame007 is given by the 005let of frame007 computed from frame003-007
            ...
            confusion matrix for frame019 is given by the 005let of frame019 computed from frame015-019
            confusion matrix for frame020 is given by the 005let of frame020 computed from frame016-020
    """
    logits_concat_shape = (
        logits_concat.shape
    )  # (batch, (time_steps - order_sprt), order_sprt + 1, num classes)
    num_classes = logits_concat_shape[-1]

    # First order_sprt+1 frames
    # (batch, order_sprt + 1, num classes)
    logits_concat_former = logits_concat[:, 0, :, :]

    for iter_idx in range(logits_concat_shape[2]):
        preds_former = torch.argmax(logits_concat_former[:, iter_idx, :], dim=-1).to(
            torch.int32
        )  # (batch,)
        if iter_idx == 0:
            seqconfmx_mult = confusion_matrix(
                preds_former, labels_concat, num_classes
            ).to(torch.int32)
            seqconfmx_mult = torch.unsqueeze(seqconfmx_mult, 0)
        else:
            seqconfmx_mult = torch.cat(
                [
                    seqconfmx_mult,
                    torch.unsqueeze(
                        confusion_matrix(preds_former, labels_concat, num_classes).to(
                            torch.int32
                        ),
                        0,
                    ),
                ],
                dim=0,
            )

    # Latter time_steps-order_sprt-1 frames
    # (batch, time_steps-order_sprt-1, num classes)
    logits_concat_latter = logits_concat[:, 1:, -1, :]

    for iter_idx in range(logits_concat_shape[1] - 1):
        preds_latter = torch.argmax(logits_concat_latter[:, iter_idx, :], dim=-1).to(
            torch.int32
        )  # (batch,)
        seqconfmx_mult = torch.cat(
            [
                seqconfmx_mult,
                torch.unsqueeze(
                    confusion_matrix(preds_latter, labels_concat, num_classes).to(
                        torch.int32
                    ),
                    dim=0,
                ),
            ],
            dim=0,
        )

    return seqconfmx_mult


# Functions: confusion matrix to metric
def confmx_to_metrics(confmx):
    """Calc confusion-matrix-based performance metrics.
    Args:
        confmx: A confusion matrix Tensor
            with shape (num classes, num classes).
    Return:
        dict_metrics: A dictionary of dictionaries of performance metrics.
            E.g., sensitivity of class 3 is dics_metrics["SNS"][3],
            and accuracy is dict_metrics["ACC"]["original"]
    Remark:
        - SNS: sensitivity, recall, true positive rate
        - SPC: specificity, true negative rate
        - PRC: precision
        - ACC: accuracy
        - BAC: balanced accuracy
        - F1: F1 score
        - GM: geometric mean
        - MCC: Matthews correlation coefficient. May cause overflow.
        - MK: markedness
        - e.g., The classwise accuracy of class i is dict_metric["SNS"][i].
        - "dict_metrics" contains some redundant metrics;
          e.g., for binary classification,
          dict_metric["SNS"]["macro"] = dict_metric["BAC"][0]
          = dict_metric["BAC"][1] = ...
        - Macro-averaged metrics are more robust to class-imbalance
          than micro-averaged ones, but note that macro-averaged metrics
          are sometimes equal to be ACC.
        - Most of the micro-averaged metrics are equal to or equivalent to ACC.
    """
    confmx = confmx.to(torch.int64)  # prevent from overflowing
    num_classes = confmx.shape[0]
    dict_metrics = {
        "SNS": dict(),
        "SPC": dict(),
        "PRC": dict(),
        "ACC": dict(),
        "BAC": dict(),
        "F1": dict(),
        "GM": dict(),
        "MCC": dict(),
        "MK": dict(),
    }
    TP_tot = 0
    TN_tot = 0
    FP_tot = 0
    FN_tot = 0

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix
    for i in range(num_classes):
        # Initialization
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # Calc TP, TN, FP, FN for class i
        TP = confmx[i, i]
        for j in range(num_classes):
            if j == i:
                continue
            FP += confmx[j, i]
            FN += confmx[i, j]
            for k in range(num_classes):
                if k == i:
                    continue
                TN += confmx[j, k]

        # Calc performance metrics of class i
        dict_metrics["SNS"][i] = TP / (TP + FN) if TP + FN != 0 else 0.0
        dict_metrics["SPC"][i] = TN / (TN + FP) if TN + FP != 0 else 0.0
        dict_metrics["PRC"][i] = TP / (TP + FP) if TP + FP != 0 else 0.0
        dict_metrics["ACC"][i] = (
            (TP + TN) / (TP + FN + TN + FP) if TP + FN + TN + FP != 0 else 0.0
        )
        dict_metrics["BAC"][i] = (
            dict_metrics["SNS"][i] + dict_metrics["SPC"][i]) / 2
        dict_metrics["F1"][i] = (
            2
            * (dict_metrics["PRC"][i] * dict_metrics["SNS"][i])
            / (dict_metrics["PRC"][i] + dict_metrics["SNS"][i])
            if dict_metrics["PRC"][i] + dict_metrics["SNS"][i] != 0
            else 0.0
        )
        dict_metrics["GM"][i] = np.sqrt(
            dict_metrics["SNS"][i] * dict_metrics["SPC"][i])
        dict_metrics["MCC"][i] = (
            ((TP * TN) - (FP * FN))
            / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
            if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0
            else 0.0
        )
        dict_metrics["MK"][i] = (
            dict_metrics["PRC"][i] + (TN / (TN + FN)) - 1
            if TN + FN != 0
            else dict_metrics["PRC"][i] - 1
        )

        TP_tot += TP
        TN_tot += TN
        FP_tot += FP
        FN_tot += FN

    # Calc micro- and macro- averaged metrics
    # sometimes returns nan. please fix it
    dict_metrics["SNS"]["macro"] = np.mean(
        [dict_metrics["SNS"][i] for i in range(num_classes)]
    )
    dict_metrics["SNS"]["micro"] = (
        TP_tot / (TP_tot + FN_tot) if TP_tot + FN_tot != 0 else 0.0
    )  # = original ACC.
    dict_metrics["SPC"]["macro"] = np.mean(
        [dict_metrics["SPC"][i] for i in range(num_classes)]
    )
    dict_metrics["SPC"]["micro"] = (
        TN_tot / (TN_tot + FP_tot) if TN_tot + FP_tot != 0 else 0.0
    )
    dict_metrics["PRC"]["macro"] = np.mean(
        [dict_metrics["PRC"][i] for i in range(num_classes)]
    )
    dict_metrics["PRC"]["micro"] = (
        TP_tot / (TP_tot + FP_tot) if TP_tot + FP_tot != 0 else 0.0
    )  # = original ACC.
    dict_metrics["ACC"]["macro"] = np.mean(
        [dict_metrics["ACC"][i] for i in range(num_classes)]
    )
    dict_metrics["ACC"]["micro"] = (
        (TP_tot + TN_tot) / (TP_tot + FN_tot + TN_tot + FP_tot)
        if TP_tot + FN_tot + TN_tot + FP_tot != 0
        else 0.0
    )
    dict_metrics["ACC"]["original"] = (
        (num_classes / 2) * dict_metrics["ACC"]["micro"]
    ) - ((num_classes - 2) / 2)
    dict_metrics["BAC"]["macro"] = np.mean(
        [dict_metrics["BAC"][i] for i in range(num_classes)]
    )
    dict_metrics["BAC"]["micro"] = (
        dict_metrics["SNS"]["micro"] + dict_metrics["SPC"]["micro"]
    ) / 2
    dict_metrics["F1"]["macro"] = np.mean(
        [dict_metrics["F1"][i] for i in range(num_classes)]
    )
    dict_metrics["F1"]["micro"] = (
        2
        * (dict_metrics["PRC"]["micro"] * dict_metrics["SNS"]["micro"])
        / (dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"])
        if dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"] != 0
        else 0.0
    )  # = original ACC.
    dict_metrics["GM"]["macro"] = np.mean(
        [dict_metrics["GM"][i] for i in range(num_classes)]
    )
    dict_metrics["GM"]["micro"] = np.sqrt(
        dict_metrics["SNS"]["micro"] * dict_metrics["SPC"]["micro"]
    )
    dict_metrics["MCC"]["macro"] = np.mean(
        [dict_metrics["MCC"][i] for i in range(num_classes)]
    )
    dict_metrics["MCC"]["micro"] = (
        ((TP_tot * TN_tot) - (FP_tot * FN_tot))
        / (
            np.sqrt(
                (TP_tot + FP_tot)
                * (TP_tot + FN_tot)
                * (TN_tot + FP_tot)
                * (TN_tot + FN_tot)
            )
        )
        if (TP_tot + FP_tot) * (TP_tot + FN_tot) * (TN_tot + FP_tot) * (TN_tot + FN_tot)
        != 0
        else 0.0
    )
    dict_metrics["MK"]["macro"] = np.mean(
        [dict_metrics["MK"][i] for i in range(num_classes)]
    )
    dict_metrics["MK"]["micro"] = (
        dict_metrics["PRC"]["micro"] + (TN_tot / (TN_tot + FN_tot)) - 1
        if TN_tot + FN_tot != 0
        else 0.0
    )

    return dict_metrics


def average_dicts(dict_list):
    """
    This function takes a list of dictionaries as input and returns a new dictionary that
    contains the average of the values for each key across all the input dictionaries.
    The keys in the input dictionaries should be identical.

    Args:
    - dict_list(list of dictionary)

    Returns:
    - averaged dict_list

    Raise:
    - AssertionError if not all dictionaries have the same keys.
    """
    keys = dict_list[0].keys()
    assert all(
        d.keys() == keys for d in dict_list
    ), "All dictionaries must have the same keys."

    return {
        key: torch.mean(torch.stack(values))
        for key, values in zip(dict_list[0], zip(*[d.values() for d in dict_list]))
    }


def seqconfmx_to_macro_ave_sns(seqconfmx):
    """Calc confusion-matrix-based performance metrics.
        V2 supports accuracy and implements a different calc criterion
        of the macro-averaged metrics. V2's output is np.ndarray, not Tensor.
    Important Note (February 24th, 2021):
        If no example belonging to a class comes in `seqconfmx`,
        i.e., all the entries in the raw corresponding to that class are zero,
        then the classwise metric of that class is assumed to be ZERO.
        However, the macro-averaged metric IGNORES such empty classes (V2),
        while in `seqconfmx_to_metrics` (V1), the macro-averaged metrics
        assume that the classwise metrics of the empty classes are ZERO
        (i.e., do not ignore them), which may significantly degradate
        the macro-averaged metrics (e.g., when the sample size used for `seqconfmx`
        is much smaller than the number of classes).
    Args:
        seqconfmx: A series of confusion matrix Tensors
            with shape (series length (arbitrary), num classes, num classes).
    Return:
        dict_metrics: A dictionary of performance metrics including
            classwise, macro-averaged, and micro-averaged metrics.
    Remarks:
        - Examples:
            dics_metrics["SNS"][k] = sensitivity of class 3, where
                k = 0, 1, 2, ..., num classes - 1.
            dict_metrics["SNS"][num classes] = macro-averaged sensitivity.
            dict_metrics["SNS"][num classes + 1] = micro-averaged sensitivity, which
                is equal to accuracy.
    """
    time_steps = seqconfmx.shape[0]
    seqconfmx = seqconfmx.to(torch.float32)  # avoids overflow
    classwise_sample_size = torch.sum(seqconfmx, dim=2)
    # shape = (time_steps, num cls)

    effective_entries = torch.not_equal(classwise_sample_size, 0)
    mask = torch.where(effective_entries)
    # tuple of (column#0 indices, column#1 indices)
    mask = torch.stack(mask, dim=1)
    # A Tensor of integer indices.
    # shape = (num of classes with non-zero sample sizes, 2)
    # 2 means [row index, column index]
    # E.g.,
    #    <tf.Tensor: id=212, shape=(1250, 2), dtype=int64, numpy=
    #    array([[  0,   1],
    #        [  0,   5],
    #        [  0,   6],
    #        ...,
    #        [ 49,  90],
    #        [ 49,  99],
    #        [ 49, 100]])>
    # Usage:
    #    NEW_TENSOR = tf.reshape(tf.gather_nd(TENSOR, mask), [time_steps, -1]),
    #    where TENSOR.shape is (time_steps, num_classes).
    #    NEW_TENSOR has shape (time_steps, num of classes with non-zero sample sizes).
    #    TENSOR can be SNS, TP, TN, FP, FN, etc., as shown below.

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix

    TP = torch.diagonal(seqconfmx, dim1=1, dim2=2)
    # (time_steps, num cls)
    FP = torch.sum(seqconfmx, dim=1) - TP
    # (time_steps, num cls)
    FN = torch.sum(seqconfmx, dim=2) - TP
    # (time_steps, num cls)
    TN = torch.sum(seqconfmx, dim=1)
    # (time_steps, num cls)
    TN = torch.sum(TN, dim=1, keepdims=True)
    # (time_steps, 1)
    TN = TN.repeat(1, 3) - (TP + FP + FN)
    # (time_steps, num cls)

    # Sensitivity (Recall, Classwise accuracy)
    ############################################################
    # Calc classwise, macro-ave, micro-ave metrics
    SNS = TP / (TP + FN + 1e-12)
    # (time_steps, num cls)
    gathered_SNS = SNS[mask[:, 0], mask[:, 1]]

    reshaped_SNS = gathered_SNS.reshape(time_steps, -1)

    macroave_sns = torch.mean(reshaped_SNS, dim=1, keepdims=True)
    # (time_steps, 1)

    return macroave_sns


################################################
# Functions For SPRT
################################################


# Functions: Calc LLRs (logit -> LLR)
def calc_llrs(logits_concat):
    """Calculate the frame-by-frame log-likelihood ratio matrices.
        Used to calculate LLR(x^(1), ..., x^(t))
        with N-th order posteriors (the N-th order TANDEM formula).
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (time_steps - order_sprt), order_sprt + 1, num classes).
            This is the output from
            datasets.data_processing.sequential_concat(
            logit_slice, labels_slice).
    Returns:
        llr_matrix: A Tensor
            with shape (batch, time_steps, num classes, num classes).
            matrix[i, j] = log(p_i/p_j), where p_i and p_j are the likelihood
            of class i and that of j, resp.
            They are anti-symmetric matrices.
    Remarks:
        - The LLRs returned are the LLRs used in the
          "order_sprt"-th order SPRT; the LLRs unnecessary to calculate the
          "order_sprt"-th order SPRT are not included.
        - "time_steps" and "order_sprt" are automatically calculated
          using logits_concat.shape.
    """
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    time_steps = int(logits_concat_shape[1] + order_sprt)
    num_classes = int(logits_concat_shape[3])
    assert num_classes > 1, "num_classes={} must > 1".format(num_classes)
    logits1 = torch.unsqueeze(logits_concat, dim=-1)
    # (batch, time_steps - order, order + 1, num cls, 1)
    logits2 = torch.unsqueeze(logits_concat, dim=-2)
    # (batch, time_steps - order, order + 1, 1, num cls)
    list_llrs = []

    # i.i.d. LLR (for 0th order SPRT)
    if order_sprt == 0:
        llrs_all_frames = (
            logits1[:, :, order_sprt, :, 0:] - logits2[:, :, order_sprt, 0:, :]
        )
        # (batch, time_steps, num cls, num cls)
        for iter_frame in range(time_steps):
            llrs = torch.sum(llrs_all_frames[:, : iter_frame + 1, :, :], dim=1)
            # (batch, num cls, num cls)
            list_llrs.append(torch.unsqueeze(llrs, dim=1))

    # N-th order LLR (for N-th order SPRT)
    else:
        for iter_frame in range(time_steps):
            if iter_frame < order_sprt + 1:
                llrs = (
                    logits1[:, 0, iter_frame, :, 0:] -
                    logits2[:, 0, iter_frame, 0:, :]
                )
                # (batch, num cls, num cls)
                list_llrs.append(torch.unsqueeze(llrs, dim=1))

            else:
                llrs1 = (
                    logits1[:, : iter_frame -
                            order_sprt + 1, order_sprt, :, 0:]
                    - logits2[:, : iter_frame -
                              order_sprt + 1, order_sprt, 0:, :]
                )
                # (batch, iter_frame-order_sprt, num cls, num cls)
                llrs1 = torch.sum(llrs1, dim=1)  # (batch, num cls, num cls)
                llrs2 = (
                    logits1[:, 1: iter_frame - order_sprt +
                            1, order_sprt - 1, :, 0:]
                    - logits2[:, 1: iter_frame -
                              order_sprt + 1, order_sprt - 1, 0:, :]
                )
                # (batch, iter_frame-order_sprt-1, num cls, num cls)
                llrs2 = torch.sum(llrs2, dim=1)  # (batch, num cls, num cls)
                llrs = llrs1 - llrs2  # (batch, num cls, num cls)
                list_llrs.append(torch.unsqueeze(llrs, dim=1))

    # (batch, time_steps, num cls, num cls)
    llr_matrix = torch.cat(list_llrs, dim=1)

    return llr_matrix


def calc_llr_abserr(estimated_llrs, ground_truth_llrs):
    """
    Calculate absolute estimation error of LLR matrix.
    The estimated and ground-truth LLR matrix must have the same shape.

    Args:
    - estimated_llrs (numpy array): an estimated likelihood ratio matrix
      of size (batch_size, time_steps, num_classes, num_classes)
    - ground_truth_llrs (numpy array): the ground-truth likelihood ratio matrix
      of size (batch_size, time_steps, num_classes, num_classes)

    Returns:
    - abserr (numpy array): absolute error matrix of the same size as the LLR matrices.
    """
    assert estimated_llrs.shape == ground_truth_llrs.shape

    abserr = torch.abs(estimated_llrs - ground_truth_llrs)
    return abserr


def calc_urgency_signal(u_sgnl_concat):
    """Calculate the urgency signal from fragmented signals.
    Args:
    - u_sgnl_concat: A Tensor with shape (bs, time-order_sprt, order_sprt+1, 1)
    Returns:
    - u_sgnl: A Tensor with shape (bs, time, 1)
    Raises:
    """
    _device = u_sgnl_concat.device
    sgnl_shape = u_sgnl_concat.shape
    bs = sgnl_shape[0]
    order_sprt = sgnl_shape[2] - 1
    time = sgnl_shape[1] + order_sprt

    u_sgnl = torch.zeros((bs, time, 1), device=_device)
    for i in range(sgnl_shape[1]):
        u_sgnl[:, i: i + order_sprt + 1, :] += u_sgnl_concat[:, i, :, :]

    for i in range(time):
        if i < order_sprt and (time - i) > order_sprt:  # at the beginning
            u_sgnl[:, i, :] /= i + 1
        elif i >= order_sprt and (time - i) > order_sprt:  # at the middle
            u_sgnl[:, i, :] /= order_sprt + 1
        elif (time - i) <= order_sprt:  # at the end
            u_sgnl[:, i, :] /= time - i

    return u_sgnl


def calc_oblivious_llrs(logits_concat):
    """Calculate the frame-by-frame log-likelihood ratio matrices.
        Used to calculate LLR(x^(t-N), ..., x^(t))
        i.e., (the N-th order TANDEMsO formula).
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (time_steps - order_sprt), order_sprt + 1, num classes).
            This is the output from
            datasets.data_processing.sequential_concat(
            logit_slice, labels_slice).
    Returns:
        llr_matrix: A Tensor
            with shape (batch, time_steps, num classes, num classes).
            matrix[i, j] = log(p_i/p_j), where p_i and p_j are the likelihood
            of class i and that of j, resp.
            They are anti-symmetric matrices.
    Remarks:
        - The LLRs returned are the LLRs used in the
          "order_sprt"-th order SPRT; the LLRs unnecessary to calculate the
          "order_sprt"-th order SPRT are not included.
        - "time_steps" and "order_sprt" are automatically calculated
          from logits_concat.shape.
    """
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    time_steps = int(logits_concat_shape[1] + order_sprt)
    num_classes = int(logits_concat_shape[3])
    assert num_classes > 1, "num_classes={} must > 1".format(num_classes)

    logits1 = torch.unsqueeze(logits_concat, dim=-1)
    # (batch, time_steps - order, order + 1, num cls, 1)
    logits2 = torch.unsqueeze(logits_concat, dim=-2)
    # (batch, time_steps - order, order + 1, 1, num cls)
    list_llrs = []

    # i.i.d. SPRT (0th-order SPRT)
    if order_sprt == 0:
        llrs_all_frames = (
            logits1[:, :, order_sprt, :, 0:] - logits2[:, :, order_sprt, 0:, :]
        )
        # (batch, time_steps, num cls, num cls)
        llr_matrix = llrs_all_frames  # oblivious!!

    # N-th order LLR (for N-th order oblivious SPRT)
    else:
        for iter_frame in range(time_steps):
            if iter_frame < order_sprt + 1:
                llrs = (
                    logits1[:, 0, iter_frame, :, 0:] -
                    logits2[:, 0, iter_frame, 0:, :]
                )
                # (batch, num cls, num cls)
                list_llrs.append(torch.unsqueeze(llrs, 1))

            else:
                llrs1 = (
                    logits1[:, iter_frame - order_sprt, order_sprt, :, 0:]
                    - logits2[:, iter_frame - order_sprt, order_sprt, 0:, :]
                )
                # (batch, num cls, num cls)
                # removed two colons and two "+1" to be oblivious
                # llrs1 = torch.sum(llrs1, 1) # (batch, num cls, num cls)
                # llrs2 = logits1[:, 1:iter_frame - order_sprt + 1, order_sprt-1, :, 0:]\
                #    - logits2[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 0:, :]
                #    # (batch, iter_frame-order_sprt-1, num cls, num cls)
                # llrs2 = torch.sum(llrs2, 1) # (batch, num cls, num cls)
                llrs = llrs1  # - llrs2 # (batch, num cls, num cls)
                list_llrs.append(torch.unsqueeze(llrs, 1))

        # (batch, time_steps, num cls, num cls)
        llr_matrix = torch.cat(list_llrs, dim=1)

    return llr_matrix


# Functions: threshold (LLR -> thresh)
def threshold_generator(llrs, num_thresh, sparsity):
    """Generates sequences of sigle-valued threshold matrices.
    Args:
        llrs: A Tensor with shape
            [batch, time_steps, num classes, num classes].
            Anti-symmetric matrix.
        num_thresh: An integer, the number of threholds.
            1 => thresh = minLLR
            2 => thresh = minLLR, maxLLR
            3 => thresh = minLLR, (minLLR+maxLLR)/2, maxLLR
            ... (linspace float numbers).
        sparsity: "linspace", "logspace", "unirandom", or "lograndom".
            Linearly spaced, logarithmically spaced, uniformly random,
            or log-uniformly random thresholds are generated
            between min LLR and max LLR.
    Returns:
        thresh: A Tensor with shape
            (num_thresh, time_steps, num classes, num classes).
            In each matrix,
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
    Remarks:
        - The threshold values are in [min |LLR| (!= 0), max |LLR|].
        - For reference, we show the Wald's approximation:
          If alpha is a float in (0, 0.5) (FPR) and
          beta is a float in (0, 0.5) (FNR),
          then thresh
          = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)].
    """
    _device = llrs.device

    llrs_shape = llrs.shape
    num_classes = llrs_shape[-1]

    # Remove 0 LLRs in off-diags
    tri = torch.ones_like(llrs)
    triu = torch.triu(tri)  # Upper triangular part.
    tril = torch.tril(tri)  # Lower triangular part.
    llrs -= 1e-12 * (triu - tril)

    # Calc non-zero max and min of |LLRs|
    llrs_abs = torch.abs(llrs)
    llrs_max = torch.max(llrs_abs)
    # max |LLRs|
    tmp = torch.eye(num_classes, device=_device) * llrs_max
    tmp = tmp.reshape([1, 1, num_classes, num_classes])
    llrs_min = torch.min(llrs_abs + tmp)
    # strictly positive (non-zero) minimum of LLRs

    assert llrs_min > 0.0

    # Single-valued threshold matrix
    if sparsity == "linspace":
        thresh = torch.linspace(
            llrs_min.item(), llrs_max.item(), num_thresh, device=_device
        )
        # (num thresh,)

    elif sparsity == "logspace":
        thresh = torch.exp(
            torch.linspace(
                torch.log(llrs_min).item(),
                torch.log(llrs_max).item(),
                num_thresh,
                device=_device,
            )
        )
        # (num thresh,)

    elif sparsity == "unirandom":
        thresh = torch.rand(num_thresh, device=_device)
        thresh = torch.sort(thresh)[0]
        thresh = ((llrs_max - llrs_min) * thresh) + llrs_min
        # (num thresh,), ascending order

    elif sparsity == "lograndom":
        thresh = torch.rand(num_thresh, device=_device)
        thresh = torch.sort(thresh)[0]
        thresh = torch.exp(
            ((torch.log(llrs_max) - torch.log(llrs_min))
             * thresh) + torch.log(llrs_min)
        )
        # (num thresh,). Ascending order.
    else:
        raise ValueError

    return thresh


def tile_constant_threshold(
    thresh: torch.Tensor, batch_size: int, time_steps: int, num_classes: int
) -> torch.Tensor:
    """ """
    _device = thresh.device
    num_thresh = thresh.shape[0]

    # convert to (num thresh, batch_size, time_steps, num cls, num cls)
    thresh = thresh.view(num_thresh, 1, 1, 1, 1)
    thresh = thresh.tile(1, batch_size, time_steps, num_classes, num_classes).to(
        _device
    )

    # diagonal all-zero, else one
    mask = 1 - torch.eye(num_classes, device=_device)

    thresh = mask * (thresh + 1e-11)
    # Avoids 0 threholds, which may occur
    # when logits for different classes have the same value,
    # e.g., 0, due to loss of significance.
    # This operation may cause sparsity of SAT curve
    # if llr_min is << 1e-11, but such a case is ignorable
    # in practice, according to my own experience.
    return thresh  # (num_thresh, time_steps, num classes, num classes).


def thresh_truncated_MSPRT(thresh: torch.Tensor) -> torch.Tensor:
    """
    Set the last threshold in the time dimension to zero, with a small value on off-diagonal elements.

    Args:
        thresh: A PyTorch Tensor with shape
            (num thresholds, time_steps, num class, num class) or
            (batch_size, num thresholds, time_steps, num class, num class).

    Returns:
        A PyTorch Tensor with the same shape as the input, with the last threshold in the
        time dimension set to zero (with a small value on off-diagonal elements).
    """
    ndim = len(thresh.shape)
    if ndim not in [4, 5]:
        raise ValueError("thresh must have 4 or 5 dimensions")

    _device = thresh.device
    num_classes = thresh.shape[-1]

    # diagonal all-zero, else one
    mask = 1 - torch.eye(num_classes, device=_device)

    last_thresh_shape = list(thresh.shape[:-3]) + [1, num_classes, num_classes]
    last_thresh = torch.zeros(last_thresh_shape, device=_device)
    last_thresh += mask * 1e-11

    return torch.cat([thresh[..., :-1, :, :], last_thresh], dim=-3)


def get_upper_triangle(scores_full):
    """
    Get the upper triangle of the scores matrix.

    Args:
        scores_full: A PyTorch Tensor with shape
            (batch_size, num thresholds, time_steps, num_classes, num_classes).
    Returns:
        scores_vec: A PyTorch Tensor with shape
            (batch_size, num thresholds, time_steps, kC2) where k is the num_classes.
    """
    assert len(scores_full.shape) == 5, "scores_full must have 5 dimensions"
    num_classes = scores_full.shape[-1]

    upper_triangle_indices = torch.triu_indices(
        num_classes, num_classes, offset=1)

    scores_vec = scores_full[
        :, :, :, upper_triangle_indices[0], upper_triangle_indices[1]
    ]

    return scores_vec


def hyperspherical_threshold(
    thresh: torch.Tensor,
    kappa: torch.Tensor,
) -> torch.Tensor:
    """
    Create a tapering threshold with hypersphere function.

    Args:
    - thresh (PyTorch Tensor): (num_thresh, batch_size, time_steps, num_classes, num_classes)
    - kappa (PyTorch Tensor): (num_thresh, batch_size, time_steps, 1)

    Returns:
    - tapering_thresh (PyTorch Tensor): (num_thresh, batch_size, time_steps, num_classes, num_classes)
    """
    assert len(kappa.shape) == 4 and kappa.shape[-1] == 1
    assert len(thresh.shape) == 5

    num_thresh, batch_size, time_steps, _, num_classes = thresh.shape
    _device = thresh.device

    # time steps
    t = torch.linspace(0, time_steps - 1, time_steps, device=_device)
    # expand to 5-dim
    t = t.view(1, 1, -1, 1, 1)

    # expand to 5-dim
    kappa = kappa.view(num_thresh, batch_size, time_steps, 1, 1)

    # Broadcasting and element-wise operations
    tapering_thresh = thresh * (1 - t / (time_steps - 1)) ** (torch.exp(kappa))

    # set diagonal to zero
    mask = 1 - torch.eye(num_classes, device=_device)
    tapering_thresh = mask * (tapering_thresh + 1e-11)

    return tapering_thresh


def check_diag_zeros_positive_offdiag(thresh_mtx: torch.Tensor) -> None:
    """
    Check if the diagonal elements of a threshold matrix are zero and
    the off-diagonal elements are positive.

    Args:
        thresh_mtx: A PyTorch Tensor with shape
            (num thresholds, time_steps, num class, num class) or
            (batch_size, num thresholds, time_steps, num class, num class).

    Raises:
        ValueError: If the diagonal elements of thresh_mtx are not 0.
        ValueError: If the off-diagonal elements of thresh_mtx are not positive.
    """
    num_classes = thresh_mtx.shape[-1]
    _device = thresh_mtx.device
    ndim = len(thresh_mtx.shape)

    if ndim not in [4, 5]:
        raise ValueError("thresh_mtx must have 4 or 5 dimensions")

    for i in range(num_classes):
        diag_mask = [slice(None)] * (ndim - 2) + [i, i]
        if not (thresh_mtx[tuple(diag_mask)] == 0).all():
            raise ValueError(
                "The diag elements of thresh_mtx must be 0."
                + f"\nNow thresh_mtx = {thresh_mtx}"
            )

    tmp = torch.eye(num_classes, device=_device)
    tmp = tmp.reshape(1, 1, *([1] * (ndim - 4)), num_classes, num_classes)
    tmp_th = thresh_mtx + tmp.expand_as(thresh_mtx)
    if not (tmp_th > 0).all():
        raise ValueError(
            "The off-diag elements of thresh_mtx must be positive."
            + f"\nNow thresh_mtx = {thresh_mtx}"
        )


# Function: Matrix SPRT (LLR, thresh -> confmx)
def truncated_MSPRT(llr_mtx, labels_concat, thresh_mtx):
    """Truncated Matrix-SPRT.
    Args:
        llr_mtx: A Tensor with shape
            (batch, time_steps, num classes, num classes).
            Anti-symmetric matrices.
        labels_concat: A Tensor with shape (batch,).
        thresh_mtx: A Tensor with shape
            (num thresholds, time_steps, num class, num class).
            Diag must be 0. Off diag must be strictly positive.
            To be checked in this function.
    Returns:
        confmx: A Tensor with shape (num thresh, classes, num classes).
            Confusion matrix.
        mht: A Tensor with shape (num thresh,). Mean hitting time as a whole.
        vht: A Tensor with shape (num thresh,). Variance of hitting times as a whole.
        trt: A Tensor with shape (num thresh,). Truncation rete as a whole.
    """
    """ check shape match, then values """
    """ care about exactly zero LLRs: Done """
    thresh_mtx_shape = thresh_mtx.shape
    num_thresh = thresh_mtx_shape[0]
    time_steps = thresh_mtx_shape[1]
    num_classes = thresh_mtx_shape[2]
    batch_size = llr_mtx.shape[0]

    # Sanity check of thresholds
    check_diag_zeros_positive_offdiag(thresh_mtx)
    thresh_mtx = thresh_mtx.to(torch.float32)

    # Reshape and calc scores
    llr_mtx = torch.unsqueeze(llr_mtx, dim=0)
    # (1, batch, time_steps, num cls, num cls)
    # to admit the num-thresh axis.
    thresh_mtx = torch.unsqueeze(thresh_mtx, dim=1)
    # (num thresh, 1, time_steps, num cls, num cls)
    # to admit the batch axis.
    tri = torch.ones_like(llr_mtx)
    triu = torch.triu(tri)  # Upper triangular part.
    tril = torch.tril(tri)  # Lower triangular part.
    llr_mtx -= 1e-10 * (triu - tril)
    # (1, batch, time_steps , num cls, num cls)
    # To avoid double hit due to the values exactly equal to zero
    # in scores or when doing truncation, LLRs of the last frame.
    scores = torch.min(llr_mtx - thresh_mtx, -1)
    # (num thresh, batch, time_steps, num cls)
    # Values are non-positive.
    """ assert 1: for each thresh, batch, and time_steps, 
                  the num of 0 is 0 or at most 1 in the last axis direction
        assert 2: values <= 0
    """

    # Calc all predictions and waits
    preds_all = torch.sign(scores) + 1
    # 0:wait, 1:hit (one-hot vector)
    # (num thresh, batch, time_steps, num cls)
    """assert actually one-hot"""

    # Calc truncation rate
    hit_or_wait_all_frames = 1 - preds_all  # wait=1, hit=0
    trt = torch.mean(torch.prod(hit_or_wait_all_frames, dim=(2, 3)), dim=1)
    # (num thresh,)

    if time_steps == 1:
        # Forced decision
        preds_last = torch.sign(torch.min(llr_mtx, -1)) + 1
        # 0: wait, 1: hit (one-hot vector)
        # (1, batch, time_steps=1, num cls)
        """assert check shape"""
        """assert check all the data points in the batch is are now one-hot vectors."""
        preds_last = preds_last.repeat(num_thresh, 1, 1, 1)
        preds_all_trunc = preds_last
        # (num thresh, batch, 1, num cls)

        # Calc hitting times
        mht = torch.tensor(1.0, dtype=torch.float32)
        vht = torch.tensor(0.0, dtype=torch.float32)

        # Calc confusion matrices
        preds = preds_all_trunc[:, :, 0, :]
        # (num thresh, batch, 1, num cls): one-hot vectors

        labels_oh = torch.nn.functional.one_hot(
            labels_concat, num_classes)  # dim=1
        # (batch, num cls)
        labels_oh = torch.unsqueeze(labels_oh, dim=0)
        labels_oh = labels_oh.repeat(num_thresh, 1, 1)
        # (num thresh, batch, num cls)

        preds = torch.unsqueeze(preds, dim=-2)
        labels_oh = torch.unsqueeze(labels_oh, dim=-1)
        confmx = torch.sum(labels_oh * preds, dim=1).to(torch.int32)
        # (num thresh, num cls, num cls): label axis x pred axis

    else:
        # Forced decision
        preds_last = torch.sign(torch.min(llr_mtx[:, :, -1, :, :], -1)) + 1
        # 0: wait, 1: hit (one-hot vector)
        # (1, batch, num cls)
        """assert check shape"""
        """assert check all the data points in the batch is are now one-hot vectors."""
        preds_last = torch.unsqueeze(preds_last, 2)
        # (1, batch, 1, num cls)
        preds_last = preds_last.repeat(num_thresh, 1, 1, 1)
        # (num thresh, batch, 1, num cls)
        preds_all_trunc = torch.cat(
            [preds_all[:, :, :-1, :], preds_last], dim=2)
        # (num thresh, batch, time_steps - 1, num cls)
        # + (num thresh, batch, 1, num cls)
        # = (num thresh, batch, time_steps, num cls)
        # Now, preds_all_trunc[i, j, t, :] for fixed i and j is
        # a one-hot vector for t = time_steps - 1
        # and
        # filled with 0 or a one-hot vector for t != time_steps - 1.
        """ assert: check this """

        # Calc mean hitting time
        mask = torch.tensor(
            [i + 1 for i in range(time_steps)][::-1], dtype=torch.float32
        )
        mask = mask.repeat(num_thresh * batch_size * num_classes)
        mask = mask.reshape(num_thresh, batch_size, num_classes, time_steps)
        mask = mask.permute(0, 1, 3, 2)
        masked = preds_all_trunc * mask
        # (num thresh, batch, time_steps, num cls)
        hitidx = torch.max(masked, dim=2)
        # (num thresh, batch, num cls)
        hittimes = time_steps - torch.max(hitidx, dim=2) + 1
        # (num thresh, batch)
        mht = torch.mean(hittimes, dim=1)
        vht = torch.var(hittimes, dim=1)
        # (num thresh,)

        # Calc confusion matrix
        preds = torch.argmax(hitidx, dim=2)
        # (num thresh, batch,)
        preds = torch.nn.functional.one_hot(preds, num_classes)  # dim=2
        # (num thresh, batch, num cls)

        labels_oh = torch.nn.functional.one_hot(
            labels_concat, num_classes)  # dim=1
        # (batch, num cls)
        labels_oh = torch.unsqueeze(labels_oh, dim=0)
        labels_oh = labels_oh.repeat(num_thresh, 1, 1)
        # (num thresh, batch, num cls)

        preds = torch.unsqueeze(preds, dim=-2)
        labels_oh = torch.unsqueeze(labels_oh, dim=-1)
        confmx = torch.sum(labels_oh * preds, dim=1).to(torch.int32)
        # (num thresh, num cls, num cls): label axis x pred axis

    return confmx, mht, vht, trt


# Functions: LLR -> confmx
def llr_sequential_confmx(llrs, labels_concat):
    """For optuna and NP test.
        Calculate the frame-by-frame confusion matrices
        based on the log-likelihood ratios.
    Args:
        llrs: A Tensor
            with shape (batch, time_steps, num classes, num classes).
        labels_concat: A non-one-hot label Tensor with shape (batch,).
            This is the output from
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Returns:
        seqconfmx_llr: A Tensor with shape (time_steps, num classes, num classes).
            The sequential confusion matrices of framewise LLRs with thresh=0.
    """
    llrs_shape = llrs.shape
    time_steps = llrs_shape[1]
    num_classes = llrs_shape[2]

    # To avoid double hit due to the values exactly equal to zero in LLRs.
    tri = torch.ones_like(llrs)
    triu = torch.triu(tri)  # Upper triangular part.
    tril = torch.tril(tri)  # Lower triangular part.
    llrs -= 1e-12 * (triu - tril)
    # (batch, time_steps , num cls, num cls)

    # Forced decision
    preds = torch.sign(torch.min(llrs, -1)[0]) + 1
    # 0: wait, 1: hit (one-hot vector)
    # (batch, time_steps, num cls)

    # Calc confusion matrices
    labels_oh = torch.nn.functional.one_hot(
        labels_concat, num_classes)  # dim=1
    # (batch, num cls)
    labels_oh = torch.unsqueeze(labels_oh, dim=1)
    labels_oh = labels_oh.repeat(1, time_steps, 1)
    # (batch, time_steps, num cls)

    preds = torch.unsqueeze(preds, dim=-2)
    labels_oh = torch.unsqueeze(labels_oh, dim=-1)
    seqconfmx = torch.sum(labels_oh * preds, dim=0).to(torch.int32)
    # (time_steps, num cls, num cls): label axis x pred axis

    return seqconfmx


def NP_test(llrs, labels, length=None):
    """Neyman-Pearson Test.
    Args:
        llrs: A Tensor
            with shape (batch, time_steps, num classes, num classes).
        labels: A Tensor with shape (batch,).
        length: A integer or None. If this is not None,
            it shouldb be 1 <= length <= time_steps,
            and scores after the length-th frame are thrown away.
    Returns:
        A Tensor with shape (num classes, num classes) if length is not None,
        else (time_steps, num classes, num classes).
    Remark:
        - Currently, only threshold = 0 is supported.
        - Note that the NP test uses the likelihood ratio of the sequence,
          not that of a frame in the sequence;
          that's why the input arg must be LLRs, not arbitrary scores.
    """
    if length is None:
        return llr_sequential_confmx(llrs, labels)

    else:
        assert 1 <= length <= llrs.shape[1]
        return llr_sequential_confmx(llrs, labels)[length - 1]


# Functions: Utils
def confmx_to_bac(confmx):
    """confx -> BAC
    Args:
        confmx: A Tensor with shape (num classes, num classes)
    Returns:
        bac: A scalar Tensor, balanced accuracy.
            Independend of num of classes (>= 2).
    """
    dict_metrics = confmx_to_metrics(confmx)
    bac = dict_metrics["SNS"]["macro"]
    return bac


def seqconfmx_to_list_metrics(seqconfmx):
    """Pair function A: sequential confmx -> metrics
    Transforms a Tensor of confusion matirces with shape
    (LEN, num classes, num classes) to a list (with length LEN)
    of dictionaries of metrics, where LEN is undetermined."""
    sequence_length = seqconfmx.shape[0]
    list_metrics = [None] * sequence_length
    for iter_idx in range(sequence_length):
        list_metrics[iter_idx] = confmx_to_metrics(seqconfmx[iter_idx])

    return list_metrics


def list_metrics_to_list_bac(list_metrics):
    """Pair function B: metrics -> BACs
        Input: [confmx_to_metrics(...), confmx_to_metrics(...), ...].
    Arg:
        list_metrics: A list of dictionaries.
    Return:
        list_bacs: A list of floats with the same length as list_metric's.
    """
    list_bacs = [None] * len(list_metrics)
    for iter_idx, iter_dict in enumerate(list_metrics):
        bac = iter_dict["SNS"]["macro"]
        list_bacs[iter_idx] = bac

    return list_bacs


# Functions: other statistical tests
def binary_np_test(llrs, labels, length=None, thresh=0.0):
    """Neyman-Pearson Test.
    Args:
        llrs: A Tensor with shape (batch, time_steps).
        labels: A Tensor with shape (batch,).
        length: A integer or None. If this is not None, it shouldb be 1 <= length <= time_steps, and scores after the length-th frame are thrown away.
        thresh: A float.
    Returns:
        confmx: A Tensor with shape (2, 2). Binary classification is assumed.
    Remark:
        - Note that the NP test uses the likelihood ratio of a sequence, not a frame; that's why the input arg is LLRs, not scores, which is not equivalent to LLRs in general.
    """
    llrs_shape = llrs.shape
    time_steps = llrs_shape[1]
    if not (length is None):
        assert 1 <= length <= time_steps

    # Calc predictions
    llr = llrs[:, length - 1] - thresh
    preds = torch.round(torch.sigmoid(llr))

    # Calc confusion matrix
    confmx = confusion_matrix(labels, preds, num_classes=2)

    return confmx


def binary_avescr_test(scores, labels, length=None, thresh=0.0):
    """Score Average Test.
    Args:
        scores: A Tensor with shape (batch, time_steps).
        labels: A Tensor with shape (batch,).
        length: A integer or None. If this is not None,
            it shouldb be 1 <= length <= time_steps,
            and scores after the length-th frame are thrown away.
        thresh: A float.
    Returns:
        confmx: A Tensor with shape (2, 2). Binary classification is assumed.
    Remark:
        - If scores are drawn from non-i.i.d. distribution,
          e.g., LSTM sequential outputs,
          then temporally averaging scores may be detrimental.
    """
    scores_shape = scores.shape
    time_steps = scores_shape[1]
    if not (length is None):
        assert 1 <= length <= time_steps
        scores = scores[:, :length]

    # Calc predictions
    score = torch.mean(scores, dim=1) - thresh
    preds = torch.round(torch.sigmoid(score))

    # Calc confusion matrix
    confmx = confusion_matrix(labels, preds, num_classes=2)

    return confmx
