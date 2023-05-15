import os
import pdb
from typing import Callable, Tuple
from termcolor import colored
import math
import pyfiglet
from loguru import logger
import pyfiglet
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
from datasets.data_processing import sequential_concat_logits
from datasets.data_processing import sequential_slice_data
from utils.misc import extract_params_from_config, ErrorHandler
from utils.hyperparameter_tuning import format_model_name
from utils.performance_metrics import (
    check_diag_zeros_positive_offdiag,
    thresh_truncated_MSPRT,
    threshold_generator,
    tile_constant_threshold,
)
from models.losses import compute_llrs, calc_scores, restore_lost_significance


def import_model(config, device, tb_writer=None):
    """
    Select LSTM- or TRANSFORMER- based temporal integrator.
    If IS_RESUME=True, load a model from saved checkpoint.

    Args:
    - config (dict): dictionary that specifies model backbone.
    - device (torch.device): specifies computation device to which data and model are moved.
    - tb_writer (torch.utils.tensorboard.SummaryWriter): optional, provide to add a network graph to TensorBoard.

    Returns:
    - model (PyTorch model)
    """

    # check if necessary parameters are defined in the config file
    requirements = set(["MODEL_BACKBONE", "IS_COMPILE", "MODE"])
    conf = extract_params_from_config(requirements, config)

    model_name = format_model_name(conf.model_backbone)

    if "LSTM" in model_name:
        model = B2BsqrtTANDEM(config)
    elif "TRANSFORMER" in model_name:
        model = TANDEMformer(config)
    else:
        raise ValueError(f"Unsupported model {conf.model_backbone} found!")

    # print model layers and #parameters
    model.summarize(tb_writer, device)
    # Note that model.to(device) is actually redundant here because
    # torchinfo.summary moves the model to device. Keeping it for educational purpose.
    model.to(device)
    logger.info("model moved onto: " + colored(f"{device}.", "yellow"))

    # compile for optimization
    if conf.is_compile:
        logger.info("compiled the model for optimization.")
        return torch.compile(model, mode=conf.mode)
    else:
        return model


def load_pretrained_states(model, optimizer, config) -> None:
    # check if necessary parameters are defined in the config file
    requirements = set(["IS_RESUME", "PATH_RESUME"])
    conf = extract_params_from_config(requirements, config)

    # If resume
    if conf.is_resume:
        assert os.path.exists(
            conf.path_resume
        ), f"Checkpoint does not exist: {conf.path_resume}"

        checkpoint = torch.load(conf.path_resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"model and optimizer states were loaded from: {conf.path_resume}")
    else:
        return


# custom LSTM to accept new activation functions
class LSTM(nn.Module):
    """
    PyTorch implementation of a Long Short-Term Memory (LSTM) module.

    Args:
        feat_dim (int): Number of input features.
        hidden_size (int): Number of hidden units in the LSTM layer.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, time_steps, feat_dim).

    Outputs:
        output (torch.Tensor): Output tensor of shape (batch_size, time_steps, hidden_size).

    References:
        - Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
    """

    def __init__(
        self,
        feat_dim,
        hidden_size,
        input_activation=nn.Sigmoid(),
        output_activation=nn.Tanh(),
    ):
        """
        Initializes the LSTM layer.

        Args:
            feat_dim (int): Number of input features.
            hidden_size (int): Number of hidden units in the LSTM layer.
            input_activation (callable): Input activation function a.k.a. recurrent activation function (default: torch.sigmoid).
            output_activation (callable): Output activation function (default: torch.tanh).
        """
        super(LSTM, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.input_activation = input_activation
        self.output_activation = output_activation

        # Initialize weight matrices and bias vectors
        self.W_i = nn.Parameter(torch.Tensor(feat_dim, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_f = nn.Parameter(torch.Tensor(feat_dim, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c = nn.Parameter(torch.Tensor(feat_dim, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(feat_dim, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights and biases of the LSTM layer using Xavier initialization.
        """
        # Initialize weights and biases using Xavier initialization
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.U_i)
        nn.init.constant_(self.b_i, 0)

        nn.init.xavier_uniform_(self.W_f)
        nn.init.xavier_uniform_(self.U_f)
        nn.init.constant_(self.b_f, 0)

        nn.init.xavier_uniform_(self.W_c)
        nn.init.xavier_uniform_(self.U_c)
        nn.init.constant_(self.b_c, 0)

        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.U_o)
        nn.init.constant_(self.b_o, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Passes the input tensor through the LSTM layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, feat_dim).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, time_steps, hidden_size).
        """
        # Initialize hidden and cell state tensors
        batch_size, time_steps, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Iterate over time steps
        output = []
        for t in range(time_steps):
            x_t = x[:, t, :]

            # Input gate
            i_t = self.input_activation(
                torch.matmul(x_t, self.W_i) + torch.matmul(h_t, self.U_i) + self.b_i
            )

            # Forget gate
            f_t = self.input_activation(
                torch.matmul(x_t, self.W_f) + torch.matmul(h_t, self.U_f) + self.b_f
            )

            # Output gate
            o_t = self.input_activation(
                torch.matmul(x_t, self.W_o) + torch.matmul(h_t, self.U_o) + self.b_o
            )

            # Cell state
            c_tilda_t = self.output_activation(
                torch.matmul(x_t, self.W_c) + torch.matmul(h_t, self.U_c) + self.b_c
            )
            c_t = f_t * c_t + i_t * c_tilda_t

            h_t = o_t * self.output_activation(c_t)

            output.append(h_t.unsqueeze(1))

        # Stack output tensors and return
        output = torch.cat(output, dim=1)
        return output


class PositionalEncoding(nn.Module):
    """
    A PyTorch module that applies a learnable positional encoding to the input tensor.
    Adapted from the official document:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Attributes:
        dropout: nn.Dropout, the dropout layer applied to the output tensor.
        feat_dim: int, the number of features in the input tensor.
        pe: torch.Tensor, the positional encoding tensor.

    Methods:
        forward: Applies the positional encoding to the input tensor and returns the output.

    """

    def __init__(self, feat_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes a new PositionalEncoding module.

        Args:
            feat_dim: int, the number of features in the input tensor.
            dropout: float, the dropout probability for the output tensor.
            max_len: int, the maximum length of the input sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.feat_dim = feat_dim
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feat_dim, 2) * (-math.log(10000.0) / feat_dim)
        )
        pe = torch.zeros(max_len, 1, feat_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # register the pe tensor as a buffer, which means that it will be included in the model's
        # state dictionary but won't be considered as a model parameter that needs to be optimized
        # during training.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, batch_first: bool = True) -> Tensor:
        """
        Applies the positional encoding to the input tensor and returns the output.

        Args:
            x: Tensor, the input tensor with shape [time_steps, batch_size, embedding_dim] if batch_first=False
                   else [batch_size, time_steps, embedding_dim] if batch_first=True
            batch_first: bool, whether the input tensor has batch_size as its first dimension.

        Returns:
            Tensor, the output tensor with the positional encoding added.
        """
        assert (
            x.shape[-1] == self.feat_dim
        ), f"Expected last dimension of x to be {self.feat_dim}, but got {x.shape[-1]} instead"
        assert (
            x.ndim == 3
        ), f"Expecting a Tensor with dimensions batch_size, time_steps, and embedding_dim but got {x.shape=}."
        x = x.permute(1, 0, 2) if batch_first else x
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2) if batch_first else x


# custom activation functions
def b2bsqrt(x, alpha=1.0):
    alpha = torch.tensor(alpha)
    return torch.sign(x) * (torch.sqrt(alpha + torch.abs(x)) - torch.sqrt(alpha))


def b2bsqrtv2(x, alpha=1.0, beta=0.01):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    # add a linear component for anti-gradient vanishing
    return torch.sign(x) * (
        torch.sqrt(alpha + torch.abs(x)) + beta * torch.abs(x) - torch.sqrt(alpha)
    )


def b2bcbrt(x, alpha=1.0, gamma=2.0):
    alpha = torch.tensor(alpha)
    gamma = torch.tensor(gamma)
    return torch.sign(x) * (
        torch.pow(alpha + torch.abs(x), 2 / 3) - torch.pow(alpha, 2 / 3)
    )


def b2bexp(x, alpha=0.03, beta=0.1, tau=1000):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    tau = torch.tensor(tau)
    return torch.sign(x) * (
        -alpha * torch.exp(-torch.abs(x) / tau) + beta * torch.abs(x) + alpha
    )


def tanhplus(x, alpha=10.0, beta=0.02, tau=100.0):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    tau = torch.tensor(tau)
    # add a linear component for anti-gradient vanishing
    return alpha * torch.tanh(x / tau) + beta * x


def dullrelu(x, beta=0.05):
    beta = torch.tensor(beta)
    return beta * torch.maximum(torch.zeros_like(x), x)


def b2blog(x):
    return torch.sign(x) * (torch.log(1 + torch.abs(x)))


activations = {
    "b2bsqrt": b2bsqrt,
    "b2bsqrtv2": b2bsqrtv2,
    "b2bcbrt": b2bcbrt,
    "b2bexp": b2bexp,
    "tanhplus": tanhplus,
    "dullrelu": dullrelu,
    "b2blog": b2blog,
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "sine": torch.sin,
}


def get_activation(activation: str) -> Callable:
    """
    Returns the activation function specified by the argument `activation`.

    Parameters:
    - activation (str): The name of the activation function. Supported values are:
    'b2bsqrt', 'b2bcbrt', 'b2bexp', 'tanhplus', 'dullrelu', 'B2Blog', 'relu', 'sigmoid', and 'tanh'.

    Returns:
    - Callable: The specified activation function.

    Raises:
        ValueError: If the specified activation function is not supported.
    """
    if activation.lower() not in activations:
        raise ValueError(f"{activation} activation function is not supported!")
    return activations[activation.lower()]


class BaseTANDEM(nn.Module):
    """
    Base PyTorch model that defines utility functions that are common to all models.

    This class inherits from `nn.Module`. All TANDEM models should inherit from this class
    instead of directly inheriting from `nn.Module` to make use of the `summarize` function.
    """

    def __init__(self, config, feature_size):
        super().__init__()

        # check if necessary parameters are defined in the config file
        requirements = set(
            [
                "NUM_CLASSES",
                "TIME_STEPS",
                "FEAT_DIM",
                "ORDER_SPRT",
                "ACTIVATION_FC",
                "BATCH_SIZE",
                "OBLIVIOUS",
                "NUM_THRESH",
                "SPARSITY",
                "DROPOUT",
                "IS_NORMALIZE",
                "IS_ADAPTIVE_LOSS",
                "IS_POSITIONAL_ENCODING",
                "IS_TRAINABLE_ENCODING",
                "OPTIMIZATION_CANDIDATES",
            ]
        )
        conf = extract_params_from_config(requirements, config)

        self.num_classes = conf.num_classes
        self.time_steps = conf.time_steps
        self.feat_dim = conf.feat_dim
        self.order_sprt = conf.order_sprt
        self.batch_size = conf.batch_size
        self.oblivious = conf.oblivious
        self.dropout = nn.Dropout(p=conf.dropout)
        self.num_thresh = conf.num_thresh
        self.sparsity = conf.sparsity

        # number of elements in the upper triangle (kC2 where k is the num_classes)
        self.vec_dim = self.num_classes * (self.num_classes - 1) // 2

        self.is_normalize = conf.is_normalize
        if self.is_normalize:
            self.layer_norm = nn.LayerNorm(feature_size)

        self.is_positional_encoding = conf.is_positional_encoding
        self.is_trainable_encoding = conf.is_trainable_encoding
        if self.is_trainable_encoding:
            self.trainable_pos_encoding = nn.Parameter(
                torch.randn(self.order_sprt + 1, self.feat_dim)
            )
        else:
            self.pos_encoder = PositionalEncoding(conf.feat_dim, conf.dropout)

        # for debug. delete this later
        self.pos_encoder = PositionalEncoding(conf.feat_dim, conf.dropout)

        if conf.is_adaptive_loss:
            self.adaptive_loss_weights = nn.Parameter(
                torch.randn(len(conf.optimization_candidates))
            )

        # FC layers
        self.activation_logit = get_activation(conf.activation_fc)
        self.fc_logits = nn.Linear(
            in_features=feature_size, out_features=conf.num_classes, bias=True
        )

    def normalized_sum_pool(self, x: Tensor, max_time_steps: int) -> Tensor:
        """
        NSP layer for precise sequential density ratio estimation.
        (Ebihara+, ICASSP 2023)

        Args:
        -x (Tensor): size (effective_batch_size, effective_time_steps, feat_dim)
         effective_time_steps \in {1, max_time_steps}
        -max_time_steps (int): maximum time_steps to consider. e.g) order_sprt + 1.

        Return:
        torch.Tensor of the size size as x.
        """
        return torch.sum(x, dim=1) / max_time_steps

    def features_to_logits(self, x, time_steps):
        """
        Args:
            x: A Tensor. Output of network with shape=(batch, time_steps, self.width_lstm).
        Returns:
        - logit_stack (PyTorch Tensor): (batch, time_steps, num_classes)
        """
        logit_stack = []
        for i in range(time_steps):
            y = x[:, i, :]
            if self.is_normalize:
                y = self.layer_norm(y)
            # final FC layer
            y_class = self.activation_logit(y)
            y_class = self.fc_logits(y)
            logit_stack.append(y_class)

        logit_stack = torch.stack(logit_stack, dim=1)
        return logit_stack

    def logits_to_llrs(self, logits: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        Args:
            logits (PyTorch Tensor): (effective_batch_size, effective_time_steps, num_classes)
            time_steps (int): full-length time_steps >= effective_time_steps.


        """
        logits_concat = sequential_concat_logits(logits, time_steps=time_steps)
        llrs = compute_llrs(logits_concat, oblivious=self.oblivious)

        return llrs

    def markov_time_slicer(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
        - inputs (PyTorch Tensor): (batch_size, time_steps, feat_dim)

        Returns:
        - x_slice (PyTorch Tensor): (batch_size * (time_steps - order_sprt), order_sprt + 1, feat_dim)
        - eff_batch_size (int): effective batch size
        - eff_time_steps (int): effective time steps
        """

        x_slice = sequential_slice_data(inputs, self.order_sprt)

        shapes = x_slice.shape
        eff_batch_size = shapes[0]
        eff_time_steps = shapes[1]

        return x_slice, eff_batch_size, eff_time_steps

    def create_threshold(self, llrs: torch.Tensor) -> torch.Tensor:
        """ """
        thresh = threshold_generator(
            llrs, self.num_thresh, self.sparsity
        )  # (num_thresh,)

        thresh = tile_constant_threshold(
            thresh, self.batch_size, self.time_steps, self.num_classes
        )

        thresh = thresh_truncated_MSPRT(thresh)

        check_diag_zeros_positive_offdiag(thresh)

        return thresh

    def distance_between_llrs_and_thresholds(
        self, llrs: torch.Tensor, thresh: torch.Tensor
    ) -> torch.Tensor:
        """ """
        # Change LLRs and threshold scales
        # llrs, thresh = normalize_LLR_thresh_scale(llrs, thresh)
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

        # scores_full: llr_mtx - thresh_mtx,
        # size:(num_thresh, batch_size, time_steps, num_classes, num_classes)
        scores_full = calc_scores(llrs, thresh)

        return scores_full

    def features_to_final_scores(
        self, outputs, effective_time_steps
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """
        # Make logits
        logits = self.features_to_logits(outputs, time_steps=effective_time_steps)

        # compute log-likelihood ratio. Also convert to full-time steps
        llrs = self.logits_to_llrs(logits, self.time_steps)

        thresh = self.create_threshold(llrs)

        scores_full = self.distance_between_llrs_and_thresholds(llrs, thresh)

        return llrs, logits, thresh, scores_full

    def summarize(self, tb_writer, device):
        """ """
        logger.info(
            "\n" + pyfiglet.figlet_format("SPRT-    \nTANDEM", font="banner3-D")
        )
        logger.info(colored("official PyTorch implementation.", "blue"))

        self.eval()

        example_input = torch.randn(self.batch_size, self.time_steps, self.feat_dim).to(
            device
        )

        with torch.no_grad():
            # summary table by torchinfo
            # caution: this moves the model to the device
            sum_str = str(summary(self, example_input.shape, device=device, verbose=0))

            example_output = self.forward(example_input)

            # export to tensorboard
            if tb_writer:
                tb_writer.add_graph(self, example_input)
        example_output = (
            example_output[0] if isinstance(example_output, tuple) else example_output
        )

        logger.info("Network summary:\n" + sum_str)
        logger.info(f"Example input shape: {example_input.shape}")
        logger.info(f"Example output shape: {example_output.shape}")


class B2BsqrtTANDEM(BaseTANDEM):
    def __init__(self, config):
        # check if necessary parameters are defined in the config file
        requirements = set(
            [
                "WIDTH_LSTM",
                "ACTIVATION_OUTPUT",
                "ACTIVATION_INPUT",
            ]
        )
        conf = extract_params_from_config(requirements, config)

        super().__init__(config, feature_size=conf.width_lstm)

        # Parameters
        self.width_lstm = conf.width_lstm

        # activation functions
        self.activation_output = get_activation(conf.activation_output)
        self.activation_input = get_activation(conf.activation_input)

        self.rnn = LSTM(
            feat_dim=self.feat_dim,
            hidden_size=self.width_lstm,
            input_activation=self.activation_input,
            output_activation=self.activation_output,
        )

    def forward(self, inputs):
        """Calc logits.
        Args:
            inputs: A Tensor with shape=(batch, time_steps, feature dimension). E.g. (128, 20, 784) for nosaic MNIST.
            training: A boolean. Training flag used in BatchNormalization and dropout.
        Returns:
            outputs: A Tensor with shape=(batch, time_steps, num_classes).
        """
        # get sequence fragments 'x_slice', effective batch size, and effective time steps
        x_slice, eff_batch_size, eff_time_steps = self.markov_time_slicer(inputs)

        # Positinal encoding
        if self.is_positional_encoding:
            if self.is_trainable_encoding:
                final_inputs = x_slice + self.trainable_pos_encoding.unsqueeze(
                    0
                ).repeat(eff_batch_size, 1, 1)
            else:
                # (batch_size, time_steps, feat_dim)
                final_inputs = self.pos_encoder(x_slice, batch_first=True)
        else:
            final_inputs = x_slice
        # Feedforward
        # (batch_size, (effective) time_steps = order_sprt + 1, width_lstm)
        outputs = self.rnn(final_inputs)

        return self.features_to_final_scores(outputs, eff_time_steps)


class TANDEMformer(BaseTANDEM):
    """
    Transformer with the Normalized Summation Pooling (NSP) layer for precise SDRE.
    """

    def __init__(self, config):
        # check if necessary parameters are defined in the config file
        requirements = set(
            ["NUM_BLOCKS", "NUM_HEADS", "FF_DIM", "MLP_UNITS", "DROPOUT", "FEAT_DIM"]
        )
        conf = extract_params_from_config(requirements, config)

        super().__init__(config, feature_size=conf.mlp_units)

        self.decoder = nn.Linear(conf.feat_dim, conf.mlp_units)

        # main Transformer
        encoder_layers = TransformerEncoderLayer(
            conf.feat_dim,
            conf.num_heads,
            dim_feedforward=conf.ff_dim,
            dropout=conf.dropout,
            activation=self.activation_logit,
            batch_first=True,
            norm_first=False,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=conf.num_blocks
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.fc_logits.bias.data.zero_()
        self.fc_logits.weight.data.uniform_(-initrange, initrange)

    def transformer_classifier(self, inputs, max_time_steps):
        # pooling in time dimension: NSP layer
        inputs = self.normalized_sum_pool(inputs, max_time_steps)
        inputs = self.decoder(inputs)
        inputs = self.dropout(inputs)
        return inputs

    def meta_classifier(self, inputs, max_time_steps):
        # pooling in time dimension: NSP layer
        inputs = self.normalized_sum_pool(inputs, max_time_steps)
        inputs = self.dropout(inputs)
        return inputs

    def metanet(self, inputs: Tensor):
        """
        inputs: (num_thresh * batch_size, time_steps, self.vec_dim)
        """
        outputs_pool = []
        for i in range(self.time_steps):
            targinputs = inputs[:, : i + 1, :]

            # mix the input matrix
            targoutputs = self.meta_encoder(targinputs)

            outputs_pool.append(
                self.meta_classifier(targoutputs, max_time_steps=self.time_steps)
            )
        # convert from (num_thresh * batch_size, time_steps, 1) to
        outputs = torch.stack(outputs_pool, dim=1)

        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        # get sequence fragments 'x_slice', effective batch size, and effective time steps
        x_slice, eff_batch_size, eff_time_steps = self.markov_time_slicer(inputs)

        outputs_pool = []
        for i in range(eff_time_steps):
            targinputs = x_slice[:, : i + 1, :]

            # Positinal encoding
            if self.is_positional_encoding:
                if self.is_trainable_encoding:
                    final_inputs = targinputs + self.trainable_pos_encoding.unsqueeze(
                        0
                    )[:, : i + 1, :].repeat(eff_batch_size, 1, 1)

                else:
                    final_inputs = self.pos_encoder(targinputs, batch_first=True)
            else:
                final_inputs = targinputs
            # mix the input matrix
            targoutputs = self.transformer_encoder(final_inputs)

            outputs_pool.append(
                self.transformer_classifier(targoutputs, max_time_steps=eff_time_steps)
            )
        # (batch_size, time_steps, mlp_feats)
        outputs = torch.stack(outputs_pool, dim=1)

        return self.features_to_final_scores(outputs, eff_time_steps)
