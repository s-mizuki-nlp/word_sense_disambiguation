#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Tuple, Optional, Dict
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L

class _SenseCodeBaseLoss(L._Loss):

    def __init__(self, distance_metric: str,
                 size_average=None, reduce=None, reduction='mean'):

        super(_SenseCodeBaseLoss, self).__init__(size_average, reduce, reduction)

        self._distance_metric = distance_metric
        if distance_metric == "mse":
            self._func_distance = self._mse
        elif distance_metric == "focal-mse":
            self._func_distance = self._focal_mse
        elif distance_metric == "mae":
            self._func_distance = self._mae
        elif distance_metric == "focal-mae":
            self._func_distance = self._focal_mae
        elif distance_metric == "hinge":
            self._func_distance = self._hinge_distance
        elif distance_metric == "binary-cross-entropy":
            self._func_distance = self._bce
        else:
            raise AttributeError(f"unsupported distance metric was specified: {distance_metric}")

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    @property
    def scale(self):
        return 1.0

    def _mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(u, v, reduction=self.reduction)

    def _focal_mse(self, u, v) -> torch.Tensor:
        errors = (u - v)**2
        weights = errors / (errors.sum() + 1E-7)
        losses = weights * errors
        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses

    def _mae(self, u, v) -> torch.Tensor:
        return F.l1_loss(u, v, reduction=self.reduction)

    def _focal_mae(self, u, v) -> torch.Tensor:
        errors = torch.abs(u - v)
        weights = errors / (errors.sum() + 1E-7)
        losses = weights * errors
        if self.reduction == "mean":
            return torch.mean(losses)
        elif self.reduction == "sum":
            return torch.sum(losses)
        elif self.reduction == "none":
            return losses

    def _hinge_distance(self, y_pred, y_true) -> torch.Tensor:
        hinge_loss = F.relu(y_pred - y_true)
        if self.reduction == "mean":
            return torch.mean(hinge_loss)
        elif self.reduction == "sum":
            return torch.sum(hinge_loss)
        elif self.reduction == "none":
            return hinge_loss
        else:
            raise NotImplementedError(f"unsupported reduction method was specified: {self.reduction}")

    def _bce(self, u, v) -> torch.Tensor:
        return F.binary_cross_entropy(u, v, reduction=self.reduction)

    def _intensity_to_probability(self, t_intensity):
        return torch.exp(self._intensity_to_log_probability(t_intensity))

    def _intensity_to_log_probability(self, t_intensity, eps=1E-15):
        # t_intensity can be either one or two dimensional tensor.
        dtype, device = self._dtype_and_device(t_intensity)
        pad_shape = t_intensity.shape[:-1] + (1,)

        t_pad_begin = torch.zeros(pad_shape, dtype=dtype, device=device)
        t_pad_end = torch.ones(pad_shape, dtype=dtype, device=device)

        t_nonbreak_probs = 1.0 - torch.cat((t_pad_begin, t_intensity), dim=-1)
        t_break_probs = torch.cat((t_intensity, t_pad_end), dim=-1)

        t_log_prob = torch.log(t_break_probs + eps) + torch.cumsum( torch.log(t_nonbreak_probs + eps), dim=-1 )

        return t_log_prob

    def calc_soft_code_length(self, t_prob_c: torch.Tensor):
        log_length = self.calc_log_soft_code_length(t_prob_c)
        return torch.exp(log_length)

    def calc_log_soft_code_length(self, t_prob_c: torch.Tensor):
        dtype, device = self._dtype_and_device(t_prob_c)
        n_digits, n_ary = t_prob_c.shape[1:]

        t_idx_zero = torch.tensor(0, device=device)
        t_p_c_zero = torch.index_select(t_prob_c, dim=-1, index=t_idx_zero).squeeze()

        t_log_prob_break = self._intensity_to_log_probability(t_p_c_zero)
        idx_nonzero = torch.tensor(range(1, n_digits+1), device=device)
        t_log_prob_break_nonzero = torch.index_select(t_log_prob_break, dim=-1, index=idx_nonzero)
        t_at_n = torch.arange(1, n_digits+1, dtype=dtype, device=device).unsqueeze(dim=0)

        t_log_length = torch.logsumexp(torch.log(t_at_n) + t_log_prob_break_nonzero, dim=-1)
        return t_log_length

    @staticmethod
    def _manual_reduction(losses: torch.Tensor, reduction: str):
        if reduction == "sum":
            loss = torch.sum(losses)
        elif reduction.endswith("mean"):
            loss = torch.mean(losses)
        elif reduction == "none":
            loss = losses
        return loss

    def forward(self, u: torch.Tensor, v: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """
        apply pre-specified distance function and reduction within mini-batch.

        @param u: 1D tensor. (n_batch,)
        @param v: 1D tensor. (n_batch,)
        @param weight: Optional 1D tensor which is used for sample-wise weight. (n_batch, )
        @return: loss value (scalar when 'reduction="mean"')
        """
        if weight is None:
            return self._func_distance(u, v)
        else:
            n_sample = weight.shape[0]
            # normalize weights
            weight = n_sample * weight / (torch.sum(weight) + 1E-7)
            # apply manual reduction
            _reduction = self.reduction
            self.reduction = "none"
            losses = self._func_distance(u, v) * weight
            loss = self._manual_reduction(losses, reduction=_reduction)
            self.reduction = _reduction
            return loss


class HyponymyScoreLoss(_SenseCodeBaseLoss):

    def __init__(self, normalize_by_ground_truth_code_length: bool = False,
                 log_scale: bool = False,
                 weight_by_ground_truth_code_length: bool = False,
                 margin_on_code_length_penalty: float = 0.0,
                 distance_metric: str = "mse",
                 label_smoothing_factor: Optional[float] = None,
                 size_average=None, reduce=None, reduction='mean') -> None:

        super(HyponymyScoreLoss, self).__init__(
                    distance_metric=distance_metric,
                    size_average=size_average, reduce=reduce, reduction=reduction)

        if (not log_scale) and (not normalize_by_ground_truth_code_length):
            warnings.warn("we recommend you to set `normalize_by_ground_truth_code_length=True` for non-logarithmic scale.")

        self._normalize_by_ground_truth_code_length = normalize_by_ground_truth_code_length
        self._weight_by_ground_truth_code_length = weight_by_ground_truth_code_length
        self._label_smoothing_factor = label_smoothing_factor
        self._log_scale = log_scale
        self._margin_on_code_length_penalty = margin_on_code_length_penalty

    def _one_hot_encoding(self, t_codes: torch.Tensor, n_ary: int, label_smoothing_factor: Optional[float] = None) -> torch.Tensor:
        """
        apply one-hot encoding and optionally label smoothing.

        @param t_codes:
        @param n_ary:
        @param label_smoothing_factor:
        @return:
        """
        t_one_hots = F.one_hot(t_codes, num_classes=n_ary).type(torch.float)
        label_smoothing_factor = self._label_smoothing_factor if label_smoothing_factor is None else label_smoothing_factor
        if label_smoothing_factor is not None:
            max_prob = 1.0 - label_smoothing_factor
            min_prob = label_smoothing_factor / (n_ary - 1)
            t_one_hots = torch.clip(t_one_hots, min=min_prob, max=max_prob)
        return t_one_hots

    def _calc_break_intensity(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # x: hypernym, y: hyponym
        n_digits, n_ary = t_prob_c_x.shape[1:]

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_nonzero = torch.tensor(range(1, n_ary), device=t_prob_c_x.device)
        t_p_c_x_nonzero = torch.index_select(t_prob_c_x, dim=-1, index=idx_nonzero)
        t_p_c_y_nonzero = torch.index_select(t_prob_c_y, dim=-1, index=idx_nonzero)

        ret = 1.0 - torch.sum(t_p_c_x_nonzero * t_p_c_y_nonzero, dim=-1)
        return ret

    def calc_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        log_probs = self.calc_log_ancestor_probability(t_prob_c_x, t_prob_c_y)
        return torch.exp(log_probs)

    def calc_log_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor, eps=1E-15):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze(dim=-1)
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze(dim=-1)

        # t_p_c_*_nonzero: (n_batch, n_digits, n_ary-1)
        idx_nonzero = torch.tensor(range(1,n_ary), device=device)
        t_p_c_x_nonzero = torch.index_select(t_prob_c_x, dim=-1, index=idx_nonzero)
        t_p_c_y_nonzero = torch.index_select(t_prob_c_y, dim=-1, index=idx_nonzero)

        # t_beta: (n_batch, n_digits)
        t_beta = t_p_c_x_zero*(1.- t_p_c_y_zero)

        # t_gamma_hat: (n_batch, n_digits)
        t_gamma_hat = torch.sum(t_p_c_x_nonzero*t_p_c_y_nonzero, dim=-1)
        # prepend 1.0 at the beginning
        # pad_shape: (n_batch, 1)
        pad_shape = t_gamma_hat.shape[:-1] + (1,)
        t_pad_begin = torch.ones(pad_shape, dtype=dtype, device=device)
        # t_gamma: (n_batch, n_digits)
        t_gamma = torch.narrow(torch.cat((t_pad_begin, t_gamma_hat), dim=-1), dim=-1, start=0, length=n_digits)
        # t_log_prob: (n_batch,)
        t_log_prob_score = torch.log(t_beta+eps) + torch.cumsum(torch.log(t_gamma+eps), dim=-1)
        t_log_prob = torch.logsumexp(t_log_prob_score, dim=-1)

        return t_log_prob

    def calc_hard_common_ancestor_length(self, t_code_gt: torch.Tensor, t_code_pred: torch.Tensor, mask_value: int = 0) -> torch.LongTensor:
        where_match = torch.logical_and(t_code_gt == t_code_pred, t_code_gt != mask_value).type(torch.long)
        common_prefix_length = torch.cumprod(where_match, dim=-1).sum(dim=-1).type(torch.float)
        return common_prefix_length

    def calc_log_soft_lowest_common_ancestor_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        t_break_intensity = self._calc_break_intensity(t_prob_c_x, t_prob_c_y)
        # t_log_prob_break: (n_batch, n_digits+1)
        t_log_prob_break = self._intensity_to_log_probability(t_break_intensity)

        # t_log_prob_break_nonzero: (n_batch, n_digits) = t_log_prob_break[:,1:]
        idx_nonzero = torch.tensor(range(1, n_digits+1), device=t_prob_c_x.device)
        t_log_prob_break_nonzero = torch.index_select(t_log_prob_break, dim=-1, index=idx_nonzero)

        # t_at_n: (1, n_digits)
        t_at_n = torch.arange(1, n_digits+1, dtype=dtype, device=device).unsqueeze(dim=0)
        # t_log_lca: (n_batch,); t_log_lca[b] = E_{N~Pr{N|x}}[N]
        t_log_lca = torch.logsumexp(torch.log(t_at_n) + t_log_prob_break_nonzero, dim=-1)

        return t_log_lca

    def calc_soft_lowest_common_ancestor_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        t_log_lca = self.calc_log_soft_lowest_common_ancestor_length(t_prob_c_x, t_prob_c_y)
        return torch.exp(t_log_lca)

    def calc_synonym_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        log_probs = self.calc_log_synonym_probability(t_prob_c_x, t_prob_c_y)
        return torch.exp(log_probs)

    def calc_log_synonym_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor, eps=1E-15):

        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze(dim=-1)
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze(dim=-1)

        # t_p_c_*_nonzero: (n_batch, n_digits, n_ary-1)
        idx_nonzero = torch.tensor(range(1,n_ary), device=device)
        t_p_c_x_nonzero = torch.index_select(t_prob_c_x, dim=-1, index=idx_nonzero)
        t_p_c_y_nonzero = torch.index_select(t_prob_c_y, dim=-1, index=idx_nonzero)

        # t_gamma_hat: (n_batch, n_digits)
        # t_gamma_hat = torch.sum(t_prob_c_x*t_prob_c_y, dim=-1) - t_p_c_x_zero*t_p_c_y_zero
        t_gamma_hat = torch.sum(t_p_c_x_nonzero*t_p_c_y_nonzero, dim=-1)
        # prepend 1.0 at the beginning
        # pad_shape: (n_batch, 1)
        pad_shape = t_gamma_hat.shape[:-1] + (1,)
        t_pad_ones = torch.ones(pad_shape, dtype=dtype, device=device)
        # t_gamma: (n_batch, n_digits+1)
        t_gamma = torch.cat((t_pad_ones, t_gamma_hat), dim=-1)

        # t_delta: (n_batch, n_digits)
        t_delta_hat = t_p_c_x_zero*t_p_c_y_zero
        # append 1.0 at the end.
        t_delta = torch.cat((t_delta_hat, t_pad_ones), dim=-1)

        # t_log_prob_d_score: (n_batch, n_digits) -> delta[b][d] * \prod{gamma[b][:]}
        t_log_prob_score = torch.log(t_delta+eps) + torch.cumsum(torch.log(t_gamma+eps), dim=-1)
        # t_log_prob: (n_batch) -> log( \sum{exp(score[b][:]})
        t_log_prob = torch.logsumexp(t_log_prob_score, dim=-1)

        return t_log_prob

    def calc_soft_hyponymy_score(self, t_prob_c_gt: torch.Tensor, t_prob_c_pred: torch.Tensor, log: bool = False):
        # calculate soft hyponymy score
        # x: hypernym, y: hyponym
        # t_prob_c_*[b,n,v] = Pr{C_n=v|x_b}; t_prob_c_*: (n_batch, n_digits, n_ary)

        # l_ground_truth, l_prediction = code lengths of the ground truth and predicted ones.
        l_ground_truth = self.calc_log_soft_code_length(t_prob_c_gt)
        l_prediction = self.calc_log_soft_code_length(t_prob_c_pred)
        l_lca = self.calc_log_soft_lowest_common_ancestor_length(t_prob_c_gt, t_prob_c_pred)

        if log:
           pass
        else:
            l_ground_truth = torch.exp(l_ground_truth)
            l_prediction = torch.exp(l_prediction)
            l_lca = torch.exp(l_lca)

        # score = | lca - l_gt | + max(0, l_pred - l_gt - margin)
        if (not log) and self._normalize_by_ground_truth_code_length:
            score = torch.abs(l_lca/l_ground_truth - 1.0) + \
                    torch.clip(l_prediction/l_ground_truth - 1.0 - self._margin_on_code_length_penalty, min=0.0)
        else:
            score = torch.abs(l_lca - l_ground_truth) + \
                    torch.clip(l_prediction - l_ground_truth - self._margin_on_code_length_penalty, min=0.0)

        return score

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.LongTensor, eps: float = 1E-15) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param input_code_probabilities: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param target_codes: list of (hypernym index, hyponym index, hyponymy score) tuples
        """

        # dtype, device = self._dtype_and_device(input_code_probabilities)
        n_digits, n_ary = input_code_probabilities.shape[1:]

        # x: ground-truth, y: predicted probability

        # clamp values so that it won't produce nan value.
        t_prob_c_y = torch.clamp(input_code_probabilities, min=eps, max=(1.0 - eps))

        # convert to one-hot encoding
        t_prob_c_x = self._one_hot_encoding(t_codes=target_codes, n_ary=n_ary, label_smoothing_factor=self._label_smoothing_factor)

        y_pred = self.calc_soft_hyponymy_score(t_prob_c_gt=t_prob_c_x, t_prob_c_pred=t_prob_c_y, log=self._log_scale)

        y_true = torch.zeros_like(y_pred)
        if self._weight_by_ground_truth_code_length:
            # weights[b] = n_digits - code_length(target_codes[b])
            l_ground_truth = self.calc_log_soft_code_length(t_prob_c_x)
            weights = n_digits - l_ground_truth
        else:
            weights = None

        loss = super().forward(y_pred, y_true, weights)

        return loss


class EntailmentProbabilityLoss(HyponymyScoreLoss):

    def __init__(self,
                 entail_probability_weight: float = 1.0,
                 synonym_probability_weight: float = 0.0,
                 label_smoothing_factor: Optional[float] = None,
                 size_average=None, reduce=None, reduction='mean',
                 loss_metric: str = "cross_entropy",
                 focal_loss_gamma: float = 1.0, focal_loss_normalize_weight: bool = False) -> None:
        """
        compute entailment/synonymy proabbility-based loss between code probabilities and ground-truth codes.

        @param synonym_probability_weight: relative weight of the synonymy class probs. DEFAULT: None (=0.0)
        @param label_smoothing_factor: smoothing factor of the one-hot encoding of ground-truth codes. DEFAULT: None (=disabled)
        @param size_average: deprecated.
        @param reduce: deprecated.
        @param reduction: redunction method of sample losses.
        @param loss_metric: sample-wise weighting method. DEFUALT: cross-entropy (=uniform weights)
        @param focal_loss_gamma: hyper-paramer of focal loss weighting method.
        @param focal_loss_normalize_weight:
        """
        super(EntailmentProbabilityLoss, self).__init__(
                    distance_metric="binary-cross-entropy",
                    size_average=size_average, reduce=reduce, reduction=reduction)
        accepted_loss_metric = ("cross_entropy", "focal_loss", "dice_loss")
        assert loss_metric in accepted_loss_metric, f"`loss_metric` must be one of these: {','.join(accepted_loss_metric)}"
        if label_smoothing_factor is not None:
            warnings.warn("label smoothing may not work due to local minima. use with caution.")

        self._label_smoothing_factor = label_smoothing_factor
        self._entail_probability_weight = entail_probability_weight
        self._synonym_probability_weight = synonym_probability_weight
        self._loss_metric = loss_metric
        self._focal_loss_gamma = focal_loss_gamma
        self._focal_loss_normalize_weight = focal_loss_normalize_weight

    def _cross_entropy_loss(self, y_log_probs: torch.Tensor):
        dtype, device = self._dtype_and_device(y_log_probs)
        y_weights = torch.ones_like(y_log_probs, dtype=dtype, device=device)
        loss = -1.0 * y_weights * y_log_probs
        return loss

    def _focal_loss(self, y_log_probs: torch.Tensor):
        y_weights = (1.0 - torch.exp(y_log_probs)) ** self._focal_loss_gamma
        if self._focal_loss_normalize_weight:
            y_weights = len(y_log_probs) * y_weights / torch.sum(y_weights)
        loss = -1.0 * y_weights * y_log_probs
        return loss

    def _dice_loss(self, y_probs: torch.Tensor):
        gamma = 1.0
        adjusted_y_probs = ((1.0 - y_probs)**self._focal_loss_gamma) * y_probs
        loss = 1.0 - (2. * adjusted_y_probs + gamma) / (adjusted_y_probs + 1 + gamma)
        return loss

    def _compute_loss(self, y_probs: torch.Tensor, log:bool = False):
        y_log_probs = torch.log(y_probs) if not log else y_probs

        if self._loss_metric == "cross_entropy": # cross-entropy loss
            losses = self._cross_entropy_loss(y_log_probs)
        elif self._loss_metric == "focal_loss": # focal loss
            losses = self._focal_loss(y_log_probs)
        elif self._loss_metric == "dice_loss": # dice loss [Li+, 2020]
            losses = self._dice_loss(torch.exp(y_log_probs))
        else:
            raise NotImplementedError(f"unknown loss metric: {self._loss_metric}")
        return losses

    def calc_log_null_probability(self, t_prob_c: torch.Tensor, eps=1E-15) -> torch.Tensor:
        """
        calculate log probability of all code is zero. this probability may be useful for avoiding local minima.

        @param t_prob_c: code probability distribution.
        """
        # t_p_c_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c.device)
        t_p_c_zero = torch.index_select(t_prob_c, dim=-1, index=idx_zero).squeeze()

        # t_log_prob: (n_batch,)
        t_log_prob = torch.sum(torch.log(t_p_c_zero+eps), dim=-1)

        return t_log_prob

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.LongTensor, eps: float = 1E-15) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param input_code_probabilities: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param target_codes: list of (hypernym index, hyponym index, hyponymy score) tuples
        """
        # target_codes: hypernym(=x), input_code_probabilities: hyponym(=y)

        dtype, device = self._dtype_and_device(input_code_probabilities)
        n_ary = input_code_probabilities.shape[-1]

        # clamp values so that it won't produce nan value.
        t_prob_c_y = torch.clamp(input_code_probabilities, min=eps, max=(1.0 - eps))

        # one-hot encoding and optionally label smoothing
        t_prob_c_x = self._one_hot_encoding(t_codes=target_codes, n_ary=n_ary, label_smoothing_factor=self._label_smoothing_factor)

        # calculate {entailment, synonym, other} probabilities
        y_log_prob_entail = self.calc_log_ancestor_probability(t_prob_c_x, t_prob_c_y)
        y_log_prob_synonym = self.calc_log_synonym_probability(t_prob_c_x, t_prob_c_y)
        y_log_prob_other = torch.log(1.0 - (torch.exp(y_log_prob_entail) + torch.exp(y_log_prob_synonym)) + eps)

        # compute the entailment probability as the objective.
        y_log_probs = self._entail_probability_weight * y_log_prob_entail + self._synonym_probability_weight * y_log_prob_synonym

        # compute loss using various sample-wise weighting methods (e.g., focal loss)
        losses = self._compute_loss(y_log_probs, log=True)
        loss = self._manual_reduction(losses, reduction=self.reduction)

        return loss


class CrossEntropyLossWrapper(L.CrossEntropyLoss):

    def __init__(self, digit_aware_weights: bool = False, n_digits: Optional[int] = None,
                 weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        if digit_aware_weights:
            assert n_digits is not None, f"you must specify `n_digits` when digit_aware_weights=True"
            reduction = "none"
            digit_weights_ = torch.arange(start=n_digits, end=0, step=-1)
            digit_weights_ = digit_weights_ / digit_weights_.sum()
            # digit_weights: (1, n_digits)
            self.digit_weights = digit_weights_.unsqueeze(0)

        super().__init__(weight, size_average, ignore_index, reduce, reduction)
        self.digit_aware_weights = digit_aware_weights

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.Tensor, eps=1E-15) -> torch.Tensor:
        """

        @param input_code_probabilities: shape: (n_batch, n_digits, n_ary). t[b,d,a] = P(C_d=a|x_b)
        @param target_codes: shape: (n_batch, n_digits). t[b,d] = c_d \in {0,1,...,n_ary-1}
        """
        input_score = torch.log(input_code_probabilities + eps).swapaxes(1,2)

        if self.digit_aware_weights:
            device = input_code_probabilities.device
            # log_losses: (n_batch, n_digits)
            log_losses = super().forward(input_score, target_codes)
            log_loss = torch.mean(log_losses * self.digit_weights.to(device))
        else:
            log_loss = super().forward(input_score, target_codes)
        return log_loss

    @property
    def scale(self):
        return 1.0


class FocalCrossEntropyLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma: float = 1.0, along_with_axes: str = "sample", alpha=None, ignore_index=-100, reduction="mean"):
        set_valid_axies = {"sample", "digit"}
        assert along_with_axes in set_valid_axies, f"unknown `along_with` parameter. it must be {set_valid_axies}"

        super().__init__(weight=alpha, ignore_index=ignore_index, reduction="none")
        self.along_with_axes = along_with_axes
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.Tensor, eps=1E-15) -> torch.Tensor:
        # input_code_probabilities: (n_batch, n_digits, n_ary)
        input_score = torch.log(input_code_probabilities+eps).swapaxes(1,2)

        # it returns sample and class wise log cross entropy.
        # cross_entropy: (n_batch, n_digits); cross_entropy[b][d] = -log(input_code_probabilities[b][d][target_codes[b][d]])
        cross_entropy = super().forward(input_score, target_codes)

        # focal_loss_probs: (n_batch, n_digits) or (n_batch,)
        # loss: (n_batch,)
        if self.along_with_axes == "digit":
            # loss[b] = -\sum_{d}{(1-p(Y_d=y_d))^{\gamma}lnp(Y_d=y_d)}
            focal_loss_probs = torch.exp(-cross_entropy)
            loss = (torch.pow(1 - focal_loss_probs, self.gamma) * cross_entropy).sum(dim=-1)
        elif self.along_with_axes == "sample":
            # loss[b] = -(1-p(Y=y))^{\gamma} \cdot \sum_{d}lnp(Y_d=y_d)
            focal_loss_probs = torch.exp(-torch.sum(cross_entropy, dim=-1))
            loss = torch.pow(1 - focal_loss_probs, self.gamma) * cross_entropy.sum(dim=-1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
