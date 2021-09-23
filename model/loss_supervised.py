#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Tuple, Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L

class CodeLengthPredictionLoss(L._Loss):

    def __init__(self, scale: float = 1.0, normalize_code_length: bool = False,
                 distance_metric: str = "scaled-mse",
                 size_average=None, reduce=None, reduction='mean'):

        super(CodeLengthPredictionLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale
        self._normalize_code_length = normalize_code_length

        self._distance_metric = distance_metric
        if distance_metric == "mse":
            self._func_distance = self._mse
        elif distance_metric == "scaled-mse":
            self._func_distance = self._scaled_mse
        elif distance_metric == "standardized-mse":
            self._func_distance = self._standardized_mse
        elif distance_metric == "autoscaled-mse":
            self._func_distance = self._auto_scaled_mse
        elif distance_metric == "positive-autoscaled-mse":
            self._func_distance = self._positive_auto_scaled_mse
        elif distance_metric == "batchnorm-mse":
            self._func_distance = self._batchnorm_mse
            self._m = nn.BatchNorm1d(1)
        elif distance_metric == "mae":
            self._func_distance = self._mae
        elif distance_metric == "scaled-mae":
            self._func_distance = self._scaled_mae
        elif distance_metric == "cosine":
            self._func_distance = self._cosine_distance
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
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def _standardize(self, vec: torch.Tensor, dim=-1):
        means = vec.mean(dim=dim, keepdim=True)
        stds = vec.std(dim=dim, keepdim=True)
        return (vec - means) / (stds + 1E-6)

    def _scale_dynamic(self, vec: torch.Tensor, dim=-1):
        stds = vec.std(dim=dim, keepdim=True)
        return vec / (stds + 1E-6)

    def _mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(u, v, reduction=self.reduction)

    def _scaled_mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(self._scale_dynamic(u), self._scale_dynamic(v), reduction=self.reduction)

    def _standardized_mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(self._standardize(u), self._standardize(v), reduction=self.reduction)

    def _auto_scaled_mse(self, u, v) -> torch.Tensor:
        # assume u and y is predicted and ground-truth values, respectively.
        scale = torch.sum(u.detach()*v.detach()) / (torch.sum(u.detach()**2)+1E-6)
        if scale > 0:
            loss = F.mse_loss(scale*u, v, reduction=self.reduction)
        else:
            loss = self._scaled_mse(u, v)
        return loss

    def _positive_auto_scaled_mse(self, u, v) -> torch.Tensor:
        # assume u and y is predicted and ground-truth values, respectively.
        scale = max(1.0, torch.sum(u.detach()*v.detach()) / (torch.sum(u.detach()**2)+1E-6))
        loss = F.mse_loss(scale*u, v, reduction=self.reduction)
        return loss

    def _mae(self, u, v) -> torch.Tensor:
        return F.l1_loss(u, v, reduction=self.reduction)

    def _scaled_mae(self, u, v) -> torch.Tensor:
        return F.l1_loss(self._scale_dynamic(u), self._scale_dynamic(v), reduction=self.reduction)

    def _cosine_distance(self, u, v, dim=0, eps=1e-8) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(u, v, dim, eps)

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
        # t_intensity can be either one or two dimensional tensor.
        dtype, device = self._dtype_and_device(t_intensity)
        pad_shape = t_intensity.shape[:-1] + (1,)

        t_pad_begin = torch.zeros(pad_shape, dtype=dtype, device=device)
        t_pad_end = torch.ones(pad_shape, dtype=dtype, device=device)

        t_prob = torch.cumprod(1.0 - torch.cat((t_pad_begin, t_intensity), dim=-1), dim=-1) * torch.cat((t_intensity, t_pad_end), dim=-1)

        return t_prob

    def calc_soft_code_length(self, t_prob_c: torch.Tensor):
        t_p_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0, device=t_prob_c.device)).squeeze()
        n_digits = t_p_c_zero.shape[-1]
        dtype, device = self._dtype_and_device(t_prob_c)

        t_p_at_n = self._intensity_to_probability(t_p_c_zero)
        t_at_n = torch.arange(n_digits+1, dtype=dtype, device=device)

        ret = torch.sum(t_p_at_n * t_at_n, dim=-1)
        return ret

    def forward(self, t_prob_c_batch: torch.Tensor, lst_code_length_tuple: List[Tuple[int, float]]) -> torch.Tensor:
        """
        evaluates L2 loss of the predicted code length and true code length in a normalized scale.

        :param t_prob_c_batch: probability array of p(c_n=m|x); (n_batch, n_digits, n_ary)
        :param lst_hyponymy_tuple: list of (entity index, entity depth) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx = torch.tensor([tup[0] for tup in lst_code_length_tuple], dtype=torch.long, device=device)
        y_true = torch.tensor([tup[1] for tup in lst_code_length_tuple], dtype=dtype, device=device)

        # t_prob_c_batch: (N_b, N_digits, N_ary); t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        t_prob_c = torch.index_select(t_prob_c_batch, dim=0, index=t_idx)

        # y_pred: (len(lst_code_length_tuple),)
        # y_true: (len(lst_code_length_tuple),)
        y_pred = self.calc_soft_code_length(t_prob_c=t_prob_c)

        # scale ground-truth value and predicted value
        if self._normalize_code_length:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            n_digits = t_prob_c_batch.shape[1]
            y_pred /= n_digits
            # scale ground-truth value by the user-specified value.
            y_true *= self._normalize_coef_for_gt

        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale


class HyponymyScoreLoss(CodeLengthPredictionLoss):

    def __init__(self, scale: float = 1.0, normalize_hyponymy_score: bool = False,
                 distance_metric: str = "mse",
                 size_average=None, reduce=None, reduction='mean') -> None:

        super(HyponymyScoreLoss, self).__init__(scale=scale,
                    distance_metric=distance_metric,
                    size_average=size_average, reduce=reduce, reduction=reduction)

        self._normalize_hyponymy_score = normalize_hyponymy_score

    def _calc_break_intensity(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # x: hypernym, y: hyponym

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

        ret = 1.0 - (torch.sum(t_prob_c_x * t_prob_c_y, dim=-1) - t_p_c_x_zero * t_p_c_y_zero)
        return ret

    def calc_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()
        # t_beta: (n_batch, n_digits)
        t_beta = t_p_c_x_zero*(1.- t_p_c_y_zero)

        # t_gamma_hat: (n_batch, n_digits)
        t_gamma_hat = torch.sum(t_prob_c_x*t_prob_c_y, dim=-1) - t_p_c_x_zero*t_p_c_y_zero
        # prepend 1.0 at the beginning
        # pad_shape: (n_batch, 1)
        pad_shape = t_gamma_hat.shape[:-1] + (1,)
        t_pad_begin = torch.ones(pad_shape, dtype=dtype, device=device)
        # t_gamma: (n_batch, n_digits)
        t_gamma = torch.narrow(torch.cat((t_pad_begin, t_gamma_hat), dim=-1), dim=-1, start=0, length=n_digits)
        # t_prob: (n_batch,)
        t_prob = torch.sum(t_beta*torch.cumprod(t_gamma, dim=-1), dim=-1)

        return t_prob

    def calc_log_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor, eps=1E-15):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

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

    def calc_soft_lowest_common_ancestor_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        t_break_intensity = self._calc_break_intensity(t_prob_c_x, t_prob_c_y)
        t_prob_break = self._intensity_to_probability(t_break_intensity)

        t_at_n = torch.arange(n_digits+1, dtype=dtype, device=device)
        ret = torch.sum(t_prob_break * t_at_n, dim=-1)

        return ret

    def calc_soft_hyponymy_score(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # calculate soft hyponymy score
        # x: hypernym, y: hyponym
        # t_prob_c_*[b,n,v] = Pr{C_n=v|x_b}; t_prob_c_*: (n_batch, n_digits, n_ary)

        # l_hyper, l_hypo = hypernym / hyponym code length
        l_hyper = self.calc_soft_code_length(t_prob_c_x)
        l_hypo = self.calc_soft_code_length(t_prob_c_y)
        # alpha = probability of hyponymy relation
        alpha = self.calc_ancestor_probability(t_prob_c_x, t_prob_c_y)
        # beta = probability of identity relation
        beta = self.calc_synonym_probability(t_prob_c_x, t_prob_c_y)
        # l_lca = length of the lowest common ancestor
        l_lca = self.calc_soft_lowest_common_ancestor_length(t_prob_c_x, t_prob_c_y)

        score = (alpha+beta) * (l_hypo - l_hyper) + (1. - (alpha + beta)) * (l_lca - l_hyper)

        return score

    def calc_synonym_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):

        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

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

        # t_prob: (n_batch)
        t_prob = torch.sum(t_delta*torch.cumprod(t_gamma, dim=-1), dim=-1)

        return t_prob

    def calc_log_synonym_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor, eps=1E-15):

        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

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

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.LongTensor) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param input_code_probabilities: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param target_codes: list of (hypernym index, hyponym index, hyponymy score) tuples
        """
        # x: hypernym, y: hyponym

        dtype, device = self._dtype_and_device(input_code_probabilities)
        n_digits, n_ary = input_code_probabilities.shape[1:]

        # clamp values so that it won't produce nan value.
        t_prob_c_y = torch.clamp(input_code_probabilities, min=1E-5, max=(1.0 - 1E-5))

        # convert to one-hot encoding
        t_prob_c_x = F.one_hot(target_codes, num_classes=n_ary).type(torch.float)

        y_pred = self.calc_soft_hyponymy_score(t_prob_c_x, t_prob_c_y)

        # scale ground-truth value and predicted value
        if self._normalize_hyponymy_score:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            y_pred /= n_digits

        y_true = torch.zeros_like(y_pred)
        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale


class EntailmentProbabilityLoss(HyponymyScoreLoss):

    def __init__(self, scale: float = 1.0,
                 synonym_probability_weight: Optional[float] = None,
                 label_smoothing_factor: Optional[float] = None,
                 size_average=None, reduce=None, reduction='mean',
                 loss_metric: str = "cross_entropy",
                 focal_loss_gamma: float = 1.0, focal_loss_normalize_weight: bool = False) -> None:
        """
        compute entailment/synonymy proabbility-based loss between code probabilities and ground-truth codes.

        @param scale: scaling coefficient of the total loss.
        @param synonym_probability_weight: relative weight of the synonymy class probs. DEFAULT: None (=0.0)
        @param label_smoothing_factor: smoothing factor of the one-hot encoding of ground-truth codes. DEFAULT: None (=disabled)
        @param size_average: deprecated.
        @param reduce: deprecated.
        @param reduction: redunction method of sample losses.
        @param loss_metric: sample-wise weighting method. DEFUALT: cross-entropy (=uniform weights)
        @param focal_loss_gamma: hyper-paramer of focal loss weighting method.
        @param focal_loss_normalize_weight:
        """
        super(EntailmentProbabilityLoss, self).__init__(scale=scale,
                    distance_metric="binary-cross-entropy",
                    size_average=size_average, reduce=reduce, reduction=reduction)
        accepted_loss_metric = ("cross_entropy", "focal_loss", "dice_loss")
        assert loss_metric in accepted_loss_metric, f"`loss_metric` must be one of these: {','.join(accepted_loss_metric)}"
        self._label_smoothing_factor = label_smoothing_factor
        self._synonym_probability_weight = 0.0 if synonym_probability_weight is None else synonym_probability_weight
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

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.LongTensor, eps: float = 1E-5) -> torch.Tensor:
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

        # convert to one-hot encoding
        t_prob_c_x = F.one_hot(target_codes, num_classes=n_ary).type(torch.float)
        if self._label_smoothing_factor is not None:
            max_prob = 1.0 - self._label_smoothing_factor
            min_prob = self._label_smoothing_factor / (n_ary - 1)
            t_prob_c_x = torch.clip(t_prob_c_x, min=min_prob, max=max_prob)

        # calculate {entailment, synonym, other} probabilities
        # y_log_prob_entail = self.calc_log_ancestor_probability(t_prob_c_x, t_prob_c_y)
        # y_log_prob_synonym = self.calc_log_synonym_probability(t_prob_c_x, t_prob_c_y)
        # y_prob_other = 1.0 - (torch.exp(y_log_prob_entail) + torch.exp(y_log_prob_synonym))
        y_prob_entail = self.calc_ancestor_probability(t_prob_c_x, t_prob_c_y)
        y_prob_synonym = self.calc_synonym_probability(t_prob_c_x, t_prob_c_y)
        y_prob_other = 1.0 - (y_prob_entail + y_prob_synonym)

        # clamp values so that it won't produce nan value.
        # y_prob_entail = torch.clamp(y_prob_entail, min=eps, max=(1.0-eps))
        # y_prob_synonym = torch.clamp(y_prob_synonym, min=eps, max=(1.0-eps))
        # y_prob_other = torch.clamp(y_prob_other, min=eps, max=(1.0-eps))

        # compute the entailment probability as the objective.
        w = self._synonym_probability_weight
        y_log_probs = (1.0 - w) * y_log_prob_entail + w * y_log_prob_synonym

        # compute loss using various sample-wise weighting methods (e.g., focal loss)
        losses = self._compute_loss(y_log_probs, log=True)

        # reduction
        if self.reduction == "sum":
            loss = torch.sum(losses)
        elif self.reduction.endswith("mean"):
            loss = torch.mean(losses)
        elif self.reduction == "none":
            loss = losses

        return loss * self._scale


class CrossEntropyLossWrapper(L.CrossEntropyLoss):

    def forward(self, input_code_probabilities: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
        """

        @param input_code_probabilities: shape: (n_batch, n_digits, n_ary). t[b,d,a] = P(C_d=a|x_b)
        @param target_codes: shape: (n_batch, n_digits). t[b,d] = c_d \in {0,1,...,n_ary-1}
        """
        input_score = torch.log(input_code_probabilities).swapaxes(1,2)

        return super().forward(input_score, target_codes)