#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Dict, Any, Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from .loss_unsupervised import CodeValueMutualInformationLoss
from .onmt.global_attention import GlobalAttention

class SimpleEncoder(nn.Module):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 dtype=torch.float32, **kwargs):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype

        self._build()

    def _build(self):

        self.x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)
        self.lst_h_to_z = nn.ModuleList([nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary) for n in range(self._n_digits)])

    def forward(self, input_x: torch.Tensor):

        t_h = torch.tanh(self.x_to_h(input_x))
        lst_z = [torch.log(F.softplus(h_to_z(t_h))) for h_to_z in self.lst_h_to_z]
        lst_prob_c = [F.softmax(t_z, dim=1) for t_z in lst_z]

        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def calc_code_probability(self, input_x: torch.Tensor, **kwargs):
        return self.forward(input_x)

    @property
    def has_discretizer(self):
        return False

    def _adjust_code_probability_to_monotone_increasing(self, probs: torch.Tensor, probs_prev: Union[None, torch.Tensor],
                                                        eps: float = 1E-6):
        # if Pr{C_{d-1}} is not given, we don't adjust the probability.
        if probs_prev is None:
            return probs

        dtype, device = self._dtype_and_device(probs)
        n_ary = self._n_ary

        # adjust Pr{Cd=0} using stick-breaking process
        # probs_zero_*: (n_batch, 1)
        probs_zero = torch.index_select(probs, dim=-1, index=torch.tensor(0, device=device))
        probs_zero_prev = torch.index_select(probs_prev, dim=-1, index=torch.tensor(0, device=device))
        probs_zero_adj = probs_zero_prev + (1.0 - probs_zero_prev)*probs_zero

        # probs_nonzero_*: (n_batch, n_ary-1)
        probs_nonzero = torch.index_select(probs, dim=-1, index=torch.arange(1, n_ary, dtype=torch.long, device=device))
        adjust_factor = (1.0 - probs_zero_adj) / ((1.0 - probs_zero) + eps)
        probs_nonzero_adj = adjust_factor * probs_nonzero

        # concatenate adjusted probabilities
        probs_adj = torch.cat((probs_zero_adj, probs_nonzero_adj), dim=-1)
        probs_adj = torch.clip(probs_adj, min=eps, max=1.0-eps)

        return probs_adj


class LSTMEncoder(SimpleEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 n_dim_emb_code: Optional[int] = None,
                 teacher_forcing: bool = True,
                 apply_argmax_on_inference: bool = False,
                 input_entity_vector: bool = False,
                 discretizer: Optional[nn.Module] = None,
                 global_attention_type: Optional[str] = None,
                 code_embeddings_type: str = "time_distributed",
                 trainable_beginning_of_code: bool = True,
                 prob_zero_monotone_increasing: bool = False,
                 dtype=torch.float32,
                 **kwargs):

        super(SimpleEncoder, self).__init__()

        if teacher_forcing:
            if discretizer is not None:
                warnings.warn("discretizer will not be used when `teacher_forcing` is enabled.")
            discretizer = None
        else:
            if discretizer is None:
                warnings.warn("student forcing will be applied without stochastic sampling.")

        self._discretizer = discretizer
        self._n_dim_emb = n_dim_emb
        self._n_dim_emb_code = n_dim_emb if n_dim_emb_code is None else n_dim_emb_code
        self._n_dim_hidden = n_dim_emb if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype
        self._n_ary_internal = None
        self._global_attention_type = global_attention_type
        self._teacher_forcing = teacher_forcing
        self._apply_argmax_on_inference = apply_argmax_on_inference
        self._input_entity_vector = input_entity_vector
        self._code_embeddings_type = code_embeddings_type
        self._trainable_beginning_of_code = trainable_beginning_of_code
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        if self._input_entity_vector == False:
            warnings.warn("entity vector input is disabled. we recommend you to input initial states instead.")

        self._build()

    def _build(self):

        # ([x;e_t],h_t) or (e_t,h_t) -> h_t
        if self._input_entity_vector:
            input_size = self._n_dim_emb + self._n_dim_emb_code
        else:
            input_size = self._n_dim_emb_code
        self._lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=self._n_dim_hidden, bias=True)

        # y_t -> e_{t+1}; t=0,1,...,N_d-2
        ## embedding for beginning-of-code
        if self._trainable_beginning_of_code:
            init_value = torch.full((self._n_dim_emb_code,), dtype=torch.float, fill_value=0.01)
            self._embedding_code_boc = nn.Parameter(init_value, requires_grad=True)
        ## embeddings for specific values
        if self._code_embeddings_type == "time_distributed": # e_t = Embed(o_t;\theta)
            self._embedding_code = nn.Linear(in_features=self._n_ary, out_features=self._n_dim_emb_code, bias=False)
        elif self._code_embeddings_type == "time_dependent": # e_t = Embed(o_t;\theta_t)
            lst_layers = [nn.Linear(in_features=self._n_ary, out_features=self._n_dim_emb_code, bias=False) for _ in range(self._n_digits-1)]
            self._embedding_code = nn.ModuleList(lst_layers)
        else:
            raise AssertionError(f"unknown `code_embeddings_type` value: {self._code_embeddings_type}")

        # h_t -> a_t
        if self._global_attention_type is None:
            self._global_attention = None
        else:
            if self._global_attention_type == "none":
                self._global_attention = None
            if self._global_attention_type in ("dot", "general", "mlp"):
                assert self._n_dim_emb == self._n_dim_hidden, \
                    f"dimension of input embeddings and hidden layer must be same: {self._n_dim_emb} != {self._n_dim_hidden}"
                self._global_attention = GlobalAttention(dim=self._n_dim_hidden, attn_type=self._global_attention_type)
            else:
                raise AssertionError(f"unknown `global_attention_type` value: {self._global_attention_type}")

        # [h_t;a_t] or h_t -> z_t
        if self._global_attention is not None: # z_t = FF([h_t;a_t];\theta)
            self._h_to_z = nn.Linear(in_features=self._n_dim_hidden*2, out_features=self._n_ary, bias=True)
        else: # z_t = FF(h_t;\theta_t)
            self._h_to_z = nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary, bias=True)

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def _init_states(self, n_batch: int, dtype, device):
        h_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)
        c_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)

        if self._trainable_beginning_of_code:
            e_0 = torch.tile(self._embedding_code_boc, (n_batch, 1) ).to(device)
        else:
            e_0 = torch.zeros((n_batch, self._n_dim_emb_code), dtype=dtype, device=device)

        return h_0, c_0, e_0

    def forward(self, entity_vectors: torch.Tensor,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_lengths: Optional[torch.Tensor] = None,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                on_inference: bool = False, **kwargs):
        """

        @param entity_vectors: input embeddings. shape: (n_batch, n_dim_emb)
        @param ground_truth_synset_codes: ground-truth codes. shape: (n_batch, n_digits)
        @param init_states: input state vectors. tuple of (h_0,c_0) tensors. shape: (n_batch, n_dim_hidden)
        @param context_embeddings: context subword embeddings. shape: (n_batch, n_seq_len, n_dim_hidden)
        @param on_inference: inference(True) or training(False)
        @return: tuple of (sampled codes, code probabilities). shape: (n_batch, n_digits, n_ary)
        """
        dtype, device = self._dtype_and_device(entity_vectors)
        n_batch = entity_vectors.shape[0]

        if on_inference:
            ground_truth_synset_codes = None
        elif ground_truth_synset_codes is not None:
            ground_truth_synset_codes = F.one_hot(ground_truth_synset_codes, num_classes=self._n_ary).type(torch.float)

        # initialize variables
        lst_prob_c = []
        lst_latent_code = []
        h_d, c_d, e_d = self._init_states(n_batch, dtype, device)
        if init_states is not None:
            h_d, c_d = init_states[0], init_states[1]

        # input: (n_batch, n_dim_emb + n_dim_emb_code)
        for d in range(self._n_digits):
            if self._input_entity_vector:
                input = torch.cat([entity_vectors, e_d], dim=-1)
            else:
                input = e_d
            # h_d, c_d: (n_batch, n_dim_hidden)
            h_d, c_d = self._lstm_cell(input, (h_d, c_d))

            # compute Pr{c_d=d|c_{<d}} = softmax( FF([h_t;a_t]) )
            if isinstance(self._global_attention, GlobalAttention):
                src = h_d.unsqueeze(1)
                a_d, _ = self._global_attention.forward(source=src, memory_bank=context_embeddings,
                                                     memory_lengths=context_sequence_lengths)
                a_d = a_d.squeeze(0)
                h_and_a_d = torch.cat([h_d, a_d], dim=-1)
                t_z_d_dash = self._h_to_z(h_and_a_d)
            else:
                t_z_d_dash = self._h_to_z(h_d)
            t_z_d = torch.log(F.softplus(t_z_d_dash) + 1E-6)
            t_prob_c_d = F.softmax(t_z_d, dim=-1)

            # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
            if self._prob_zero_monotone_increasing:
                prob_c_prev = lst_prob_c[-1] if len(lst_prob_c) > 0 else None
                t_prob_c_d = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

            # compute the relaxed code of current digit: c_d
            if on_inference:
                if self._apply_argmax_on_inference:
                    t_latent_code_d = F.one_hot(t_prob_c_d.argmax(dim=-1), num_classes=self._n_ary).type(dtype)
                else:
                    # empirically, embedding based on the probability produces better result then argmax.
                    # I guess it minimizes difference on training and inference.
                    t_latent_code_d = t_prob_c_d
            else:
                ## on training -> stochastic sampling if discretizer is available.
                if self.has_discretizer:
                    t_latent_code_d = self._discretizer(t_prob_c_d)
                else:
                    t_latent_code_d = t_prob_c_d

            # compute the embeddings of previous code
            if d != (self._n_digits - 1):
                if on_inference:
                    o_d = t_latent_code_d.detach()
                else:
                    # on training
                    if self._teacher_forcing:
                        # teacher forcing
                        # o_d = input_y[:, d, :]
                        o_d = torch.index_select(ground_truth_synset_codes, dim=1, index=torch.tensor(d, device=device)).squeeze()
                    else:
                        # student forcing (detached previous output)
                        o_d = t_latent_code_d.detach()

                # calculate embeddings
                if self._code_embeddings_type == "time_distributed":
                    e_d = self._embedding_code(o_d)
                elif self._code_embeddings_type == "time_dependent":
                    e_d = self._embedding_code[d](o_d)
                else:
                    e_d = None
            else:
                e_d = None

            # store computed results
            lst_prob_c.append(t_prob_c_d)
            lst_latent_code.append(t_latent_code_d)

        # stack code probability and latent code.
        # t_prob_c: (n_batch, n_digits, n_ary)
        t_prob_c = torch.stack(lst_prob_c, dim=1)
        # t_latent_code: (n_batch, n_digits, n_ary)
        t_latent_code = torch.stack(lst_latent_code, dim=1)

        return t_latent_code, t_prob_c

    @property
    def has_discretizer(self):
        return self._discretizer is not None

    @property
    def discretizer(self):
        return self._discretizer

    @property
    def n_digits(self):
        return self._n_digits

    @property
    def n_ary(self):
        return self._n_ary

    @property
    def n_dim_hidden(self):
        return self._n_dim_hidden

    @property
    def teacher_forcing(self):
        return self._teacher_forcing

    @property
    def global_attention_type(self):
        return self._global_attention_type


class TransformerEncoder(SimpleEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 normalize_digit_embeddings: bool,
                 n_layer: int = 4,
                 n_head: int = 4,
                 dropout: float = 0.1,
                 n_dim_emb_digits: Optional[int] = None,
                 time_distributed: bool = True,
                 how_digit_embeddings: str = "add",
                 prob_zero_monotone_increasing: bool = False,
                 dtype=torch.float32,
                 **kwargs):

        raise NotImplementedError(f"not implemented yet.")

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_emb_digits = n_dim_emb if n_dim_emb_digits is None else n_dim_emb_digits
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._n_layer = n_layer
        self._n_head = n_head
        self._dropout = dropout
        self._dtype = dtype
        self._time_distributed = time_distributed
        self._how_digit_embeddings = how_digit_embeddings
        self._normalize_digit_embeddings = normalize_digit_embeddings
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        self._build()

    def _build(self):

        # digit embeddings
        cfg_embedding_layer = {
            "num_embeddings":self._n_digits,
            "embedding_dim":self._n_dim_emb_digits,
            "max_norm":1.0 if self._normalize_digit_embeddings else None
        }
        self._embedding_digit = nn.Embedding(**cfg_embedding_layer)

        if self._how_digit_embeddings == "add":
            n_dim_transformer = self._n_dim_emb
        elif self._how_digit_embeddings == "concat":
            n_dim_transformer = self._n_dim_emb + self._n_dim_emb_digits
        else:
            raise NotImplementedError(f"unknown `how_digit_embeddings` value: {self._how_digit_embeddings}")

        # transformer layers
        cfg_transformer_layer = {
            "d_model":n_dim_transformer,
            "nhead":self._n_head,
            "dim_feedforward":n_dim_transformer*4,
            "dropout":self._dropout
        }
        layer = nn.TransformerEncoderLayer(**cfg_transformer_layer)
        self._transformer = nn.TransformerEncoder(layer, num_layers=self._n_layer)

        # prediction layer
        if self._time_distributed:
            self._prediction_layer = nn.Linear(in_features=n_dim_transformer, out_features=self._n_ary, bias=True)
        else:
            lst_layers = [nn.Linear(in_features=n_dim_transformer, out_features=self._n_ary, bias=True) for _ in range(self._n_digits)]
            self._prediction_layer = nn.ModuleList(lst_layers)

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def forward(self, input_x: torch.Tensor, on_inference: bool = False):
        dtype, device = self._dtype_and_device(input_x)

        # input_x: (N_batch, N_dim_emb)
        n_batch = input_x.shape[0]

        # transformer layer inputs
        # x_in: (N_batch, N_digits, N_dim_transformer)

        ## entity embeddings
        x_in_entity = input_x.unsqueeze(1).repeat(1, self._n_digits, 1)
        ## digit embeddings
        dummy_digits = torch.arange(self._n_digits, dtype=torch.long, device=device)
        x_in_digits = self._embedding_digit.forward(dummy_digits).unsqueeze(0).repeat(n_batch,1,1)
        ## merge two embeddings
        if self._how_digit_embeddings == "add":
            x_in = x_in_entity + x_in_digits
        elif self._how_digit_embeddings == "concat":
            x_in = torch.cat((x_in_entity, x_in_digits), dim=-1)
        else:
            raise NotImplementedError(f"unknown `how_digit_embeddings` value: {self._how_digit_embeddings}")

        # transformer
        # transformer's input shape must be (N_digits, N_batch, N_dim_transformer)
        # h_out: (N_batch, N_digits, N_dim_transformer)
        h_out = self._transformer.forward(x_in.transpose(1,0)).transpose(0,1)

        # compute Pr{c_d}
        # t_prob_c: (n_batch, n_digits, n_ary)
        if self._time_distributed:
            lst_prob_c = [F.softmax(self._prediction_layer(h_out[:,idx_d,:]), dim=-1) for idx_d in range(self._n_digits)]
        else:
            lst_prob_c = [F.softmax(layer(h_out[:,idx_d,:]), dim=-1) for idx_d, layer in enumerate(self._prediction_layer)]

        # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
        if self._prob_zero_monotone_increasing:
            for d in range(self._n_digits):
                prob_c_prev = lst_prob_c[d-1] if d > 0 else None
                t_prob_c_d = lst_prob_c[d]
                lst_prob_c[d] = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

        # stack each digits
        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def calc_code_probability(self, input_x: torch.Tensor, adjust_code_probability: bool = False, **kwargs):
        t_prob_c = self.forward(input_x, on_inference=True)
        if adjust_code_probability:
            t_prob_c = CodeValueMutualInformationLoss.calc_adjusted_code_probability(t_prob_c)

        return t_prob_c
