#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Dict, Any, Union, Tuple, List, Iterable

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .loss_unsupervised import CodeValueMutualInformationLoss
from .onmt.global_attention import GlobalAttention
from .encoder_internal import PositionalEncoding

class BaseEncoder(nn.Module):

    def __init__(self, n_ary: int):
        super().__init__()
        self._n_ary = n_ary

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

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


class LSTMEncoder(BaseEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 pos_tagset: Iterable[str],
                 n_dim_hidden: Optional[int] = None,
                 n_dim_emb_code: Optional[int] = None,
                 teacher_forcing: bool = True,
                 input_entity_vector: bool = False,
                 discretizer: Optional[nn.Module] = None,
                 global_attention_type: Optional[str] = None,
                 code_embeddings_type: str = "time_distributed",
                 prob_zero_monotone_increasing: bool = False,
                 **kwargs):

        super().__init__(n_ary=n_ary)

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
        # self._n_ary = n_ary
        self._pos_tagset = sorted(list(pos_tagset))
        self._n_ary_internal = None
        self._global_attention_type = global_attention_type
        self._teacher_forcing = teacher_forcing
        self._input_entity_vector = input_entity_vector
        self._code_embeddings_type = code_embeddings_type
        self._trainable_beginning_of_code = True
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        if self._input_entity_vector == False:
            warnings.warn("entity vector input is disabled. we recommend you to input initial states instead.")

        self._build()

    def _build(self):

        # assign part-of-speech index
        self._pos_index = {}
        for idx, pos in enumerate(self._pos_tagset):
            self._pos_index[pos] = idx

        # ([x;e_t],h_t) or (e_t,h_t) -> h_t
        if self._input_entity_vector:
            input_size = self._n_dim_emb + self._n_dim_emb_code
        else:
            input_size = self._n_dim_emb_code
        self._lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=self._n_dim_hidden, bias=True)

        # y_t -> e_{t+1}; t=0,1,...,N_d-2
        ## embedding for beginning-of-code: (n_pos, n_dim_emb_code)
        if self._trainable_beginning_of_code:
            init_value = torch.full((len(self._pos_index), self._n_dim_emb_code), dtype=torch.float, fill_value=0.01)
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

    def _init_states(self, n_batch: int, lst_pos: List[str], dtype, device):
        h_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)
        c_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)

        if self._trainable_beginning_of_code:
            lst_embeddings = [self._embedding_code_boc[self._pos_index[pos],:] for pos in lst_pos]
            e_0 = torch.stack(lst_embeddings, dim=0).to(device)
        else:
            e_0 = torch.zeros((n_batch, self._n_dim_emb_code), dtype=dtype, device=device)

        return h_0, c_0, e_0

    def forward(self, pos: List[str], entity_vectors: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_lengths: Optional[torch.Tensor] = None,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                on_inference: bool = False,
                apply_argmax_on_inference: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @param pos: list of part-of-speech tags. i.e., ["n","n","v",...,"n"]
        @param entity_vectors: input embeddings. shape: (n_batch, n_dim_emb)
        @param ground_truth_synset_codes: ground-truth codes. shape: (n_batch, n_digits)
        @param init_states: input state vectors. tuple of (h_0,c_0) tensors. shape: (n_batch, n_dim_hidden)
        @param context_embeddings: context subword embeddings. shape: (n_batch, n_seq_len, n_dim_hidden)
        @param on_inference: inference(True) or training(False)
        @return: tuple of (sampled codes, code probabilities). shape: (n_batch, n_digits, n_ary)
        """
        if isinstance(pos, str):
            pos = [pos]

        if entity_vectors is not None:
            t = entity_vectors
        elif init_states is not None:
            t = init_states[0]
        else:
            raise AssertionError(f"neighter `entity_vectors` nor `init_states` is available.")
        dtype, device = self._dtype_and_device(t)
        n_batch = t.shape[0]

        if on_inference:
            ground_truth_synset_codes = None
        elif ground_truth_synset_codes is not None:
            ground_truth_synset_codes = F.one_hot(ground_truth_synset_codes, num_classes=self._n_ary).type(torch.float)

        # initialize variables
        lst_prob_c = []
        lst_latent_code = []
        h_d, c_d, e_d = self._init_states(n_batch, pos, dtype, device)
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
                if apply_argmax_on_inference:
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

    @property
    def input_entity_vector(self):
        return self._input_entity_vector

    def summary(self):
        #ToDo: implement summary
        ret = {}
        if self.has_discretizer:
            ret["discretizer"] = self._discretizer.summary()
        return ret


class TransformerEncoder(BaseEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_layer: int, n_head: int,
                 pos_tagset: Iterable[str],
                 layer_normalization: bool = False,
                 discretizer: Optional[nn.Module] = None,
                 norm_digit_embeddings: bool = False,
                 ignore_trailing_zero_digits: bool = False,
                 trainable_positional_encoding: bool = False,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 prob_zero_monotone_increasing: bool = False,
                 **kwargs):

        super().__init__(n_ary=n_ary)

        self._n_dim_hidden = n_dim_emb
        self._n_digits = n_digits
        # self._n_ary = n_ary
        self._pos_tagset = sorted(list(pos_tagset))
        self._discretizer = discretizer
        self._n_layer = n_layer
        self._n_head = n_head
        self._layer_normalization = layer_normalization
        self._dropout = dropout
        self._norm_digit_embeddings = norm_digit_embeddings
        self._ignore_trailing_zero_digits = ignore_trailing_zero_digits
        self._trainable_positional_encoding = trainable_positional_encoding
        self._batch_first = batch_first
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        self._build()

    def _build(self):

        # assign part-of-speech index
        self._pos_index = {}
        for idx, pos in enumerate(self._pos_tagset):
            self._pos_index[pos] = self._n_ary + idx

        # digit embeddings
        cfg_emb_layer = {
            "num_embeddings":self._n_ary + len(self._pos_tagset), # n_pos will be used for beginning-of-sense-code symbol.
            "embedding_dim":self._n_dim_hidden,
            "max_norm":1.0 if self._norm_digit_embeddings else None,
            "padding_idx":0 if self._ignore_trailing_zero_digits else None
        }
        self._emb_layer = nn.Embedding(**cfg_emb_layer)

        # positional encodings
        cfg_pe_layer = {
            "d_model":self._n_dim_hidden,
            "dropout":self._dropout,
            "max_len":self._n_digits,
            "trainable":self._trainable_positional_encoding
        }
        self._pe_layer = PositionalEncoding(**cfg_pe_layer)

        # transformer decoder
        cfg_transformer_decoder_layer = {
            "d_model":self._n_dim_hidden,
            "nhead":self._n_head,
            "dim_feedforward": self._n_dim_hidden * 4,
            "dropout":self._dropout
        }
        layer = nn.TransformerDecoderLayer(**cfg_transformer_decoder_layer)

        if self._layer_normalization:
            norm = nn.LayerNorm(normalized_shape=self._n_dim_hidden)
        else:
            norm = None
        self._decoder = nn.TransformerDecoder(layer, num_layers=self._n_layer, norm=norm)

        # logits layer
        self._softmax_logit_layer = nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary, bias=True)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self._emb_layer.weight, -initrange, initrange)
        nn.init.zeros_(self._softmax_logit_layer.weight)

    def create_sequence_inputs(self, lst_pos: List[str], device,
                               ground_truth_synset_codes: Optional[Union[List[List[int]], torch.Tensor]] = None) -> torch.Tensor:
        # 1. prepend PoS index
        # t_boc: (n_batch, 1)
        lst_pos_idx = [self._pos_index[pos] for pos in lst_pos]
        t_boc = torch.LongTensor(lst_pos_idx, device=device).unsqueeze(dim=-1)

        if ground_truth_synset_codes is None:
            return t_boc

        # 2. concat with one-shifted ground-truth codes.
        # ground_truth_synset_codes: (n_batch, <=n_digits)
        if isinstance(ground_truth_synset_codes, list):
            ground_truth_synset_codes = torch.LongTensor(ground_truth_synset_codes, device=device)
        # input sequence length must be less or equal to n_digits.
        # t_inputs: (n_batch, <n_digits)
        t_inputs = torch.cat((t_boc, ground_truth_synset_codes[:,:self._n_digits-1]), dim=-1)
        return t_inputs

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        r"""
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        ref: https://pytorch.org/docs/1.10.0/generated/torch.nn.Transformer.html
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


    def forward_base(self, input_sequence: torch.Tensor, entity_embeddings: torch.Tensor,
                     entity_sequence_mask: torch.Tensor,
                     on_inference: bool,
                     **kwargs):
        """

        @param input_sequence: (n_batch, <=n_digits). sequence of the input codes.
        @param entity_embeddings: (n_batch, max(entity_span), n_emb). stack of the subword embeddings within entity span.
        @param entity_sequence_mask: (n_batch, max(entity_span)). mask of the entity embeddings.
        @param kwargs:
        @return:
        """

        n_digits = min(self._n_digits, input_sequence.shape[-1])
        # compute target embeddings: (n_batch, n_digits, n_dim_emb)
        t_emb = self._emb_layer.forward(input_sequence) * math.sqrt(self._n_dim_hidden)
        tgt = self._pe_layer.forward(t_emb)
        # prepare subsequent masks: (n_digits, n_digits)
        if on_inference:
            tgt_mask = None
        else:
            _, device = self._dtype_and_device(input_sequence)
            tgt_mask = self.generate_square_subsequent_mask(sz=n_digits).to(device)

        # prepare memory embeddings and masks
        memory = entity_embeddings
        memory_key_padding_mask = entity_sequence_mask

        if self._batch_first:
            tgt = tgt.swapaxes(0, 1)
            memory = memory.swapaxes(0, 1)

        # compute transformer decoder
        h_out = self._decoder.forward(tgt=tgt, tgt_mask=tgt_mask, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        if self._batch_first:
            h_out = h_out.swapaxes(0, 1)

        # compute Pr{Y_d|y_{<d}}
        # lst_code_probs: List[tensor(n_batch, n_ary)]
        lst_code_probs = [F.softmax(self._softmax_logit_layer(h_out[:, idx_d, :]), dim=-1) for idx_d in range(n_digits)]

        # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
        if self._prob_zero_monotone_increasing:
            for d in range(n_digits):
                prob_c_prev = lst_code_probs[d-1] if d > 0 else None
                t_prob_c_d = lst_code_probs[d]
                lst_code_probs[d] = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

        # stack all digits
        # t_code_prob: (n_batch, n_digits, n_ary)
        t_code_prob = torch.stack(lst_code_probs, dim=1)

        # sample continuous relaxed codes
        # (2021-11-11) this will never be used because we do not support student forcing on training so far.
        if self.has_discretizer and (not on_inference):
            t_latent_code = self._discretizer.forward(t_code_prob)
        else:
            t_latent_code = t_code_prob

        return t_latent_code, t_code_prob

    def forward_inference_greedy(self, pos: List[str], entity_embeddings: torch.Tensor,
                                 entity_sequence_mask: torch.Tensor,
                                 **kwargs):
        # decoding using greedy search
        # return: t_code_probs: (n_batch, n_digits, n_ary). t_code_probs[n,d] = Pr(Y|\hat{y_{<d}})
        # return: t_codes: (n_batch, n_digits). t_codes[n,d] = argmax_{a}{Pr(Y=a|\hat{y_{<d}})} = \hat{y_{d}}

        dtype, device = self._dtype_and_device(entity_embeddings)

        for digit in range(self._n_digits):
            if digit == 0:
                input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device, ground_truth_synset_codes=None)
            else:
                input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device, ground_truth_synset_codes=t_codes)
            # t_code_probs_upto_d: (n_batch, digit+1, n_ary)
            _, t_code_probs_upto_d = self.forward_base(input_sequence=input_sequence,
                                                     entity_embeddings=entity_embeddings,
                                                     entity_sequence_mask=entity_sequence_mask,
                                                     on_inference=True,
                                                     **kwargs)

            # t_code_probs_d: (n_batch, 1, n_ary)
            t_code_probs_d = t_code_probs_upto_d[:, digit, :].unsqueeze(1)
            # t_codes_d: (n_batch, 1)
            t_codes_d = t_code_probs_d.argmax(dim=-1)

            # concat with previous sequences
            if digit == 0:
                t_codes, t_code_probs = t_codes_d, t_code_probs_d
            else:
                t_codes = torch.cat([t_codes, t_codes_d], dim=-1)
                t_code_probs = torch.cat([t_code_probs, t_code_probs_d], dim=1)

        return t_codes, t_code_probs

    def forward_training(self, pos: List[str],
                         entity_embeddings: torch.Tensor,
                         entity_sequence_mask: torch.Tensor,
                         ground_truth_synset_codes: torch.Tensor,
                         **kwargs):
        dtype, device = self._dtype_and_device(entity_embeddings)
        input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device,
                                                     ground_truth_synset_codes=ground_truth_synset_codes)
        return self.forward_base(input_sequence=input_sequence,
                                 entity_embeddings=entity_embeddings,
                                 entity_sequence_mask=entity_sequence_mask, on_inference=False,
                                 **kwargs)

    def forward(self, pos: List[str], entity_embeddings: torch.Tensor,
                entity_sequence_mask: torch.Tensor,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                on_inference: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(pos, str):
            pos = [pos]

        if not on_inference:
            return self.forward_training(pos=pos, entity_embeddings=entity_embeddings,
                                         entity_sequence_mask=entity_sequence_mask, ground_truth_synset_codes=ground_truth_synset_codes)
        else:
            return self.forward_inference_greedy(pos=pos, entity_embeddings=entity_embeddings,
                                                 entity_sequence_mask=entity_sequence_mask, **kwargs)

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
        return True

    def summary(self):
        ret = {
            "class_name": self.__class__.__name__,
            "n_head": self._n_head,
            "n_layer": self._n_layer,
            "pos_tagset": self._pos_tagset,
            # "n_digits": self.n_digits,
            # "n_ary": self.n_ary
        }
        if self.has_discretizer:
            ret["discretizer"] = self._discretizer.summary()
        return ret