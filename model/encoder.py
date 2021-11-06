#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Dict, Any, Union, Tuple, List, Iterable

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
                 n_dim_hidden: Optional[int] = None,
                 n_dim_emb_code: Optional[int] = None,
                 teacher_forcing: bool = True,
                 input_entity_vector: bool = False,
                 discretizer: Optional[nn.Module] = None,
                 global_attention_type: Optional[str] = None,
                 code_embeddings_type: str = "time_distributed",
                 trainable_beginning_of_code: bool = True,
                 prob_zero_monotone_increasing: bool = False,
                 **kwargs):

        super(BaseEncoder, self).__init__()

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
        self._n_ary_internal = None
        self._global_attention_type = global_attention_type
        self._teacher_forcing = teacher_forcing
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

    def _init_states(self, n_batch: int, dtype, device):
        h_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)
        c_0 = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)

        if self._trainable_beginning_of_code:
            e_0 = torch.tile(self._embedding_code_boc, (n_batch, 1) ).to(device)
        else:
            e_0 = torch.zeros((n_batch, self._n_dim_emb_code), dtype=dtype, device=device)

        return h_0, c_0, e_0

    def forward(self, entity_vectors: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_lengths: Optional[torch.Tensor] = None,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                on_inference: bool = False,
                apply_argmax_on_inference: bool = False,
                **kwargs):
        """

        @param entity_vectors: input embeddings. shape: (n_batch, n_dim_emb)
        @param ground_truth_synset_codes: ground-truth codes. shape: (n_batch, n_digits)
        @param init_states: input state vectors. tuple of (h_0,c_0) tensors. shape: (n_batch, n_dim_hidden)
        @param context_embeddings: context subword embeddings. shape: (n_batch, n_seq_len, n_dim_hidden)
        @param on_inference: inference(True) or training(False)
        @return: tuple of (sampled codes, code probabilities). shape: (n_batch, n_digits, n_ary)
        """
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
                 discretizer: Optional[nn.Module] = None,
                 norm_digit_embeddings: bool = False,
                 ignore_trailing_zero_digits: bool = False,
                 trainable_positional_encoding: bool = False,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 prob_zero_monotone_increasing: bool = False,
                 **kwargs):

        super().__init__()

        self._n_dim_emb = n_dim_emb
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._pos_tagset = set(pos_tagset)
        self._discretizer = discretizer
        self._n_layer = n_layer
        self._n_head = n_head
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
            "embedding_dim":self._n_dim_emb,
            "max_norm":1.0 if self._norm_digit_embeddings else None,
            "padding_idx":0 if self._ignore_trailing_zero_digits else None
        }
        self._emb_layer = nn.Embedding(**cfg_emb_layer)

        # positional encodings
        cfg_pe_layer = {
            "d_model":self._n_dim_emb,
            "dropout":self._dropout,
            "max_len":self._n_digits,
            "trainable":self._trainable_positional_encoding
        }
        self._pe_layer = PositionalEncoding(**cfg_pe_layer)

        # transformer decoder
        cfg_transformer_decoder_layer = {
            "d_model":self._n_dim_emb,
            "nhead":self._n_head,
            "dim_feedforward":self._n_dim_emb*4,
            "dropout":self._dropout
        }
        layer = nn.TransformerDecoderLayer(**cfg_transformer_decoder_layer)
        self._decoder = nn.TransformerDecoder(layer, num_layers=self._n_layer)

        # logits layer
        self._softmax_logit_layer = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_ary, bias=True)

    def create_teacher_forcing_inputs(self, ground_truth_synset_codes: torch.Tensor, lst_pos: List[str]):
        n_batch, n_digits = ground_truth_synset_codes.shape
        dtype, device = self._dtype_and_device(ground_truth_synset_codes)

        # prepend PoS index, trim tail digit.
        lst_pos_idx = [self._pos_index[pos] for pos in lst_pos]
        # t_boc: (n_batch, 1)
        t_boc = torch.LongTensor(lst_pos_idx, device=device).unsqueeze(dim=-1)
        # t_inputs: (n_batch, n_digits)
        t_inputs = torch.cat((t_boc, ground_truth_synset_codes[:,:n_digits-1]))
        return t_inputs

    def subsequent_mask(self, n_seq_length: int):
        "Mask out subsequent positions of target embeddings."
        attn_shape = (n_seq_length, n_seq_length)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 1

    def forward_training(self, pos: List[str], entity_embeddings: torch.Tensor,
                entity_sequence_mask: torch.Tensor,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                **kwargs):
        # crate inputs
        t_inputs = self.create_teacher_forcing_inputs(ground_truth_synset_codes, pos)

        # compute target embeddings: (n_batch, n_digits, n_dim_emb)
        tgt = self._pe_layer.forward( self._emb_layer.forward(t_inputs) )
        # prepare subsequent masks: (n_digits, n_digits)
        tgt_mask = self.subsequent_mask(n_seq_length=self._n_digits)

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
        lst_code_probs = [F.softmax(self._softmax_logit_layer(h_out[:, idx_d, :]), dim=-1) for idx_d in range(self._n_digits)]

        # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
        if self._prob_zero_monotone_increasing:
            for d in range(self._n_digits):
                prob_c_prev = lst_code_probs[d-1] if d > 0 else None
                t_prob_c_d = lst_code_probs[d]
                lst_code_probs[d] = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

        # stack all digits
        # t_code_prob: (n_batch, n_digits, n_ary)
        t_code_prob = torch.stack(lst_code_probs, dim=1)

        # sample continuous relaxed codes
        if self.has_discretizer:
            t_latent_code = self._discretizer.forward(t_code_prob)
        else:
            t_latent_code = t_code_prob

        return t_latent_code, t_code_prob

    def forward_inference_greedy(self, pos: List[str], entity_embeddings: torch.Tensor,
                                 entity_sequence_mask: torch.Tensor,
                                 apply_argmax_on_inference: bool,
                                 **kwargs):
        return None, None

    def forward(self, pos: List[str], entity_embeddings: torch.Tensor,
                entity_sequence_mask: torch.Tensor,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                on_inference: bool = False,
                apply_argmax_on_inference: bool = False,
                **kwargs):

        if not on_inference:
            return self.forward_training(pos, entity_embeddings, entity_sequence_mask, ground_truth_synset_codes, **kwargs)
        else:
            return self.forward_inference_greedy(pos, entity_embeddings, entity_sequence_mask, apply_argmax_on_inference, **kwargs)

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
    def teacher_forcing(self):
        return True
