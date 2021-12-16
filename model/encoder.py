#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os.path
import warnings
from typing import Optional, Dict, Any, Union, Tuple, List, Iterable

import io, math, copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from dataset import WSDTaskDataset
from dataset.lexical_knowledge import SynsetDataset

from .onmt.global_attention import GlobalAttention
from .encoder_internal import \
    PositionalEncoding
from .embedding_layer import HashCodeAwareEmbedding, PositionAwareEmbedding


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
                 pos_index: Dict[str, int],
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
        self._pos_index = pos_index
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

        if entity_vectors is not None:
            t = entity_vectors
        elif init_states is not None:
            t = init_states[0]
        else:
            raise AssertionError(f"neighter `entity_vectors` nor `init_states` is available.")
        dtype, device = self._dtype_and_device(t)
        n_batch = t.shape[0]

        if on_inference and (ground_truth_synset_codes is None):
            on_generation = True
        else:
            on_generation = False

        if ground_truth_synset_codes is not None:
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
            if on_generation:
                if apply_argmax_on_inference:
                    t_latent_code_d = F.one_hot(t_prob_c_d.argmax(dim=-1), num_classes=self._n_ary).type(dtype)
                else:
                    # empirically, embedding based on the probability produces better result then argmax.
                    # I guess it minimizes difference on training and inference.
                    t_latent_code_d = t_prob_c_d
            else:
                if on_inference:
                    ## on inference but not generation -> conditional probability Pr{Y_d|y_{<d}}
                    t_latent_code_d = torch.index_select(ground_truth_synset_codes, dim=1, index=torch.tensor(d, device=device)).squeeze()
                else:
                    ## on training -> stochastic sampling if discretizer is available.
                    if self._teacher_forcing:
                        t_latent_code_d = torch.index_select(ground_truth_synset_codes, dim=1, index=torch.tensor(d, device=device)).squeeze()
                    elif self.has_discretizer:
                        t_latent_code_d = self._discretizer(t_prob_c_d)
                    else:
                        t_latent_code_d = t_prob_c_d

            # compute the embeddings of previous code
            if d != (self._n_digits - 1):
                # calculate embeddings
                o_d = t_latent_code_d.detach()
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
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 n_head: int,
                 pos_index: Dict[str, int],
                 softmax_logit_layer: nn.Module,
                 sequence_direction: str = "left_to_right",
                 memory_encoder_input_feature: str = "entity",
                 layer_normalization: bool = False,
                 norm_digit_embeddings: bool = False,
                 ignore_trailing_zero_digits: bool = False,
                 trainable_positional_encoding: bool = False,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 prob_zero_monotone_increasing: bool = False,
                 embedding_layer_type: str = "default",
                 **kwargs):

        super().__init__(n_ary=n_ary)

        AVAILABLE_VALUES = ("left_to_right", "both", "right_to_left")
        assert sequence_direction in AVAILABLE_VALUES, f"invalid `sequence_direction` value: {AVAILABLE_VALUES}"

        if kwargs.get("teacher_forcing", False):
            warnings.warn(f"`teacher_forcing=True` is invalid for TransformerEncoder module.")
        if num_encoder_layers > 0:
            set_features = {"entity","context"}
            if memory_encoder_input_feature not in set_features:
                raise ValueError(f"`memory_encoder_input_feature` must be: {set_features}")
        else:
            memory_encoder_input_feature = None

        self._n_dim_hidden = n_dim_emb
        self._n_digits = n_digits
        self._softmax_logit_layer = softmax_logit_layer
        # self._n_ary = n_ary
        self._pos_index = pos_index
        self._num_encoder_layers = num_encoder_layers
        self._memory_encoder_input_feature = memory_encoder_input_feature
        self._num_decoder_layers = num_decoder_layers
        self._n_head = n_head
        self._layer_normalization = layer_normalization
        self._dropout = dropout
        self._norm_digit_embeddings = norm_digit_embeddings
        self._ignore_trailing_zero_digits = ignore_trailing_zero_digits
        self._trainable_positional_encoding = trainable_positional_encoding
        self._batch_first = batch_first
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing
        self._embedding_layer_type = embedding_layer_type
        self._sequence_direction = sequence_direction

        self._kwargs = kwargs

        self._build()

    def _build(self):

        # digit embeddings
        cfg_emb_layer = {
            "embedding_dim":self._n_dim_hidden,
            "max_norm":1.0 if self._norm_digit_embeddings else None,
            "padding_idx":0 if self._ignore_trailing_zero_digits else None,
            "num_embeddings": self._n_ary + len(self._pos_index) # n_pos will be used for beginning-of-sense-code symbol.
        }
        if self._embedding_layer_type == "default":
            self._emb_layer = PositionAwareEmbedding(n_seq_len=None, **cfg_emb_layer)
        elif self._embedding_layer_type == "position_aware":
            self._emb_layer = PositionAwareEmbedding(n_seq_len=self._n_digits, **cfg_emb_layer)
        elif self._embedding_layer_type == "hash":
            self._emb_layer = HashCodeAwareEmbedding(n_seq_len=self._n_digits,
                                                     num_buckets=self._kwargs.get("num_buckets", 10000),
                                                     num_embeddings=self._n_synset_code_prefix+1,
                                                     num_hashes=self._kwargs.get("num_hashes", 2),
                                                     embedding_dim=self._n_dim_hidden,
                                                     append_weight=True,
                                                     replace_trailing_zeroes=False # False = fill with zeroes
                                                     )
        else:
            raise AssertionError(f"unknown `embedding_layer_type` value: {self._embedding_layer_type}")

        # positional encodings
        cfg_pe_layer = {
            "d_model":self._n_dim_hidden,
            "dropout":self._dropout,
            "max_len":self._n_digits,
            "trainable":self._trainable_positional_encoding
        }
        self._pe_layer = PositionalEncoding(**cfg_pe_layer)

        if self._layer_normalization:
            norm = nn.LayerNorm(normalized_shape=self._n_dim_hidden)
        else:
            norm = None

        # entity token embeddings encoder
        if self._num_encoder_layers > 0:
            cfg_transformer_encoder_layer = {
                "d_model":self._n_dim_hidden,
                "nhead":self._n_head,
                "dim_feedforward": self._n_dim_hidden * 4,
                "dropout":self._dropout
            }
            encoder_layer = nn.TransformerEncoderLayer(**cfg_transformer_encoder_layer)
            self._memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self._num_encoder_layers, norm=norm)
        else:
            self._memory_encoder = None

        # sense code decoder
        cfg_transformer_decoder_layer = {
            "d_model":self._n_dim_hidden,
            "nhead":self._n_head,
            "dim_feedforward": self._n_dim_hidden * 4,
            "dropout":self._dropout
        }
        decoder_layer = nn.TransformerDecoderLayer(**cfg_transformer_decoder_layer)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=self._num_decoder_layers, norm=norm)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self._emb_layer.init_weights(-initrange, initrange)
        self._softmax_logit_layer.init_weights()

    def create_sequence_inputs(self, lst_pos: List[str], device,
                               ground_truth_synset_codes: Optional[Union[List[List[int]], torch.Tensor]] = None,
                               trim: bool = True,
                               mask_index: int = 0) -> torch.Tensor:

        if ground_truth_synset_codes is None:
            n_batch = len(lst_pos)
            n_synset_code_len = 0
        elif isinstance(ground_truth_synset_codes, list):
            n_batch = len(ground_truth_synset_codes)
            n_synset_code_len = len(ground_truth_synset_codes[0])
            ground_truth_synset_codes = torch.LongTensor(ground_truth_synset_codes, device="cpu").to(device)
        else:
            n_batch = ground_truth_synset_codes[0]
            n_synset_code_len = ground_truth_synset_codes.shape[-1]
        n_seq_len = min(self._n_digits, n_synset_code_len + 1)

        if self._sequence_direction == "both":
            t_inputs = torch.full(size=(n_batch, n_seq_len), fill_value=mask_index, device="cpu", dtype=torch.long).to(device)

        elif self._sequence_direction == "left_to_right":
            # 1. prepend PoS index
            # t_boc: (n_batch, 1)
            lst_pos_idx = [self._pos_index[pos] for pos in lst_pos]
            t_boc = torch.LongTensor(lst_pos_idx, device="cpu").unsqueeze(dim=-1).to(device)

            if ground_truth_synset_codes is None:
                return t_boc

            # 2. concat with right-shifted ground-truth codes.
            # e.g. ground_truth = [1,2,3,4,0,0]; t_inputs = [idx(pos),1,2,3,4,0]
            # ground_truth_synset_codes: (n_batch, <=n_digits)
            # t_inputs: (n_batch, <n_digits)
            t_inputs = torch.cat((t_boc, ground_truth_synset_codes), dim=-1)
            # trim the input sequence so that it would not exceed the number of digits.
            if trim:
                t_inputs = t_inputs[:,:self._n_digits]

        elif self._sequence_direction == "right_to_left":
            # 1. prepare to-be-appended
            t_eoc = torch.full(size=(n_batch, 1), fill_value=mask_index, device="cpu", dtype=torch.long).to(device)

            if ground_truth_synset_codes is None:
                return t_eoc

            # left-shifted ground-truth codes.
            # e.g. ground_truth = [1,2,3,4,0,0]; t_inputs = [2,3,4,0,0,0]
            # ground_truth_synset_codes: (n_batch, <=n_digits)
            t_inputs = torch.cat((ground_truth_synset_codes[:,1:], t_eoc), dim=-1)

        return t_inputs

    def setup_sense_code_prefix_index(self, synset_dataset: SynsetDataset):
        sense_code_prefix_index = synset_dataset.sense_code_prefix_index

        if hasattr(self._softmax_logit_layer, "sense_code_prefix_index"):
            self._softmax_logit_layer.sense_code_prefix_index = sense_code_prefix_index
            print(f"sense code prefix index is set: {self._softmax_logit_layer.__class__.__name__} -> {len(self._softmax_logit_layer.sense_code_prefix_index)}")
        if hasattr(self._emb_layer, "sense_code_prefix_index"):
            self._emb_layer.sense_code_prefix_index = sense_code_prefix_index
            print(f"sense code prefix index is set: {self._emb_layer.__class__.__name__} -> {len(self._emb_layer.sense_code_prefix_index)}")

    def setup_sense_code_prefix_statistics(self, trainset: WSDTaskDataset, synset_dataset: Optional[SynsetDataset] = None,
                                           use_prefix_index_as_lookup_key: bool = True,
                                           path_cache: str = None):
        if not hasattr(self._softmax_logit_layer, "sense_code_prefix_statistics"):
            print(f"logit layer doesn't support prefix statistics. do nothing.")
            return
        else:
            logit_adjustment = getattr(self._softmax_logit_layer, "logit_adjustment", False)
            if logit_adjustment:
                synset_dataset = trainset.synset_dataset if synset_dataset is None else synset_dataset

                if (path_cache is not None) and os.path.exists(path_cache):
                    print(f"loading prefix statistics from cache file: {path_cache}")
                    def _key_str_to_int(x):
                        if isinstance(x, dict):
                            return {int(k):v for k,v in x.items()}
                        return x
                    with io.open(path_cache, mode="r") as ifs:
                        prefix_stats = json.load(ifs, object_hook=_key_str_to_int)
                    print("validating compatibility of loaded prefix statistics with synset dataset.")
                    synset_dataset.validate_synset_code_prefix_statistics(prefix_statistics=prefix_stats)
                else:
                    print(f"building prefix statistics from trainset...")
                    prefix_stats = synset_dataset.count_synset_code_prefix_next_values(trainset, use_index_as_lookup_key=use_prefix_index_as_lookup_key)
                    if (path_cache is not None) and not(os.path.exists(path_cache)):
                        print(f"prefix statistics will be saved as: {path_cache}")
                        with io.open(path_cache, mode="w") as ofs:
                            json.dump(prefix_stats, ofs)

                self._softmax_logit_layer.sense_code_prefix_statistics = prefix_stats
                assert self._softmax_logit_layer.sense_code_prefix_statistics is not None, f"something went wrong: {self._softmax_logit_layer.__dict__}"
                print(f"sense code prefix stats has been set: {len(self._softmax_logit_layer.sense_code_prefix_statistics)}")
            else:
                print(f"`logit_adjustment` is configured as False. do nothing.")

    @staticmethod
    def generate_square_subsequent_mask(sz: int, left_to_right: bool) -> torch.Tensor:
        r"""
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        ref: https://pytorch.org/docs/1.10.0/generated/torch.nn.Transformer.html
        """
        if left_to_right:
            mask_tensor = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        else:
            mask_tensor = torch.tril(torch.full((sz, sz), float("-inf")), diagonal=-1)
        return mask_tensor

    def _extract_entity_embeddings_from_context_embeddings(self,
                                    context_embeddings: torch.Tensor,
                                    subword_spans: List[List[List[int]]],
                                    padding_value: float = 0.0) -> torch.Tensor:
        """
        extracts entity subword embeddings from context embeddings specified by subword_spans argument.

        @param context_embeddings: (n_batch, n_max_seq_length, n_dim).
        @param subword_spans: list of the list of span of the subwords within the entity. i.e. [[[2,4],[4,5]]]. subword_spans[batch][word][subword]
        @return: (n_batch, n_max_entity_span, n_dim)
        """
        lst_subword_ranges = [(subword_span[0][0], subword_span[-1][1]) for subword_span in subword_spans]
        it = zip(lst_subword_ranges, context_embeddings)
        lst_entity_embeddings = [context[slice(*subword_span),:] for subword_span, context in it]
        entity_embeddings = pad_sequence(lst_entity_embeddings, batch_first=True, padding_value=padding_value)
        return entity_embeddings

    def forward_base_memory_encoder(self, entity_embeddings: torch.Tensor,
                                    entity_sequence_mask: torch.Tensor,
                                    context_embeddings: Optional[torch.Tensor] = None,
                                    context_sequence_mask: Optional[torch.Tensor] = None,
                                    subword_spans: List[List[List[int]]] = None,
                                    **kwargs):
        # prepare memory embeddings
        if self.has_memory_encoder:
            if self._memory_encoder_input_feature == "entity":
                src = entity_embeddings
                src_key_padding_mask = entity_sequence_mask
            elif self._memory_encoder_input_feature == "context":
                src = context_embeddings
                src_key_padding_mask = context_sequence_mask

            if self._batch_first:
                src = src.swapaxes(0, 1)

            memory = self._memory_encoder.forward(src=src, src_key_padding_mask=src_key_padding_mask)
            if self._batch_first:
                memory = memory.swapaxes(0, 1)

            if self._memory_encoder_input_feature == "context":
                memory = self._extract_entity_embeddings_from_context_embeddings(context_embeddings=memory,
                                                                                 subword_spans=subword_spans)
        else:
            memory = entity_embeddings

        # memory: (n_batch, n_max_entity_span, n_dim)
        return memory

    def forward_base(self, input_sequence: torch.Tensor,
                     entity_embeddings: torch.Tensor,
                     entity_sequence_mask: torch.Tensor,
                     context_embeddings: Optional[torch.Tensor] = None,
                     context_sequence_mask: Optional[torch.Tensor] = None,
                     subword_spans: List[List[List[int]]] = None,
                     **kwargs):
        """

        @param input_sequence: (n_batch, <=n_digits). sequence of the input codes.
        @param entity_embeddings: (n_batch, max(entity_span), n_emb). stack of the subword embeddings within entity span.
        @param entity_sequence_mask: (n_batch, max(entity_span)). mask of the entity embeddings.
        @param apply_autoregressive_mask: autoregressive (=left-to-right) prediction (True) or not (False)
        @param kwargs:
        @return:
        """

        _, device = self._dtype_and_device(input_sequence)
        n_digits = min(self._n_digits, input_sequence.shape[-1])

        # compute target embeddings: (n_batch, n_digits, n_dim_emb)
        t_emb = self._emb_layer.forward(input_sequence) * math.sqrt(self._n_dim_hidden)
        tgt = self._pe_layer.forward(t_emb)

        # prepare memory embeddings and masks
        memory = self.forward_base_memory_encoder(entity_embeddings=entity_embeddings,
                                                  entity_sequence_mask=entity_sequence_mask,
                                                  context_embeddings=context_embeddings,
                                                  context_sequence_mask=context_sequence_mask,
                                                  subword_spans=subword_spans)
        memory_key_padding_mask = entity_sequence_mask

        # prepare subsequent masks: (n_digits, n_digits)
        if self._sequence_direction == "left_to_right":
            tgt_mask = self.generate_square_subsequent_mask(sz=n_digits, left_to_right=True).to(device)
        elif self._sequence_direction == "right_to_left":
            tgt_mask = self.generate_square_subsequent_mask(sz=n_digits, left_to_right=False).to(device)
        elif self._sequence_direction == "both":
            tgt_mask = None

        # compute decoding
        if self._batch_first:
            tgt = tgt.swapaxes(0, 1)
            memory = memory.swapaxes(0, 1)
        h_out = self._decoder.forward(tgt=tgt, tgt_mask=tgt_mask, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        if self._batch_first:
            h_out = h_out.swapaxes(0, 1)

        # compute Pr{Y_d|y_{<d}}
        # t_code_probs: (n_batch, n_digits, n_ary)
        t_logits = self._softmax_logit_layer(t_representation=h_out, input_sequence=input_sequence)
        t_code_prob = F.softmax(t_logits, dim=-1)

        # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
        if self._prob_zero_monotone_increasing:
            lst_t_code_prob = [t_prob for t_prob in t_code_prob.swapaxes(0,1)]
            for d in range(n_digits):
                prob_d_prev = lst_t_code_prob[d-1] if d > 0 else None
                prob_d = lst_t_code_prob[d]
                lst_t_code_prob[d] = self._adjust_code_probability_to_monotone_increasing(probs=prob_d, probs_prev=prob_d_prev)
            t_code_prob = torch.stack(lst_t_code_prob, dim=1)

        # we never apply stochastic sampling because transformer does not accept student forcing.
        t_latent_code = t_code_prob

        return t_latent_code, t_code_prob

    def forward_greedy_decoding(self, pos: List[str], entity_embeddings: torch.Tensor,
                                entity_sequence_mask: torch.Tensor,
                                context_embeddings: Optional[torch.Tensor] = None,
                                context_sequence_mask: Optional[torch.Tensor] = None,
                                subword_spans: List[List[List[int]]] = None,
                                **kwargs):
        # decoding using greedy search
        # return: t_code_probs: (n_batch, n_digits, n_ary). t_code_probs[n,d] = Pr(Y|\hat{y_{<d}})
        # return: t_codes: (n_batch, n_digits). t_codes[n,d] = argmax_{a}{Pr(Y=a|\hat{y_{<d}})} = \hat{y_{d}}

        dtype, device = self._dtype_and_device(entity_embeddings)
        n_batch = len(pos)

        if self._sequence_direction == "both":
            dummy_inputs = torch.zeros((n_batch, self._n_digits), device=device, dtype=torch.long)
            input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device, ground_truth_synset_codes=dummy_inputs)
            _, t_code_probs = self.forward_base(input_sequence=input_sequence,
                                                entity_embeddings=entity_embeddings,
                                                entity_sequence_mask=entity_sequence_mask,
                                                context_embeddings=context_embeddings,
                                                context_sequence_mask=context_sequence_mask,
                                                subword_spans=subword_spans,
                                                **kwargs)
            t_codes = t_code_probs.argmax(dim=-1)

        else:
            for digit in range(self._n_digits):
                if digit == 0:
                    input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device, ground_truth_synset_codes=None)
                else:
                    input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device, ground_truth_synset_codes=t_codes)

                # t_code_probs_upto_d: (n_batch, digit+1, n_ary)
                _, t_code_probs_upto_d = self.forward_base(input_sequence=input_sequence,
                                                           entity_embeddings=entity_embeddings,
                                                           entity_sequence_mask=entity_sequence_mask,
                                                           context_embeddings=context_embeddings,
                                                           context_sequence_mask=context_sequence_mask,
                                                           subword_spans=subword_spans,
                                                           **kwargs)

                if self._sequence_direction == "left_to_right":
                    target_digit = digit
                elif self._sequence_direction == "right_to_left":
                    target_digit = 0
                # t_code_probs_d: (n_batch, 1, n_ary)
                t_code_probs_d = t_code_probs_upto_d[:, target_digit, :].unsqueeze(1)
                # t_codes_d: (n_batch, 1)
                t_codes_d = t_code_probs_d.argmax(dim=-1)

                # concat with previous sequences
                if digit == 0:
                    t_codes, t_code_probs = t_codes_d, t_code_probs_d
                else:
                    if self._sequence_direction == "left_to_right":
                        t_codes = torch.cat([t_codes, t_codes_d], dim=-1)
                        t_code_probs = torch.cat([t_code_probs, t_code_probs_d], dim=1)
                    elif self._sequence_direction == "right_to_left":
                        t_codes = torch.cat([t_codes_d, t_codes], dim=-1)
                        t_code_probs = torch.cat([t_code_probs_d, t_code_probs], dim=1)

        return t_codes, t_code_probs

    def forward(self, pos: List[str], entity_embeddings: torch.Tensor,
                entity_sequence_mask: torch.Tensor,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_mask: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                subword_spans: List[List[List[int]]] = None,
                on_inference: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        if on_inference and (ground_truth_synset_codes is None):
            on_generation = True
        else:
            on_generation = False

        if on_generation:
            # if ground-truth is not given, it generates sense code from scratch.
            return self.forward_greedy_decoding(pos=pos,
                                                entity_embeddings=entity_embeddings,
                                                entity_sequence_mask=entity_sequence_mask,
                                                context_embeddings=context_embeddings,
                                                context_sequence_mask=context_sequence_mask,
                                                subword_spans=subword_spans,
                                                **kwargs)
        else:
            # if ground-truth is given, it computes conditional probability: Pr{Y_d|y_{<d}}
            # regardless of on_inference is true or not.
            dtype, device = self._dtype_and_device(entity_embeddings)
            input_sequence = self.create_sequence_inputs(lst_pos=pos, device=device,
                                                         ground_truth_synset_codes=ground_truth_synset_codes)
            return self.forward_base(input_sequence=input_sequence,
                                     entity_embeddings=entity_embeddings,
                                     entity_sequence_mask=entity_sequence_mask,
                                     context_embeddings=context_embeddings,
                                     context_sequence_mask=context_sequence_mask,
                                     subword_spans=subword_spans,
                                     **kwargs)

    @property
    def has_discretizer(self):
        return False

    @property
    def has_memory_encoder(self):
        return self._memory_encoder is not None

    @property
    def entity_embeddings_encoder(self):
        return self._memory_encoder

    @property
    def discretizer(self):
        return None

    @property
    def n_digits(self):
        return self._n_digits

    @property
    def n_ary(self):
        return self._n_ary

    @property
    def n_synset_code_prefix(self):
        return self._n_synset_code_prefix

    @property
    def n_dim_hidden(self):
        return self._n_dim_hidden

    @property
    def teacher_forcing(self):
        return True

    def summary(self):
        ret = {
            "class_name": self.__class__.__name__,
            "n_dim_hidden": self._n_dim_hidden,
            "n_head": self._n_head,
            "sequence_direction": self._sequence_direction,
            "num_decoder_layers": self._num_decoder_layers,
            "num_encoder_layers": self._num_encoder_layers,
            "memory_encoder_input_feature": self._memory_encoder_input_feature,
            "pos_index": self._pos_index,
            "layer_normalization": self._layer_normalization,
            "embedding_layer_type": self._embedding_layer_type,
            "logit_layer_class": self._softmax_logit_layer.__class__.__name__
        }
        if hasattr(self._softmax_logit_layer, "summary"):
            ret["logit_layer"] = self._softmax_logit_layer.summary()
        return ret