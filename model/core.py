#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple
import warnings

import numpy
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import ExitStack
from dataset import utils
from .encoder import LSTMEncoder, TransformerEncoder
from .attention import EntityVectorEncoder, InitialStatesEncoder


class HierarchicalCodeEncoder(nn.Module):

    def __init__(self, encoder: [LSTMEncoder, TransformerEncoder],
                 entity_vector_encoder: Optional[EntityVectorEncoder] = None,
                 initial_states_encoder: Optional[InitialStatesEncoder] = None,
                 **kwargs):

        super(HierarchicalCodeEncoder, self).__init__()
        self._encoder = encoder
        self._encoder_class_name = encoder.__class__.__name__
        self._entity_vector_encoder = entity_vector_encoder
        self._initial_states_encoder = initial_states_encoder
        if self._encoder_class_name == "TransformerEncoder":
            if entity_vector_encoder is not None:
                warnings.warn(f"{self._encoder_class_name} does not use `entity_vector_encoder`.")
            if initial_states_encoder is not None:
                warnings.warn(f"{self._encoder_class_name} does not use `initial_states_encoder`.")
            self._entity_vector_encoder = None
            self._initial_states_encoder = None

    @property
    def n_ary(self):
        return self._encoder.n_ary

    @property
    def n_digits(self):
        return self._encoder.n_digits

    @property
    def n_n_code_prefix(self):
        return self._encoder.n_synset_code_prefix

    @property
    def n_dim_hidden(self):
        return self._encoder.n_dim_hidden

    @property
    def teacher_forcing(self):
        return getattr(self._encoder, "teacher_forcing", False)

    @property
    def use_entity_vector_encoder(self):
        return self._entity_vector_encoder is not None

    @property
    def use_initial_state_encoder(self):
        return self._initial_states_encoder is not None

    @property
    def global_attention_type(self):
        return getattr(self._encoder, "global_attention_type", False)

    @property
    def has_discretizer(self):
        return self._encoder.has_discretizer

    def summary(self, include_submodules: bool = False, flatten: bool = False):
        ret = {
            "n_ary": self.n_ary,
            "n_digits": self.n_digits,
            "n_dim": self.n_dim_hidden,
            "entity_vector_encoder": self.use_entity_vector_encoder,
            "initial_state_encoder": self.use_initial_state_encoder,
            "global_attention_type": self.global_attention_type,
            "teacher_forcing": self.teacher_forcing,
            "has_discretizer": self.has_discretizer
        }
        if include_submodules:
            ret["encoder"] = self._encoder.summary()
            if self.use_entity_vector_encoder:
                ret["entity_vector_encoder"] = self._entity_vector_encoder.summary()
            if self.use_initial_state_encoder:
                ret["initial_state_encoder"] = self._initial_states_encoder.summary()

            if flatten:
                ret = pd.json_normalize(ret)

        return ret

    def forward_by_lstm_encoder(self, pos: Optional[List[str]] = None,
                                entity_span_avg_vectors: Optional[torch.Tensor] = None,
                                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                                entity_embeddings: Optional[torch.Tensor] = None,
                                entity_sequence_mask: Optional[torch.BoolTensor] = None,
                                context_embeddings: Optional[torch.Tensor] = None,
                                context_sequence_mask: Optional[torch.BoolTensor] = None,
                                context_sequence_lengths: Optional[torch.LongTensor] = None,
                                on_inference: bool = False,
                                apply_argmax_on_inference: bool = False,
                                **kwargs):

        # entity vectors
        ## 1. calculate entity vectors using encoder.
        if self.use_entity_vector_encoder:
            entity_vectors = self._entity_vector_encoder.forward(entity_embeddings=entity_embeddings,
                                                                 context_embeddings=context_embeddings,
                                                                 entity_sequence_mask=entity_sequence_mask,
                                                                 context_sequence_mask=context_sequence_mask)
        ## 2. use entity span averaged vectors as it is.
        elif entity_span_avg_vectors is not None:
            entity_vectors = entity_span_avg_vectors
        ## 3. encoder does not use entity vector.
        elif self._encoder.input_entity_vector == False:
            entity_vectors = None
        else:
            raise ValueError(f"We couldn't prepare entity vectors.")

        # initial states
        if self.use_initial_state_encoder:
            init_states = self._initial_states_encoder.forward(entity_embeddings=entity_embeddings,
                                                               context_embeddings=context_embeddings,
                                                               entity_sequence_mask=entity_sequence_mask,
                                                               context_sequence_mask=context_sequence_mask)
        else:
            init_states = None

        # ground truth codes
        if not self.teacher_forcing:
            ground_truth_synset_codes = None

        # encoder and discretizer
        t_latent_code, t_code_prob = self._encoder.forward(pos=pos,
                                                           entity_vectors=entity_vectors,
                                                           ground_truth_synset_codes=ground_truth_synset_codes,
                                                           context_embeddings=context_embeddings,
                                                           context_sequence_lengths=context_sequence_lengths,
                                                           init_states=init_states,
                                                           on_inference=on_inference,
                                                           apply_argmax_on_inference=apply_argmax_on_inference)
        return t_latent_code, t_code_prob

    def forward_by_tf_encoder(self, pos: Optional[List[str]] = None,
                              ground_truth_synset_codes: Optional[torch.Tensor] = None,
                              ground_truth_synset_code_prefixes: Optional[torch.Tensor] = None,
                              entity_embeddings: Optional[torch.Tensor] = None,
                              entity_sequence_mask: Optional[torch.BoolTensor] = None,
                              context_embeddings: Optional[torch.Tensor] = None,
                              context_sequence_mask: Optional[torch.Tensor] = None,
                              subword_spans: List[List[List[int]]] = None,
                              on_inference: bool = False,
                              **kwargs):
        t_latent_code, t_code_prob = self._encoder.forward(pos=pos,
                                                           entity_embeddings=entity_embeddings,
                                                           entity_sequence_mask=entity_sequence_mask,
                                                           ground_truth_synset_codes=ground_truth_synset_codes,
                                                           ground_truth_synset_code_prefixes=ground_truth_synset_code_prefixes,
                                                           context_embeddings=context_embeddings,
                                                           context_sequence_mask=context_sequence_mask,
                                                           subword_spans=subword_spans,
                                                           on_inference=on_inference,
                                                           **kwargs)

        return t_latent_code, t_code_prob

    def forward(self, pos: Optional[List[str]] = None,
                entity_span_avg_vectors: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                ground_truth_synset_code_prefixes: Optional[torch.Tensor] = None,
                entity_embeddings: Optional[torch.Tensor] = None,
                entity_sequence_mask: Optional[torch.BoolTensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_mask: Optional[torch.BoolTensor] = None,
                context_sequence_lengths: Optional[torch.LongTensor] = None,
                subword_spans: Optional[List[List[List[int]]]] = None,
                requires_grad: bool = True,
                on_inference: bool = False,
                apply_argmax_on_inference: bool = False,
                **kwargs):

        if pos is not None:
            if isinstance(pos, str):
                pos = [pos]
        if subword_spans is not None:
            element = subword_spans[0][0]
            if isinstance(element, int):
                subword_spans = [subword_spans]

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            if self._encoder_class_name == "LSTMEncoder":
                t_latent_code, t_code_prob = self.forward_by_lstm_encoder(pos=pos,
                                                                          entity_span_avg_vectors=entity_span_avg_vectors,
                                                                          ground_truth_synset_codes=ground_truth_synset_codes,
                                                                          entity_embeddings=entity_embeddings,
                                                                          entity_sequence_mask=entity_sequence_mask,
                                                                          context_embeddings=context_embeddings,
                                                                          context_sequence_mask=context_sequence_mask,
                                                                          context_sequence_lengths=context_sequence_lengths,
                                                                          on_inference=on_inference,
                                                                          apply_argmax_on_inference=apply_argmax_on_inference,
                                                                          **kwargs)
            elif self._encoder_class_name == "TransformerEncoder":
                t_latent_code, t_code_prob = self.forward_by_tf_encoder(pos=pos,
                                                                        entity_embeddings=entity_embeddings,
                                                                        entity_sequence_mask=entity_sequence_mask,
                                                                        context_embeddings=context_embeddings,
                                                                        context_sequence_mask=context_sequence_mask,
                                                                        ground_truth_synset_codes=ground_truth_synset_codes,
                                                                        ground_truth_synset_code_prefixes=ground_truth_synset_code_prefixes,
                                                                        subword_spans=subword_spans,
                                                                        on_inference=on_inference,
                                                                        **kwargs)
            else:
                raise NotImplementedError(f"unknown encoder module: {self._encoder_class_name}")

        return t_latent_code, t_code_prob

    def _numpy_to_tensor_bulk(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                kwargs[key] = utils.numpy_to_tensor(value)
        return kwargs

    def _predict(self, **kwargs):
        """
        predict conditional probability and sense code. it takes both ground-truth synset code and postag into account.

        @param kwargs:
        @return: t_code: (n_batch, n_digits), t_code_prob: (n_batch, n_digits, n_ary)
        """
        assert kwargs.get("ground_truth_synset_codes",None) is not None, \
            f"`ground_truth_synset_codes` is missing."
        assert kwargs.get("ground_truth_synset_code_prefixes",None) is not None, \
            f"`ground_truth_synset_code_prefixes` is missing."
        t_code, t_code_prob = self.forward(**kwargs, requires_grad=False, on_inference=True)
        if t_code.ndim == 3:
            t_code = t_code.argmax(dim=-1)
        return t_code, t_code_prob

    def predict(self, **kwargs):
        """
        predict conditional probability and sense code. it takes both ground-truth synset code and postag into account.

        @param kwargs:
        @return: t_code: (n_batch, n_digits), t_code_prob: (n_batch, n_digits, n_ary)
        """
        kwargs = self._numpy_to_tensor_bulk(**kwargs)
        t_code, t_code_prob = self._predict(**kwargs)
        v_code, v_code_prob = tuple(map(utils.tensor_to_numpy, (t_code, t_code_prob)))
        return v_code, v_code_prob

    def _encode(self, **kwargs):
        """
        encode features into sense code. greedy decoding is applied. it ignores ground-truth synset codes but takes postag into account.

        @param kwargs:
        @return:
        """
        kwargs["ground_truth_synset_codes"] = None
        kwargs["ground_truth_synset_code_prefixes"] = None
        t_code, t_code_prob = self.forward(**kwargs, requires_grad=False, on_inference=True, apply_argmax_on_inference=True)
        if t_code.ndim == 3:
            t_code = t_code.argmax(dim=-1)
        return t_code, t_code_prob

    def encode(self, **kwargs):
        """
        encode features into sense code. greedy decoding is applied. it ignores ground-truth synset codes but takes postag into account.

        @param kwargs:
        @return: t_code: (n_batch, n_digits), t_code_prob: (n_batch, n_digits, n_ary)
        """
        kwargs = self._numpy_to_tensor_bulk(**kwargs)
        t_code, t_code_prob = self._encode(**kwargs)
        v_code, v_code_prob = tuple(map(utils.tensor_to_numpy, (t_code, t_code_prob)))
        return v_code, v_code_prob