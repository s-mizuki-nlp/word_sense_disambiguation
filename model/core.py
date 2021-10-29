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
from .encoder import LSTMEncoder
from .attention import EntityVectorEncoder, InitialStatesEncoder

class HierarchicalCodeEncoder(nn.Module):

    def __init__(self, encoder: LSTMEncoder,
                 entity_vector_encoder: Optional[EntityVectorEncoder] = None,
                 initial_states_encoder: Optional[InitialStatesEncoder] = None,
                 **kwargs):

        super(HierarchicalCodeEncoder, self).__init__()
        self._encoder = encoder
        self._encoder_class_name = encoder.__class__.__name__
        self._entity_vector_encoder = entity_vector_encoder
        self._initial_states_encoder = initial_states_encoder

    @property
    def n_ary(self):
        return self._encoder.n_ary

    @property
    def n_digits(self):
        return self._encoder.n_digits

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

    @property
    def temperature(self):
        return getattr(self._encoder.discretizer, "temperature", None)

    @temperature.setter
    def temperature(self, value):
        if self.temperature is not None:
            setattr(self._encoder.discretizer, "temperature", value)

    def summary(self, include_submodules: bool = False, flatten: bool = False):
        ret = {
            "n_ary": self.n_ary,
            "n_digits": self.n_digits,
            "n_dim": self.n_dim_hidden,
            "use_entity_vector_encoder": self.use_entity_vector_encoder,
            "use_initial_state_encoder": self.use_initial_state_encoder,
            "global_attention_type": self.global_attention_type,
            "teacher_forcing": self.teacher_forcing,
            "has_discretizer": self.has_discretizer
        }
        if include_submodules:
            ret["encoder"] = self._encoder.summary()
            if self.use_entity_vector_encoder:
                ret["entity_vector_encoder"] = self._entity_vector_encoder.summary()
            if self.use_initial_state_encoder:
                ret["use_initial_state_encoder"] = self._initial_states_encoder.summary()

            if flatten:
                ret = pd.json_normalize(ret)

        return ret

    def forward(self, entity_span_avg_vectors: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                entity_embeddings: Optional[torch.Tensor] = None,
                entity_sequence_mask: Optional[torch.BoolTensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_mask: Optional[torch.BoolTensor] = None,
                context_sequence_lengths: Optional[torch.LongTensor] = None,
                requires_grad: bool = True,
                on_inference: bool = False,
                **kwargs):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

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
            if self._encoder_class_name == "LSTMEncoder":
                t_latent_code, t_code_prob = self._encoder.forward(entity_vectors=entity_vectors,
                                                                   ground_truth_synset_codes=ground_truth_synset_codes,
                                                                   context_embeddings=context_embeddings,
                                                                   context_sequence_lengths=context_sequence_lengths,
                                                                   init_states=init_states,
                                                                   on_inference=on_inference)
            else:
                raise NotImplementedError("Not implemented yet.")

        return t_latent_code, t_code_prob

    def _numpy_to_tensor_bulk(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                kwargs[key] = utils.numpy_to_tensor(value)
        return kwargs

    def _predict(self, **kwargs):
        kwargs["ground_truth_synset_codes"] = None
        return self.forward(**kwargs, requires_grad=False, on_inference=True)

    def predict(self, **kwargs):
        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            kwargs = self._numpy_to_tensor_bulk(**kwargs)
            t_latent_code, t_code_prob = self._predict(**kwargs)
        return tuple(map(utils.tensor_to_numpy, (t_latent_code, t_code_prob)))

    def encode(self, **kwargs):
        v_latent_code, v_code_prob = self.predict(**kwargs)
        v_code = np.argmax(v_code_prob, axis=-1)
        return v_code
