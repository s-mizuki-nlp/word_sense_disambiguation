#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Tuple
import warnings

import numpy
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
                 discretizer: nn.Module,
                 entity_vector_encoder: Optional[EntityVectorEncoder] = None,
                 initial_states_encoder: Optional[InitialStatesEncoder] = None,
                 **kwargs):

        super(HierarchicalCodeEncoder, self).__init__()
        self._encoder = encoder
        self._encoder_class_name = encoder.__class__.__name__
        self._entity_vector_encoder = entity_vector_encoder
        self._initial_states_encoder = initial_states_encoder

        built_in_discretizer = getattr(self._encoder, "use_built_in_discretizer", False)
        if built_in_discretizer:
            self._discretizer = self._encoder.built_in_discretizer
        else:
            self._discretizer = discretizer

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
        return getattr(self._encoder, "teacher_forcing", True)

    @property
    def use_entity_vector_encoder(self):
        return self._entity_vector_encoder is not None

    @property
    def use_initial_state_encoder(self):
        return self._initial_states_encoder is not None

    @property
    def temperature(self):
        return getattr(self._discretizer, "temperature", None)

    @temperature.setter
    def temperature(self, value):
        if self.temperature is not None:
            setattr(self._discretizer, "temperature", value)

    def forward(self, entity_span_avg_vectors: Optional[torch.Tensor] = None,
                ground_truth_synset_codes: Optional[torch.Tensor] = None,
                entity_embeddings: Optional[torch.Tensor] = None,
                entity_sequence_mask: Optional[torch.BoolTensor] = None,
                entity_sequence_lengths: Optional[torch.LongTensor] = None,
                context_embeddings: Optional[torch.Tensor] = None,
                context_sequence_mask: Optional[torch.BoolTensor] = None,
                context_sequence_lengths: Optional[torch.LongTensor] = None,
                requires_grad: bool = True, enable_discretizer: bool = True, **kwargs):

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
            else:
                raise ValueError(f"We couldn't obtain entity vectors.")

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
                                                                   on_inference=False)
                if self._encoder.use_built_in_discretizer:
                    pass
                else:
                    if enable_discretizer:
                        t_latent_code = self._discretizer.forward(t_code_prob)
                    else:
                        t_latent_code = t_code_prob
            else:
                raise NotImplementedError("Not implemented yet.")

        return t_latent_code, t_code_prob

    def _numpy_to_tensor(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, utils.Array_like):
                kwargs[key] = utils.numpy_to_tensor(value)
        return kwargs

    def _predict(self, **kwargs):
        return self.forward(**kwargs, requires_grad=False, enable_discretizer=False)

    def predict(self, **kwargs):
        kwargs = self._numpy_to_tensor(**kwargs)
        t_latent_code, t_code_prob = self._predict(**kwargs)
        return tuple(map(utils.tensor_to_numpy, (t_latent_code, t_code_prob)))

    def encode(self, adjust_code_probability: bool = False, **kwargs):
        v_code_prob = self.encode_soft(**kwargs, adjust_code_probability=adjust_code_probability)
        v_code = np.argmax(v_code_prob, axis=-1)
        return v_code

    def _encode_soft(self, adjust_code_probability: bool = False, **kwargs):
        t_code_prob = self._encoder.calc_code_probability(**kwargs, adjust_code_probability=adjust_code_probability)
        return t_code_prob

    def encode_soft(self, adjust_code_probability: bool = False, **kwargs):
        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            kwargs = self._numpy_to_tensor(**kwargs)
            t_prob = self._encode_soft(**kwargs, adjust_code_probability=adjust_code_probability)
        return t_prob.cpu().numpy()


class MaskedAutoEncoder(HierarchicalCodeEncoder):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discretizer: nn.Module, masked_value: int = 0, normalize_output_length: bool = True, dtype=torch.float32, **kwargs):
        """

        :param encoder:
        :param decoder:
        :param discretizer:
        :param masked_value: the code value that is masked (=ignored) during decoding process.
            currently valid value is `0`, otherwise it will raise error.
            in future, it may accept multiple values or digit-dependent values.
        :param normalize_output_length:
        :param dtype:
        :param kwargs:
        """
        if not normalize_output_length:
            warnings.warn("it is recommended to enable output length normalization.")

        super(MaskedAutoEncoder, self).__init__(encoder, decoder, discretizer, normalize_output_length, dtype)

        assert masked_value == 0, "currently `mased_value` must be `0`, otherwise it will raise error."
        self._masked_value = masked_value

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def _build_mask_tensor(self, masked_value: int, dtype, device):

        mask_shape = (1, self.n_digits, self.n_ary)
        mask_tensor = torch.ones(mask_shape, dtype=dtype, device=device, requires_grad=False)
        mask_tensor[:,:,masked_value] = 0.0

        return mask_tensor

    def forward(self, t_x: torch.Tensor, requires_grad: bool = True, enable_discretizer: bool = True):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            # encoder and discretizer
            if self._encoder_class_name == "AutoRegressiveLSTMEncoder":
                if self._encoder.use_built_in_discretizer:
                    t_latent_code, t_code_prob = self._encoder.forward(t_x)
                else:
                    _, t_code_prob = self._encoder.forward(t_x)
                    if enable_discretizer:
                        t_latent_code = self._discretizer.forward(t_code_prob)
                    else:
                        t_latent_code = t_code_prob
            else:
                t_code_prob = self._encoder.forward(t_x)
                if enable_discretizer:
                    t_latent_code = self._discretizer.forward(t_code_prob)
                else:
                    t_latent_code = t_code_prob

            # mask intermediate representation
            dtype, device = self._dtype_and_device(t_x)
            mask = self._build_mask_tensor(self._masked_value, dtype, device)
            t_decoder_input = t_latent_code * mask

            # decoder
            t_x_dash = self._decoder.forward(t_decoder_input)

            # length-normalizer
            if self._normalize_output_length:
                t_x_dash = self._normalize(x=t_x, x_dash=t_x_dash)

        return t_latent_code, t_code_prob, t_x_dash
