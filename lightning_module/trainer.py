#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Callable, Union, Any
import warnings
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.modules.loss import _Loss
from torch import optim
from torch.optim import Adam
from torch_optimizer import RAdam
from pytorch_lightning import LightningModule

from model import HierarchicalCodeEncoder
from model.loss_supervised import HyponymyScoreLoss, EntailmentProbabilityLoss, CrossEntropyLossWrapper
from custom.optimizer import init_adam_with_warmup_optimizer

class SenseCodeTrainer(LightningModule):

    def __init__(self,
                 model: HierarchicalCodeEncoder,
                 loss_supervised: Union[HyponymyScoreLoss, EntailmentProbabilityLoss, CrossEntropyLossWrapper],
                 optimizer_params: Dict[str, Any],
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 ):

        super().__init__()

        self._loss_supervised = loss_supervised
        self._scale_loss_supervised = loss_supervised.scale

        self._model = model
        self._encoder = model._encoder
        self.n_ary = model.n_ary
        self.n_digits = model.n_digits

        # ToDo: implement hyper-parameter export feature on encoder when saving hyper-parameters are helpful.
        hparams = {"n_ary":model.n_ary, "n_digits":model.n_digits, "encoder.n_dim_hidden":model.n_dim_hidden}
        self.save_hyperparameters(hparams)

        self._optimizer_class_name = optimizer_params.pop("class_name")
        self._optimizer_params = optimizer_params
        # auxiliary function that is solely used for validation
        self._aux_hyponymy_score = HyponymyScoreLoss()
        self._aux_cross_entropy = CrossEntropyLossWrapper()

        # set model parameter scheduler
        if model_parameter_schedulers is None:
            self._model_parameter_schedulers = {}
        else:
            self._model_parameter_schedulers = model_parameter_schedulers

        if loss_parameter_schedulers is None:
            self._loss_parameter_schedulers = {}
        else:
            self._loss_parameter_schedulers = loss_parameter_schedulers

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).to(self._device)

    def _get_model_device(self):
        return (next(self._model.parameters())).device

    def configure_optimizers(self):
        if self._optimizer_class_name == "Adam":
            opt = Adam(self.parameters(), **self._optimizer_params)
        elif self._optimizer_class_name == "RAdam":
            opt = RAdam(self.parameters(), **self._optimizer_params)
        elif self._optimizer_class_name == "AdamWithWarmup":
            opt = init_adam_with_warmup_optimizer(self._model, **self._optimizer_params)
        else:
            _optimizer_class = getattr(optim, self._optimizer_class_name)
            opt = _optimizer_class(params=self.parameters(), **self._optimizer_params)
        return opt

    @property
    def metrics(self) -> Dict[str, str]:
        map_metric_to_validation = {
            "hp/common_prefix_length":"val_cond_cpl",
            "hp/cross_entropy":"val_cross_entropy",
            "hp/relative_common_prefix_length":"val_cond_cpl_vs_gt_ratio",
            "hp/inclusion_probs":"val_code_inclusion_probability"
        }
        return map_metric_to_validation

    def on_train_start(self) -> None:
        init_metrics = {metric_name:0 for metric_name in self.metrics.keys()}
        self.logger.log_hyperparams(params=self.hparams, metrics=init_metrics)

    def forward(self, x):
        t_codes, t_code_probs = self._model.forward(x)
        return t_codes, t_code_probs

    def on_save_checkpoint(self, checkpoint):
        device = self._get_model_device()
        if device != torch.device("cpu"):
            # convert device to cpu. it changes self._model instance itself.
            _ = self._model.to(device=torch.device("cpu"))
        # save model dump
        checkpoint["model_dump"] = pickle.dumps(self._model)
        # then revert back if necessary.
        if device != torch.device("cpu"):
            # revert to original device (probably cuda).
            _ = self._model.to(device=device)

    @classmethod
    def load_model_from_checkpoint(cls, weights_path: str, on_gpu: bool, map_location=None, fix_model_missing_attributes: bool = True):
        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        model = pickle.loads(checkpoint["model_dump"])
        if on_gpu:
            model = model.cuda(device=map_location)
        model_state_dict = {key.replace("_model.", ""):param for key, param in checkpoint["state_dict"].items() if key.startswith("_model")}
        model.load_state_dict(model_state_dict, strict=False)

        # fix model attributes for backward compatibility.
        if fix_model_missing_attributes:
            model = cls.fix_missing_attributes(model)

        return model

    @classmethod
    def fix_missing_attributes(cls, model):
        if getattr(model._encoder, "internal_layer_class_type", None) is None:
            if model._encoder.__class__.__name__ == "LSTMEncoder":
                # d1bae73
                if hasattr(model._encoder, "_apply_argmax_on_inference"):
                    delattr(model._encoder, "_apply_argmax_on_inference")
                # 78e74aa
                if not hasattr(model._encoder, "_pos_index"):
                    p_emb_code = getattr(model._encoder, "_embedding_code_boc", None)
                    if p_emb_code is not None:
                        if p_emb_code.ndim == 1:
                            p_emb_code_ = torch.nn.Parameter(data=p_emb_code.unsqueeze(0), requires_grad=True)
                            setattr(model._encoder, "_pos_index", {"n":0, "v":0})
                            setattr(model._encoder, "_embedding_code_boc", p_emb_code_)
            elif model._encoder.__class__.__name__ == "TransformerEncoder":
                pass
            else:
                pass

        return model

    def _update_model_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_epochs

        for parameter_name, scheduler_function in self._model_parameter_schedulers.items():
            if scheduler_function is None:
                continue

            current_value = getattr(self._model, parameter_name, None)
            if current_value is not None:
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(self._model, parameter_name, new_value)

                if verbose:
                    print(f"{parameter_name}: {current_value:.2f} -> {new_value:.2f}")

    def _update_loss_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_epochs

        for loss_name, dict_property_scheduler in self._loss_parameter_schedulers.items():
            # get loss layer
            if not loss_name.startswith("_"):
                loss_name = "_" + loss_name
            loss_layer = getattr(self, loss_name, None)
            if loss_layer is None:
                continue

            # get property name and apply scheduler function
            for property_name, scheduler_function in dict_property_scheduler.items():
                if scheduler_function is None:
                    continue

                # check if property exists
                if not hasattr(loss_layer, property_name):
                    continue

                current_value = getattr(loss_layer, property_name, None)
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(loss_layer, property_name, new_value)

                if verbose:
                    print(f"{loss_name}.{property_name}: {current_value:.2f} -> {new_value:.2f}")

    def training_step(self, batch, batch_idx):

        current_step = self.trainer.global_step / (self.trainer.max_epochs * self.trainer.num_training_batches)
        self._update_model_parameters(current_step, verbose=False)
        self._update_loss_parameters(current_step, verbose=False)

        # forward computation
        t_codes, t_code_probs = self._model.forward(**batch)

        # choose the representation which is used for loss computation
        if self._model.teacher_forcing:
            code_repr = t_code_probs
        else:
            if self._model.has_discretizer:
                code_repr = t_codes
            else:
                code_repr = t_code_probs

        # (required) supervised loss
        loss_supervised = self._loss_supervised.forward(target_codes=batch["ground_truth_synset_codes"], input_code_probabilities=code_repr)

        loss = loss_supervised

        dict_losses = {
            "train_loss_supervised": loss_supervised / self._scale_loss_supervised,
            "train_loss": loss
        }
        self.log_dict(dict_losses)
        return loss

    def evaluate_metrics(self, target_codes: torch.Tensor, conditional_code_probs: torch.Tensor, generated_codes: Optional[torch.Tensor] = None, eps: float = 1E-15):
        """

        @param target_codes: (n_batch, n_digits). ground-truth sense codes.
        @param conditional_code_probs: (n_batch, n_digits, n_ary). conditional probability. Pr(Y_d|y_{<d})
        @param generated_codes: (n_batch, n_digits). generated sense code (using greedy decoding).
        @param eps:
        @return:
        """
        # one-hot encoding without smoothing
        n_ary = self._model.n_ary
        t_code_probs_gt = self._aux_hyponymy_score._one_hot_encoding(t_codes=target_codes, n_ary=n_ary, label_smoothing_factor=0.0)

        # code lengths
        t_code_length_gt = (target_codes != 0).sum(axis=-1).type(torch.float)
        t_soft_code_length_pred = self._aux_hyponymy_score.calc_soft_code_length(conditional_code_probs)

        # common prefix lengths using conditional probability
        t_cond_cpl = self._aux_hyponymy_score.calc_soft_lowest_common_ancestor_length(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_lca_vs_gt_ratio = t_cond_cpl / t_code_length_gt
        t_pred_vs_gt_ratio = t_soft_code_length_pred / t_code_length_gt

        # common prefix lengths using generated codes
        if generated_codes is not None:
            if generated_codes.ndim == 3:
                t_code_pred = generated_codes.argmax(dim=-1)
            else:
                t_code_pred = generated_codes
            t_gen_cpl_batch = self._aux_hyponymy_score.calc_hard_common_ancestor_length(t_code_gt=target_codes, t_code_pred=t_code_pred)
            t_gen_cpl = torch.mean(t_gen_cpl_batch)
        else:
            t_gen_cpl = None

        # entailment probability
        t_prob_entail = self._aux_hyponymy_score.calc_ancestor_probability(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_prob_synonym = self._aux_hyponymy_score.calc_synonym_probability(t_prob_c_x=t_code_probs_gt, t_prob_c_y=conditional_code_probs)
        t_prob_inclusion = t_prob_synonym + t_prob_entail

        # cross entropy
        t_cross_entropy = self._aux_cross_entropy.forward(input_code_probabilities=conditional_code_probs, target_codes=target_codes)

        # code diversity
        code_probability_divergence = torch.mean(np.log(n_ary) + torch.sum(conditional_code_probs * torch.log(conditional_code_probs + eps), axis=-1), axis=-1)

        metrics = {
            "val_cross_entropy":t_cross_entropy,
            "val_gen_cpl":t_gen_cpl,
            "val_cond_cpl":torch.mean(t_cond_cpl),
            "val_cond_cpl_vs_gt_ratio":torch.mean(t_lca_vs_gt_ratio),
            "val_code_length_mean":torch.mean(t_soft_code_length_pred),
            "val_code_inclusion_probability":torch.mean(t_prob_inclusion),
            "val_code_length_std":torch.std(t_soft_code_length_pred),
            "val_code_length_pred_vs_gt_ratio":torch.mean(t_pred_vs_gt_ratio),
            "val_code_probability_divergence":torch.mean(code_probability_divergence)
        }
        return metrics

    def validation_step(self, batch, batch_idx):

        # forward computation without back-propagation
        t_target_codes = batch["ground_truth_synset_codes"]
        # conditional probability
        _, t_code_probs = self._model._predict(**batch)
        # sense code generation and its probability
        t_codes_greedy, t_code_probs_greedy = self._model._encode(**batch)

        # (required) supervised loss
        loss_supervised = self._loss_supervised.forward(target_codes=t_target_codes, input_code_probabilities=t_code_probs)

        loss = loss_supervised

        metrics = {
            "val_loss": loss
        }

        # analysis metrics
        ## based on continuous relaxation
        metrics_repr = self.evaluate_metrics(target_codes=t_target_codes, conditional_code_probs=t_code_probs, generated_codes=t_codes_greedy)
        metrics.update(metrics_repr)

        self.log_dict(metrics)

        # copy metrics to hyper parameters
        for metric_name, validation_metric_name in self.metrics.items():
            self.log(metric_name, metrics_repr[validation_metric_name])

        return None

    def test_step(self, batch, batch_idx):
        # ToDo: call WSD evaluator
        # self._evaluator
        pass

    def on_epoch_start(self):
        pass
