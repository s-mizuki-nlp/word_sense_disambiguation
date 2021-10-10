#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Callable, Union
import warnings
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
import pytorch_lightning as pl

from model import HierarchicalCodeEncoder
from model.loss_supervised import HyponymyScoreLoss, EntailmentProbabilityLoss, CrossEntropyLossWrapper


class UnsupervisedTrainer(pl.LightningModule):

    def __init__(self,
                 model: HierarchicalCodeEncoder,
                 loss_supervised: Union[HyponymyScoreLoss, EntailmentProbabilityLoss, CrossEntropyLossWrapper],
                 dataloader_train: Optional[DataLoader] = None,
                 dataloader_val: Optional[DataLoader] = None,
                 dataloader_test: Optional[DataLoader] = None,
                 learning_rate: Optional[float] = 0.001,
                 optimizer_class: Optional[Optimizer] = None,
                 use_sampled_code_repr_for_loss_computation: bool = False,
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None
                 ):

        super().__init__()

        self._use_sampled_code_repr_for_loss_computation = use_sampled_code_repr_for_loss_computation

        self._loss_supervised = loss_supervised
        self._scale_loss_supervised = loss_supervised.scale

        self._model = model
        self._encoder = model._encoder
        self.n_ary = model.n_ary
        self.n_digits = model.n_digits

        self._learning_rate = learning_rate
        self._dataloaders = {
            "train": dataloader_train,
            "val": dataloader_val,
            "test": dataloader_test
        }
        self._optimizer_class = optimizer_class
        # auxiliary function that is solely used for validation
        self._auxiliary = HyponymyScoreLoss()

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
        if self._optimizer_class is None:
            opt = Adam(self.parameters(), lr=self._learning_rate)
        elif self._optimizer_class.__name__ == "SAM":
            opt = self._optimizer_class(self.parameters(), Adam, lr=self._learning_rate)
        else:
            opt = self._optimizer_class(params=self.parameters(), lr=self._learning_rate)
        return opt

    def train_dataloader(self):
        return self._dataloaders["train"]

    def val_dataloader(self):
        return self._dataloaders["val"]

    def test_dataloader(self):
        return self._dataloaders["test"]

    def forward(self, x):
        t_codes, t_code_probs = self._model.forward(x)
        return t_codes, t_code_probs

    def _evaluate_code_stats(self, t_code_prob):

        _EPS = 1E-6
        n_ary = self._model.n_ary
        soft_code_length = self._auxiliary.calc_soft_code_length(t_code_prob)
        code_probability_divergence = torch.mean(np.log(n_ary) + torch.sum(t_code_prob * torch.log(t_code_prob + _EPS), axis=-1), axis=-1)

        metrics = {
            "val_soft_code_length_mean":torch.mean(soft_code_length),
            "val_soft_code_length_std":torch.std(soft_code_length),
            "val_code_probability_divergence":torch.mean(code_probability_divergence)
        }
        return metrics

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_metrics = defaultdict(list)
        for output in outputs:
            for key, value in output["log"].items():
                avg_metrics[key].append(value)
        for key, values in avg_metrics.items():
            avg_metrics[key] = torch.stack(values).mean()
        return {'avg_val_loss': avg_loss, 'log': avg_metrics}

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
        state_dict = {key.replace("_model.", ""):param for key, param in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)

        # fix model attributes for backward compatibility.
        if fix_model_missing_attributes:
            model = cls.fix_missing_attributes(model)

        return model

    @classmethod
    def fix_missing_attributes(cls, model):
        if getattr(model._encoder, "internal_layer_class_type", None) is None:
            if model._encoder.__class__.__name__ == "LSTMEncoder":
                pass
            elif model._encoder.__class__.__name__ == "TransformerEncoder":
                pass
            else:
                pass

        return model

    def _update_model_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_nb_epochs

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
            current_step = self.current_epoch / self.trainer.max_nb_epochs

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

        current_step = self.trainer.global_step / (self.trainer.max_nb_epochs * self.trainer.total_batches)
        self._update_model_parameters(current_step, verbose=False)
        self._update_loss_parameters(current_step, verbose=False)

        # forward computation
        t_codes, t_code_probs = self._model.forward(**batch)

        # choose the representation which is used for loss computation
        if self._use_sampled_code_repr_for_loss_computation:
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

    def validation_step(self, batch, batch_idx):

        # forward computation without back-propagation
        t_target_codes = batch["ground_truth_synset_codes"]
        t_codes, t_code_probs = self._model._predict(**batch)
        code_repr = t_codes

        # (required) supervised loss
        loss_reconst = self._loss_supervised.forward(target_codes=t_target_codes, input_code_probabilities=code_repr)

        # analysis metrics
        ## lowest common ancestor lengths
        metric_lca_lengths = self._auxiliary.calc_log_soft_lowest_common_ancestor_length()

        # (optional) mutual information loss
        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_probs)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32, device=t_code_probs.device)

        loss = loss_reconst + loss_hyponymy + loss_non_hyponymy + loss_code_length + loss_mi

        metrics = {
            "val_loss_reconst": loss_reconst / self._scale_loss_reconst,
            "val_loss_mutual_info": loss_mi / self._scale_loss_mi,
            "val_loss_hyponymy": loss_hyponymy / self._scale_loss_supervised,
            "val_loss_non_hyponymy": loss_non_hyponymy / self._scale_loss_non_hyponymy,
            "val_loss_hyponymy_positive": loss_hyponymy_positive / self._scale_loss_supervised, # self._scale_loss_non_hyponymy,
            "val_loss_code_length": loss_code_length / self._scale_loss_code_length,
            "val_loss": loss
        }
        metrics_repr = self._evaluate_code_stats(t_code_probs)
        metrics.update(metrics_repr)

        return {"val_loss":loss, "log":metrics}

    def on_epoch_start(self):
        pass
