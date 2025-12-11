import math
from collections import defaultdict
from functools import partial
from typing import Any, Union

import wandb
import lightning as L
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities.model_summary import summarize
from lion_pytorch import Lion
from matplotlib import pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from evenet.network.evenet_model import EveNetModel
from evenet.network.loss.assignment import convert_target_assignment

from evenet.network.metrics.general_comparison import GenericMetrics
from evenet.network.metrics.classification import ClassificationMetrics
from evenet.network.metrics.classification import shared_step as cls_step, shared_epoch_end as cls_end
from evenet.network.metrics.assignment import get_assignment_necessaries as get_ass, predict
from evenet.network.metrics.assignment import shared_step as ass_step, shared_epoch_end as ass_end
from evenet.network.metrics.assignment import SingleProcessAssignmentMetrics
from evenet.network.metrics.generation import GenerationMetrics
from evenet.network.metrics.generation import shared_step as gen_step, shared_epoch_end as gen_end
from evenet.network.metrics.segmentation import SegmentationMetrics
from evenet.network.metrics.segmentation import shared_step as seg_step, shared_epoch_end as seg_end
from evenet.network.loss.famo import FAMO
from evenet.utilities.ema import EMA

from evenet.utilities.debug_tool import time_decorator, log_function_stats
from evenet.utilities.task_scheduler import ProgressiveTaskScheduler
from evenet.utilities.tool import get_transition, check_param_overlap, print_params_used_by_loss, safe_load_state

from evenet.utilities.logger import LocalLogger
import logging


def get_total_gradient(module, norm_type="l1"):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            if norm_type == "l1":
                total += param.grad.abs().sum().item()
            elif norm_type == "l2":
                total += (param.grad ** 2).sum().item()
    return total


class EveNetEngine(L.LightningModule):
    def __init__(self, global_config, world_size=1, total_events=1024, total_val_events=1024):
        super().__init__()

        self.current_stage = None
        self.l = logging.getLogger(__name__)

        self.aggregator = None
        self.sampler = None
        self.steps_per_epoch = None
        self.steps_per_epoch_val = None
        self.total_steps = None
        self.current_step = None  # hack global_step due to incorrect behavior in lightning for multiple optimizers
        self.classification_metrics_train = None
        self.classification_metrics_valid = None
        self.classification_metrics_train_cross_term = None
        self.classification_metrics_valid_cross_term = None
        self.assignment_metrics_train = None
        self.assignment_metrics_valid = None
        self.generation_metrics_train = None
        self.generation_metrics_valid = None
        self.segmentation_metrics_train = None
        self.segmentation_metrics_valid = None
        self.model_parts = {}
        self.model: Union[EveNetModel, None] = None
        self.config = global_config
        self.world_size = world_size
        self.total_events = total_events
        self.total_val_events = total_val_events

        self.pretrain_ckpt_path: str = global_config.options.Training.pretrain_model_load_path

        self.num_classes: list[str] = (
            signal[0] if isinstance((signal := global_config.event_info.class_label.get("EVENT", {}).get("signal")),
                                    list)
            else [1]
        )

        ###### Initialize Keys for Data Inputs #####
        self.input_keys = ["x", "x_mask", "conditions", "conditions_mask"]
        self.target_classification_key = 'classification'
        self.target_regression_key = 'regression-data'
        self.target_regression_mask_key = 'regression-mask'
        self.target_assignment_key = 'assignments-indices'
        self.target_assignment_mask_key = 'assignments-mask'

        ###### Initialize Model Components Configs #####
        self.component_cfg = global_config.options.Training.Components

        self.classification_cfg = self.component_cfg.Classification
        self.regression_cfg = self.component_cfg.Regression
        self.assignment_cfg = self.component_cfg.Assignment
        self.global_generation_cfg = self.component_cfg.GlobalGeneration
        self.recon_generation_cfg = self.component_cfg.ReconGeneration
        self.truth_generation_cfg = self.component_cfg.TruthGeneration
        self.segmentation_cfg = self.component_cfg.Segmentation
        self.generation_include = self.global_generation_cfg.include or self.recon_generation_cfg.include or self.truth_generation_cfg.include

        self.target_segmentation_cls_key = 'segmentation-full-class' if self.segmentation_cfg.use_full_mask else 'segmentation-class'
        self.target_segmentation_reg_key = 'segmentation-momentum'
        self.target_segmentation_data_mask_key = 'segmentation-data'

        ###### Initialize Normalizations and Balance #####
        self.normalization_dict: dict = torch.load(self.config.options.Dataset.normalization_file)
        self.balance_dict: dict = self.normalization_dict
        if self.config.options.Dataset.get("balance_file", None) is not None:
            self.balance_dict = torch.load(self.config.options.Dataset.balance_file)

        self.class_weight = None
        if self.classification_cfg.include:
            base = list(self.balance_dict["class_balance"])
            self.class_weight = self.balance_dict["class_balance"]
            # If extra class weights are provided, validate and combine
            extra = getattr(self.classification_cfg, "class_weight", None)
            if extra is not None:
                extra = list(extra)
                if len(extra) != len(base):
                    raise ValueError(
                        f"class_weight must have length {len(base)}, "
                        f"but got {len(extra)}"
                    )
                self.class_weight = torch.tensor([b * e for b, e in zip(base, extra)], dtype=torch.float32)

        self.assignment_weight = None
        self.subprocess_balance = None
        if self.assignment_cfg.include:
            self.assignment_weight = self.balance_dict["particle_balance"]
            self.subprocess_balance = self.balance_dict["subprocess_balance"]

        self.segmentation_cls_balance = None
        if self.segmentation_cfg.include:
            self.segmentation_cls_balance = self.balance_dict[
                "segment_full_class_balance"] if self.segmentation_cfg.use_full_mask else self.balance_dict[
                "segment_class_balance"]

        self.l.info(f"normalization dicts initialized")

        ###### Initialize Assignment Necessaries ######
        self.ass_args = None
        if self.assignment_cfg.include:
            self.l.info(f"permutation indices configured")
            self.permutation_indices = dict()
            self.num_targets = dict()

            self.ass_args = get_ass(global_config.event_info)

        #######  Initialize Diffusion Settings ##########
        self.ema_model: Union[EMA, None] = None
        self.ema_cfg = global_config.options.Training.EMA

        self.diffusion_every_n_epochs = global_config.options.Training.diffusion_every_n_epochs
        self.diffusion_every_n_steps = global_config.options.Training.diffusion_every_n_steps

        self.global_diffusion_steps = self.global_generation_cfg.diffusion_steps
        self.point_cloud_diffusion_steps = self.recon_generation_cfg.diffusion_steps
        self.neutrino_diffusion_steps = self.truth_generation_cfg.diffusion_steps

        ###### Initialize Loss ######
        self.include_famo: bool = self.config.options.Training.get("FAMO", {}).get("turn_on", False)
        self.famo_detailed_loss: bool = self.config.options.Training.FAMO.get("detailed_loss", False)
        self.apply_event_weight: bool = self.config.options.Training.get("apply_event_weight", False)

        self.cls_loss = None
        if self.classification_cfg.include:
            import evenet.network.loss.classification as cls_loss
            self.cls_loss = cls_loss.loss
            self.l.info(f"classification loss initialized")

        self.reg_loss = None
        if self.regression_cfg.include:
            import evenet.network.loss.regression as reg_loss
            self.reg_loss = reg_loss.loss
            self.l.info(f"regression loss initialized")

        self.ass_loss = None
        if self.assignment_cfg.include:
            import evenet.network.loss.assignment as ass_loss
            assignment_loss_partial = partial(
                ass_loss.loss,
                focal_gamma=self.assignment_cfg.focal_gamma,
                particle_balance=self.assignment_weight,
                process_balance=self.subprocess_balance,
                **self.ass_args['loss']
            )
            self.ass_loss = assignment_loss_partial
            self.l.info(f"assignment loss initialized")

        self.seg_loss = None
        if self.segmentation_cfg.include:
            from evenet.network.loss.segmentation import loss as seg_loss
            self.seg_loss = seg_loss
            self.l.info(f"segmentation loss initialized")

        self.seg_loss = None
        if self.segmentation_cfg.include:
            from evenet.network.loss.segmentation import loss as seg_loss
            self.seg_loss = seg_loss
            print(f"{self.__class__.__name__} segmentation loss initialized")

        self.gen_loss = None
        if self.generation_include:
            import evenet.network.loss.generation as gen_loss
            self.gen_loss = gen_loss.loss
            self.l.info(f"generation loss initialized")

        ###### Initialize Optimizers ######
        self.hyper_par_cfg = {
            'batch_size': global_config.platform.batch_size,
            'epoch': global_config.options.Training.total_epochs,
            'lr_factor': global_config.options.Training.learning_rate_factor,
            'warm_up_factor': global_config.options.Training.learning_rate_warm_up_factor,
            'weight_decay': global_config.options.Training.weight_decay,
            'decoupled_weight_decay': global_config.options.Training.decoupled_weight_decay,
        }
        self.automatic_optimization = False

        ###### For general log ######
        self.general_log = GenericMetrics()
        self.log_gradient_step = global_config.options.Training.get("log_gradient_step", 100)
        self.simplified_log: bool = global_config.get('logger', {}).get("wandb", {}).get("simplified", False)
        self.local_logger: Union[None, LocalLogger] = None

        self.eval_metrics_every_n_epochs: int = global_config.options.Training.get("eval_metrics_every_n_epochs", 1)
        self.eval_metrics: bool = False

        # only save loss during prediction
        self.save_loss_predict: bool = self.config.options.get('prediction', {}).get('save_loss', False)

        ###### Progressive Training ######
        self.task_scheduler: Union[ProgressiveTaskScheduler, None] = None
        self.progressive_index = 0
        self.current_schedule = None

        ###### Last ######
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        pass

    @time_decorator()
    def calculate_loss(
            self, inputs, outputs,
            batch, device, loss_head_dict, update_metric, event_weight, batch_idx, schedules, batch_size,
    ):
        loss_raw: dict[str, torch.Tensor] = {}
        loss_detailed_dict = {}

        if self.classification_cfg.include and outputs["classification"]:
            scaled_cls_loss = cls_step(
                target_classification=batch[self.target_classification_key].to(device=device),
                cls_output=next(iter(outputs["classification"].values())),
                cls_loss_fn=self.cls_loss,
                class_weight=self.class_weight.to(device=device),
                loss_dict=loss_head_dict,
                loss_scale=self.classification_cfg.loss_scale,
                metrics=self.classification_metrics_train if self.training else self.classification_metrics_valid,
                device=device,
                update_metric=update_metric,
                event_weight=event_weight,
            )

            loss_raw["classification"] = scaled_cls_loss

        if self.classification_cfg.include_cross_term and outputs["classification-noised"]:
            scaled_cls_loss_cross_term = cls_step(
                target_classification=batch[self.target_classification_key].to(device=device),
                cls_output=next(iter(outputs["classification-noised"].values())),
                cls_loss_fn=self.cls_loss,
                class_weight=self.class_weight.to(device=device),
                event_weight=(outputs["alpha"] * outputs["alpha"] * event_weight) if event_weight is not None else (
                        outputs["alpha"] * outputs["alpha"]),
                loss_dict=loss_head_dict,
                loss_scale=self.classification_cfg.loss_scale_cross_term,
                metrics=self.classification_metrics_train_cross_term if self.training else self.classification_metrics_valid_cross_term,
                device=device,
                update_metric=update_metric,
                loss_name="classification-noised",
            )

            loss_raw["classification-noised"] = scaled_cls_loss_cross_term  # TODO: check if this is correct for famo

        if self.regression_cfg.include and outputs["regression"]:
            target_regression = batch[self.target_regression_key].to(device=device)
            target_regression_mask = batch[self.target_regression_mask_key].to(device=device)
            reg_output = outputs["regression"]
            reg_output = torch.cat([v.view(batch_size, -1) for v in reg_output.values()], dim=-1)
            reg_loss = self.reg_loss(
                predict=reg_output,
                target=target_regression.float(),
                mask=target_regression_mask.float(),
                event_weight=event_weight,
            ).mean()
            # loss = loss + reg_loss * self.regression_cfg.loss_scale

            loss_head_dict["regression"] = reg_loss

            loss_raw["regression"] = reg_loss * self.regression_cfg.loss_scale

        ass_predicts = None
        if self.assignment_cfg.include and outputs["assignments"]:
            ass_targets = batch[self.target_assignment_key].to(device=device)
            ass_targets_mask = batch[self.target_assignment_mask_key].to(device=device)
            scaled_ass_loss, ass_predicts = ass_step(
                ass_loss_fn=self.ass_loss,
                loss_dict=loss_head_dict,
                loss_detailed_dict=loss_detailed_dict,
                assignment_loss_scale=self.assignment_cfg.assignment_loss_scale,
                detection_loss_scale=self.assignment_cfg.detection_loss_scale,
                process_names=self.config.event_info.process_names,
                assignments=outputs["assignments"],
                detections=outputs["detections"],
                targets=ass_targets,
                targets_mask=ass_targets_mask,
                batch_size=batch_size,
                device=device,
                metrics=self.assignment_metrics_train if self.training else self.assignment_metrics_valid,
                point_cloud=inputs['x'],
                point_cloud_mask=inputs['x_mask'],
                subprocess_id=inputs["subprocess_id"],
                update_metric=update_metric,
                event_weight=event_weight,
                **self.ass_args['step']
            )

            loss_raw['assignment'] = scaled_ass_loss.flatten()[0]

        if self.generation_include and outputs["generations"] != dict():
            scaled_gen_loss, detailed_gen_loss = gen_step(
                batch=batch,
                outputs=outputs["generations"],
                gen_metrics=self.generation_metrics_train if self.training else self.generation_metrics_valid,
                model=self.model,
                global_loss_scale=self.global_generation_cfg.loss_scale,
                event_loss_scale=self.recon_generation_cfg.loss_scale,
                invisible_loss_scale=self.truth_generation_cfg.loss_scale,
                device=device,
                num_steps_global=self.global_diffusion_steps,
                num_steps_point_cloud=self.point_cloud_diffusion_steps,
                num_steps_neutrino=self.neutrino_diffusion_steps,
                diffusion_on=(
                        not self.training
                        and ((self.current_epoch % self.diffusion_every_n_epochs) == (
                        self.diffusion_every_n_epochs - 1))
                        and ((batch_idx % self.diffusion_every_n_steps) == 0)
                ),
                loss_head_dict=loss_head_dict,
                invisible_padding=self.model.invisible_padding,
                update_metric=update_metric,
                event_weight=event_weight,
                schedules=schedules,
            )

            loss_raw["generation"] = scaled_gen_loss
            if "generation-truth" in loss_head_dict:
                loss_raw["generation-truth"] = loss_head_dict["generation-truth"]
            if "generation-recon" in loss_head_dict:
                loss_raw["generation-recon"] = loss_head_dict["generation-recon"]

        if self.segmentation_cfg.include and outputs["segmentation-cls"] is not None:
            seg_targets_cls = batch[self.target_segmentation_cls_key].to(device=device)
            seg_target_mask = batch[self.target_segmentation_data_mask_key].to(device=device)
            class_weight = self.segmentation_cls_balance.to(device=device)
            scaled_seg_loss = seg_step(
                target_classification=seg_targets_cls,
                target_mask=seg_target_mask,
                predict_classification=outputs["segmentation-cls"],
                predict_mask=outputs["segmentation-mask"],
                point_cloud_mask=inputs['x_mask'],
                seg_loss_fn=self.seg_loss,
                class_weight=class_weight,
                class_label=inputs["subprocess_id"],
                metrics=self.segmentation_metrics_train if self.training else self.segmentation_metrics_valid,
                loss_dict=loss_head_dict,
                mask_loss_scale=self.segmentation_cfg.mask_loss_scale,
                dice_loss_scale=self.segmentation_cfg.dice_loss_scale,
                cls_loss_scale=self.segmentation_cfg.cls_loss_scale,
                loss_name="segmentation",
                update_metrics=update_metric,
                event_weight=event_weight,
                aux_outputs=outputs.get("segmentation-aux", None)
            )
            loss_raw["segmentation"] = scaled_seg_loss
            loss_head_dict["segmentation"] = scaled_seg_loss

        return loss_raw, loss_detailed_dict, ass_predicts

    @time_decorator()
    def shared_step(
            self, batch: Any, batch_idx: int,
            loss_head_dict: dict,
            update_metric: bool = True
    ):

        batch_size = batch["x"].shape[0]
        device = self.device
        if self.apply_event_weight:
            event_weight = batch.get("event_weight", None)
        else:
            event_weight = None

        self.current_stage = self.task_scheduler.get_current_stage(self.current_epoch).get("name", "default")
        current_parameters = self.task_scheduler.get_current_parameters(
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            batches_per_epoch=self.steps_per_epoch if self.training else self.steps_per_epoch_val,
        )
        task_weights = current_parameters["loss_weights"]
        train_parameters = current_parameters["train_parameters"]

        # logging training parameters
        for name, val in train_parameters.items():
            if not self.simplified_log:
                if self.global_rank == 0:
                    if self.training:
                        self.log(f"progressive/train/{name}", val, prog_bar=False, sync_dist=False)
                    else:
                        self.log(f"progressive/val/{name}", val, prog_bar=False, sync_dist=False)

        if self.simplified_log:
            self.local_logger.log_real(
                train_parameters,
                step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                training=self.training,
                prefix="progressive"
            )

        inputs = {
            key: value.to(device=device) for key, value in batch.items()
        }

        schedules = {k: v for (k, v) in self.model.schedule_flags}
        # if self.training:
        # Evaluate which tasks are active based on weights
        task_gate = {
            "generation": task_weights.get("generation-recon", 0) > 0,
            "neutrino_generation": task_weights.get("generation-truth", 0) > 0,
            "deterministic": (
                                     task_weights.get("classification", 0)
                                     + task_weights.get("regression", 0)
                                     + task_weights.get("assignment", 0)
                                     + task_weights.get("segmentation", 0)
                             ) > 0,
        }

        # Combine schedule flags and weight gating
        schedules = {
            key: schedules[key] and task_gate.get(key, False)
            for key in schedules
        }

        self.current_schedule = schedules

        # logging schedules
        for key, value in schedules.items():
            if not self.simplified_log:
                self.log(f"progressive/schedule-{key}", int(value), prog_bar=False, sync_dist=True)

        if self.simplified_log:
            self.local_logger.log_real(
                schedules,
                step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                training=self.training,
                prefix="progressive/schedule-"
            )

        outputs = self.model.shared_step(
            batch=inputs,
            batch_size=batch_size,
            train_parameters=train_parameters,
            schedules=[(key, value) for key, value in schedules.items()],
        )

        loss_raw, loss_detailed_dict, ass_predicts = self.calculate_loss(
            inputs, outputs,
            batch, device, loss_head_dict, update_metric, event_weight, batch_idx, schedules, batch_size,
        )

        self.general_log.update(loss_detailed_dict, is_train=self.training)

        loss = torch.zeros(1, device=self.device, requires_grad=True)

        # if self.global_rank == 0: print(f"[Step {self.current_step}] loss_1", flush=True)
        for name, loss_val in loss_raw.items():
            weight = task_weights.get(name, 0.0)
            loss_raw[name] = loss_val * weight
            if weight > 0:
                # if loss and loss_raw is not nan, else give it 0
                if not torch.isfinite(loss_raw[name]).all():
                    self.l.warning(
                        f"[Step {self.current_step}] Non-finite loss detected: {name} - "
                        f"loss: {loss}, loss_raw: {loss_raw[name]}"
                    )
                    continue

                loss = loss + loss_raw[name]

                if self.training:
                    if self.global_rank == 0:
                        self.log(f'progressive/loss_weight/{name}', weight, prog_bar=False, sync_dist=False)

        # if self.global_rank == 0: print(f"[Step {self.current_step}] loss_2", flush=True)
        if self.training:
            if not self.simplified_log:
                self.log('progressive/loss', loss, prog_bar=False, sync_dist=True)
            else:
                self.local_logger.log_real(
                    metrics={"loss": loss.item()},
                    step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                    training=self.training,
                    prefix="progressive"
                )

        # if self.global_rank == 0: print(f"[Step {self.current_step}] loss_3", flush=True)
        if self.training:
            self.log_loss(loss, loss_head_dict, loss_detailed_dict, prefix="train", batch_idx=batch_idx)
            self.l.info(
                f"[Epoch {self.current_epoch:03d}][Step {batch_idx} / {self.steps_per_epoch}] loss: {loss.item():.5f}")
        else:
            self.log_loss(loss, loss_head_dict, loss_detailed_dict, prefix="val", batch_idx=batch_idx)
            self.l.info(
                f"[Epoch {self.current_epoch:03d}][Step {batch_idx} / {self.steps_per_epoch_val}] loss: {loss.item():.5f}")

        # if self.global_rank == 0: print(f"[Step {self.current_step}] loss_4", flush=True)
        # return loss, loss_head_dict, loss_detailed_dict, ass_predicts, loss_raw, [outputs["full_input_point_cloud"],outputs["full_global_conditions"]]
        return loss, loss_head_dict, loss_detailed_dict, ass_predicts, loss_raw

    def log_loss(self, loss, loss_head, loss_dict, prefix: str, batch_idx):
        for name, val in loss_head.items():
            if not self.simplified_log:
                self.log(f"{prefix}/{name}", val, prog_bar=False, sync_dist=True)

        if self.simplified_log:
            self.local_logger.log_real(
                loss_head,
                step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                training=self.training,
                prefix=prefix
            )

        for name, val in loss_dict.items():
            for n, v in val.items():
                if v is None:
                    # print(f"[Rank {self.global_rank}] Skipping log for {n}/{prefix}/{name}: value is None")
                    continue
                if not torch.is_tensor(v):
                    v = torch.tensor(v, device=self.device)
                if not torch.isfinite(v).all():
                    # print(f"[Rank {self.global_rank}] Non-finite value in {n}/{prefix}/{name}: {v}")
                    continue
                if not self.simplified_log:
                    self.log(f"{n}/{prefix}/{name}", v, prog_bar=False, sync_dist=True)
                else:
                    self.local_logger.log_real(
                        {f"{n}/{prefix}/{name}": v.item()},
                        step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                        training=self.training,
                    )

        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True)
        if self.local_logger is not None:
            self.local_logger.log_real(
                {"loss": loss.item()},
                step=self.current_step, epoch=self.current_epoch, batch=batch_idx,
                training=self.training,
                prefix=prefix
            )

        # if self.global_rank == 0: print(f"[Step {self.current_step}] loss_3.3", flush=True)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # print(f"train batch end: {batch_idx}")
        pass

    @time_decorator()
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        optimizers = list(self.optimizers())
        schedulers = self.lr_schedulers()

        self.current_step = int(schedulers[0].state_dict().get("last_epoch", self.current_step))
        step = self.current_step
        batch_size = batch["x"].shape[0]

        # print(f"[Step {step}] train step start", flush=True)

        gradient_heads, loss_head = self.prepare_heads_loss()
        # print(f"[Step {step}] share step start", flush=True)
        loss, loss_head, loss_dict, _, loss_raw = self.shared_step(
            batch=batch, batch_idx=batch_idx,
            loss_head_dict=loss_head,
            update_metric=self.eval_metrics,
        )
        # print(f"[Step {step}] share step end", flush=True)
        final_loss = loss
        famo_logs = None
        if self.include_famo:
            in_loss = loss_head if self.famo_detailed_loss else loss_raw
            final_loss, famo_logs = self.model.famo.step(in_loss)

        # === Manual optimization ===
        for opt in optimizers:
            opt.zero_grad()

        task_losses, shared_params, task_param_sets, gen_global_loss = self.prepare_mtl_parameters(loss_head)
        #
        # # check_param_overlap(
        # #     task_param_sets=task_param_sets,
        # #     task_names=list(task_losses.keys()),
        # #     model=self.model,
        # #     current_step=self.current_step,
        # #     check_every=1000,
        # #     verbose=False,
        # # )
        # # for task, loss in task_losses.items():
        # #     print(f"[Task {task}] Loss: {loss.item()}")
        # #     print_params_used_by_loss(loss, self.model)
        #
        if self.current_step % self.log_gradient_step == 0 and self.global_rank == 0:
            self.log_task_gradient(task_losses, shared_params)

        # # === Backward for EveNet Main Part (multitask) ===
        # task_losses = list(task_losses.values())
        # mtl_backward(
        #     task_losses,
        #     features=shared_output,
        #     aggregator=self.aggregator,
        #     tasks_params=task_param_sets,
        #     shared_params=shared_params,
        #     retain_graph=True,
        #     parallel_chunk_size=1,
        # )
        # # === Sync gradients (excluding GlobalGeneration) ===
        # self.sync_gradients_ddp(
        #     self.model,
        #     exclude_modules=(getattr(self.model, "GlobalGeneration", None),) if gen_global_loss else ()
        # )
        # # === Backward for EveNet Global Part (GlobalGeneration) ===
        # if gen_global_loss:
        #     gen_global_loss.mean().backward()

        # print(f"[Step {step}] Loss: {final_loss.item()}")
        # self.safe_manual_backward(loss.mean())
        self.safe_manual_backward(final_loss.mean())
        # print(f"[Step {step}] Backward done")

        # === Check for Gradients ===
        clip_grad_norm_(self.model.parameters(), 1.0)
        if self.current_step % self.log_gradient_step == 0:
            self.check_gradient(gradient_heads)

        # === Step optimizers ===
        for opt in optimizers:
            opt.step()

        for sch in schedulers:
            sch.step()

        # === Update EMA ===
        if self.ema_model is not None:
            update_epoch = self.current_epoch >= self.ema_cfg.get("start_epoch", 0)
            update_step = self.current_step % self.ema_cfg.get("update_step", 1) == 0

            current_parameters = self.task_scheduler.get_current_parameters(
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                batches_per_epoch=self.steps_per_epoch,
            )

            train_parameters = current_parameters["train_parameters"]
            if update_epoch and update_step:
                self.ema_model.update(self.model, decay_=train_parameters.get("ema_decay", None))

        # -------------------------------------
        # logging
        # -------------------------------------
        if self.include_famo:
            with torch.no_grad():
                loss, loss_head, loss_dict, _, loss_raw = self.shared_step(
                    batch=batch, batch_idx=batch_idx,
                    loss_head_dict=loss_head,
                    update_metric=False,
                )
                in_loss = loss_head if self.famo_detailed_loss else loss_raw
                self.model.famo.update(in_loss)

            for k, v in famo_logs.items():
                self.log(f"famo/{k}", v, prog_bar=False, sync_dist=True)
            self.log("train/famo-loss", final_loss.mean(), prog_bar=True, sync_dist=True)

        # self.current_step += 1
        # print(f"[Step {step}] train step done", flush=True)
        return final_loss.mean()

    # @time_decorator
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        step = self.current_step
        epoch = self.current_epoch

        _, loss_head = self.prepare_heads_loss()
        loss, loss_head, loss_dict, _, loss_raw = self.shared_step(
            batch=batch, batch_idx=batch_idx,
            loss_head_dict=loss_head,
            update_metric=self.eval_metrics,
        )

        return loss.mean()

    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_size = batch["x"].shape[0]
        device = self.device

        inputs = {
            key: value.to(device=device) for key, value in batch.items()
        }

        if self.apply_event_weight:
            event_weight = batch.get("event_weight", None)
        else:
            event_weight = None

        outputs = self.model.shared_step(
            batch=inputs,
            batch_size=batch_size,
            train_parameters=None,
        )

        if self.save_loss_predict:
            _, loss_head_dict = self.prepare_heads_loss()

            loss_raw, _, _ = self.calculate_loss(
                inputs, outputs,
                batch, device, loss_head_dict, update_metric=False, event_weight=event_weight, batch_idx=batch_idx,
                schedules={}, batch_size=batch_size,
            )

            outputs['losses'] = loss_raw
        else:
            outputs["generations"] = None
            outputs["classification-noised"] = None
            outputs['regression-noised'] = None
            outputs.pop('alpha')

        if self.config.options.prediction.get('save_point_cloud', False):
            outputs["full_input_point_cloud"] = inputs['x']

        extra_save = self.config.options.prediction.get('extra_save', {})

        for key in extra_save:
            if key in batch:
                outputs[key] = batch[key]

        if self.assignment_cfg.include:
            outputs["assignment_target"], outputs["assignment_target_mask"] = convert_target_assignment(
                targets=batch["assignments-indices"],
                targets_mask=batch["assignments-mask"],
                event_particles=self.ass_args['loss']['event_particles'],
                num_targets=self.ass_args['loss']['num_targets']
            )

            outputs["assignment_prediction"] = {}
            for process in self.config.event_info.process_names:
                outputs["assignment_prediction"][process] = predict(
                    assignments=outputs["assignments"].pop(process),
                    detections=outputs["detections"].pop(process),
                    product_symbolic_groups=self.ass_args['step']['product_symbolic_groups'][process],
                    event_permutations=self.ass_args['step']['event_permutations'][process],
                )

        if self.segmentation_cfg.include:
            outputs['truth_segmentation-full-class'] = inputs['segmentation-full-class']
            outputs['truth_segmentation-class'] = inputs['segmentation-class']
            outputs['truth_segmentation-mask'] = inputs[self.target_segmentation_data_mask_key]

        if self.truth_generation_cfg.include:
            outputs["neutrinos"] = {
                "predict": {},
                "target": {}
            }
            data_shape = inputs['x_invisible'].shape
            feature_names = self.config.event_info.invisible_feature_names

            predict_for_neutrino = partial(
                self.model.predict_diffusion_vector,
                mode="neutrino",
                cond_x=inputs,
                noise_mask=inputs["x_invisible_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )

            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_neutrino,
                normalize_fn=self.model.invisible_normalizer,
                eta=1.0,
                num_steps=self.neutrino_diffusion_steps,
                use_tqdm=False,
                process_name=f"Neutrino",
                remove_padding=True,
            )

            for i in range(data_shape[-1]):
                outputs["neutrinos"]["predict"][feature_names[i]] = generated_distribution[..., i]
                outputs["neutrinos"]["target"][feature_names[i]] = inputs['x_invisible'][..., i]

        return outputs

    def check_gradient(self, gradient_heads: dict[str, torch.nn.Module]):
        for name, module in gradient_heads.items():
            grad_mag = get_total_gradient(module)
            num_params = sum(p.numel() for p in module.parameters() if p.grad is not None)
            grad_avg = grad_mag / num_params if num_params > 0 else 0.0

            if self.global_rank == 0:
                self.log(f"grad_head/{name}", grad_avg, prog_bar=False, sync_dist=False)

    def log_task_gradient(self, task_loss_dict, shared_params):
        """
        Computes and logs pairwise cosine similarity between task gradients
        on shared parameters using torch.autograd.grad (non-intrusive and efficient).

        Args:
            task_loss_dict (dict): Mapping from task name to loss tensor.
            shared_params (iterable): Parameters shared across tasks (e.g., backbone).
        """
        total_dim = sum(p.numel() for p in shared_params)
        grad_vectors = {}

        for task_name, loss in task_loss_dict.items():
            if loss.requires_grad is False or loss.detach().item() == 0.0:
                grad_vec = torch.zeros(total_dim, device=loss.device)
                grad_vectors[task_name] = grad_vec
                continue

            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                allow_unused=False,
                create_graph=False
            )
            flat = [g.view(-1) for g in grads if g is not None]
            grad_vec = torch.cat(flat) if flat else torch.zeros(total_dim, device=loss.device)
            grad_vectors[task_name] = grad_vec

        for task_name, grad_vec in grad_vectors.items():
            grad_norm = grad_vec.norm(p=2)
            self.log(
                f"grad_norm/{task_name}",
                grad_norm,
                on_step=True,
                logger=True,
                sync_dist=False
            )

        # Stack into [T, D]
        task_names = list(grad_vectors.keys())
        all_grads = torch.stack([grad_vectors[name] for name in task_names], dim=0)  # [T, D]

        # Normalize manually, avoid NaNs for zero vectors
        norms = all_grads.norm(p=2, dim=1, keepdim=True)  # [T, 1]
        nonzero = norms > 0
        safe_norms = torch.where(nonzero, norms, torch.ones_like(norms))  # avoid div-by-zero
        normalized_grads = all_grads / safe_norms  # [T, D]

        # Compute cosine sim matrix [T, T]
        sim_matrix = normalized_grads @ normalized_grads.T
        for i, name_i in enumerate(task_names):
            for j in range(i, len(task_names)):
                if i == j:
                    continue
                self.log(
                    f"cos_grad_sim/{name_i}_vs_{task_names[j]}",
                    sim_matrix[i, j],
                    on_step=True,
                    logger=True,
                    sync_dist=False
                )

    def sync_gradients_ddp(self, model, average=True, exclude_modules=()):
        if not torch.distributed.is_initialized():
            return

        buckets = defaultdict(list)

        excluded = set()
        for module in exclude_modules:
            excluded.update(p for p in module.parameters())

        # Group gradients by (device, dtype)
        for param in model.parameters():
            if param.grad is not None and param not in excluded:
                key = (param.grad.device, param.grad.dtype)
                buckets[key].append(param.grad)

        for (_, _), grads in buckets.items():
            for grad in grads:
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                if average:
                    grad /= torch.distributed.get_world_size()

    # @time_decorator
    def on_fit_start(self) -> None:
        self.current_step = 0

        if self.simplified_log:
            if len(self.loggers) < 2:
                raise ValueError(
                    "Simplified logging requires at least two loggers: WandbLogger and LocalLogger."
                )

            self.local_logger = self.loggers[1]

            self.l.info(f"Simplified logging enabled")

        if self.classification_cfg.include:
            self.classification_metrics_train = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )
            self.classification_metrics_valid = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )

        if self.classification_cfg.include_cross_term:
            self.classification_metrics_train_cross_term = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )
            self.classification_metrics_valid_cross_term = ClassificationMetrics(
                num_classes=len(self.num_classes), normalize=True, device=self.device
            )

        if self.assignment_cfg.include:
            def make_assignment_metrics():
                return {
                    process: SingleProcessAssignmentMetrics(
                        device=self.device,
                        event_permutations=self.config.event_info.event_permutations[process],
                        event_symbolic_group=self.config.event_info.event_symbolic_group[process],
                        event_particles=self.config.event_info.event_particles[process],
                        product_symbolic_groups=self.config.event_info.product_symbolic_groups[process],
                        ptetaphienergy_index=self.config.event_info.ptetaphienergy_index,
                        process=process
                    ) for process in self.config.event_info.process_names
                }

            self.assignment_metrics_train = make_assignment_metrics()
            self.assignment_metrics_valid = make_assignment_metrics()

        if self.generation_include:
            generation_kwargs = {
                "class_names": self.config.event_info.class_label.get("EVENT", {}).get("signal", ["a"])[0],
                "sequential_feature_names": self.config.event_info.sequential_feature_names,
                "invisible_feature_names": self.config.event_info.invisible_feature_names,
                "device": self.device,
                "global_generation": self.global_generation_cfg.include,
                "point_cloud_generation": self.recon_generation_cfg.include,
                "neutrino_generation": self.truth_generation_cfg.include,
                "special_bin_configs": self.config.options.Metrics.get("Generation-Binning", {}),
                "target_global_index": self.config.event_info.generation_target_indices,
                "target_global_names": self.config.event_info.generation_target_names,
                "use_generation_result": self.recon_generation_cfg.get("use_generation_result", False),
                "target_event_index": self.config.event_info.generation_pc_indices,
            }

            self.generation_metrics_train = GenerationMetrics(**generation_kwargs)
            self.generation_metrics_valid = GenerationMetrics(**generation_kwargs)

        if self.segmentation_cfg.include:
            segmentation_kwargs = {
                "device": self.device,
                "clusters_label": self.config.event_info.segment_label,
                "processes_labels": self.config.event_info.process_names,
            }
            self.segmentation_metrics_train = SegmentationMetrics(**segmentation_kwargs)
            self.segmentation_metrics_valid = SegmentationMetrics(**segmentation_kwargs)

    def on_fit_end(self) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.general_log.reduce_across_gpus(
            device=device
        )

        if self.global_rank == 0:
            self.l.info(f"On fit end")

            figs = self.general_log.plot_all()

            for metric, fig in figs.items():
                self.logger.experiment.log({
                    f"General/{metric}": wandb.Image(fig)
                })
                plt.close(fig)

            # debug time information
            log_function_stats(self.logger)

            summary = summarize(self, max_depth=3)
            columns = ["Name", "Type", "Params"]
            data = [
                [str(name), str(type_), int(num)]
                for name, type_, num in zip(summary.layer_names, summary.layer_types, summary.param_nums)
            ]
            self.logger.log_table(
                key="model summary",
                columns=columns,
                data=data,
            )

            wandb.finish()  # ensure everything is flushed
            self.l.info(f"W&B logging done and finished.")

        self.l.info(f"On fit end all end")

    def on_validation_start(self):
        self.eval_metrics = (self.current_epoch + 1) % self.eval_metrics_every_n_epochs == 0

        self.l.info(f"[Epoch {self.current_epoch:03d}] ▶️ Validation Start | eval_metrics: {self.eval_metrics}")
        pass

    def on_train_epoch_start(self) -> None:
        self.eval_metrics = (self.current_epoch + 1) % self.eval_metrics_every_n_epochs == 0

        self.l.info(f"[Epoch {self.current_epoch:03d}] ▶️ Training Start | eval_metrics: {self.eval_metrics}")
        pass

    @time_decorator()
    def on_validation_epoch_end(self) -> None:
        if self.classification_cfg.include and self.current_schedule.get("deterministic", False) and self.eval_metrics:
            cls_end(
                global_rank=self.global_rank,
                metrics_valid=self.classification_metrics_valid,
                metrics_train=self.classification_metrics_train,
                num_classes=self.num_classes,
                logger=self.logger.experiment,
                module=self
            )

        if self.classification_cfg.include_cross_term and self.current_schedule.get("deterministic",
                                                                                    False) and self.eval_metrics:
            cls_end(
                global_rank=self.global_rank,
                metrics_valid=self.classification_metrics_valid_cross_term,
                metrics_train=self.classification_metrics_train_cross_term,
                num_classes=self.num_classes,
                logger=self.logger.experiment,
                prefix="cross-term-"
            )

        if self.assignment_cfg.include and self.current_schedule.get("deterministic", False) and self.eval_metrics:
            ass_end(
                global_rank=self.global_rank,
                metrics_valid=self.assignment_metrics_valid,
                metrics_train=self.assignment_metrics_train,
                logger=self.logger.experiment,
            )

        if self.generation_include and (
                self.current_schedule.get("generation", False) or self.current_schedule.get("neutrino_generation",
                                                                                            False)) and self.eval_metrics:
            gen_end(
                global_rank=self.global_rank,
                metrics_valid=self.generation_metrics_valid,
                metrics_train=self.generation_metrics_train,
                logger=self.logger.experiment
            )

        if self.segmentation_cfg.include and self.current_schedule.get("deterministic", False):
            seg_end(
                global_rank=self.global_rank,
                metrics_valid=self.segmentation_metrics_valid,
                metrics_train=self.segmentation_metrics_train,
                logger=self.logger.experiment
            )

        self.general_log.finalize_epoch(is_train=False)

        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                print("val current stage:", self.current_stage)
                logger.flush_metrics(stage=self.current_stage)

        self.l.info(f"[Epoch {self.current_epoch:03d}] ✅ Validation Complete")

    @time_decorator()
    def on_train_epoch_end(self) -> None:
        self.general_log.finalize_epoch(is_train=True)

        for logger in self.loggers:
            if isinstance(logger, LocalLogger):
                print("train current stage:", self.current_stage)
                logger.flush_metrics(stage=self.current_stage)

        self.l.info(f"[Epoch {self.current_epoch:03d}] ✅ Training Complete")

    @time_decorator()
    def safe_manual_backward(self, loss, *args, **kwargs):
        super().manual_backward(loss, *args, **kwargs)
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             print(f"🚨 Gradient in {name} is NaN or Inf!")
        #             raise ValueError("Gradient check failed.")

    # @time_decorator
    def configure_optimizers(self):
        cfg = self.hyper_par_cfg
        lr_factor = cfg.get('lr_factor', 1.0)
        batch_size = cfg['batch_size']
        epochs = cfg['epoch']
        warm_up_factor = cfg.get('warm_up_factor', 0.5)

        # Distributed training info
        world_size = self.world_size
        dataset_size = self.total_events
        self.steps_per_epoch = math.ceil(dataset_size / (batch_size * world_size))
        self.steps_per_epoch_val = math.ceil(self.total_val_events / (batch_size * world_size))
        warmup_steps = warm_up_factor * self.steps_per_epoch
        self.total_steps = epochs * self.steps_per_epoch

        if self.global_rank == 0:
            self.l.info("--> Optimizer Configuration:")
            self.l.info(f"  world_size     : {world_size}")
            self.l.info(f"  dataset_size   : {dataset_size}")
            self.l.info(f"  steps_per_epoch: {self.steps_per_epoch}")
            self.l.info(f"  warmup_steps   : {warmup_steps}")
            self.l.info(f"  total_steps    : {self.total_steps}")
            self.l.info(f"  batch_size     : {batch_size}")
            self.l.info(f"  warm_up_factor : {warm_up_factor}")

        betas = (0.95, 0.99)

        def create_optim_schedule(
                p, base_lr, base_wd,
                decoupled_wd: bool = False,
                warm_up: bool = True, optimizer_type: str = "lion"
        ):
            scaled_lr = base_lr * math.sqrt(world_size) / lr_factor
            scaled_weight_decay = base_wd / math.sqrt(world_size) * lr_factor

            if optimizer_type.lower() == "adamw":
                optimizer = torch.optim.AdamW(
                    p,
                    lr=scaled_lr,
                    # betas=betas,
                    weight_decay=scaled_weight_decay
                )
            elif optimizer_type.lower() == "lion":
                optimizer = Lion(
                    p,
                    lr=scaled_lr,
                    betas=betas,
                    weight_decay=scaled_weight_decay,
                    decoupled_weight_decay=decoupled_wd,
                )
            else:  # Default using Lion
                optimizer = Lion(
                    p,
                    lr=scaled_lr,
                    betas=betas,
                    weight_decay=scaled_weight_decay,
                    decoupled_weight_decay=decoupled_wd,
                )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps if warm_up else 0,
                num_training_steps=self.total_steps,
                num_cycles=0.5
            )
            return optimizer, scheduler

        optimizers, schedulers = [], []
        for name, modules in self.model_parts.items():
            self.l.info(f"Configuring optimizer/scheduler for {name} with modules: {modules['modules']}")

            if not modules:
                continue
            valid_modules = [getattr(self.model, m) for m in modules['modules'] if m is not None]
            params = [p for m in valid_modules for p in m.parameters()]

            if len(params) == 0:
                self.l.warning(f"No parameters found for {name}. Skipping optimizer/scheduler configuration.")
                continue

            opt, sch = create_optim_schedule(
                params,
                base_lr=modules['lr'],
                base_wd=modules['weight_decay'],
                warm_up=modules['warm_up'],
                optimizer_type=modules["optimizer_type"],
                decoupled_wd=modules["decoupled_wd"],
            )
            optimizers.append(opt)
            schedulers.append({
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": f"lr-{name}"
            })

        # Add FAMO optimizer
        if hasattr(self.model, "famo") and hasattr(self.model.famo, "optimizer"):
            self.l.info(f"[FAMO] --> Adding FAMO optimizer")
            optimizers.append(self.model.famo.optimizer)

        # Add Scheduler
        self.task_scheduler = ProgressiveTaskScheduler(
            config=self.config.options.Training.ProgressiveTraining,  # assuming you store the YAML-loaded config here
            total_epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            model_parts=self.model_parts  # assuming this is defined
        )

        return optimizers, schedulers

    # @time_decorator
    def configure_model(self) -> None:
        self.l.info(f"Configuring model on device {self.device}")
        if self.model is not None:
            return

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            if "A100" in name:
                torch.set_float32_matmul_precision("medium")
                self.l.info(" --> A100 detected, using medium precision for matmul")

        self.model = EveNetModel(
            config=self.config,
            device=self.device,
            classification=self.classification_cfg.include,
            regression=self.regression_cfg.include,
            global_generation=self.global_generation_cfg.include,
            point_cloud_generation=self.recon_generation_cfg.include,
            neutrino_generation=self.truth_generation_cfg.include,
            assignment=self.assignment_cfg.include,
            segmentation=self.segmentation_cfg.include,
            normalization_dict=self.normalization_dict,
        )

        if self.global_rank == 0:
            ema_enable = self.ema_cfg.get("enable", False)
            ema_replace = self.ema_cfg.get("replace_model_after_load", False)
            if self.global_rank == 0:
                self.l.info(f"[Model] --> EMA enabled: {ema_enable}")
                self.l.warning(f"[Model] --> EMA replace model after load: {ema_replace}")

            if self.pretrain_ckpt_path is not None:
                self.l.warning(f"[Model] --> Loading pretrained weights from: {self.pretrain_ckpt_path}")
                ckpt = torch.load(self.pretrain_ckpt_path, map_location=self.device)

                if ema_enable and 'ema_state_dict' in ckpt and ema_replace:
                    safe_load_state(self.model, ckpt['ema_state_dict'])
                else:
                    safe_load_state(self.model, ckpt['state_dict'])

            # EMA
            if ema_enable:
                self.ema_model = EMA(
                    model=self.model, decay=self.ema_cfg.get("decay", 0.999)
                )

        # Define Freezing
        # self.model.freeze_module("Classification", self.classification_cfg.get("freeze", {}))
        # self.model.freeze_module("Regression", self.regression_cfg.get("freeze", {}))
        # self.model.freeze_module("Assignment", self.assignment_cfg.get("freeze", {}))

        # Define model part groups
        self.model_parts = {}

        for key, cfg in self.component_cfg.items():
            group = cfg.get("optimizer_group", None)
            if not group:
                continue

            module_attr = getattr(self.model, key, None)
            if not module_attr:
                self.l.warning(f"⚠️ Warning: No module set for '{key}'. Skipping.")
                continue

            # Add to group
            if group not in self.model_parts:
                self.model_parts[group] = {
                    'lr': cfg['learning_rate'],
                    'warm_up': cfg.get('warm_up', True),
                    'optimizer_type': cfg.get('optimizer_type', 'lion'),
                    'weight_decay': cfg.get('weight_decay', 0.0),
                    'decoupled_wd': cfg.get('decoupled_weight_decay', False),
                    'modules': [],
                }
            self.model_parts[group]['modules'].append(key)

        self.l.info(f"[Model] --> Model parts: {self.model_parts}")

        ### Initialize FAMO ###
        famo_task_list = ["classification", "regression", "assignment", "generation", "segmentation"]
        if self.include_famo and self.famo_detailed_loss:
            famo_task_list = self.config.options.Training.FAMO.detailed_loss_list

        self.model.register_module('famo', FAMO(
            task_list=famo_task_list,
            lr=self.config.options.Training.FAMO.get("lr", 0.025),
            device=self.device,
            turn_on=self.include_famo,
            logits_bound=self.config.options.Training.FAMO.get("logits_bound", 1.0),
        ))
        self.l.warning(f"FAMO Applied? --> {self.include_famo}")

        from evenet.utilities.diffusion_sampler import DDIMSampler
        self.sampler = DDIMSampler(device=self.device)

        ### Initialize Jacobian Descent ###
        # self.aggregator = UPGrad()

    def on_save_checkpoint(self, checkpoint):
        orig_model = getattr(self.model, "_orig_mod", self.model)

        # Save current model state_dict
        checkpoint["state_dict"] = {f"model.{k}": v for k, v in orig_model.state_dict().items()}

        # Optionally replace what’s saved (but not in memory) with EMA weights
        if self.ema_model is not None and self.ema_cfg.get("replace_model_at_end", False):
            ema_sd = {f"model.{k}": v for k, v in self.ema_model.state_dict().items()}
            checkpoint["state_dict"] = ema_sd
            self.l.warning(f"[Model] --> Saved EMA weights instead of current model weights")

        # Always save EMA weights separately as well
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'], device=self.device)
            self.l.info(f"[Model] --> Loading EMA model")
            if self.ema_cfg.get("replace_model_after_load", False):
                self.ema_model.copy_to(self.model)
                self.l.warning(f"[Model] --> Replacing model with EMA model after loading checkpoint")

    def prepare_heads_loss(self):
        gradient_heads = {}
        loss_heads = {}

        gradient_heads["PET"] = self.model.PET
        gradient_heads["GlobalEmbedding"] = self.model.GlobalEmbedding
        gradient_heads["ObjectEncoder"] = self.model.ObjectEncoder

        if self.classification_cfg.include:
            gradient_heads["classification"] = self.model.Classification
            loss_heads["classification"] = torch.zeros(1, device=self.device, requires_grad=True)

        if self.regression_cfg.include:
            gradient_heads["regression"] = self.model.Regression
            loss_heads["regression"] = torch.zeros(1, device=self.device, requires_grad=True)

        if self.global_generation_cfg.include:
            gradient_heads["generation-global"] = self.model.GlobalGeneration
            loss_heads["generation-global"] = torch.zeros(1, device=self.device, requires_grad=True)

        if self.recon_generation_cfg.include:
            gradient_heads["generation-recon"] = self.model.ReconGeneration
            loss_heads["generation-recon"] = torch.zeros(1, device=self.device, requires_grad=True)

        if self.truth_generation_cfg.include:
            gradient_heads["generation-truth"] = self.model.TruthGeneration
            loss_heads["generation-truth"] = torch.zeros(1, device=self.device, requires_grad=True)

        if self.assignment_cfg.include:
            gradient_heads["Assignment"] = self.model.Assignment
            assignment_heads = self.model.Assignment.multiprocess_assign_head
            gradient_heads.update({
                f'assignment_{name}': assignment_heads[name]
                for name in assignment_heads.keys()
            })
            loss_heads.update({
                f'assignment_{name}': torch.zeros(1, device=self.device, requires_grad=True)
                for name in assignment_heads.keys()
            })

        if self.segmentation_cfg.include:
            gradient_heads["segmentation"] = self.model.Segmentation
            loss_heads["segmentation"] = torch.zeros(1, device=self.device, requires_grad=True)

        return gradient_heads, loss_heads

    # For Torch JD
    def prepare_mtl_parameters(self, loss_head: dict[str, torch.Tensor]):

        task_losses = {}
        task_loss_global = None

        if "classification" in loss_head:
            task_losses["classification"] = loss_head["classification"]

        seg_losses = {
            'segmentation': ['segmentation'],
            'segmentation-mask': [
                'segmentation-mask', 'segmentation-mask-aux', 'segmentation-dice', 'segmentation-dice-aux'
            ],
            'segmentation-cls': ['segmentation-cls', 'segmentation-cls-aux'],
        }

        for task_name, loss_names in seg_losses.items():
            present = [name for name in loss_names if name in loss_head]
            if present:  # only create the grouped loss if at least one component is present
                task_losses[task_name] = sum(loss_head[name] for name in present)

        # if "assignment" in loss_head or "detection" in loss_head:
        #     task_losses["assignment"] = (
        #             loss_head.get("assignment", 0.0) + loss_head.get("detection", 0.0)
        #     )

        if "classification" in loss_head or "assignment" in loss_head or "detection" in loss_head:
            task_losses["deterministic"] = (
                    loss_head.get("classification", 0.0) + loss_head.get("assignment", 0.0) +
                    loss_head.get("detection", 0.0)
            )

        if "classification-noised" in loss_head:
            task_losses["cross_term"] = loss_head["classification-noised"]

        for loss_name in loss_head.keys():
            if loss_name.startswith("assignment_"):
                task_losses[loss_name] = loss_head[loss_name]

        if "regression" in loss_head:
            task_losses["regression"] = loss_head["regression"]

        if "generation-truth" in loss_head:
            task_losses["generation-truth"] = loss_head["generation-truth"]

        if "generation-recon" in loss_head:
            task_losses["generation-recon"] = loss_head["generation-recon"]

        ### Individual Loss
        if "generation-global" in loss_head:
            task_loss_global = loss_head["generation-global"]

        def filter_trainable(params):
            return [p for p in params if p.requires_grad]

        shared_params = filter_trainable(
            list(self.model.PET.parameters()) +
            list(self.model.GlobalEmbedding.parameters())
        )

        task_param_sets = []
        for loss_name in task_losses:
            # if loss_name == "classification":
            #     task_params = (
            #             filter_trainable(self.model.ObjectEncoder.parameters()) +
            #             filter_trainable(self.model.Classification.parameters())
            #     )
            # if hasattr(self.model, "Assignment") and hasattr(self.model.Assignment, "multiprocess_assign_head"):
            #     task_params += filter_trainable(
            #         chain.from_iterable(
            #             v.parameters() for v in self.model.Assignment.multiprocess_assign_head.values()
            #         )
            #     )
            # elif loss_name == "assignment":
            #     task_params = (
            #             filter_trainable(self.model.ObjectEncoder.parameters()) +
            #             filter_trainable(self.model.Assignment.parameters())
            #     )
            if loss_name == "deterministic":
                task_params = filter_trainable(self.model.ObjectEncoder.parameters())
                if hasattr(self.model, "Classification"):
                    task_params += filter_trainable(self.model.Classification.parameters())
                if hasattr(self.model, "Assignment"):
                    task_params += filter_trainable(self.model.Assignment.parameters())

            elif loss_name == "regression":
                task_params = (
                        filter_trainable(self.model.ObjectEncoder.parameters()) +
                        filter_trainable(self.model.Regression.parameters())
                )
            elif loss_name == "generation-truth":
                task_params = (
                    filter_trainable(self.model.TruthGeneration.parameters())
                )
            elif loss_name == "generation-recon":
                task_params = (
                    filter_trainable(self.model.ReconGeneration.parameters())
                )
            # No need, will use automatic backward
            # elif loss_name == "generation-global":
            #     task_params = (
            #         filter_trainable(self.model.GlobalGeneration.parameters())
            #     )
            else:
                # print(f"[TorchJD] Unhandled loss name: {loss_name}")
                # raise NotImplementedError(f"[TorchJD] Unhandled loss name: {loss_name}")
                task_params = []
                pass

            task_param_sets.append(task_params)

        return task_losses, shared_params, task_param_sets, task_loss_global
