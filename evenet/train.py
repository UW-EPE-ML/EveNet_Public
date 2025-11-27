import os
import argparse
from pathlib import Path

import ray
import ray.train
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichModelSummary
from lightning.pytorch.profilers import PyTorchProfiler

from evenet.control.global_config import global_config
from evenet.shared import make_process_fn, prepare_datasets, EveNetTrainCallback
from evenet.engine import EveNetEngine
from evenet.utilities.logger import LocalLogger, setup_logging


def train_func(cfg):
    batch_size = cfg['batch_size']
    max_epochs = cfg['epochs']
    prefetch_batches = cfg['prefetch_batches']
    total_events = cfg['total_events']
    total_val_events = cfg['total_val_events']
    world_rank = ray.train.get_context().get_world_rank()
    global_config.load_yaml(cfg['global_config_path'], current_dir=cfg['current_dir'])

    log_cfg = cfg.get('logger', {})
    loggers = []
    wandb_config = log_cfg.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "EveNet"),
        name=wandb_config.get("run_name", None),
        tags=wandb_config.get("tags", []),
        entity=wandb_config.get("entity", None),
        config=global_config.to_logger(),
        id=wandb_config.get("id", None),
    )
    loggers.append(wandb_logger)

    local_logger = None
    if 'local' in log_cfg:
        local_logger = LocalLogger(
            rank=world_rank,
            **log_cfg['local'],
        )
        loggers.append(local_logger)

    tmp_log_dir = os.path.join(os.getcwd(), "logs")
    setup_logging(rank=world_rank, log_dir=local_logger.log_dir if local_logger else tmp_log_dir)

    dataset_configs = {
        'batch_size': batch_size,
        'prefetch_batches': prefetch_batches,
        'local_shuffle_buffer_size': batch_size * prefetch_batches,
    }

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")

    train_ds_loader = train_ds.iter_torch_batches(**dataset_configs)
    val_ds_loader = val_ds.iter_torch_batches(**dataset_configs)

    # Model
    model = EveNetEngine(
        global_config=global_config,
        world_size=ray.train.get_context().get_world_size(),
        total_events=total_events,
        total_val_events=total_val_events,
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=50,
        mode="min",
        verbose=True,
        dirpath=global_config.options.Training.model_checkpoint_save_path,
        save_last="link",
        auto_insert_metric_name=False,
        filename="epoch={epoch}_train={train/loss:.4f}_val={val/loss:.4f}",
    )
    early_stop_callback = EarlyStopping(
        **cfg.get("early_stopping", {}),
    )

    accelerator_config = {
        "accelerator": "auto",
        "devices": "auto",
    }
    # if this is macOS, set the accelerator to "cpu"
    if os.uname().sysname == "Darwin":
        accelerator_config["accelerator"] = "cpu"
        accelerator_config["devices"] = 1

    trainer = L.Trainer(
        max_epochs=max_epochs,
        strategy=RayDDPStrategy(find_unused_parameters=True, timeout=180),
        plugins=[RayLightningEnvironment()],
        callbacks=[
            EveNetTrainCallback(),
            checkpoint_callback,
            early_stop_callback,
            LearningRateMonitor(),
            RichModelSummary(max_depth=3),
        ],
        enable_progress_bar=True,
        logger=loggers,
        # val_check_interval=10,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        # profiler=PyTorchProfiler(
        #     dirpath=global_config.options.Training.model_checkpoint_save_path,
        #     filename=f"profiler_{world_rank}",
        # ),
        **accelerator_config,
    )

    trainer = prepare_trainer(trainer)

    ckpt_path = None
    if global_config.options.Training.model_checkpoint_load_path is not None:
        ckpt_path = global_config.options.Training.model_checkpoint_load_path
        # print(f"Loading checkpoint from {ckpt_path}")

    trainer.fit(
        model,
        train_dataloaders=train_ds_loader,
        val_dataloaders=val_ds_loader,
        ckpt_path=ckpt_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")
    # argument for loading all dataset files into RAM
    parser.add_argument("--load_all", action="store_true", help="Load all dataset files into RAM")
    parser.add_argument("--ray_dir", type=str, default="~/ray_results")
    return parser


def main(args: argparse.Namespace) -> None:
    assert (
            "WANDB_API_KEY" in os.environ
    ), 'Please set WANDB_API_KEY="abcde" when running this script.'

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}",
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            # "TORCH_NCCL_BLOCKING_WAIT": "1",
            # "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_TIMEOUT": "180",
            # "NCCL_DEBUG_SUBSYS": "ALL",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000000",
        }
    }

    # Expand ~ and convert to absolute path
    config_path = os.path.abspath(os.path.expanduser(args.config))
    # Check existence
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    # Load your config
    global_config.load_yaml(config_path)
    global_config.display()

    if "logger" not in global_config._global_config:
        raise KeyError("Missing required config key: 'logger'")

    platform_info = global_config.platform

    ray.init(
        runtime_env=runtime_env,
    )

    base_dir = Path(platform_info.data_parquet_dir)
    base_val_dir = None if "data_parquet_val_dir" not in platform_info else Path(platform_info.data_parquet_val_dir)

    process_fn = make_process_fn(base_dir)
    train_ds, valid_ds, total_events, total_val_events = prepare_datasets(
        base_dir=base_dir,
        process_event_batch_partial=process_fn,
        platform_info=platform_info,
        load_all_in_ram=args.load_all,
        base_val_dir=base_val_dir,
        predict=False,
    )

    run_config = RunConfig(
        name="EveNet-Training",
        storage_path=args.ray_dir,
    )

    # Schedule four workers for DDP training (1 GPU/worker by default)
    scaling_config = ScalingConfig(
        num_workers=platform_info.number_of_workers,
        resources_per_worker=platform_info.resources_per_worker,
        use_gpu=platform_info.get("use_gpu", True),
    )

    trainer_config = {
        "batch_size": platform_info.batch_size,
        "epochs": global_config.options.Training.epochs,
        "prefetch_batches": platform_info.prefetch_batches,
        'logger': {
            **global_config.logger,
        },
        "total_events": total_events,
        "total_val_events": total_val_events,
        "early_stopping": global_config.options.Training.EarlyStopping,
        "global_config_path": config_path,
        "current_dir": os.getcwd(),
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=trainer_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={
            "train": train_ds,
            "validation": valid_ds,
        },
    )

    result = trainer.fit()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def cli() -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args()
    main(args)


if __name__ == '__main__':
    cli()
