"""GWen image-only VLM supervised fine-tuning."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import MiniMindVParquetDataset, SFTDataset, VLMSFTDataset, vlm_collate_fn
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import (
    Logger,
    amp_context,
    configure_vision_token_ids,
    configure_torch_speed,
    cleanup_distributed,
    cosine_lr_lambda,
    create_dataloader,
    format_eta,
    config_from_checkpoint,
    load_checkpoint,
    load_model_weights,
    load_resume,
    load_tokenizer,
    make_scaler,
    maybe_no_sync,
    resolve_dtype,
    save_checkpoint,
    set_seed,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)


class MixedVLMSFTBatchDataset(Dataset):
    def __init__(
        self,
        llm_dataset: Dataset,
        vlm_dataset: Dataset,
        batch_size: int,
        llm_batch_fraction: float,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0.0 < llm_batch_fraction < 1.0:
            raise ValueError("--llm_sft_batch_fraction must be between 0 and 1 for mixed VLM SFT")
        if batch_size < 2:
            raise ValueError("--batch_size must be at least 2 for mixed VLM SFT")
        self.llm_dataset = llm_dataset
        self.vlm_dataset = vlm_dataset
        self.batch_size = batch_size
        self.llm_per_batch = max(1, min(batch_size - 1, int(round(batch_size * llm_batch_fraction))))
        self.vlm_per_batch = batch_size - self.llm_per_batch
        self.seed = seed
        self.rank = rank
        self.world_size = max(1, world_size)
        self.global_batches = max(
            math.ceil(len(llm_dataset) / self.llm_per_batch),
            math.ceil(len(vlm_dataset) / self.vlm_per_batch),
        )
        self.local_batches = max(1, (self.global_batches + self.world_size - 1 - self.rank) // self.world_size)
        self.start_local_batch = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        self.llm_order = torch.randperm(len(self.llm_dataset), generator=generator).tolist()
        self.vlm_order = torch.randperm(len(self.vlm_dataset), generator=generator).tolist()

    def set_start_batch(self, start_batch: int) -> None:
        self.start_local_batch = max(0, min(int(start_batch), self.local_batches))

    def __len__(self) -> int:
        return max(0, self.local_batches - self.start_local_batch) * self.batch_size

    def __getitem__(self, idx: int):
        local_batch_idx = self.start_local_batch + idx // self.batch_size
        slot = idx % self.batch_size
        global_batch_idx = local_batch_idx * self.world_size + self.rank
        if slot < self.llm_per_batch:
            llm_idx = self.llm_order[(global_batch_idx * self.llm_per_batch + slot) % len(self.llm_order)]
            return self.llm_dataset[llm_idx]
        vlm_slot = slot - self.llm_per_batch
        vlm_idx = self.vlm_order[(global_batch_idx * self.vlm_per_batch + vlm_slot) % len(self.vlm_order)]
        return self.vlm_dataset[vlm_idx]


def parse_args():
    parser = argparse.ArgumentParser(description="GWen VLM SFT")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--vision_model_path", default="models/siglip2-base-p32-256-ve")
    parser.add_argument("--llm_checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_format", default="jsonl", choices=["jsonl", "minimind_v_parquet"])
    parser.add_argument("--data_source", default="vlm_sft", choices=["vlm_sft", "llm_sft_vlm_sft"])
    parser.add_argument("--llm_sft_data_path", default="")
    parser.add_argument("--llm_sft_batch_fraction", type=float, default=0.5)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_llm_mode", default="edge", choices=["edge", "full"])
    parser.add_argument("--train_llm_edge_layers", type=int, default=1)
    parser.add_argument("--vlm_rope_type", default="rope", choices=["mrope", "rope"])
    parser.add_argument("--rotary_dim", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_interval_steps", type=int, default=0)
    parser.add_argument("--resume_path", default="")
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow missing/skipped text checkpoint weights")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_speed()
    env = setup_distributed(args.device)
    set_seed(args.seed + env["rank"])
    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(os.path.join(args.out_dir, "vlm_sft.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    ckpt = load_checkpoint(args.llm_checkpoint, map_location="cpu")
    config = config_from_checkpoint(
        ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    configure_vision_token_ids(config, tokenizer)
    config.vision_model_name = args.vision_model_path
    config.vlm_rope_type = args.vlm_rope_type
    config.rotary_dim = args.rotary_dim
    checkpoint_prefix = f"vlm_sft-{config.vlm_rope_type}"
    model = GWenForCausalLM(config, vision_model_path=args.vision_model_path).to(env["device"])
    load_info = load_model_weights(model, ckpt, env["device"], strict=False)
    missing_keys = load_info.get("missing_keys", [])
    missing = len([key for key in missing_keys if not key.startswith("vision_projector.")])
    unexpected = len(load_info.get("unexpected_keys", []))
    skipped = len(load_info.get("skipped_shape_mismatch", {}))
    if env["is_main"]:
        logger.info(f"Loaded LLM checkpoint: missing={missing} unexpected={unexpected} skipped_shape={skipped}")
    if (missing or unexpected or skipped) and not args.allow_partial_load:
        raise RuntimeError(
            "Checkpoint did not fully match the VLM SFT model. "
            "Use the same tokenizer/config, or pass --allow_partial_load only for experiments."
        )
    for param in model.visual.parameters():
        param.requires_grad = False
    if args.train_llm_mode == "full":
        for param in model.model.parameters():
            param.requires_grad = True
        for param in model.vision_projector.parameters():
            param.requires_grad = True
        if not config.tie_word_embeddings:
            for param in model.lm_head.parameters():
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.vision_projector.parameters():
            param.requires_grad = True
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.model.norm.parameters():
            param.requires_grad = True
        if not config.tie_word_embeddings:
            for param in model.lm_head.parameters():
                param.requires_grad = True
        edge_layers = max(0, min(args.train_llm_edge_layers, len(model.model.layers) // 2))
        if edge_layers > 0:
            for layer in list(model.model.layers[:edge_layers]) + list(model.model.layers[-edge_layers:]):
                for param in layer.parameters():
                    param.requires_grad = True

    dtype, dtype_name = resolve_dtype(args.dtype, env["device"])
    if args.use_compile and hasattr(torch, "compile"):
        model.model = torch.compile(model.model)
    model = wrap_ddp(model, env)

    image_processor = SiglipImageProcessor.from_pretrained(args.vision_model_path)
    image_token_count = config.image_grid_size * config.image_grid_size
    if args.data_format == "jsonl":
        vlm_dataset = VLMSFTDataset(
            args.data_path,
            tokenizer,
            image_processor,
            max_length=args.max_seq_len,
            image_root=args.image_root,
            image_token_count=image_token_count,
        )
    else:
        vlm_dataset = MiniMindVParquetDataset(
            args.data_path,
            tokenizer,
            image_processor,
            max_length=args.max_seq_len,
            image_token_count=image_token_count,
        )
    if args.data_source == "llm_sft_vlm_sft":
        if not args.llm_sft_data_path:
            raise ValueError("--llm_sft_data_path is required when --data_source llm_sft_vlm_sft")
        llm_dataset = SFTDataset(args.llm_sft_data_path, tokenizer, max_length=args.max_seq_len)
        dataset = MixedVLMSFTBatchDataset(
            llm_dataset,
            vlm_dataset,
            batch_size=args.batch_size,
            llm_batch_fraction=args.llm_sft_batch_fraction,
            seed=args.seed,
            rank=env["rank"],
            world_size=env["world_size"],
        )
        loader_kwargs = {}
        if args.num_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=vlm_collate_fn,
            **loader_kwargs,
        )
        sampler = None
    else:
        dataset = vlm_dataset
        loader, sampler = create_dataloader(
            dataset,
            args.batch_size,
            env["distributed"],
            num_workers=args.num_workers,
            collate_fn=vlm_collate_fn if args.data_format == "minimind_v_parquet" else None,
        )
    total_steps = args.max_steps if args.max_steps > 0 else max(1, len(loader) * args.epochs // args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unwrap_model(model).parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, total_steps, min(args.warmup_steps, max(1, total_steps // 5))),
    )
    scaler = make_scaler(dtype_name, env["device"])
    step = 0
    start_epoch = 0
    start_batch = 0
    resume_path = args.resume_path or os.path.join(args.out_dir, "vlm_sft_resume.pth")
    if args.from_resume and os.path.exists(resume_path):
        step, start_epoch, start_batch = load_resume(resume_path, model, optimizer, scheduler, scaler, env["device"])
        logger.info(f"Resumed from {resume_path} at step={step}, epoch={start_epoch}, batch_idx={start_batch}")

    stats = unwrap_model(model).get_param_breakdown()
    logger.banner(
        f"GWen VLM SFT: total={stats['total']/1e6:.1f}M "
        f"trainable={unwrap_model(model).get_num_trainable_params()/1e6:.2f}M "
        f"train_llm_mode={args.train_llm_mode} dtype={dtype_name} data_source={args.data_source} "
        f"vlm_rope_type={config.vlm_rope_type}"
    )
    if env["is_main"] and args.data_source == "llm_sft_vlm_sft":
        logger.info(
            f"Mixed VLM SFT batch: llm={dataset.llm_per_batch} vlm={dataset.vlm_per_batch} "
            f"batch_size={args.batch_size}"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        supports_batch_offset = hasattr(dataset, "set_start_batch")
        if epoch == start_epoch and start_batch > 0 and not supports_batch_offset:
            raise RuntimeError(
                "Exact fast batch-offset resume is currently implemented for mixed VLM SFT only. "
                "Use --data_source llm_sft_vlm_sft or resume from an epoch checkpoint."
            )
        if sampler is not None:
            if epoch == start_epoch and start_batch > 0:
                raise RuntimeError(
                    "Exact fast batch-offset resume is currently implemented for mixed VLM SFT only. "
                    "Use --data_source llm_sft_vlm_sft or resume from an epoch checkpoint."
                )
            sampler.set_epoch(epoch)
        elif hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
            if supports_batch_offset:
                dataset.set_start_batch(start_batch if epoch == start_epoch else 0)
        log_loss_sum = 0.0
        log_loss_count = 0
        epoch_start = time.time()
        iter_per_epoch = len(loader)
        total_batches_for_log = dataset.local_batches if supports_batch_offset else iter_per_epoch
        for batch_idx, batch in enumerate(loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            actual_batch_idx = batch_idx + (start_batch if epoch == start_epoch and supports_batch_offset else 0)
            sync_grad = (actual_batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
            with maybe_no_sync(model, env["distributed"] and not sync_grad):
                input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
                labels = batch["labels"].to(env["device"], non_blocking=True)
                attention_mask = batch["attention_mask"].to(env["device"], non_blocking=True)
                pixel_values = batch.get("pixel_values")
                pixel_values = pixel_values.to(env["device"], non_blocking=True) if pixel_values is not None else None
                image_indices = batch.get("image_indices")
                image_indices = image_indices.to(env["device"], non_blocking=True) if image_indices is not None else None
                with amp_context(dtype_name, dtype, env["device"]):
                    loss = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_indices=image_indices,
                        labels=labels,
                        attention_mask=attention_mask,
                    )["loss"]
                    loss = loss / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            current_loss = loss.item() * args.gradient_accumulation_steps
            log_loss_sum += current_loss
            log_loss_count += 1
            current_batch = actual_batch_idx + 1
            if env["is_main"] and (current_batch % args.log_interval == 0 or current_batch == 1):
                avg_loss = log_loss_sum / max(1, log_loss_count)
                elapsed = time.time() - epoch_start
                eta = (elapsed / max(1, batch_idx + 1)) * max(0, iter_per_epoch - batch_idx - 1)
                logger.info(
                    f"VLM SFT Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{total_batches_for_log}) "
                    f"avg_loss={avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)} step={step}"
                )
                log_loss_sum = 0.0
                log_loss_count = 0
            if sync_grad:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unwrap_model(model).parameters()), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if env["is_main"] and args.save_interval_steps > 0 and step % args.save_interval_steps == 0:
                    step_path = os.path.join(args.out_dir, f"{checkpoint_prefix}-{step}.pth")
                    save_checkpoint(
                        step_path,
                        model,
                        config,
                        optimizer,
                        scheduler,
                        scaler,
                        step=step,
                        epoch=epoch,
                        train_args=vars(args),
                        extra={"batch_idx": actual_batch_idx + 1},
                    )
                    logger.info(f"Saved checkpoint {step_path}")
        if env["is_main"]:
            save_checkpoint(
                os.path.join(args.out_dir, f"{checkpoint_prefix}-epoch{epoch+1}.pth"),
                model,
                config,
                optimizer,
                scheduler,
                scaler,
                step=step,
                epoch=epoch + 1,
                train_args=vars(args),
                extra={"batch_idx": 0},
            )
        if args.max_steps > 0 and step >= args.max_steps:
            break

    if env["is_main"]:
        final_path = os.path.join(args.out_dir, f"{checkpoint_prefix}-final.pth")
        save_checkpoint(final_path, model, config, optimizer, scheduler, scaler, step=step, epoch=args.epochs, train_args=vars(args), extra={"batch_idx": 0})
        logger.info(f"Done. Saved {final_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
