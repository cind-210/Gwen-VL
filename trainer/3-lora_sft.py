"""GWen LoRA supervised fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import SFTDataset
from model.model_lora import LoRAConfig, apply_lora_to_model, count_lora_params, save_lora_checkpoint
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import (
    Logger,
    amp_context,
    configure_torch_speed,
    cleanup_distributed,
    cosine_lr_lambda,
    create_dataloader,
    format_eta,
    get_config,
    config_from_checkpoint,
    load_checkpoint,
    load_model_weights,
    load_tokenizer,
    make_scaler,
    maybe_no_sync,
    resolve_dtype,
    set_seed,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GWen LoRA SFT")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--pretrain_path", required=True)
    parser.add_argument("--data_path", default="./dataset/sft.jsonl")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--max_seq_len", "--max_length", type=int, default=768)
    parser.add_argument("--linear_attention_backend", default="auto", choices=["auto", "gdn", "full"])
    parser.add_argument("--gdn_kernel_backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--gated_attention", default="auto", choices=["auto", "none", "headwise", "elementwise", "sigmoid"])
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_speed()
    env = setup_distributed(args.device)
    set_seed(args.seed + env["rank"])
    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(os.path.join(args.out_dir, "lora_sft.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    pretrain_ckpt = load_checkpoint(args.pretrain_path, map_location="cpu")
    config = config_from_checkpoint(
        pretrain_ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    if args.linear_attention_backend != "auto":
        config.linear_attention_backend = args.linear_attention_backend
    config.gdn_kernel_backend = args.gdn_kernel_backend
    if args.gated_attention != "auto":
        config.gated_attention = args.gated_attention
        config.attn_output_gate = args.gated_attention != "none"
    model = GWenForCausalLM(config).to(env["device"])
    load_model_weights(model, pretrain_ckpt, env["device"], strict=False)
    lora_config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    replaced = apply_lora_to_model(model, lora_config)
    dtype, dtype_name = resolve_dtype(args.dtype, env["device"])
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = wrap_ddp(model, env)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader, sampler = create_dataloader(dataset, args.batch_size, env["distributed"], num_workers=args.num_workers)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, len(loader) * args.epochs // args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unwrap_model(model).parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, total_steps, min(args.warmup_steps, max(1, total_steps // 5))),
    )
    scaler = make_scaler(dtype_name, env["device"])

    logger.banner(
        f"GWen LoRA SFT {args.config}: trainable={count_lora_params(unwrap_model(model))/1e6:.2f}M "
        f"modules={len(replaced)} dtype={dtype_name}"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_start_time = time.time()
        iter_per_epoch = len(loader)
        for batch_idx, batch in enumerate(loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            sync_grad = (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
            with maybe_no_sync(model, env["distributed"] and not sync_grad):
                input_ids = batch["input_ids"].to(env["device"])
                labels = batch["labels"].to(env["device"])
                attention_mask = batch["attention_mask"].to(env["device"])
                with amp_context(dtype_name, dtype, env["device"]):
                    loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)["loss"]
                    loss = loss / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            current_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss_sum += current_loss
            epoch_loss_count += 1
            current_batch = batch_idx + 1

            if env["is_main"] and (current_batch % args.log_interval == 0 or current_batch == 1):
                avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
                elapsed = time.time() - epoch_start_time
                eta = (elapsed / max(1, current_batch)) * max(0, iter_per_epoch - current_batch)
                logger.info(
                    f"LoRA SFT Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{iter_per_epoch}) "
                    f"loss={current_loss:.4f} avg_loss = {avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)}"
                )

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
        if args.max_steps > 0 and step >= args.max_steps:
            break

    if env["is_main"]:
        final = os.path.join(args.out_dir, f"lora_r{args.lora_r}.pth")
        save_lora_checkpoint(unwrap_model(model), lora_config, final)
        logger.info(f"Done. Saved {final}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
