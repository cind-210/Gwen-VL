"""GWen full-parameter supervised fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import SFTDataset
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import (
    Logger,
    amp_context,
    configure_vision_token_ids,
    configure_torch_speed,
    cleanup_distributed,
    cosine_lr_lambda,
    create_dataloader,
    evaluate_loss,
    format_eta,
    get_config,
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


def parse_args():
    parser = argparse.ArgumentParser(description="GWen Full SFT")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--pretrain_path", required=True)
    parser.add_argument("--data_path", default="./dataset/sft.jsonl")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--max_seq_len", "--max_length", type=int, default=768)
    parser.add_argument(
        "--linear_attention_backend",
        default="auto",
        choices=["auto", "gdn", "full"],
        help="auto keeps the backend saved in the pretrain checkpoint",
    )
    parser.add_argument("--gdn_kernel_backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--gated_attention", default="auto", choices=["auto", "none", "headwise", "elementwise", "sigmoid"])
    parser.add_argument("--vlm_rope_type", default="rope", choices=["mrope", "rope"])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", "--accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_data_path", default="")
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_interval_steps", type=int, default=0)
    parser.add_argument("--resume_path", default="")
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow missing/skipped pretrain weights")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_speed()
    env = setup_distributed(args.device)
    set_seed(args.seed + env["rank"])
    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(os.path.join(args.out_dir, "sft.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    pretrain_ckpt = load_checkpoint(args.pretrain_path, map_location="cpu")
    config = config_from_checkpoint(
        pretrain_ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    configure_vision_token_ids(config, tokenizer)
    if args.linear_attention_backend != "auto": # 如果是auto的话就用预训练的，否则用命令行指定的
        config.linear_attention_backend = args.linear_attention_backend
    config.gdn_kernel_backend = args.gdn_kernel_backend
    if args.gated_attention != "auto": # 如果是auto的话就用预训练的，否则用命令行指定的
        config.gated_attention = args.gated_attention
        config.attn_output_gate = args.gated_attention != "none"
    config.vlm_rope_type = args.vlm_rope_type
    config.dropout = args.dropout
    checkpoint_prefix = f"sft-{config.vlm_rope_type}"
    model = GWenForCausalLM(config).to(env["device"])
    load_info = load_model_weights(model, pretrain_ckpt, env["device"], strict=False)
    missing = len(load_info.get("missing_keys", []))
    unexpected = len(load_info.get("unexpected_keys", []))
    skipped = len(load_info.get("skipped_shape_mismatch", {}))
    if env["is_main"]:
        logger.info(
            f"Loaded pretrain checkpoint: backend={config.linear_attention_backend} "
            f"gdn_kernel={config.gdn_kernel_backend} "
            f"missing={missing} unexpected={unexpected} skipped_shape={skipped}"
        )
    if (missing or unexpected or skipped) and not args.allow_partial_load:
        raise RuntimeError(
            "Pretrain checkpoint did not fully match the SFT model. "
            "This usually means config/backend/tokenizer mismatch. "
            "Use the same --linear_attention_backend as pretrain, or pass --allow_partial_load only for experiments."
        )
    dtype, dtype_name = resolve_dtype(args.dtype, env["device"])
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = wrap_ddp(model, env)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader, sampler = create_dataloader(dataset, args.batch_size, env["distributed"], num_workers=args.num_workers)
    eval_loader = None
    if args.eval_data_path:
        eval_dataset = SFTDataset(args.eval_data_path, tokenizer, max_length=args.max_seq_len)
        eval_loader, _ = create_dataloader(
            eval_dataset,
            args.batch_size,
            env["distributed"],
            shuffle=False,
            num_workers=args.num_workers,
        )
    total_steps = args.max_steps if args.max_steps > 0 else max(1, len(loader) * args.epochs // args.gradient_accumulation_steps)
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 5))

    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, total_steps, warmup_steps),
    )
    scaler = make_scaler(dtype_name, env["device"])

    step = 0
    start_epoch = 0
    start_batch = 0
    resume_path = args.resume_path or os.path.join(args.out_dir, "sft_resume.pth")
    if args.from_resume and os.path.exists(resume_path):
        step, start_epoch, start_batch = load_resume(resume_path, model, optimizer, scheduler, scaler, env["device"])

    stats = unwrap_model(model).get_param_breakdown()
    logger.banner(f"GWen SFT {args.config}: total={stats['total']/1e6:.1f}M dtype={dtype_name}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        log_loss_sum = 0.0
        log_loss_count = 0
        epoch_start_time = time.time()
        log_start_time = epoch_start_time
        log_tokens = 0
        iter_per_epoch = len(loader)
        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if args.max_steps > 0 and step >= args.max_steps:
                break
            sync_grad = (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
            with maybe_no_sync(model, env["distributed"] and not sync_grad):
                input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
                labels = batch["labels"].to(env["device"], non_blocking=True)
                attention_mask = batch["attention_mask"].to(env["device"], non_blocking=True)
                with amp_context(dtype_name, dtype, env["device"]):
                    loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)["loss"]
                    loss = loss / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            current_loss = loss.item() * args.gradient_accumulation_steps
            log_loss_sum += current_loss
            log_loss_count += 1
            current_batch = batch_idx + 1
            log_tokens += input_ids.numel() * env["world_size"]

            if env["is_main"] and (current_batch % args.log_interval == 0 or current_batch == 1):
                avg_loss = log_loss_sum / max(1, log_loss_count)
                elapsed = time.time() - epoch_start_time
                log_elapsed = max(time.time() - log_start_time, 1e-6)
                tokens_per_s = log_tokens / log_elapsed
                effective_tokens = args.batch_size * env["world_size"] * args.max_seq_len * args.gradient_accumulation_steps
                eta = (elapsed / max(1, current_batch)) * max(0, iter_per_epoch - current_batch)
                logger.info(
                    f"SFT Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{iter_per_epoch}) "
                    f"avg_loss = {avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)} "
                    f"tokens/s={tokens_per_s:.0f} optimizer_step={step} "
                    f"effective_tokens_per_step={effective_tokens}"
                )
                log_start_time = time.time()
                log_tokens = 0
                log_loss_sum = 0.0
                log_loss_count = 0

            if sync_grad:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
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
                        extra={"batch_idx": batch_idx + 1},
                    )
                    logger.info(f"Saved checkpoint {step_path}")
                if eval_loader is not None and args.eval_interval > 0 and step % args.eval_interval == 0:
                    eval_loss = evaluate_loss(model, eval_loader, env, dtype_name, dtype, args.eval_steps)
                    if env["is_main"]:
                        logger.info(f"Eval step={step} loss={eval_loss:.4f}")
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
