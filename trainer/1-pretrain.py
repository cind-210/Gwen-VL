"""GWen pretraining with packed JSONL text."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import LazyPretrainDataset, PackedPretrainDataset
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM, print_accel_info
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
    parser = argparse.ArgumentParser(description="GWen Pretrain")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--data_path", default="./dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--dataset_mode", default="lazy", choices=["lazy", "packed"])
    parser.add_argument("--data_cache_dir", default="", help="Tokenized pretrain cache directory")
    parser.add_argument("--no_data_cache", action="store_true", help="Disable tokenized pretrain cache")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--linear_attention_backend", default="gdn", choices=["gdn", "full"])
    parser.add_argument("--gdn_kernel_backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--gated_attention", default="sigmoid", choices=["none", "headwise", "elementwise", "sigmoid"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", "--accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_data_path", default="")
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--resume_path", default="")
    parser.add_argument("--from_resume", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_speed()
    env = setup_distributed(args.device)
    set_seed(args.seed + env["rank"]) # 
    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(os.path.join(args.out_dir, "pretrain.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    config = get_config(args.config, args.max_seq_len)
    configure_vision_token_ids(config, tokenizer)
    config.linear_attention_backend = args.linear_attention_backend
    config.gdn_kernel_backend = args.gdn_kernel_backend
    config.gated_attention = args.gated_attention
    config.attn_output_gate = args.gated_attention != "none"
    config.dropout = args.dropout
    model = GWenForCausalLM(config).to(env["device"])
    dtype, dtype_name = resolve_dtype(args.dtype, env["device"])
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = wrap_ddp(model, env)

    stats = unwrap_model(model).get_param_breakdown()
    logger.banner(
        f"GWen pretrain {args.config}: total={stats['total']/1e6:.1f}M "
        f"embedding={stats['embedding']/1e6:.1f}M body={stats['body']/1e6:.1f}M "
        f"backend={config.linear_attention_backend} gdn_kernel={config.gdn_kernel_backend} gate={config.gated_attention}"
    )
    if env["is_main"]:
        print_accel_info()
    logger.info(
        f"Preparing dataset: path={args.data_path} seq_len={args.max_seq_len} "
        f"mode={args.dataset_mode} cache={'off' if args.no_data_cache or args.dataset_mode == 'lazy' else 'on'}"
    )

    if args.dataset_mode == "lazy":
        dataset = LazyPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    else:
        dataset = PackedPretrainDataset(
            args.data_path,
            tokenizer,
            max_length=args.max_seq_len,
            cache_dir=args.data_cache_dir or None,
            use_cache=not args.no_data_cache,
            build_cache=env["is_main"] or not env["distributed"],
            logger=logger,
        )
    if env["distributed"] and args.dataset_mode == "packed":
        torch.distributed.barrier()

    loader, sampler = create_dataloader(dataset, args.batch_size, env["distributed"], num_workers=args.num_workers)
    eval_loader = None
    if args.eval_data_path:
        eval_dataset = PackedPretrainDataset(
            args.eval_data_path,
            tokenizer,
            max_length=args.max_seq_len,
            cache_dir=args.data_cache_dir or None,
            use_cache=not args.no_data_cache,
            build_cache=env["is_main"] or not env["distributed"],
            logger=logger,
        )
        if env["distributed"]:
            torch.distributed.barrier()
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

    start_step = 0
    start_epoch = 0
    resume_path = args.resume_path or os.path.join(args.out_dir, "pretrain_resume.pth")
    if args.from_resume and os.path.exists(resume_path):
        start_step, start_epoch = load_resume(resume_path, model, optimizer, scheduler, scaler, env["device"])
        logger.info(f"Resumed from {resume_path} at step={start_step}, epoch={start_epoch}")
    logger.info(
        f"Dataset blocks={len(dataset)} batch={args.batch_size} accum={args.gradient_accumulation_steps} "
        f"world={env['world_size']} total_steps={total_steps} dtype={dtype_name}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_start_time = time.time()
        log_start_time = epoch_start_time
        log_tokens = 0
        iter_per_epoch = len(loader)
        for batch_idx, batch in enumerate(loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            is_accum_boundary = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            sync_grad = is_accum_boundary or (batch_idx + 1) == len(loader)
            with maybe_no_sync(model, enabled=env["distributed"] and not sync_grad):
                input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
                labels = batch["labels"].to(env["device"], non_blocking=True)
                attention_mask = None if args.dataset_mode == "packed" else batch["attention_mask"].to(env["device"], non_blocking=True)
                with amp_context(dtype_name, dtype, env["device"]):
                    out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    loss = out["loss"] / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            current_loss = loss.item() * args.gradient_accumulation_steps
            epoch_loss_sum += current_loss
            epoch_loss_count += 1
            current_batch = batch_idx + 1
            log_tokens += input_ids.numel() * env["world_size"]

            if env["is_main"] and (current_batch % args.log_interval == 0 or current_batch == 1):
                avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
                elapsed = time.time() - epoch_start_time
                log_elapsed = max(time.time() - log_start_time, 1e-6)
                tokens_per_s = log_tokens / log_elapsed
                effective_tokens = args.batch_size * env["world_size"] * args.max_seq_len * args.gradient_accumulation_steps
                eta = (elapsed / max(1, current_batch)) * max(0, iter_per_epoch - current_batch)
                logger.info(
                    f"Pretrain Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{iter_per_epoch}) "
                    f"loss={current_loss:.4f} avg_loss = {avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)} "
                    f"tokens/s={tokens_per_s:.0f} optimizer_step={step} "
                    f"effective_tokens_per_step={effective_tokens}"
                )
                log_start_time = time.time()
                log_tokens = 0

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
                if eval_loader is not None and args.eval_interval > 0 and step % args.eval_interval == 0:
                    eval_loss = evaluate_loss(model, eval_loader, env, dtype_name, dtype, args.eval_steps)
                    if env["is_main"]:
                        logger.info(f"Eval step={step} loss={eval_loss:.4f}")
        if env["is_main"]:
            save_checkpoint(
                os.path.join(args.out_dir, f"pretrain_epoch{epoch+1}.pth"),
                model,
                config,
                optimizer,
                scheduler,
                scaler,
                step=step,
                epoch=epoch + 1,
                train_args=vars(args),
            )
        if args.max_steps > 0 and step >= args.max_steps:
            break

    if env["is_main"]:
        final_path = os.path.join(args.out_dir, "pretrain_final.pth")
        save_checkpoint(final_path, model, config, optimizer, scheduler, scaler, step=step, epoch=args.epochs, train_args=vars(args))
        logger.info(f"Done. Saved {final_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
