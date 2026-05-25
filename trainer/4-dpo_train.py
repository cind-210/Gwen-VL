"""GWen Direct Preference Optimization."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import DPODataset
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
    save_checkpoint,
    set_seed,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)


def sequence_logprobs(model, input_ids, labels, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"][:, :-1, :]
    target = labels[:, 1:]
    mask = target.ne(-100)
    safe_target = target.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(dim=-1)


def dpo_loss(policy_model, ref_model, batch, beta: float):
    chosen_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    chosen_mask = batch["chosen_attention_mask"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    rejected_mask = batch["rejected_attention_mask"]

    policy_chosen = sequence_logprobs(policy_model, chosen_ids, chosen_labels, chosen_mask)
    policy_rejected = sequence_logprobs(policy_model, rejected_ids, rejected_labels, rejected_mask)
    with torch.no_grad():
        ref_chosen = sequence_logprobs(ref_model, chosen_ids, chosen_labels, chosen_mask)
        ref_rejected = sequence_logprobs(ref_model, rejected_ids, rejected_labels, rejected_mask)
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    losses = -F.logsigmoid(beta * logits)
    rewards_chosen = beta * (policy_chosen - ref_chosen).detach()
    rewards_rejected = beta * (policy_rejected - ref_rejected).detach()
    return losses.mean(), rewards_chosen.mean(), rewards_rejected.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="GWen DPO")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--sft_path", required=True)
    parser.add_argument("--data_path", default="./dataset/dpo.jsonl")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--max_seq_len", "--max_length", type=int, default=768)
    parser.add_argument("--linear_attention_backend", default="auto", choices=["auto", "gdn", "full"])
    parser.add_argument("--gdn_kernel_backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--gated_attention", default="auto", choices=["auto", "none", "headwise", "elementwise", "sigmoid"])
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
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
    logger = Logger(os.path.join(args.out_dir, "dpo.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    sft_ckpt = load_checkpoint(args.sft_path, map_location="cpu")
    config = config_from_checkpoint(
        sft_ckpt,
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
    policy_model = GWenForCausalLM(config).to(env["device"])
    ref_model = GWenForCausalLM(config).to(env["device"])
    load_model_weights(policy_model, sft_ckpt, env["device"], strict=False)
    load_model_weights(ref_model, sft_ckpt, env["device"], strict=False)
    for param in ref_model.parameters():
        param.requires_grad_(False)
    ref_model.eval()

    lora_config = None
    if args.use_lora:
        lora_config = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
        apply_lora_to_model(policy_model, lora_config)
        logger.info(f"LoRA DPO trainable params: {count_lora_params(policy_model)/1e6:.2f}M")

    dtype, dtype_name = resolve_dtype(args.dtype, env["device"])
    if args.use_compile and hasattr(torch, "compile"):
        policy_model = torch.compile(policy_model)
    policy_model = wrap_ddp(policy_model, env)
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader, sampler = create_dataloader(dataset, args.batch_size, env["distributed"], num_workers=args.num_workers)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, len(loader) * args.epochs // args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unwrap_model(policy_model).parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, total_steps, min(args.warmup_steps, max(1, total_steps // 5))),
    )
    scaler = make_scaler(dtype_name, env["device"])

    logger.banner(f"GWen DPO {args.config}: pairs={len(dataset)} beta={args.beta} dtype={dtype_name}")
    policy_model.train()
    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_start_time = time.time()
        iter_per_epoch = len(loader)
        for batch_idx, raw_batch in enumerate(loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            batch = {k: v.to(env["device"]) for k, v in raw_batch.items()}
            sync_grad = (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
            with maybe_no_sync(policy_model, env["distributed"] and not sync_grad):
                with amp_context(dtype_name, dtype, env["device"]):
                    loss, chosen_reward, rejected_reward = dpo_loss(policy_model, ref_model, batch, args.beta)
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
                    f"DPO Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{iter_per_epoch}) "
                    f"loss={current_loss:.4f} avg_loss = {avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)}"
                )

            if sync_grad:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(policy_model).parameters(), args.grad_clip)
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
        if args.use_lora:
            final = os.path.join(args.out_dir, f"dpo_lora_r{args.lora_r}.pth")
            save_lora_checkpoint(unwrap_model(policy_model), lora_config, final)
        else:
            final = os.path.join(args.out_dir, "dpo_final.pth")
            save_checkpoint(final, policy_model, config, optimizer, scheduler, scaler, step=step, epoch=args.epochs, train_args=vars(args))
        logger.info(f"Done. Saved {final}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
