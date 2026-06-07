"""Experimental GWen GRPO / RLAIF training."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import RLAIFDataset
from model.model_gwen import CONFIG_PRESETS, GWenForCausalLM
from trainer.common import (
    config_from_checkpoint,
    load_checkpoint,
    load_model_weights,
    load_tokenizer,
    save_checkpoint,
)


def compute_rewards(model, prompt_ids, response_ids):
    """Toy reward: negative perplexity plus a small length bonus."""
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    out = model(input_ids=full_ids, labels=full_ids)
    ppl = torch.exp(out["loss"])
    return -ppl + math.log(max(1, response_ids.size(-1)))


def grpo_loss(model, ref_model, prompt_ids, response_ids, advantages, clip_eps=0.2, beta=0.04):
    """Simplified GRPO-style loss for experiments."""
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    labels = full_ids.clone()
    prompt_len = prompt_ids.size(-1)
    labels[:, :prompt_len] = -100

    out = model(input_ids=full_ids, labels=labels)
    log_p = -out["loss"] * (labels != -100).sum()

    with torch.no_grad():
        out_ref = ref_model(input_ids=full_ids, labels=labels)
        log_p_ref = -out_ref["loss"] * (labels != -100).sum()

    ratio = torch.exp(log_p - log_p_ref)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    kl = (log_p_ref - log_p).mean() * beta
    return -torch.min(ratio * advantages, clipped * advantages).mean() + kl


@torch.no_grad()
def generate_response(model, prompt_ids, eos_token_id=None, max_new_tokens=128, temperature=0.8):
    """Generate one rollout from the current policy."""
    generated = prompt_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=generated)
        logits = out["logits"][:, -1, :]
        if temperature and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="GWen experimental GRPO")
    parser.add_argument("--config", type=str, default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", type=str, default="model/tokenizer_mini8k")
    parser.add_argument("--sft_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./dataset/rlaif.jsonl")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--max_length", "--max_seq_len", type=int, default=512)
    parser.add_argument("--linear_attention_backend", default="auto", choices=["auto", "gdn", "full"])
    parser.add_argument("--gdn_kernel_backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--gated_attention", default="auto", choices=["auto", "none", "headwise", "elementwise", "sigmoid"])
    parser.add_argument("--rotary_dim", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(args.tokenizer_path)
    sft_ckpt = load_checkpoint(args.sft_path, map_location="cpu")
    config = config_from_checkpoint(
        sft_ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_length,
        tokenizer_vocab_size=len(tokenizer),
    )
    if args.linear_attention_backend != "auto":
        config.linear_attention_backend = args.linear_attention_backend
    config.gdn_kernel_backend = args.gdn_kernel_backend
    if args.gated_attention != "auto":
        config.gated_attention = args.gated_attention
        config.attn_output_gate = args.gated_attention != "none"
    config.rotary_dim = args.rotary_dim

    model = GWenForCausalLM(config).to(device)
    ref_model = GWenForCausalLM(config).to(device)
    load_model_weights(model, sft_ckpt, device, strict=False)
    load_model_weights(ref_model, sft_ckpt, device, strict=False)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"RLAIF dataset: {len(dataset)} samples")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.train()
    step = 0
    t_start = time.time()

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            prompt_ids = batch["prompt_ids"].to(device)

            rollouts = []
            rewards_list = []
            for _ in range(args.num_rollouts):
                resp = generate_response(model, prompt_ids, eos_token_id=tokenizer.eos_token_id)
                resp_ids = resp[:, prompt_ids.size(-1) :]
                rollouts.append(resp)
                rewards_list.append(compute_rewards(model, prompt_ids, resp_ids))

            rewards = torch.stack(rewards_list)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            total_loss = 0
            for i in range(args.num_rollouts):
                loss = grpo_loss(
                    model,
                    ref_model,
                    prompt_ids,
                    rollouts[i][:, prompt_ids.size(-1) :],
                    advantages[i : i + 1],
                )
                total_loss = total_loss + loss
            total_loss = total_loss / args.num_rollouts

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step += 1
            if step % args.log_interval == 0:
                print(
                    f"Step {step} | grpo_loss={total_loss.item():.4f} | "
                    f"mean_reward={rewards.mean().item():.4f} | elapsed={time.time() - t_start:.0f}s"
                )

    final_path = os.path.join(args.out_dir, "grpo_final.pth")
    save_checkpoint(final_path, model, config, optimizer=optimizer, step=step, epoch=args.num_epochs, train_args=vars(args))
    print(f"GRPO complete! Saved to {final_path}")


if __name__ == "__main__":
    main()
