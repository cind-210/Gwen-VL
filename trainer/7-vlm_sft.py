"""GWen image-only VLM supervised fine-tuning."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import SiglipImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import VLMSFTDataset
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
    parser = argparse.ArgumentParser(description="GWen VLM SFT")
    parser.add_argument("--config", default="gwen8k_hybrid", choices=sorted(CONFIG_PRESETS))
    parser.add_argument("--tokenizer_path", default="model/tokenizer_mini8k")
    parser.add_argument("--vision_model_path", default="models/siglip2-base-p32-256-ve")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_llm_mode", default="edge", choices=["edge", "full"])
    parser.add_argument("--train_llm_edge_layers", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow missing/skipped VLM checkpoint weights")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_torch_speed()
    env = setup_distributed(args.device)
    set_seed(args.seed + env["rank"])
    os.makedirs(args.out_dir, exist_ok=True)
    logger = Logger(os.path.join(args.out_dir, "vlm_sft.log"), enabled=env["is_main"])

    tokenizer = load_tokenizer(args.tokenizer_path)
    ckpt = load_checkpoint(args.pretrained, map_location="cpu")
    config = config_from_checkpoint(
        ckpt,
        fallback_name=args.config,
        max_seq_len=args.max_seq_len,
        tokenizer_vocab_size=len(tokenizer),
    )
    configure_vision_token_ids(config, tokenizer)
    config.vision_model_name = args.vision_model_path
    model = GWenForCausalLM(config, vision_model_path=args.vision_model_path).to(env["device"])
    load_info = load_model_weights(model, ckpt, env["device"], strict=False)
    missing = len(load_info.get("missing_keys", []))
    unexpected = len(load_info.get("unexpected_keys", []))
    skipped = len(load_info.get("skipped_shape_mismatch", {}))
    if env["is_main"]:
        logger.info(f"Loaded VLM checkpoint: missing={missing} unexpected={unexpected} skipped_shape={skipped}")
    if (missing or unexpected or skipped) and not args.allow_partial_load:
        raise RuntimeError(
            "VLM checkpoint did not fully match the VLM SFT model. "
            "Use the same tokenizer/config/vision path as VLM pretrain, or pass --allow_partial_load only for experiments."
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
    model = wrap_ddp(model, env)

    image_processor = SiglipImageProcessor.from_pretrained(args.vision_model_path)
    image_token_count = config.image_grid_size * config.image_grid_size
    dataset = VLMSFTDataset(
        args.data_path,
        tokenizer,
        image_processor,
        max_length=args.max_seq_len,
        image_root=args.image_root,
        image_token_count=image_token_count,
    )
    loader, sampler = create_dataloader(dataset, args.batch_size, env["distributed"], num_workers=args.num_workers)
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

    stats = unwrap_model(model).get_param_breakdown()
    logger.banner(
        f"GWen VLM SFT: total={stats['total']/1e6:.1f}M "
        f"trainable={unwrap_model(model).get_num_trainable_params()/1e6:.2f}M "
        f"train_llm_mode={args.train_llm_mode} dtype={dtype_name}"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_start = time.time()
        iter_per_epoch = len(loader)
        for batch_idx, batch in enumerate(loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            sync_grad = (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader)
            with maybe_no_sync(model, env["distributed"] and not sync_grad):
                input_ids = batch["input_ids"].to(env["device"], non_blocking=True)
                labels = batch["labels"].to(env["device"], non_blocking=True)
                attention_mask = batch["attention_mask"].to(env["device"], non_blocking=True)
                pixel_values = batch["pixel_values"].to(env["device"], non_blocking=True)
                with amp_context(dtype_name, dtype, env["device"]):
                    loss = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                        attention_mask=attention_mask,
                    )["loss"]
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
                elapsed = time.time() - epoch_start
                eta = (elapsed / max(1, current_batch)) * max(0, iter_per_epoch - current_batch)
                logger.info(
                    f"VLM SFT Epoch [{epoch+1}/{args.epochs}] ({current_batch}/{iter_per_epoch}) "
                    f"loss={current_loss:.4f} avg_loss={avg_loss:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} eta={format_eta(eta)} step={step}"
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
        final_path = os.path.join(args.out_dir, "vlm_sft_final.pth")
        save_checkpoint(final_path, model, config, optimizer, scheduler, scaler, step=step, epoch=args.epochs, train_args=vars(args))
        logger.info(f"Done. Saved {final_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
