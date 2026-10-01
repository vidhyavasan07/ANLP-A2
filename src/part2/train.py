"""
Part 2: pretrain a small decoder-only transformer on browndw/human-ai-parallel-corpus
with a pluggable from-scratch optimizer.

Usage:
    python train.py --optimizer adamw --lr 6e-4 --wandb_project anlp-a2-part2

Install:
    uv add torch transformers datasets wandb sacrebleu
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

# ---- your from-scratch optimizers live here; register each one as you finish it ----
from part2.adamw import AdamW  # e.g. from src.part2.optimizers.adamw import AdamW

OPTIMIZERS = {
    "adamw": lambda params, a: AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd),
    # "lion":  lambda params, a: Lion(params, lr=a.lr, ...),
    # "adafactor": ...
    # "muon": ...
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def build_data(args, tok):
    ds = load_dataset("browndw/human-ai-parallel-corpus", split="train")
    print("Columns:", ds.column_names)  # CHECK the text column name and set --text_col
    if args.max_docs:
        ds = ds.select(range(min(args.max_docs, len(ds))))

    # split by document (no chunk leakage between splits)
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    holdout = split["test"].train_test_split(test_size=0.5, seed=args.seed)
    parts = {"train": split["train"], "val": holdout["train"], "test": holdout["test"]}

    def tokenize(batch):
        return {"ids": [tok(t).input_ids + [tok.eos_token_id] for t in batch[args.text_col]]}

    out = {}
    for name, d in parts.items():
        d = d.map(tokenize, batched=True, remove_columns=d.column_names, num_proc=4)
        flat = torch.tensor([i for row in d["ids"] for i in row], dtype=torch.long)
        n = (len(flat) // args.seq_len) * args.seq_len
        out[name] = flat[:n].view(-1, args.seq_len)  # (num_chunks, seq_len)
        print(f"{name}: {out[name].numel():,} tokens, {out[name].shape[0]:,} chunks")
    return out


# ----------------------------------------------------------------------------
# Model (HF GPT-NeoX, randomly initialised; the transformer is NOT the point here)
# ----------------------------------------------------------------------------
def build_model(args, vocab_size):
    cfg = GPTNeoXConfig(
        vocab_size=vocab_size,
        hidden_size=args.d_model,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        intermediate_size=4 * args.d_model,
        max_position_embeddings=args.seq_len,
        rotary_pct=0.25,
        use_parallel_residual=True,
        attn_implementation="sdpa",
    )
    model = GPTNeoXForCausalLM(cfg)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model


def param_groups(model, wd):
    """No weight decay on biases / norms / embeddings."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "embed" in n else decay).append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ----------------------------------------------------------------------------
# LR schedule (manual, since torch schedulers wrap torch.optim behaviour)
# ----------------------------------------------------------------------------
def lr_at(step, total, base_lr, warmup_frac=0.02, min_ratio=0.1):
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, data, args, device, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), args.eval_bs):
        if max_batches and i // args.eval_bs >= max_batches:
            break
        x = data[i : i + args.eval_bs].to(device)
        logits = model(x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="sum")
        tot += loss.item()
        n += x[:, 1:].numel()
    model.train()
    return tot / n


@torch.no_grad()
def eval_bleu(model, tok, test, args, device):
    """Prompt with the first `prompt_len` tokens of a test chunk, greedily continue,
    and BLEU-score against the true next `gen_len` tokens. (Using HF generate is fine
    in Part 2; only Part 3 forbids it.)"""
    import sacrebleu

    model.eval()
    chunks = test[: args.bleu_samples]
    hyps, refs = [], []
    for i in range(0, len(chunks), args.eval_bs):
        x = chunks[i : i + args.eval_bs].to(device)
        prompt = x[:, : args.prompt_len]
        ref = x[:, args.prompt_len : args.prompt_len + args.gen_len]
        out = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=args.gen_len,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )[:, args.prompt_len :]
        hyps += tok.batch_decode(out, skip_special_tokens=True)
        refs += tok.batch_decode(ref, skip_special_tokens=True)
    model.train()
    return sacrebleu.corpus_bleu(hyps, [refs]).score


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="adamw", choices=list(OPTIMIZERS))
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--text_col", default="text")
    p.add_argument("--max_docs", type=int, default=0, help="debug: use only N docs")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_evals", type=int, default=10, help="evals at every 1/n_evals of the dataset")
    p.add_argument("--token_frac", type=float, default=1.0, help="fraction of the train tokens to train on (1.0 = 1x dataset)")
    p.add_argument("--bleu_samples", type=int, default=128)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--gen_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="anlp-a2")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--hf_user", default="https://huggingface.co/vidhyavasan", help="HF username/org; if set, pushes the final model to <hf_user>/anlp-a2-<optimizer>")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    data = build_data(args, tok)
    model = build_model(args, len(tok)).to(device)
    opt = OPTIMIZERS[args.optimizer](param_groups(model, args.wd), args)

    train = data["train"]
    steps_per_epoch = len(train) // args.bs
    total_steps = int(steps_per_epoch * args.token_frac)
    eval_every = max(1, total_steps // args.n_evals)  # every 0.1x of the dataset
    tokens_per_step = args.bs * args.seq_len
    print(f"total steps {total_steps}, eval every {eval_every} steps ({eval_every * tokens_per_step:,} tokens)")

    import wandb
    wandb.init(project=args.wandb_project, name=args.optimizer, config=vars(args), tags=[args.optimizer])

    perm = torch.randperm(len(train))  # one pass = 1x the dataset
    model.train()
    t0 = time.time()
    for step in range(1, total_steps + 1):
        idx = perm[(step - 1) * args.bs : step * args.bs]
        x = train[idx].to(device)

        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, total_steps, args.lr)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        tokens_seen = step * tokens_per_step
        if step % 20 == 0:
            wandb.log({"train/loss": loss.item(), "train/grad_norm": gnorm.item(),
                       "lr": opt.param_groups[0]["lr"], "tokens": tokens_seen}, step=step)

        if step % eval_every == 0 or step == total_steps:
            val = eval_loss(model, data["val"], args, device)
            bleu = eval_bleu(model, tok, data["test"], args, device)
            frac = tokens_seen / (steps_per_epoch * tokens_per_step)
            print(f"step {step} | {frac:.2f}x data | val loss {val:.4f} | test BLEU {bleu:.2f} | {time.time() - t0:.0f}s")
            wandb.log({"val/loss": val, "test/bleu": bleu, "data_frac": frac, "tokens": tokens_seen}, step=step)

    os.makedirs(f"{args.out_dir}/{args.optimizer}", exist_ok=True)
    model.save_pretrained(f"{args.out_dir}/{args.optimizer}")  # add checkpoints/ to .gitignore
    tok.save_pretrained(f"{args.out_dir}/{args.optimizer}")
    if args.hf_user:
        repo = f"{args.hf_user}/anlp-a2-{args.optimizer}"
        model.push_to_hub(repo)  # creates the repo if needed (public by default)
        tok.push_to_hub(repo)
        print(f"Pushed to https://huggingface.co/{repo}")
        wandb.summary["hf_checkpoint"] = f"https://huggingface.co/{repo}"
    wandb.finish()


if __name__ == "__main__":
    main()