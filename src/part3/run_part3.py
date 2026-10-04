"""
Evaluate the decoding strategies on ROCStories with Pythia-160M.

    python run_part3.py --text_col <column>  --n_samples 1000

Protocol (state it in your report, change it if the assignment/TAs define it differently):
  * Each story is tokenized; the first `prompt_len` tokens are the prompt, the next `gen_len`
    tokens are the reference continuation. Stories shorter than that are skipped.
  * Every strategy generates `gen_len` tokens from the prompt.
  * perplexity      = exp(mean NLL of the GENERATED tokens under Pythia-160M)
  * token accuracy  = fraction of generated tokens equal to the reference token at the same
                      position
  * a baseline row gives teacher-forced perplexity / next-token accuracy on the reference.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding import beam_search, greedy, top_k_sampling, top_p_sampling


def load_samples(args, tok):
    ds = load_dataset(args.dataset)
    split = "test" if "test" in ds else "train"
    d = ds[split]
    print(f"split={split}, columns={d.column_names}")
    if args.text_col not in d.column_names:
        raise SystemExit(f"--text_col '{args.text_col}' not found; pick one of {d.column_names}")

    total = args.prompt_len + args.gen_len
    rows = []
    for text in d[args.text_col]:
        ids = tok(text).input_ids
        if len(ids) >= total:
            rows.append(ids[:total])
        if len(rows) == args.n_samples:
            break
    print(f"{len(rows)} usable samples (need >= 1000 for the assignment)")
    x = torch.tensor(rows)
    return x[:, : args.prompt_len], x[:, args.prompt_len :]


@torch.no_grad()
def teacher_forced(model, prompt, target):
    """Per-sample mean NLL of `target` given `prompt` + earlier target tokens, and the
    per-sample accuracy of the model's argmax prediction against `target`."""
    ids = torch.cat([prompt, target], dim=1)
    logits = model(input_ids=ids).logits[:, prompt.size(1) - 1 : -1].float()
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, target[..., None]).squeeze(-1).mean(dim=1)
    acc = (logits.argmax(-1) == target).float().mean(dim=1)
    return nll, acc


def distinct_n(gens, n=2):
    grams = [tuple(g[i : i + n]) for g in gens.tolist() for i in range(len(g) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_config(model, prompts, name, kw, args, device):
    """Generate for all prompts; returns (generations (N, gen_len), seconds)."""
    torch.manual_seed(args.seed)
    outs = []
    sync(device)
    t0 = time.time()
    if name == "beam":  # beams are the batch dim, so go one prompt at a time
        for i in range(len(prompts)):
            toks, _ = beam_search(model, prompts[i : i + 1].to(device), kw["width"], args.gen_len)
            outs.append(toks[None].cpu())
    else:
        fn = {"greedy": greedy, "top_k": top_k_sampling, "top_p": top_p_sampling}[name]
        extra = {} if name == "greedy" else {**kw, "temperature": args.temperature}
        for i in range(0, len(prompts), args.bs):
            p = prompts[i : i + args.bs].to(device)
            outs.append(fn(model, p, args.gen_len, **extra).cpu())
    sync(device)
    return torch.cat(outs), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--dataset", default="hamishivi/ROCStories")
    ap.add_argument("--text_col", default="text")  # CHECK: printed column names on first run
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--prompt_len", type=int, default=32)
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_ks", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--top_ps", type=float, nargs="+", default=[0.7, 0.9, 0.95])
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/part3")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    prompts, refs = load_samples(args, tok)

    os.makedirs(f"{args.out_dir}/generations", exist_ok=True)
    results = {}

    # baseline: teacher-forced metrics on the reference continuation
    nlls, accs = [], []
    for i in range(0, len(prompts), args.bs):
        n, a = teacher_forced(model, prompts[i : i + args.bs].to(device), refs[i : i + args.bs].to(device))
        nlls.append(n.cpu()); accs.append(a.cpu())
    results["reference (teacher-forced)"] = {
        "ppl": math.exp(torch.cat(nlls).mean().item()),
        "token_acc": torch.cat(accs).mean().item(),
    }

    configs = [("greedy", {})]
    configs += [("beam", {"width": w}) for w in args.beams]
    configs += [("top_k", {"k": k}) for k in args.top_ks]
    configs += [("top_p", {"p": p}) for p in args.top_ps]

    for name, kw in configs:
        label = name + "".join(f"_{k}{v}" for k, v in kw.items())
        gens, secs = run_config(model, prompts, name, kw, args, device)

        nlls = []
        for i in range(0, len(gens), args.bs):
            n, _ = teacher_forced(model, prompts[i : i + args.bs].to(device), gens[i : i + args.bs].to(device))
            nlls.append(n.cpu())
        results[label] = {
            "ppl": math.exp(torch.cat(nlls).mean().item()),
            "token_acc": (gens == refs).float().mean().item(),
            "distinct_2": distinct_n(gens, 2),
            "total_sec": secs,
            "sec_per_sample": secs / len(gens),
        }
        print(label, results[label])

        with open(f"{args.out_dir}/generations/{label}.jsonl", "w") as f:  # raw generations
            for p, r, g in zip(prompts, refs, gens):
                f.write(json.dumps({
                    "prompt": tok.decode(p), "reference": tok.decode(r), "generation": tok.decode(g),
                }, ensure_ascii=False) + "\n")

    with open(f"{args.out_dir}/results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    print(f"\n{'config':28s} {'ppl':>8s} {'acc':>7s} {'dist-2':>7s} {'sec/sample':>11s}")
    for k, v in results.items():
        print(f"{k:28s} {v['ppl']:8.2f} {v['token_acc']:7.3f} {v.get('distinct_2', float('nan')):7.3f} "
              f"{v.get('sec_per_sample', float('nan')):11.4f}")


if __name__ == "__main__":
    main()
