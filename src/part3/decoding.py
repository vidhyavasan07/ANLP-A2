"""
Part 3: decoding strategies implemented from scratch on top of model.forward().
No model.generate() anywhere.

All functions take a prompt of shape (B, L) (no padding; every prompt in a batch has the
same length) and generate exactly `max_new` tokens (EOS is treated like any other token so
that every strategy produces equal-length outputs and the metrics stay comparable).
They return only the newly generated tokens, shape (B, max_new).

For simplicity and clarity these do NOT use a KV cache: each step re-runs the model on the
full sequence so far. That keeps beam search free of cache-reordering bugs.
"""
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


@torch.no_grad()
def _last_logits(model, ids):
    """Logits for the next token, shape (B, V), in fp32."""
    return model(input_ids=ids).logits[:, -1, :].float()


# ----------------------------------------------------------------------------
# Greedy
# ----------------------------------------------------------------------------
@torch.no_grad()
def greedy(model, prompt, max_new):
    ids = prompt
    for _ in range(max_new):
        nxt = _last_logits(model, ids).argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[:, prompt.size(1):]


# ----------------------------------------------------------------------------
# Top-k / top-p (nucleus) sampling
# ----------------------------------------------------------------------------
def top_k_filter(logits, k):
    """Keep the k highest logits per row, set the rest to -inf."""
    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits, p):
    """Keep the smallest set of top tokens whose cumulative probability reaches p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # drop a token if the mass BEFORE it already exceeds p (so the token that crosses p
    # is kept, and the top-1 token is always kept)
    remove = (cum - probs) > p
    sorted_logits = sorted_logits.masked_fill(remove, NEG_INF)
    # scatter back to the original vocabulary order
    return torch.full_like(logits, NEG_INF).scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def sample(model, prompt, max_new, filter_fn=None, temperature=1.0):
    """Generic ancestral sampling with an optional logit filter."""
    ids = prompt
    for _ in range(max_new):
        logits = _last_logits(model, ids) / temperature
        if filter_fn is not None:
            logits = filter_fn(logits)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[:, prompt.size(1):]


def top_k_sampling(model, prompt, max_new, k=50, temperature=1.0):
    return sample(model, prompt, max_new, lambda l: top_k_filter(l, k), temperature)


def top_p_sampling(model, prompt, max_new, p=0.9, temperature=1.0):
    return sample(model, prompt, max_new, lambda l: top_p_filter(l, p), temperature)


# ----------------------------------------------------------------------------
# Beam search (one prompt at a time; the beams are the batch dimension)
# ----------------------------------------------------------------------------
@torch.no_grad()
def beam_search(model, prompt, width, max_new):
    """prompt: (1, L). Returns (best_tokens (max_new,), best_sum_logprob scalar tensor).

    Beams are ranked by summed log-probability. All beams have the same length, so no
    length normalization is needed."""
    assert prompt.size(0) == 1
    # step 0: a single sequence -> take the top `width` first tokens
    logp = F.log_softmax(_last_logits(model, prompt), dim=-1)       # (1, V)
    V = logp.size(-1)
    scores, tok = logp[0].topk(width)                               # (W,), sorted descending
    beams = torch.cat([prompt.expand(width, -1), tok[:, None]], 1)  # (W, L+1)

    for _ in range(max_new - 1):
        logp = F.log_softmax(_last_logits(model, beams), dim=-1)    # (W, V)
        cand = (scores[:, None] + logp).reshape(-1)                 # (W*V,)
        scores, flat = cand.topk(width)                             # best W of W*V candidates
        beam_idx = torch.div(flat, V, rounding_mode="floor")        # which beam each came from
        tok = flat % V                                              # which token extends it
        beams = torch.cat([beams[beam_idx], tok[:, None]], dim=1)

    return beams[0, prompt.size(1):], scores[0]
