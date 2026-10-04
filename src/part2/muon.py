import math
import torch
from torch.optim import Optimizer  # base class only


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximately orthogonalize G (push all singular values toward 1) using a
    quintic Newton-Schulz iteration (coefficients from Keller Jordan's Muon)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    # bf16 is fast on Ampere+, but unsupported/slow on e.g. a 2080 Ti -> fall back to fp32
    dtype = torch.bfloat16 if (G.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    X = G.to(dtype)
    transposed = X.size(0) > X.size(1)
    if transposed:  # work with the wide orientation so X @ X.T is the small matrix
        X = X.T
    X = X / (X.norm() + eps)  # spectral norm <= Frobenius norm <= 1, so iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.

    A param group is either
      * use_muon=True  : 2D weight matrices -> momentum, orthogonalized update
      * use_muon=False : everything else (embeddings, output head, norms, biases)
                         -> plain AdamW update (implemented here, no torch.optim)

    adjust_lr controls the per-matrix step scale for Muon groups:
      "match_rms": 0.2 * sqrt(max(rows, cols))  (Moonlight) -> update RMS ~ AdamW's,
                   so you can reuse AdamW's lr and weight decay
      "original" : sqrt(max(1, rows/cols))      (Keller Jordan) -> wants lr ~ 0.02
    """

    def __init__(self, params, lr=6e-4, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.1, adjust_lr="match_rms",
                 adam_betas=(0.9, 0.95), adam_eps=1e-8):
        if adjust_lr not in ("match_rms", "original"):
            raise ValueError(f"Invalid adjust_lr: {adjust_lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, adjust_lr=adjust_lr,
                        adam_betas=adam_betas, adam_eps=adam_eps, use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        mom, nesterov = group["momentum"], group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]

            g = p.grad
            buf.lerp_(g, 1.0 - mom)                      # buf = mom*buf + (1-mom)*g
            g_eff = g.lerp(buf, mom) if nesterov else buf  # Nesterov look-ahead (not in-place on p.grad)

            g2d = g_eff.reshape(g_eff.size(0), -1)       # conv-style tensors -> 2D
            update = zeropower_via_newtonschulz5(g2d, steps=group["ns_steps"]).view_as(p)

            rows, cols = g2d.shape
            if group["adjust_lr"] == "match_rms":
                scale = 0.2 * math.sqrt(max(rows, cols))
            else:
                scale = max(1.0, rows / cols) ** 0.5

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            m, v = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            m.mul_(beta1).add_(p.grad, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
            denom = (v.sqrt() / math.sqrt(1.0 - beta2 ** t)).add_(eps)
            p.addcdiv_(m, denom, value=-lr / (1.0 - beta1 ** t))


def muon_param_groups(model, wd=0.1):
    """Matrices in the transformer blocks -> Muon. Embeddings, LM head, norms,
    biases -> AdamW (Muon is not meant for these)."""
    muon, adam = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in n:
            muon.append(p)
        else:
            adam.append(p)
    return [
        {"params": muon, "use_muon": True, "weight_decay": wd},
        {"params": adam, "use_muon": False, "weight_decay": 0.0},
    ]


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Newton-Schulz should push singular values toward ~1 (quintic iteration lands in ~[0.7, 1.2])
    for shape in [(64, 64), (64, 256), (256, 64)]:
        G = torch.randn(*shape)
        S = torch.linalg.svdvals(zeropower_via_newtonschulz5(G).float())
        print(f"NS {shape}: singular values in [{S.min():.3f}, {S.max():.3f}]")

    # 2) Hybrid optimizer reduces loss on a toy regression
    W = torch.nn.Parameter(torch.randn(32, 16) * 0.1)  # matrix -> Muon
    b = torch.nn.Parameter(torch.zeros(32))            # vector -> AdamW
    X, Y = torch.randn(256, 16), torch.randn(256, 32)
    opt = Muon([
        {"params": [W], "use_muon": True},
        {"params": [b], "use_muon": False, "weight_decay": 0.0},
    ], lr=1e-2)
    for i in range(201):
        opt.zero_grad()
        loss = ((X @ W.T + b - Y) ** 2).mean()
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"step {i}: loss {loss.item():.4f}")