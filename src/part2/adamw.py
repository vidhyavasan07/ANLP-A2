import math
import torch
from torch.optim import Optimizer  # allowed strictly as a base class


class AdamW(Optimizer):
    """AdamW (Loshchilov & Hutter): Adam with *decoupled* weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                # Lazy state init
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)     # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 1) Decoupled weight decay: applied to the weights directly,
                #    NOT added to the gradient (that would be Adam + L2).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2) Update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3) Bias correction
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t

                # 4) Parameter update: p -= lr * m_hat / (sqrt(v_hat) + eps)
                denom = (v.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias_c1)

        return loss


if __name__ == "__main__":
    # Sanity check against torch's reference implementation
    # (only for testing; do NOT use torch.optim.AdamW in your training code)
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(10, 10))
    w2 = torch.nn.Parameter(w1.detach().clone())
    mine = AdamW([w1], lr=1e-2, weight_decay=0.1)
    ref = torch.optim.AdamW([w2], lr=1e-2, weight_decay=0.1)

    for _ in range(50):
        x = torch.randn(32, 10)
        for w, opt in ((w1, mine), (w2, ref)):
            opt.zero_grad()
            (x @ w).pow(2).mean().backward()
            opt.step()

    print("max abs diff vs torch AdamW:", (w1 - w2).abs().max().item())