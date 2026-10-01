import math
import torch
from torch.optim import Optimizer  # base class only


class NAdamW(Optimizer):
    """NAdamW: AdamW with Nesterov momentum (Dozat, 2016) and decoupled weight decay.

    Update (t = step count, starting at 1):
        m_t   = b1 * m_{t-1} + (1 - b1) * g_t
        v_t   = b2 * v_{t-1} + (1 - b2) * g_t^2
        m_hat = b1 * m_t / (1 - b1^(t+1)) + (1 - b1) * g_t / (1 - b1^t)   # Nesterov look-ahead
        v_hat = v_t / (1 - b2^t)
        p     = p * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)

    The only difference from AdamW is m_hat: instead of the bias-corrected m_t,
    it uses a look-ahead that already applies one extra momentum step.
    Check this against the pseudocode in the paper's appendix before you report results.
    """

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
                    raise RuntimeError("NAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # moment updates
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_c1_now = 1.0 - beta1 ** t        # for the current gradient
                bias_c1_next = 1.0 - beta1 ** (t + 1)  # for the look-ahead momentum
                bias_c2 = 1.0 - beta2 ** t

                # Nesterov-corrected first moment
                m_hat = m * (beta1 / bias_c1_next) + grad * ((1.0 - beta1) / bias_c1_now)

                denom = (v.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


if __name__ == "__main__":
    # Sanity check against a plain-tensor reference of the formula above.
    # (torch.optim.NAdam uses a different momentum-decay schedule, so it won't match exactly.)
    torch.manual_seed(0)
    lr, b1, b2, eps, wd = 1e-2, 0.9, 0.999, 1e-8, 0.1

    w1 = torch.nn.Parameter(torch.randn(10, 10))
    w_ref = w1.detach().clone()
    m = torch.zeros_like(w_ref)
    v = torch.zeros_like(w_ref)
    opt = NAdamW([w1], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    for t in range(1, 51):
        x = torch.randn(32, 10)

        opt.zero_grad()
        (x @ w1).pow(2).mean().backward()
        opt.step()

        # reference: same data, autograd-free gradient of mean((xW)^2) w.r.t. W
        g = 2.0 * x.t() @ (x @ w_ref) / (32 * 10)
        w_ref = w_ref * (1 - lr * wd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = b1 * m / (1 - b1 ** (t + 1)) + (1 - b1) * g / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w_ref = w_ref - lr * m_hat / (v_hat.sqrt() + eps)

    print("max abs diff vs reference formula:", (w1.detach() - w_ref).abs().max().item())