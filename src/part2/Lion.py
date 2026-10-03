import torch
from torch.optim import Optimizer  # base class only


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum, Chen et al. 2023).

    Update:
        c_t = b1 * m_{t-1} + (1 - b1) * g_t        # interpolate momentum and gradient
        p   = p * (1 - lr * wd) - lr * sign(c_t)   # sign update + decoupled weight decay
        m_t = b2 * m_{t-1} + (1 - b2) * g_t        # momentum updated AFTER the step, with b2

    Only ONE state buffer per parameter (AdamW needs two), and every update entry
    has magnitude exactly lr, so Lion wants a smaller lr and larger weight decay than AdamW.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # update direction uses beta1 interpolation, then sign
                update = m.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # momentum uses beta2
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


if __name__ == "__main__":
    # Sanity check against a plain-tensor reference of the formula above.
    torch.manual_seed(0)
    lr, b1, b2, wd = 1e-2, 0.9, 0.99, 0.1

    w1 = torch.nn.Parameter(torch.randn(10, 10))
    w_ref = w1.detach().clone()
    m = torch.zeros_like(w_ref)
    opt = Lion([w1], lr=lr, betas=(b1, b2), weight_decay=wd)

    for t in range(1, 51):
        x = torch.randn(32, 10)

        opt.zero_grad()
        (x @ w1).pow(2).mean().backward()
        opt.step()

        g = 2.0 * x.t() @ (x @ w_ref) / (32 * 10)
        w_ref = w_ref * (1 - lr * wd)
        c = b1 * m + (1 - b1) * g
        w_ref = w_ref - lr * torch.sign(c)
        m = b2 * m + (1 - b2) * g

    print("max abs diff vs reference formula:", (w1.detach() - w_ref).abs().max().item())