"""AdamW that updates same-shaped parameters together, as one stacked tensor.

PyTorch/XLA traces every operation on the host before the chip runs the step.
With ``torch.optim.AdamW`` and ``clip_grad_norm_`` over the ~1,140 parameter
tensors of three ConvNeXt backbones, a training step took about 1 s of host
time on a TPU VM against 75 ms on the chip, so the chips sat idle. Here the
parameters are bucketed by shape (27 shapes), each bucket's parameters and
gradients are stacked, and clipping, the cross-chip average, and the AdamW
update run on the stacks; the new values are copied back row by row. A step
then takes about 190 ms of host time. The arithmetic is that of
``torch.optim.AdamW`` (amsgrad off) and ``torch.nn.utils.clip_grad_norm_``.

Two things that look equivalent are much slower on XLA and are avoided:
keeping persistent stacks that are updated in place and then sliced (each
copy from a slice of a mutated tensor costs ~0.6 ms), and concatenating
everything into one flat vector (XLA rewrites reshapes of its slices into a
reshape of the whole vector, which the TPU's tiled layout pads to tens of GB).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


class StackedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        for group in self.param_groups:
            self._bucket(group)

    @staticmethod
    def _bucket(group) -> None:
        buckets = {}
        for i, p in enumerate(group["params"]):
            buckets.setdefault((tuple(p.shape), p.dtype), []).append(i)
        group["buckets"] = list(buckets.values())

    def grads(self) -> List[torch.Tensor]:
        """Stacked gradients, one per bucket of every group (zeros where a parameter got none)."""
        out = []
        for group in self.param_groups:
            params = group["params"]
            for idx in group["buckets"]:
                out.append(torch.stack([params[i].grad if params[i].grad is not None
                                        else torch.zeros_like(params[i]) for i in idx]))
        return out

    @staticmethod
    def clip_(grads: List[torch.Tensor], max_norm: float) -> torch.Tensor:
        """In-place global-norm clipping, as clip_grad_norm_; returns the norm before clipping."""
        total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        scale = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for g in grads:
            g.mul_(scale)
        return total

    @torch.no_grad()
    def step(self, grads: Optional[List[torch.Tensor]] = None):
        grads = iter(self.grads() if grads is None else grads)
        for group in self.param_groups:
            params, (beta1, beta2) = group["params"], group["betas"]
            state = self.state[params[0]]
            if not state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = [torch.zeros((len(idx),) + params[idx[0]].shape, dtype=params[idx[0]].dtype,
                                                device=params[idx[0]].device) for idx in group["buckets"]]
                state["exp_avg_sq"] = [torch.zeros_like(m) for m in state["exp_avg"]]
            state["step"] += 1
            step = state["step"].item()  # a CPU counter: no device sync
            bias1, bias2 = 1 - beta1 ** step, 1 - beta2 ** step
            # The values that change during training reach the device as one tensor. Passed as Python floats,
            # each would become a device scalar that torch_xla caches by value, so two that round to the same
            # float32 would share one: the decay factors of the two groups once the head's learning rate has
            # decayed a little (epoch 6), or sqrt(bias2) and beta2 at step 6212 (epoch 18). The graph then has one
            # input fewer and the whole training step compiles again, which at 512 px took 9 minutes and crashed.
            decay, step_size, sqrt_bias2 = torch.tensor(
                [1 - group["lr"] * group["weight_decay"], -group["lr"] / bias1, math.sqrt(bias2)],
                dtype=torch.float32).to(params[0].device).unbind(0)
            for idx, m, v in zip(group["buckets"], state["exp_avg"], state["exp_avg_sq"]):
                grad = next(grads)
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (v.sqrt() / sqrt_bias2).add_(group["eps"])
                stack = torch.stack([params[i] for i in idx]) * decay
                stack = stack + step_size * (m / denom)  # out of place, see the docstring
                for i, row in zip(idx, stack.unbind(0)):
                    params[i].copy_(row)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None
