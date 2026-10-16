"""Helpers shared by the TPU scripts (PyTorch/XLA).

Importing this module puts the repository root on ``sys.path`` so the scripts
in ``tpu/`` can use the ``mvtree`` package without installing it. Nothing here
touches the TPU at import time: the runtime must be initialised inside the
per-chip processes started by ``torch_xla.launch``, never in the parent.
"""

from __future__ import annotations

import contextlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def require_xla() -> None:
    """Fail early with a useful message when torch_xla is missing."""
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    try:
        import torch_xla  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch_xla is not installed. On a Cloud TPU VM run `bash tpu/setup_vm.sh` "
                         "(see tpu/README.md).") from exc


def launch(fn, args: tuple, all_chips: bool = True) -> None:
    """Run ``fn(index, *args)`` once per TPU chip on this host, or once in-process."""
    if all_chips:
        import torch_xla

        torch_xla.launch(fn, args=args)
    else:
        fn(0, *args)


def share_cpu(world: int) -> None:
    """Give each chip process its share of the host's cores (building ten networks in four
    processes that each use every core is several times slower)."""
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))


def autocast(precision: str):
    """bf16 autocast on the XLA device (no loss scaling needed), or full fp32."""
    if precision == "bf16":
        import torch

        return torch.autocast("xla", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compile_count() -> int:
    """Number of XLA graph compilations so far in this process."""
    import torch_xla.debug.metrics as met

    data = met.metric_data("CompileTime")
    return int(data[0]) if data else 0
