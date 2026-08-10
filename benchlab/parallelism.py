"""How a config spreads one engine over more than one GPU.

vLLM has four independent ways to do it and they are not interchangeable:

    tensor    shards every layer across ranks; all-reduces on every forward pass
    pipeline  gives each rank a slice of the layers; ranks talk once per stage
    data      whole replicas of the model behind one endpoint; no shared tensors
    expert    for MoE models, spreads the experts across the ranks TP/DP provide

The first three multiply into the number of devices a pod needs — TP x PP x DP —
which is why the count lives here rather than being written into the Deployment
by hand. A pod that requests one GPU and is then told `--tensor-parallel-size 2`
does not fail fast: it hangs waiting for a rank that can never start, times out
after the (deliberately long) rollout deadline, and gets recorded as a failure of
whatever knob the sweep thought it was measuring.
"""

from __future__ import annotations

import dataclasses

TENSOR = "TENSOR_PARALLEL_SIZE"
PIPELINE = "PIPELINE_PARALLEL_SIZE"
DATA = "DATA_PARALLEL_SIZE"
EXPERT = "EXPERT_PARALLEL"

# The keys this module owns, in the order a label spells them out.
KEYS = (TENSOR, PIPELINE, DATA, EXPERT)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


@dataclasses.dataclass(frozen=True)
class Topology:
    """The parallelism of one config, and what it costs in devices."""

    tensor: int = 1
    pipeline: int = 1
    data: int = 1
    expert: bool = False

    @property
    def gpus(self) -> int:
        return self.tensor * self.pipeline * self.data

    @property
    def label(self) -> str:
        """Short form for tags, tables and plot legends: tp2, dp2-ep, tp2-pp2."""
        parts = [
            f"{prefix}{size}"
            for prefix, size in (
                ("tp", self.tensor),
                ("pp", self.pipeline),
                ("dp", self.data),
            )
            if size > 1
        ]
        if self.expert:
            parts.append("ep")
        # A single-GPU config is "tp1" rather than "" so it reads as a topology
        # next to the multi-GPU ones instead of looking like missing data.
        return "-".join(parts) or "tp1"

    @property
    def human(self) -> str:
        spelled = [
            f"{name} {size}"
            for name, size in (
                ("tensor-parallel", self.tensor),
                ("pipeline-parallel", self.pipeline),
                ("data-parallel", self.data),
            )
            if size > 1
        ]
        if self.expert:
            spelled.append("expert-parallel")
        noun = "GPU" if self.gpus == 1 else "GPUs"
        return f"{self.gpus} {noun}" + (f" ({', '.join(spelled)})" if spelled else "")

    @property
    def flags(self) -> list[str]:
        """vLLM flags this topology needs beyond the sizes the Deployment passes.

        Expert parallelism is a switch rather than a number, so it cannot ride
        along as `--flag "$VALUE"` like the sizes do; it is appended to
        EXTRA_ARGS instead, where the rendered ConfigMap still shows it.
        """
        return ["--enable-expert-parallel"] if self.expert else []


def _size(params: dict[str, str], key: str) -> int:
    raw = str(params.get(key, "1") or "1").strip()
    try:
        size = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a positive integer, got '{raw}'") from None
    if size < 1:
        raise ValueError(f"{key} must be at least 1, got {size}")
    return size


def _switch(params: dict[str, str], key: str) -> bool:
    raw = str(params.get(key, "") or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(f"{key} must be one of {', '.join(accepted)}, got '{raw}'")


def parse(params: dict[str, str]) -> Topology:
    """Read the topology out of a config's ConfigMap keys.

    Raises ValueError on a malformed or self-contradictory combination, so the
    caller can name the experiment that asked for it.
    """
    topology = Topology(
        tensor=_size(params, TENSOR),
        pipeline=_size(params, PIPELINE),
        data=_size(params, DATA),
        expert=_switch(params, EXPERT),
    )
    if topology.expert and topology.gpus == 1:
        raise ValueError(
            f"{EXPERT}=1 needs more than one rank to spread experts over, but "
            f"{TENSOR} x {PIPELINE} x {DATA} is 1"
        )
    return topology


def of(params: dict[str, str] | None) -> Topology:
    """The topology of a config, defaulting to one GPU when it cannot be read.

    For readers rather than writers: an imported run recorded no params at all,
    and every run predating multi-GPU support was single-GPU by construction, so
    the default is the truth for them rather than a guess.
    """
    try:
        return parse(params or {})
    except ValueError:
        return Topology()


def capacity_error(topology: Topology, available: int) -> str | None:
    """Why this topology cannot be scheduled here, or None if it can.

    A pod's GPUs all come from one node — there is no cross-node engine in this
    setup — so the ceiling is the GPU count of a single node.
    """
    if topology.gpus <= available:
        return None
    return (
        f"needs {topology.human} but this cluster has {available} GPU(s) per node "
        f"({TENSOR} x {PIPELINE} x {DATA} is the device count)"
    )
