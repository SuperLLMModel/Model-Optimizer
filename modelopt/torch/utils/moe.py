# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace-coupled MoE expert-parallel (EP) helpers.

Capability detection, EP/DP degree selection, the DP=EP all-to-all expert transport
(``torch_a2a``), and the EP-sharded model loader.

The all-to-all transport is plain ``torch.distributed.all_to_all_single`` (NCCL) wrapped around
transformers' quantizable ``_grouped_linear``. ``register_torch_a2a()`` registers
it into transformers' ``ExpertsInterface`` (lazily, from ``load_and_prepare_ep``'s a2a path -- never at
import time, so importing this module has no transformers-``ExpertsInterface`` side effect).
"""

import contextlib
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn

if TYPE_CHECKING:
    from .distributed import ModelOptSession


def require_transformers(min_version: str, feature: str) -> None:
    """Fail loudly if the installed transformers is older than ``min_version`` (the floor for ``feature``).

    Callers enforce the version where the needed APIs first exist, so a too-old transformers errors
    clearly instead of silently mis-behaving (e.g. transformers 5.5 silently shards nothing for EP).
    See each call site for the empirically-determined floor.
    """
    import transformers
    from packaging.version import parse

    have = transformers.__version__
    if parse(have) < parse(min_version):
        raise RuntimeError(
            f"{feature} requires transformers >= {min_version}, but {have} is installed. "
            f"Upgrade with: pip install -U 'transformers>={min_version}'."
        )


def _ep_config(hf_config):
    """Return the config that carries the EP plan / expert count.

    For a plain MoE causal LM this is the root config. For a VLM whose nested text submodel
    is the MoE (e.g. ``KimiK25.text_config`` -> deepseek_v3, or ``NemotronH_Omni.llm_config``
    -> nemotron_h), this is that nested config.
    """
    if getattr(hf_config, "base_model_ep_plan", None):
        return hf_config
    for _attr in ("text_config", "llm_config"):
        sub = getattr(hf_config, _attr, None)
        if sub is not None and getattr(sub, "base_model_ep_plan", None):
            return sub
    # Fall through: no plan anywhere; let callers decide.
    return hf_config


def num_experts_from_config(hf_config) -> int:
    """Return the MoE expert count from an HF config (0 if the model is not MoE).

    Recurses into ``.text_config`` / ``.llm_config`` so VLMs whose nested text submodel is the MoE
    (Kimi-K2.7, Llama-4 multimodal, NemotronH-Omni, ...) report the LM's expert count.
    """
    for cfg in (
        hf_config,
        getattr(hf_config, "text_config", None),
        getattr(hf_config, "llm_config", None),
    ):
        if cfg is None:
            continue
        for attr in ("num_experts", "num_local_experts", "n_routed_experts", "num_routed_experts"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
    return 0


def is_expert_parallel_capable(hf_config) -> bool:
    """True if the model can be expert-parallel sharded.

    Transformers exposes ``base_model_ep_plan`` on MoE configs whose experts support
    expert parallelism (e.g. gpt_oss, llama4; qwen3_moe in transformers>=5.7). Dense models
    and MoE configs without the plan return False -> use plain FSDP2. For VLMs the plan
    lives on ``text_config`` (the nested LM submodel), so check both.
    """
    return (
        bool(getattr(_ep_config(hf_config), "base_model_ep_plan", None))
        and num_experts_from_config(hf_config) > 0
    )


def choose_ep_dp_degrees(world_size: int, num_experts: int) -> tuple[int, int]:
    """Pick ``(expert_parallel, expert_data_parallel)`` degrees with ``EP * EDP == world_size``.

    EP must divide both ``num_experts`` (whole experts per rank) and ``world_size`` (a valid mesh
    factorization); we take the largest such EP, and each expert shard is then replicated over the
    remaining ``EDP = world_size // EP`` ranks. Falls back to ``(1, world_size)`` when the model is not
    MoE or no valid EP degree exists. Heuristic -- callers may override.
    """
    if num_experts <= 0 or world_size <= 1:
        return 1, world_size
    for ep in range(world_size, 0, -1):
        if world_size % ep == 0 and num_experts % ep == 0:
            return ep, world_size // ep
    return 1, world_size


def _estimate_dense_param_bytes(hf_config) -> "int | None":
    """Best-effort estimate of the DENSE (non-expert) parameter bytes from an HF config.

    embeddings + lm_head + per-layer attention/norms (+ the dense FFN when the model is NOT MoE, since a MoE's FFN is
    the expert weights). Returns ``None`` when the config lacks the fields to estimate (the caller then treats the dense
    part as 'fits'). A deliberately CONSERVATIVE upper bound (ignores GQA and tied embeddings) -- it exists to catch a
    dense part that clearly cannot be replicated, not to be exact.
    """
    cfg = getattr(hf_config, "text_config", None) or hf_config
    h = getattr(cfg, "hidden_size", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if not h or not n_layers:
        return None
    vocab = getattr(cfg, "vocab_size", None) or getattr(hf_config, "vocab_size", None) or 0
    inter = getattr(cfg, "intermediate_size", None) or 4 * h
    attn = 4 * h * h  # q,k,v,o (upper bound; ignores GQA head reduction)
    norms = 4 * h  # ~2 layernorms per layer
    ffn = (
        0 if is_expert_parallel_capable(hf_config) else 3 * h * inter
    )  # gated FFN; a MoE's FFN is expert
    per_layer = attn + norms + ffn
    embed = 2 * vocab * h  # embed + lm_head (untied upper bound)
    bytes_per = 2  # bf16/fp16 default (the pre-quant load dtype)
    return (embed + n_layers * per_layer) * bytes_per


def _dense_fits_per_gpu(
    hf_config, device_mem_bytes: "int | None" = None, headroom: float = 0.6
) -> bool:
    """Fit predicate for the dense knob.

    does the DENSE footprint fit ONE GPU (leaving ``headroom`` for activations + comm buffers + any EP-local experts)?
    CONSERVATIVE -- returns ``True`` ('fits' -> replicate / data-parallel) whenever it cannot estimate the dense size or
    the device memory, so the data-parallel default is preserved. This is the pluggable decision point of the auto-
    resolver; a heuristic or a recipe hint may substitute for it without changing callers.
    """
    dense_bytes = _estimate_dense_param_bytes(hf_config)
    if not dense_bytes:
        return True
    if device_mem_bytes is None:
        try:
            import torch

            device_mem_bytes = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
        except Exception:
            return True
    return dense_bytes < headroom * device_mem_bytes


def _resolve_ep_size(hf_config, world_size: int, ep_size) -> int:
    """Resolve the expert-parallel degree.

    An explicit ``ep_size`` wins; ``auto`` -> the largest EP degree dividing both the expert count and
    ``world_size`` for an EP-capable model, else ``1``.
    """
    if ep_size not in (None, "auto", "0", 0):
        return int(ep_size)
    if world_size > 1 and is_expert_parallel_capable(hf_config):
        ep, _ = choose_ep_dp_degrees(world_size, num_experts_from_config(hf_config))
        return ep
    return 1


def apply_distributed_support(
    model_type: str,
    hf_config,
    *,
    ckpt_path: "str | None" = None,
    trust_remote_code: bool = False,
):
    """Prepare ``hf_config`` so ``model_type`` can be loaded expert-parallel, from its model spec.

    Reads the model's :class:`~modelopt.torch.models.EPSpec` and applies it: enforce the
    transformers floor this model's expert parallelism needs, then attach the
    ``base_model_ep_plan`` it requires to whichever config carries its MoE -- the root for a plain
    MoE, the nested text/LLM config for a VLM whose experts live in its language sub-model, since
    the plan's globs are written against that module tree and a plan on the wrong config matches
    nothing at all rather than failing.

    An existing plan always wins: stock transformers ships one for some models, and overwriting it
    would substitute our idea of the layout for the model's own.

    Returns the config to load, unchanged for a model type with no spec or no EP section. The
    generic all-to-all transport is not wired here -- it self-registers in
    :func:`load_and_prepare_ep`, so nothing about it is per-model.
    """
    from modelopt.torch.models import get_spec

    spec = get_spec(model_type)
    ep = getattr(spec, "ep_spec", None) if spec is not None else None
    if ep is None:
        return hf_config

    if ep.requires_transformers:
        require_transformers(ep.requires_transformers, f"{model_type} expert-parallel PTQ")

    if ep.base_model_ep_plan:
        target = hf_config
        if ep.plan_config_attr:
            target = getattr(hf_config, ep.plan_config_attr, None)
            if target is None:
                raise ValueError(
                    f"{model_type} expects its expert-parallel plan on "
                    f"config.{ep.plan_config_attr}, which this checkpoint's config does not have."
                )
        if not getattr(target, "base_model_ep_plan", None):
            target.base_model_ep_plan = dict(ep.base_model_ep_plan)

    if ep.prepare is not None:
        hf_config = ep.prepare(hf_config, ckpt_path=ckpt_path, trust_remote_code=trust_remote_code)
    return hf_config


def decide_distribution_plan(
    hf_config,
    world_size: int,
    *,
    ep_size="auto",
    fsdp_size="auto",
    device_mem_bytes: "int | None" = None,
):
    """Decide how a HuggingFace MoE model is distributed for PTQ: a :class:`DistributionPlan`.

    Two knobs, each an explicit value or ``"auto"`` (explicit always wins):

    * ``ep_size`` -- ``auto`` -> the largest EP degree dividing both the expert count and
      ``world_size`` (``1`` for a model that cannot be expert-parallel).
    * ``fsdp_size`` -- ``auto`` -> ``1`` (replicate the dense part) when it fits one GPU, else
      ``world_size`` (FSDP2-shard it); ``"world"`` -> ``world_size``.

    On a distributed run the plan is decided ONCE, on rank 0, and broadcast; every other rank adopts rank 0's
    decision instead of making its own. The inputs are all read locally -- the env knobs, this GPU's memory,
    the config on this rank's filesystem -- so ranks deciding independently could disagree, and a
    disagreement surfaces as a hang at mesh creation rather than as an error. Collective: every rank must
    call it. If rank 0 fails to decide, every rank raises.
    """
    from .distributed import is_distributed_run

    decide_kwargs = {
        "ep_size": ep_size,
        "fsdp_size": fsdp_size,
        "device_mem_bytes": device_mem_bytes,
    }
    if not is_distributed_run():
        return _decide_distribution_plan_on_rank0(hf_config, world_size, **decide_kwargs)

    plan, error = None, None
    if dist.get_rank() == 0:
        try:
            plan = _decide_distribution_plan_on_rank0(hf_config, world_size, **decide_kwargs)
        except Exception as e:
            error = e
    # Ship the failure as text, not the exception object: the other ranks only need to stop, and an
    # exception is not guaranteed to pickle.
    decision = [(plan, None if error is None else f"{type(error).__name__}: {error}")]
    dist.broadcast_object_list(decision, src=0)
    if error is not None:
        raise error
    plan, rank0_error = decision[0]
    if rank0_error is not None:
        raise RuntimeError(f"rank 0 failed to decide the distribution plan: {rank0_error}")
    return plan


def _decide_distribution_plan_on_rank0(
    hf_config, world_size: int, *, ep_size, fsdp_size, device_mem_bytes: "int | None"
):
    """The decision itself (see :func:`decide_distribution_plan`); run on rank 0 alone when distributed."""
    from .distributed import DistributionPlan

    ep = _resolve_ep_size(hf_config, world_size, ep_size)
    if fsdp_size in (None, "auto"):
        fsdp = 1 if _dense_fits_per_gpu(hf_config, device_mem_bytes) else world_size
    elif str(fsdp_size).lower() in ("world", "full"):
        fsdp = world_size
    else:
        fsdp = int(fsdp_size)
    fsdp = max(1, min(fsdp, world_size))
    return DistributionPlan(world_size=world_size, ep_size=ep, fsdp_size=fsdp)


# ---------------------------------------------------------------------------------------------------
# DP=EP all-to-all expert transport (`torch_a2a`)
#
# The throughput-correct MoE parallelism for calibration: dense+router run data-parallel (each rank its
# own batch), experts are EP-sharded, and tokens are routed to the expert-owning rank by an all-to-all
# (dispatch) / reduce (combine). The expert matmul stays on transformers' `_grouped_linear`, so
# ModelOpt's per-expert quantizers attach and collect amax over the *global* (dispatched) token pool.
# Transport is plain `torch.distributed.all_to_all_single` (NCCL) -- works cross-node at any rank count.
# ---------------------------------------------------------------------------------------------------


# all_to_all_single addresses its flattened buffers with int32 element offsets, so a hop larger than
# this overflows them; _a2a_moe_torch chunks such transfers along the hidden dim.
_A2A_MAX_HOP_ELEMENTS = 2**31 - 1


def _a2a_moe_torch(
    group,
    hidden_states,
    topk_idx,
    topk_weights,
    num_global_experts,
    num_local_experts,
    expert_fn,
):
    """All-to-all MoE expert stage via plain `torch.distributed.all_to_all_single` (NCCL).

    Works cross-node at any rank count, with explicit token permute/unpermute.

    Flow: expand (token, expert) pairs -> route each to dest_rank = global_expert // num_local ->
    all_to_all the tokens (+ local-expert id + weight) -> local grouped_mm (quantizable) -> weight ->
    all_to_all back -> scatter-sum to original tokens.
    """
    world = group.size()
    x = hidden_states.contiguous().bfloat16()
    H = x.size(-1)  # noqa: N806
    T, top_k = topk_idx.shape  # noqa: N806
    dev = x.device

    def _one_all_to_all(output, input_, out_split, in_split):
        """One all_to_all_single hop (hidden, expert-output, AND routing weight).

        Uses the AUTOGRAD-AWARE ``torch.distributed.nn.functional.all_to_all_single`` (whose backward is
        the reverse all-to-all) when a graph is being built AND the input needs grad -- so gradients
        propagate through dispatch/combine to the routed experts (AutoQuantize gradient score estimation)
        AND, via the routing-weight hop, back to the gate/router (MoE training through the experts).
        Otherwise (forward-only calibration / inference, or a frozen input) it uses the plain, no-graph
        collective, leaving the validated production path byte-for-byte unchanged.
        """
        if torch.is_grad_enabled() and input_.requires_grad:
            # torch.distributed.nn.functional.all_to_all_single is the autograd-aware collective (backward =
            # reverse all-to-all). It is deprecated in torch>=2.13 in favor of
            # torch.distributed._functional_collectives.all_to_all_single (AsyncCollectiveTensor); migrating
            # is a follow-up -- suppress the per-call FutureWarning here to keep score-estimation logs clean.
            import warnings

            from torch.distributed.nn.functional import all_to_all_single as _ata_autograd

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, message=".*all_to_all_single.*"
                )
                return _ata_autograd(output, input_, out_split, in_split, group=group)
        dist.all_to_all_single(output, input_, out_split, in_split, group=group)
        return output

    def _all_to_all(output, input_, out_split, in_split):
        """All-to-all that chunks along the hidden dim to stay inside the int32 element limit.

        all_to_all_single, transparently CHUNKED along the hidden dim when a single transfer would
        exceed the int32 element limit. ``all_to_all_single`` addresses the flattened ``[rows, H]`` buffer
        with element offsets ``= row_offset * H``; once ``rows*H > 2**31`` that offset overflows int32 in
        the NCCL path and the collective HANGS (observed: qwen3_5_moe_text EP calib at batch8 x seq4096 ->
        Q*H = 327680*8192 = 2.68e9 > 2**31, all_to_all ran the full 3h NCCL timeout, then aborted). Split
        the LAST (hidden) dim into column slices so every hop stays < 2**31 elements; the per-rank ROW split
        sizes are identical for every slice, and cat-ing the slices is autograd-safe.
        """
        int32_max = _A2A_MAX_HOP_ELEMENTS
        if input_.dim() >= 2:
            # n_chunks (the number of collective HOPS we issue) MUST be identical on every rank, but the
            # recv size Q*H is per-rank data-dependent (routing imbalance -> different Q). Deciding n_chunks
            # from the LOCAL numel desyncs the hop count and re-deadlocks. All-reduce the MAX transfer size
            # so every rank agrees on the same split. Unconditional over the group (every rank runs this same
            # branch for a 2-D hop) so the all_reduce itself can never desync; it's a cheap scalar.
            gmax_t = torch.tensor(
                [max(output.numel(), input_.numel())], device=output.device, dtype=torch.int64
            )
            dist.all_reduce(gmax_t, op=dist.ReduceOp.MAX, group=group)
            gmax = int(gmax_t.item())
            if gmax > int32_max:
                hdim = input_.shape[-1]
                n_chunks = (gmax + int32_max - 1) // int32_max
                step = (hdim + n_chunks - 1) // n_chunks
                if torch.is_grad_enabled() and input_.requires_grad:
                    # Autograd (AutoQuantize scoring): cat is graph-safe (an in-place copy_ would break the
                    # reverse-all-to-all backward). Higher peak, but grad scoring uses far fewer tokens.
                    slices = [
                        _one_all_to_all(
                            output[..., c0 : c0 + step].contiguous(),
                            input_[..., c0 : c0 + step].contiguous(),
                            out_split,
                            in_split,
                        )
                        for c0 in range(0, hdim, step)
                    ]
                    return torch.cat(slices, dim=-1)
                # Forward-only calibration: write each hidden-column slice back into `output` IN PLACE so we
                # never hold a full second [Q,H] copy (torch.cat would). The imbalanced-rank recv buffer is
                # already tens of GiB at long seq x batch; the cat doubling is what tips calibration into OOM.
                for c0 in range(0, hdim, step):
                    c1 = min(c0 + step, hdim)
                    tmp = _one_all_to_all(
                        output[..., c0:c1].contiguous(),
                        input_[..., c0:c1].contiguous(),
                        out_split,
                        in_split,
                    )
                    output[..., c0:c1].copy_(tmp)
                return output
        return _one_all_to_all(output, input_, out_split, in_split)

    # (token, expert) pairs (drop any -1 "no selection" slots).
    token_rows = torch.arange(T, device=dev).repeat_interleave(top_k)  # [P]
    exp = topk_idx.reshape(-1).to(torch.int64)
    w = topk_weights.reshape(-1).to(x.dtype)
    valid = exp >= 0
    token_rows, exp, w = token_rows[valid], exp[valid], w[valid]
    dest = (exp // num_local_experts).to(torch.int64)  # owner rank of each pair
    local_eid = exp % num_local_experts

    order = torch.argsort(dest, stable=True)  # group pairs by dest rank
    token_rows_s, local_eid_s, w_s = token_rows[order], local_eid[order], w[order]
    send_hidden = x[token_rows_s].contiguous()  # [P, H]
    in_counts = torch.bincount(dest[order], minlength=world).to(torch.int64)  # rows -> each rank

    out_counts = torch.empty_like(in_counts)  # rows <- each rank
    # The host reads out_counts (.tolist()) immediately below to size the dispatch buffers (Q = sum(out)).
    # all_to_all_single runs on the NCCL stream; a HOST read of its output must wait for the collective, or
    # .tolist() can copy stale/uninitialized out_counts -> a garbage Q -> a size-mismatched dispatch
    # all_to_all -> NCCL out-of-bounds -> SIGSEGV (deterministic on the qwen3_5_moe_text EP path; masked on
    # heavier models only by pre-MoE timing, and masked everywhere by CUDA_LAUNCH_BLOCKING=1). Take the
    # work handle and wait() so the current stream (hence the .tolist() DtoH) orders after the exchange.
    _cnt_work = dist.all_to_all_single(out_counts, in_counts, group=group, async_op=True)
    _cnt_work.wait()
    in_list, out_list = in_counts.tolist(), out_counts.tolist()
    Q = int(sum(out_list))  # noqa: N806

    # Q (tokens RECEIVED) is what sizes every buffer below, and it is data-dependent: a balanced
    # router gives Q ~= P, but a collapsed one piles the whole cluster's tokens onto a few experts
    # and Q grows without bound. Report it HERE rather than only in the end-of-function trace --
    # the expert stage allocates O(Q*H) immediately after, so on a bad routing draw the function
    # OOMs before that trace can run and the one number that explains the failure is never printed.
    P_sent = int(send_hidden.size(0))  # noqa: N806
    if 4 * max(P_sent, 1) < Q:
        warnings.warn(
            f"[a2a] rank {dist.get_rank()} received Q={Q} tokens for {num_local_experts} local "
            f"expert(s) after sending P={P_sent} -- a {Q / max(P_sent, 1):.1f}x imbalance. The "
            f"expert stage allocates O(Q x H), so this is a likely OOM. Routing has probably "
            f"collapsed onto few experts (check the router/hidden states for NaN or blow-up).",
            stacklevel=2,
        )
    # dispatch hidden (autograd-aware when scoring) + local-expert id + weight (same split sizes)
    recv_hidden = torch.empty(Q, H, device=dev, dtype=x.dtype)
    recv_hidden = _all_to_all(recv_hidden, send_hidden, out_list, in_list)
    recv_eid = torch.empty(Q, device=dev, dtype=torch.int64)
    dist.all_to_all_single(recv_eid, local_eid_s.contiguous(), out_list, in_list, group=group)
    # Routing WEIGHT hop is gradient-carrying (its backward reaches the gate/router when training the MoE
    # through the experts), so it goes through the autograd-aware ``_all_to_all`` -- unlike the integer
    # expert-id hop above, which is a non-differentiable routing DECISION. Falls back to the plain
    # collective when ``w_s`` doesn't need grad (frozen router / forward-only), so PTQ is unchanged.
    recv_w = torch.empty(Q, device=dev, dtype=x.dtype)
    recv_w = _all_to_all(recv_w, w_s.contiguous(), out_list, in_list)

    # group received tokens by local expert -> grouped_mm (quantizable) -> weight
    rorder = torch.argsort(recv_eid, stable=True)
    counts = torch.bincount(recv_eid[rorder], minlength=num_local_experts)
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    eo = expert_fn(recv_hidden[rorder], offsets).to(x.dtype) * recv_w[rorder].unsqueeze(-1)
    eo_recv = torch.empty(Q, H, device=dev, dtype=x.dtype)
    eo_recv[rorder] = eo  # back to recv order

    # combine: all_to_all back (swap counts, autograd-aware when scoring) -> scatter-sum to original tokens
    send_back = torch.empty(send_hidden.size(0), H, device=dev, dtype=x.dtype)
    send_back = _all_to_all(send_back, eo_recv, in_list, out_list)
    out_pair = torch.empty_like(send_back)
    out_pair[order] = send_back  # back to pair order
    # Accumulate the per-token top-k scatter-sum in FP32, then round ONCE. Each token's output is a sum
    # of its top_k expert results; accumulating that in bf16 rounds at every partial add (~0.4% each), so
    # a different summation ORDER gives a different answer -- and the a2a returns partials in
    # post-all_to_all order, which differs from the native (non-EP) forward's local order. That made the
    # exported activation scales EP-dependent: on Qwen3-30B-A3B (top_k=8) 27% of input_scale values
    # differed from the single-GPU reference, by up to 28%. FP32 accumulation is order-independent to
    # ~1e-7, so every order rounds to the same bf16 value -> byte-identical scales at EP1/2/4/8
    # (validated single- and multi-node). Costs one T x H fp32 temporary; no measurable runtime change.
    _acc = torch.zeros(T, H, device=dev, dtype=torch.float32)
    _acc.index_add_(0, token_rows, out_pair.float())
    result = _acc.to(x.dtype)
    return result


def a2a_moe(
    group,
    hidden_states,
    topk_idx,
    topk_weights,
    num_global_experts,
    num_local_experts,
    expert_fn,
):
    """Route one MoE layer's expert stage by all-to-all (NCCL)."""
    return _a2a_moe_torch(
        group,
        hidden_states,
        topk_idx,
        topk_weights,
        num_global_experts,
        num_local_experts,
        expert_fn,
    )


def torch_a2a_experts_forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
    """All-to-all expert-parallel forward for PTQ calibration (DP=EP).

    Registered as the `"torch_a2a"` experts implementation on transformers' `ExpertsInterface`.

    Dense+router run data-parallel; experts are EP-sharded; tokens are routed to the expert-owning
    rank by the NCCL all-to-all above. Local experts run through the ModelOpt-quantizable
    `_grouped_linear` so per-expert weight + shared input quantizers attach over the dispatched
    (global) token pool. `process_group` (the EP group) is injected by the `torch_a2a_experts` TP style's
    `_prepare_input_fn`; the router must emit GLOBAL ids unmasked (`torch_a2a_router`).
    """
    from transformers.integrations.moe import _grouped_linear

    if process_group is None:
        raise ValueError(
            "torch_a2a experts forward requires the EP `process_group` (use the `torch_a2a_experts` "
            "TP style, which injects it via _prepare_input_fn)."
        )
    num_local = self.num_experts
    num_global = num_local * process_group.size()
    # Per-expert biases (e.g. GPT-OSS ``gate_up_proj_bias`` / ``down_proj_bias``, shape [num_local, out],
    # EP-sharded dim-0 by grouped_gemm like the weights). transformers' ``_grouped_linear`` (and ModelOpt's
    # quantized override) already accept ``bias`` and do ``out += bias`` -- transposed-safe. Bias is added
    # AFTER each expert's grouped GEMM, unquantized.
    has_bias = getattr(self, "has_bias", False)

    def expert_fn(sorted_tok, offsets):
        # ``sorted_tok`` rows are grouped by local expert with cumulative ``offsets``; the per-ROW local
        # expert id is arange(num_local) repeated by each expert's token count (= diff of offsets), so gather
        # the per-expert bias to a per-row [S, out] bias for _grouped_linear's out += bias.
        row_eid = None
        if has_bias:
            counts = torch.diff(offsets, prepend=offsets.new_zeros(1)).to(torch.long)
            row_eid = torch.repeat_interleave(
                torch.arange(num_local, device=offsets.device), counts
            )
        w1 = self.gate_up_proj if self.has_gate else self.up_proj
        b1 = (
            (self.gate_up_proj_bias if self.has_gate else self.up_proj_bias)[row_eid]
            if has_bias
            else None
        )
        proj = _grouped_linear(sorted_tok, w1, offsets, bias=b1, is_transposed=self.is_transposed)
        proj = self._apply_gate(proj) if self.has_gate else self.act_fn(proj)
        b2 = self.down_proj_bias[row_eid] if has_bias else None
        return _grouped_linear(
            proj, self.down_proj, offsets, bias=b2, is_transposed=self.is_transposed
        )

    return a2a_moe(
        process_group, hidden_states, top_k_index, top_k_weights, num_global, num_local, expert_fn
    )


_TORCH_A2A_REGISTERED = False


def register_torch_a2a():
    """Register everything the torch_a2a path needs into stock transformers (idempotent).

    Done lazily from ``load_and_prepare_ep`` -- NOT at import time, so importing this module has no
    transformers side effect:

    - the ``torch_a2a`` experts FORWARD into ``ExpertsInterface`` (``ALL_EXPERTS_FUNCTIONS``);
    - two EP parallel STYLES into ``ALL_PARALLEL_STYLES``:

      * ``torch_a2a_router`` -- emit GLOBAL unmasked expert ids (skip ``RouterParallel``'s
        per-EP-rank masking) so the a2a dispatch can route each (token, expert) to its owning rank;
      * ``torch_a2a_experts`` -- inject the EP ``process_group`` into the experts forward (4th arg)
        and skip the all-reduce (the a2a combine does the reduction).

    These two styles are functionally the stock ``megamoe_router`` / ``megamoe_experts`` (which were
    added for DeepGEMM Mega-MoE in transformers 5.10.4), reimplemented here -- ~5 lines each -- so the
    a2a path OWNS its EP styles and does NOT depend on megamoe (i.e. it stays portable below 5.10.4).
    The base classes (``RouterParallel``, ``TensorParallelLayer``) are stock wherever the EP infra
    exists (transformers >= 5.6, the framework floor enforced in ``load_and_prepare_ep``), so this
    works on any such version; ``grouped_gemm`` (expert weight sharding) is stock too.
    """
    global _TORCH_A2A_REGISTERED
    if _TORCH_A2A_REGISTERED:
        return
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        RouterParallel,
        TensorParallelLayer,
    )

    ALL_EXPERTS_FUNCTIONS.register("torch_a2a", torch_a2a_experts_forward)

    class TorchA2ARouter(RouterParallel):
        """torch_a2a router style: emit GLOBAL unmasked expert ids (== stock ``megamoe_router``).

        The base ``RouterParallel._prepare_output_fn`` remaps to local ids + masks non-local; the a2a
        dispatch needs the global ids, so this just passes the router outputs through unchanged.
        """

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return outputs

        # transformers >=5.16 renamed the hook; the _prepare_* form above is never called there.
        def transform_output_post_forward(self, module, output, mesh):
            return output

    class TorchA2AExperts(TensorParallelLayer):
        """torch_a2a experts style: inject the EP ``process_group`` (4th arg) + skip the all-reduce.

        (== stock ``megamoe_experts``.) Expert weights shard via packed_colwise/rowwise +
        grouped_gemm, so ``shard_tensor`` is a no-op here. ``_prepare_input_fn``'s 4th return is what
        ``torch_a2a_experts_forward`` receives as ``process_group``.
        """

        def _prepare_input_fn(self, mod, inputs, device_mesh):
            hidden_states, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
            return hidden_states, top_k_index, top_k_weights, device_mesh.get_group()

        def _prepare_output_fn(self, mod, outputs, device_mesh):
            return outputs

        # --- transformers >=5.16 API ---------------------------------------------------------
        # install_forward() there calls transform_inputs_pre_forward/transform_output_post_forward
        # and NEVER the _prepare_* pair above, whose base implementations are no-ops. Without these
        # two methods the EP process_group is silently not injected and torch_a2a_experts_forward
        # raises "requires the EP process_group". Additive: <5.16 ignores them.
        def should_use_local_tensors(self, module):
            # The a2a forward dispatches tokens itself and runs the LOCAL expert shard through
            # _grouped_linear; without this the grouped matmul mixes DTensor params with plain
            # activation tensors and aten._grouped_mm rejects the call.
            return True

        def transform_inputs_pre_forward(self, module, args, kwargs, mesh):
            if len(args) < 3:
                return args, kwargs
            hidden_states, top_k_index, top_k_weights = args[0], args[1], args[2]
            return (hidden_states, top_k_index, top_k_weights, mesh.get_group()), kwargs

        def transform_output_post_forward(self, module, output, mesh):
            return output

        def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
            return param[...].to(device=device, dtype=dtype)

    ALL_PARALLEL_STYLES.register("torch_a2a_router", TorchA2ARouter())
    ALL_PARALLEL_STYLES.register("torch_a2a_experts", TorchA2AExperts())
    _TORCH_A2A_REGISTERED = True


def _make_expert_param_matcher(ep_plan: dict[str, Any]):
    """Build ``is_expert(param_name) -> bool`` from the model's ``base_model_ep_plan``.

    the plan that actually drives EP sharding -- instead of a fragile ``"expert" in name`` substring heuristic.

    A param is EP-sharded (only its ``1/ep_degree`` shard resides on this rank) iff its name, with layer
    indices wildcarded exactly as HuggingFace does (``replace_layer_number_by_wildcard``) and any trailing
    ``.weight`` / ``.bias`` stripped, matches an expert plan key as a SUFFIX (tolerating the ``model.`` /
    ``language_model.model.`` prefixes the base-relative plan keys omit) OR is NESTED under one. This
    covers both per-parameter shard keys (``...experts.gate_up_proj: grouped_gemm``) and experts-container
    keys (``...experts: torch_a2a_experts`` / ``nemotron_experts_ep``). Router entries
    (``ep_router`` / ``torch_a2a_router``) are REPLICATED, not sharded, so they're excluded. If the plan is
    empty the matcher is always ``False`` (-> no EP credit -> stock warmup sizing).
    """
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    _router_styles = {"ep_router", "torch_a2a_router"}
    keys = [k for k, v in (ep_plan or {}).items() if v not in _router_styles]

    def is_expert(pname: str) -> bool:
        generic = replace_layer_number_by_wildcard(pname)
        for suffix in (".weight", ".bias"):
            if generic.endswith(suffix):
                generic = generic[: -len(suffix)]
                break
        padded = "." + generic + "."
        return any(
            generic == k or generic.endswith("." + k) or ("." + k + ".") in padded for k in keys
        )

    return is_expert


@contextlib.contextmanager
def _ep_aware_caching_allocator_warmup(
    ep_degree: int, ep_plan: dict[str, Any], dense_on_meta: bool = False
):
    """Make transformers' ``caching_allocator_warmup`` EP-aware for the duration of ``from_pretrained``.

    transformers sizes the load-time allocator warmup via ``get_total_byte_count``, which credits the
    ``// world_size`` sharding division only to params matched in ``model._tp_plan``. Under the EP load
    (``device_mesh`` + ``enable_expert_parallel``; sharding driven by ``base_model_ep_plan``) ``_tp_plan``
    is EMPTY, so every param -- experts are ~95% of a large MoE -- is counted at FULL global size. The
    warmup then pre-allocates one ``torch.empty`` clamped to ``device_total - 1.2 GiB`` (~the whole GPU)
    on every rank; torch keeps that as reserved cache, which later starves the NCCL a2a comm buffers.

    This temporarily replaces ``get_total_byte_count`` with a version that additionally credits EP-sharded
    expert params their per-rank shard (``// ep_degree``), identifying them from ``ep_plan`` (the model's
    ``base_model_ep_plan``), sizing the warmup to the real per-rank footprint (e.g. ~91 GiB instead of
    ~275 GiB for Nemotron-3-Ultra-550B EP32) -- keeping the warmup's load-speed benefit without the
    ~device-max transient.

    Always on (no disable knob): it only *sizes* an allocator pre-reservation, never model correctness; it
    falls back to stock sizing on any incompatibility (``try/except``); and a mis-estimate only nudges load
    speed (an over-credit under-sizes -> the allocator just grows during load; an under-credit sizes up to
    the same device-max clamp as stock) -- so it can never OOM worse than, or load differently from, stock.
    No-op when ``ep_degree <= 1``.
    """
    if ep_degree <= 1:
        yield
        return
    try:
        import transformers.modeling_utils as _mu
    except Exception:
        yield
        return
    orig = getattr(_mu, "get_total_byte_count", None)
    if orig is None:
        yield
        return

    is_expert = _make_expert_param_matcher(ep_plan)

    def _ep_aware(model, accelerator_device_map, hf_quantizer=None):
        try:
            from collections import defaultdict

            try:
                from transformers.integrations.tensor_parallel import _get_parameter_tp_plan
            except Exception:
                _get_parameter_tp_plan = None
            ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            tp_plan = (getattr(model, "_tp_plan", None) or {}) if ws > 1 else {}
            tied = set(getattr(model, "all_tied_weights_keys", {}) or {})
            total: dict = defaultdict(int)
            for pname, dev in accelerator_device_map.items():
                if pname in tied:
                    continue
                try:
                    p = model.get_parameter_or_buffer(pname)
                except Exception:
                    continue
                dsize = (
                    hf_quantizer.param_element_size(model, pname, p)
                    if hf_quantizer is not None
                    else p.element_size()
                )
                nb = p.numel() * dsize
                if is_expert(pname):
                    nb //= ep_degree  # EP-sharded across the ep submesh: only the local shard resides here
                elif dense_on_meta and pname.endswith((".weight", ".bias")):
                    # dense_strategy="sharded": _ep_dense_meta_load forces every dense .weight/.bias to
                    # materialize on META during from_pretrained (they are FSDP2-sharded + DCP-loaded
                    # afterwards), so they occupy NO device memory in the window this warmup sizes.
                    # Counting them at full global size is what pushed Kimi-K3 (EP64, dense=fsdp) to a
                    # 185.8 GiB estimate on a 184.31 GiB card -- over the device, so the warmup fell back
                    # to the device-max clamp it exists to avoid. Persistent buffers are NOT forced to
                    # meta by that context, so they keep their full credit here.
                    continue
                elif (
                    tp_plan
                    and _get_parameter_tp_plan is not None
                    and _get_parameter_tp_plan(pname, tp_plan, is_weight=True) is not None
                ):
                    nb //= ws  # preserve stock TP sharding credit for any tp-planned params
                total[dev] += nb
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
                gib = 1024**3
                sizes = ", ".join(f"{d}: {b / gib:.1f} GiB" for d, b in total.items())
                print(
                    f"[MEM] EP-aware warmup sizing (experts // ep_degree={ep_degree}"
                    f"{', dense on meta -> uncounted' if dense_on_meta else ''}) -> {sizes} "
                    f"(was full-model ~device-max; EP-aware plan-driven sizing)",
                    flush=True,
                )
            return total
        except Exception:
            return orig(model, accelerator_device_map, hf_quantizer)

    try:
        _mu.get_total_byte_count = _ep_aware
        yield
    finally:
        _mu.get_total_byte_count = orig


def _routed_expert_modules(model: nn.Module, ep_plan: dict[str, Any]) -> dict[str, nn.Module]:
    """Find the EP-sharded routed-expert containers from ``base_model_ep_plan``; return ``{name: module}``.

    Uses the SAME plan that drove the EP sharding to decide expert-ness (not a per-class flag or FQN
    heuristic): a module is a routed-expert container iff its name -- with layer indices wildcarded exactly
    as HuggingFace does (``replace_layer_number_by_wildcard``) -- matches, as a SUFFIX (tolerating the
    ``model.`` / ``language_model.model.`` prefixes the base-relative plan keys omit), a plan key whose
    (post-swap) style is ``torch_a2a_experts``. ``grouped_gemm`` (per-parameter shard) and
    ``torch_a2a_router`` (replicated router) entries are excluded -- only the grouped-GEMM experts
    container holds the expert quantizers.
    """
    from transformers.integrations.tensor_parallel import replace_layer_number_by_wildcard

    # a2a tags the experts container ``torch_a2a_experts``.
    expert_keys = [k for k, v in ep_plan.items() if v == "torch_a2a_experts"]
    experts = {
        name: module
        for name, module in model.named_modules()
        if any(
            (generic := replace_layer_number_by_wildcard(name)) == k or generic.endswith("." + k)
            for k in expert_keys
        )
    }

    # Quantize the grouped experts through the patched ``_grouped_linear`` and keep the fused grouped-GEMM
    # weights + the ``torch_a2a`` experts forward. If ModelOpt has a per-expert-nn.Linear quant wrapper
    # registered for the experts class (Qwen3VLMoeTextExperts, Llama4TextExperts, DbrxExperts),
    # ``mtq.quantize`` would convert the module to that wrapper -- whose forward drops the a2a-injected
    # process_group ("forward() takes 4 positional args but 5 given") and discards the grouped structure.
    # Unregister those classes so the experts stay a2a-managed (matches deepseek/glm/minimax, which have NO
    # registered wrapper).
    try:
        from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry

        for cls in {type(m) for m in experts.values()}:
            if cls in QuantModuleRegistry:
                QuantModuleRegistry.unregister(cls)
    except Exception as _e:  # pragma: no cover - defensive; never block the load on this
        print(f"[ep] WARNING: could not unregister experts quant wrapper: {_e}", flush=True)
    return experts


def _seed_parallel_states(model: nn.Module, expert_names, dense_state, expert_state) -> None:
    """Seed every module's ``_parallel_state`` for calibration's amax sync.

    The routed-expert containers named in ``expert_names`` and everything under them get
    ``expert_state``; every other module gets ``dense_state``. Seeding the plain modules works because
    quantization keeps a ``_parallel_state`` that is already present when it converts a module -- the same
    way Megatron's expert MLPs seed theirs.
    """
    prefixes = tuple(name + "." for name in expert_names)
    for name, module in model.named_modules():
        module._parallel_state = (
            expert_state if prefixes and (name + ".").startswith(prefixes) else dense_state
        )


_FINEGRAINED_FP8_PATCHED = False
_FP8_DEQUANT_DTYPE = None  # model compute dtype the dequant must emit (set by the call site)


def _patch_finegrained_fp8_dequant(target_dtype=None):
    """Correct transformers' block-FP8 dequant (``Fp8Dequantize._dequantize_one``) for DeepSeek-V3.

    transformers 5.9's ``_dequantize_one`` infers the block geometry by *dividing* the weight shape
    by the scale-grid shape and requires exact divisibility::

        block_m = rows // scale_rows  # assumes rows == scale_rows * block_m exactly

    But DeepSeek-V3 block-FP8 uses a fixed 128x128 block with a *ceil* scale grid, so a weight whose
    dimension isn't a multiple of 128 has a partial last block: e.g. ``kv_a_proj_with_mqa`` is
    (576, 7168) with a (5, 56) grid -> ``576 % 5 != 0`` raises ``ValueError``, and degenerate/empty
    scale grids from the merged/sharded expert path hit ``rows % 0`` -> ``ZeroDivisionError``.

    Replace it with a divisibility-free tile-and-crop: broadcast each per-block scale over its
    ``block`` rows/cols (block size from the real ``weight_block_size`` when the grid is inexact,
    else derived), then crop to the weight shape -- correct for both full and partial last blocks,
    2-D dense linears and (leading-dim-broadcast) per-expert tensors. Idempotent.

    transformers >= 5.10.1 ONLY: this is skipped. Upstream fixed the two show-stoppers there -- the
    loader's expert ``tensor_idx`` check became ``any(op is MergeModulelist ...)`` (bug 1) and
    ``_dequantize_one`` gained an ``output_dtype`` the loader passes as the destination dtype (bug 3) --
    and it added native FP4-unpack + ue8m0 (E8M0) scale decoding that our simpler override does NOT do
    (installing it on >= 5.10.1 would BREAK ue8m0 checkpoints like DeepSeek-V4). The only thing still
    unpatched upstream is the partial-block divisibility (bug 2), which no shipping model on >= 5.10.1
    hits here (DeepSeek-V4's dims are all 128-multiples); revisit if one does.
    """
    global _FINEGRAINED_FP8_PATCHED, _FP8_DEQUANT_DTYPE
    if target_dtype is not None:
        _FP8_DEQUANT_DTYPE = target_dtype  # keep current even if already patched
    if _FINEGRAINED_FP8_PATCHED:
        return
    import transformers
    from packaging.version import parse

    if parse(transformers.__version__) >= parse("5.10.1"):
        _FINEGRAINED_FP8_PATCHED = True  # upstream handles it (and natively supports ue8m0/fp4)
        return
    import torch
    from transformers.integrations import finegrained_fp8 as _fp8

    def _dequantize_one(self, quantized, scales):
        # FP4 path (int8 / float4_e2m1fn_x2 nibble-packed) unchanged: unpack to fp32 first.
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (
            fp4_dtype is not None and quantized.dtype == fp4_dtype
        ):
            q = self._unpack_fp4(quantized)
        else:
            q = quantized.to(torch.float32)
        rows, cols = q.shape[-2:]
        scale_rows, scale_cols = scales.shape[-2:]
        # Emit the MODEL compute dtype (bf16), NOT the scale dtype. DeepSeek's weight_scale_inv is F32;
        # transformers' original out_dtype logic would return F32 weights, which then run against bf16
        # activations -> "mat1 and mat2 have the same dtype" RuntimeError in the calibration matmuls.
        out_dtype = _FP8_DEQUANT_DTYPE or torch.bfloat16
        # Degenerate/empty scale grid (sharded/merged expert artifact): emit unscaled rather than crash.
        if scale_rows == 0 or scale_cols == 0 or scales.numel() == 0:
            return q.to(out_dtype)
        wbs = getattr(
            getattr(self.hf_quantizer, "quantization_config", None), "weight_block_size", None
        ) or (128, 128)

        def _blk(dim, sgrid, cfg_blk):
            # A ceil scale grid at the config block size (DeepSeek's uniform 128x128, incl. partial last
            # block e.g. 576 -> 5 blocks) is the common case -- trust the config block. Only a grid that
            # is NOT the ceil-at-config-block (a coarser/per-tensor scale) derives the block by division;
            # never infer from exact divisibility alone (576 % 5 and 9 % 3 both mislead).
            if sgrid == -(-dim // cfg_blk):
                return cfg_blk
            return dim // sgrid if sgrid and dim % sgrid == 0 else cfg_blk

        block_m = _blk(rows, scale_rows, wbs[0])
        block_n = _blk(cols, scale_cols, wbs[1])
        s = (
            scales.to(torch.float32)
            .repeat_interleave(block_m, dim=-2)
            .repeat_interleave(block_n, dim=-1)
        )
        s = s[..., :rows, :cols]
        return (q * s).to(out_dtype)

    _fp8.Fp8Dequantize._dequantize_one = _dequantize_one

    # --- Fix the routed-expert block-FP8 load corruption (transformers 5.9) ---------------------
    # The FP8 quantizer's update_weight_conversions PREPENDS Fp8Dequantize to each converter's
    # operations, so for the fused-MoE experts converter operations[0] becomes Fp8Dequantize instead
    # of the original MergeModulelist. The state-dict loader only assigns a per-expert tensor_idx when
    # ``isinstance(mapping.operations[0], MergeModulelist)`` (core_model_loading.py) -- with the prepend
    # that check fails, tensor_idx becomes None, and GroupedGemmParallel.shard_tensor then row-slices
    # each per-expert (2048,7168) weight to param[rank*local:(rank+1)*local] = (16,7168) and slices the
    # scales full/empty per rank. Result: gate_up_proj loads as (256,32,7168), a grouped-GEMM contraction
    # mismatch at calibration. Fix: FUSE the prepended Fp8Dequantize into the expert MergeModulelist (a
    # MergeModulelist subclass that dequantizes each per-expert (weight,scale) pair, dropping scales, then
    # stacks) so operations[0] IS a MergeModulelist again -- the loader assigns tensor_idx (whole-expert
    # selection + EP shard) and Concatenate still runs after scales are dropped. Idempotent (flag-guarded).
    try:
        from transformers.core_model_loading import MergeModulelist as _MergeModulelist
        from transformers.core_model_loading import WeightConverter as _WeightConverter
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer as _FineGrainedFP8HfQuantizer,
        )

        # Fuse the prepended Fp8Dequantize INTO the expert MergeModulelist: dequantize each per-expert
        # (weight, scale) pair -- dropping scales -- then stack, exactly as the intended
        # [Fp8Dequantize, MergeModulelist, Concatenate] chain does, but packaged as a single op that
        # *is a* MergeModulelist. That keeps operations[0] a MergeModulelist so the loader assigns each
        # per-expert its tensor_idx (whole-expert selection + EP shard) AND Concatenate still runs after
        # dequant has dropped the scales (Concatenate cats every remaining source pattern, so it must not
        # see scale keys). Net execution is identical to upstream's intent; only the op layout changes.
        class _DequantThenMerge(_MergeModulelist):
            def __init__(self, deq, dim=0):
                super().__init__(dim)
                self._deq = deq

            def convert(self, input_dict, source_patterns=None, target_patterns=None, **kwargs):
                deq_out = self._deq.convert(
                    input_dict,
                    source_patterns=source_patterns,
                    target_patterns=target_patterns,
                    **kwargs,
                )
                return super().convert(
                    deq_out,
                    source_patterns=source_patterns,
                    target_patterns=target_patterns,
                    **kwargs,
                )

        if not getattr(_FineGrainedFP8HfQuantizer, "_modelopt_uwc_patched", False):
            _orig_uwc = _FineGrainedFP8HfQuantizer.update_weight_conversions

            def _uwc_fixed(self, weight_conversions):
                convs = _orig_uwc(self, weight_conversions)
                for conv in convs:
                    ops = getattr(conv, "operations", None)
                    # Match the expert converter update_weight_conversions produced:
                    # [Fp8Dequantize, MergeModulelist, <Concatenate...>]. Fuse ops[0]+ops[1].
                    if (
                        isinstance(conv, _WeightConverter)
                        and ops
                        and len(ops) >= 2
                        and isinstance(ops[0], _fp8.Fp8Dequantize)
                        and isinstance(ops[1], _MergeModulelist)
                        and not isinstance(ops[1], _DequantThenMerge)
                    ):
                        deq, mml = ops[0], ops[1]
                        ops[:2] = [_DequantThenMerge(deq, dim=mml.dim)]
                return convs

            _FineGrainedFP8HfQuantizer.update_weight_conversions = _uwc_fixed
            _FineGrainedFP8HfQuantizer._modelopt_uwc_patched = True
    except Exception as _e:
        print(f"[ep] WARNING: could not patch update_weight_conversions: {_e}", flush=True)

    _FINEGRAINED_FP8_PATCHED = True


def _fsdp2_shard_dense(model: nn.Module, fsdp_mesh, expert_modules) -> int:
    """FSDP2-shard the model's dense modules, leaving the routed experts to EP.

    FSDP2 ``fully_shard`` the model's DENSE (non-expert) modules over ``fsdp_mesh``, leaving the routed
    experts (already EP-sharded on ``ep`` + the a2a transport) untouched -- the "FSDP2 dense + EP MoE"
    hybrid. In the EDP=1 layout ``fsdp_mesh`` is the FULL world (dense shards over every rank, the same
    ranks the experts are EP-sharded over); a 2D ``(dp, fsdp)`` ``fsdp_mesh`` gives the EDP>1 hybrid
    (dense HSDP-sharded over ``fsdp`` and replicated over ``dp``, aligned with the experts' ``(edp, ep)``).

    Wraps at the DECODER-LAYER boundary (exactly like the proven pure-FSDP2 path,
    :func:`~modelopt.torch.utils.distributed._fsdp2_wrap_modules` via HF ``_no_split_modules``) and passes
    the routed-expert params as ``ignored_params`` so FSDP shards the layer's dense params (attention,
    layernorms, router) while leaving the experts EP-sharded on the ``ep`` mesh + a2a transport. Wrapping
    the whole layer (not ``self_attn`` or individual leaves) is essential: ``fully_shard`` swaps a wrapped
    module's class (``Foo`` -> ``FSDPFoo``), and modelopt inspects classes by name/source -- wrapping
    ``self_attn`` breaks ``mtq``'s KV-cache AST registration (``inspect.getsource`` on the synthetic
    class), and wrapping the leaf ``QuantLinear``s breaks the export's name-based ``is_quantlinear``.
    Wrapping only the LAYER leaves ``self_attn`` and the q/k/v/o ``QuantLinear``s PLAIN, while
    ``_get_enclosing_fsdp_module`` still finds the layer FSDP wrapper for the weight-access/writeback
    contexts. The root (embed/lm_head/final norm) is intentionally NOT wrapped: those stay replicated
    (and unswizzled, so their quantization-exclude still applies), matching the dense-replicated baseline.

    Returns the number of layers wrapped.
    """
    from torch.distributed.fsdp import fully_shard

    from .distributed import _fsdp2_wrap_modules

    # Every routed-expert param is EP-sharded on the ``ep`` mesh; FSDP must IGNORE them so it never tries
    # to reshard them onto ``fsdp_mesh``. Ignored params not present in a given layer are simply no-ops.
    ignored_params: set = set()
    for em in expert_modules:
        ignored_params.update(em.parameters(recurse=True))

    from torch.distributed.tensor import DTensor, Replicate

    ignored_param_ids = {id(p) for p in ignored_params}  # id-based: Tensor.__eq__ is elementwise

    # Wrap only the OUTERMOST _no_split_modules. Some models (e.g. Kimi's MoonViT VLM) list BOTH a
    # container (the vision model) and its blocks in ``_no_split_modules``; ``fully_shard`` on the parent
    # already shards the children, so wrapping a child afterward hits ``DeviceMesh._concatenate`` ("Cannot
    # concatenate overlapping meshes"). Drop any wrap module nested under another.
    wrap_modules = _fsdp2_wrap_modules(model)
    wrap_ids = {id(m) for m in wrap_modules}
    nested_ids: set = set()
    for m in wrap_modules:
        for c in m.modules():
            if id(c) != id(m) and id(c) in wrap_ids:
                nested_ids.add(id(c))
    wrap_modules = [m for m in wrap_modules if id(m) not in nested_ids]

    n = 0
    for layer in wrap_modules:
        layer_ignored = set(ignored_params)  # experts + any pre-sharded dense (below)
        has_shardable = False
        for name, p in list(layer.named_parameters(recurse=True)):
            if id(p) in ignored_param_ids:
                continue
            if isinstance(p.data, DTensor):
                if all(isinstance(pl, Replicate) for pl in p.data.placements):
                    # Replicate -> plain (a free view, no gather), re-registered as a plain Parameter:
                    # assigning .data would keep the DTensor parameter, which fully_shard then tries
                    # to place on both its own mesh and the FSDP one.
                    owner_name, _, leaf = name.rpartition(".")
                    owner = layer.get_submodule(owner_name) if owner_name else layer
                    setattr(
                        owner,
                        leaf,
                        nn.Parameter(p.data.to_local(), requires_grad=p.requires_grad),
                    )
                else:
                    layer_ignored.add(
                        p
                    )  # already sharded by the loader (TP) -> leave it distributed
                    continue
            has_shardable = True
        if not has_shardable:
            continue  # nothing left to FSDP-shard in this layer (all expert / already-sharded)
        fully_shard(
            layer,
            mesh=fsdp_mesh,
            reshard_after_forward=True,
            ignored_params=layer_ignored,
        )
        n += 1
    return n


def _ckpt_quant_method(cfg):
    """Return the checkpoint's native quantization method, or ``None``.

    Return the checkpoint's native quantization method (e.g. ``"fp8"`` for block-fp8 DeepSeek-V3),
    or ``None`` if the checkpoint isn't natively quantized. Checks the top config and the VLM ``text_config``.
    """
    for c in (cfg, getattr(cfg, "text_config", None)):
        qc = getattr(c, "quantization_config", None)
        m = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if m:
            # transformers' config objects carry a QuantizationMethod enum, whose str() is not its value.
            return str(getattr(m, "value", m))
    return None


def _transformers_mesh_builder():
    """The module whose ``initialize_distributed_mesh`` ``from_pretrained`` calls, or ``None``.

    Newer transformers drops ``from_pretrained``'s ``device_mesh`` once ``distributed_config`` is
    given: it builds its own world-sized ``tp`` mesh from the config's sizes. That shards the experts
    over every rank instead of this rank's EP group whenever the EP degree is below the world size.
    Older releases honor ``device_mesh`` and have no such builder.
    """
    try:
        import transformers.distributed.mixin as mixin
    except ImportError:
        return None
    return mixin if hasattr(mixin, "initialize_distributed_mesh") else None


@contextlib.contextmanager
def _experts_sharded_over(ep_mesh):
    """Make ``from_pretrained`` shard the experts over ``ep_mesh`` on releases that build their own mesh."""
    mixin = _transformers_mesh_builder()
    if mixin is None:
        yield
        return

    def _ep_mesh(distributed_config):
        index = torch.cuda.current_device() if ep_mesh.device_type == "cuda" else None
        return torch.device(ep_mesh.device_type, index), ep_mesh

    original = mixin.initialize_distributed_mesh
    mixin.initialize_distributed_mesh = _ep_mesh
    try:
        yield
    finally:
        mixin.initialize_distributed_mesh = original


# The two ways the DENSE (non-expert) modules can be placed/loaded in the EP hybrid. Experts are ALWAYS
# EP-sharded by from_pretrained; only the dense modules vary:
_DENSE_STRATEGIES = ("replicated", "sharded")


def _resolve_dense_strategy(dense_strategy, fsdp_dense, hf_config, mo_session):
    """Resolve the ``dense_strategy`` governing how the EP hybrid places and loads dense modules.

    The strategies:

    * ``"replicated"``     -- dense stays replicated per rank (DP dense; from_pretrained default).
    * ``"sharded"``        -- dense left on ``meta`` during from_pretrained, then FSDP2-sharded + DCP-loaded
                              so it is never replicated anywhere (the FSDP2-dense path). Implemented via
                              meta-init + Distributed Checkpoint shard-load, so peak memory is one shard
                              per rank -- never the full replicated transient of a naive load-then-shard.

    Resolution order: an explicit ``dense_strategy`` wins; otherwise derive from the plan's
    ``fsdp_dense`` (``plan.is_fsdp_dense``). Block-fp8 checkpoints can't be DCP-dequantized (F8_E8M0),
    so ``sharded`` falls back to ``replicated`` for them.
    """
    if dense_strategy is None:
        dense_strategy = "sharded" if fsdp_dense else "replicated"
    if dense_strategy not in _DENSE_STRATEGIES:
        raise ValueError(
            f"dense_strategy must be one of {_DENSE_STRATEGIES}; got {dense_strategy!r}"
        )
    # block-fp8 dense can't be DCP-dequantized (F8_E8M0) -> fall back to replicated dense (from_pretrained's
    # FineGrainedFP8Config(dequantize=True) path materializes it in bf16 per rank).
    if dense_strategy == "sharded" and _ckpt_quant_method(hf_config) == "fp8":
        dense_strategy = "replicated"
        if mo_session.rank == 0:
            print(
                "[ep] block-fp8 checkpoint: sharded dense unavailable (DCP can't dequant F8_E8M0) "
                "-> dense_strategy=replicated",
                flush=True,
            )
    # "sharded" FSDP2-shards dense over the ``ep`` group and replicates over ``edp``: at EDP=1 that is the
    # full world (1D), at EDP>1 a 2D (edp, ep) HSDP layout aligned with the experts' sparse_moe_mesh. Both
    # are handled by ``load_and_prepare_ep`` (which picks the mesh) + the shared meta+DCP loader (which fills
    # [Replicate(edp), Shard(ep)] dense DTensors, exactly like the pure-FSDP2 HSDP-dense path).
    return dense_strategy


@contextlib.contextmanager
def _ep_dense_meta_load(is_expert):
    """Fully-distributed dense load for the EP hybrid.

    during ``from_pretrained``, force every non-expert weight/bias to materialize on ``meta`` (i.e. NOT materialize on
    the GPU) by making transformers' per-param ``get_device`` return ``"meta"`` for dense keys.

    transformers 5.x's ``convert_and_load_state_dict_in_model`` places EXPERT params via the tp /
    ``distributed_operation`` branch (which fuses + EP-shards them and never consults ``get_device``), and
    every OTHER (dense) param via ``param_device = get_device(device_map, key)`` -> ``spawn_materialize``.
    Returning ``"meta"`` there leaves dense params unmaterialized while experts still EP-shard normally. The
    skipped dense params are then FSDP2-sharded + DCP-loaded (:func:`_load_ep_dense_distributed`), so dense is
    never replicated per rank. Only ``.weight``/``.bias`` are forced meta -- persistent buffers (small) load
    normally. Patches ``get_device`` on both ``transformers.modeling_utils`` and the module that actually
    defines the loader (it's looked up as a module global there).
    """
    import sys as _sys

    import transformers.modeling_utils as _mu

    loader = getattr(_mu, "convert_and_load_state_dict_in_model", None)
    mods = {_mu}
    loader_module = getattr(loader, "__module__", None)
    dm = _sys.modules.get(loader_module) if loader_module else None
    if dm is not None:
        mods.add(dm)
    targets = [m for m in mods if hasattr(m, "get_device")]
    if not targets:
        yield
        return

    def _dense_weight(key) -> bool:
        k = str(key)
        return k.endswith((".weight", ".bias")) and not is_expert(k)

    saved = [(m, m.get_device) for m in targets]

    def _make(orig):
        def patched(device_map, key, *args, **kwargs):
            if _dense_weight(key):
                return "meta"
            return orig(device_map, key, *args, **kwargs)

        return patched

    try:
        for m, orig in saved:
            m.get_device = _make(orig)
        yield
    finally:
        for m, orig in saved:
            m.get_device = orig


def _load_ep_dense_distributed(model, fsdp_mesh, local_dir, is_expert, mo_session, expert_modules):
    """Fully-distributed dense load for the EP hybrid.

    FSDP2-shard the (meta) dense modules and DCP-load each rank's shard directly from the checkpoint -- dense is NEVER
    materialized replicated. Experts are already EP-sharded (loaded by ``from_pretrained``) and left untouched. Returns
    the number of dense layers sharded, like :func:`_fsdp2_shard_dense`.

    Requires the dense ``.weight``/``.bias`` params to be on ``meta`` on entry (see
    :func:`_ep_dense_meta_load`). Delegates the materialize + DCP load to the SHARED
    :func:`~modelopt.torch.utils.distributed._meta_dcp_load` (``load_names`` restricted to the dense params),
    the same loader that backs the DP and FSDP2-full paths -- here just fed a partially-meta model (dense
    meta, experts already real) and a dense-only name set.
    """
    from .distributed import _meta_dcp_load

    device = torch.device(f"cuda:{mo_session.local_rank}")
    # FSDP2-shard the dense modules. Their params are meta plain tensors -> fully_shard makes them meta Shard
    # DTensors; experts are passed as ignored_params and stay EP-sharded.
    n_dense = _fsdp2_shard_dense(model, fsdp_mesh, expert_modules)
    # Restrict the shared loader to the dense params (experts already materialized by from_pretrained must not
    # be touched). It selectively materializes only the meta dense modules (experts skipped), preserves
    # buffers across to_empty, and DCP-loads each rank's dense shard.
    dense_names = {
        n
        for n, _ in model.named_parameters()
        if n.endswith((".weight", ".bias")) and not is_expert(n)
    }
    if mo_session.rank == 0:
        print(
            f"[ep] dense-distributed load: FSDP2-sharded {n_dense} dense module(s); "
            f"meta+DCP-loading {len(dense_names)} dense param(s) (experts left EP-sharded)",
            flush=True,
        )
    _meta_dcp_load(model, local_dir, device, load_names=dense_names)
    return n_dense


def load_and_prepare_ep(
    ckpt_path: str,
    mo_session: "ModelOptSession",
    ep_degree: int,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    freeze: bool = True,
    hf_config=None,
    fsdp_dense: bool = False,
    dense_strategy: "str | None" = None,
    require_lm_head: bool = True,
) -> nn.Module:
    """Load + shard a MoE causal LM for expert-parallel PTQ via the DP=EP all-to-all transport.

    Builds a ``(dp, ep)`` device mesh and loads experts EP-sharded over the ``ep`` submesh
    (``enable_expert_parallel=True`` -- each rank owns ``num_experts / ep`` experts as grouped-GEMM
    shards), while ATTENTION and dense params stay REPLICATED (so EP has no KV-head-divisibility cap).
    The model's ``base_model_ep_plan`` is swapped to the ``torch_a2a`` styles: the router emits GLOBAL
    unmasked expert ids and the experts forward routes each token to its expert-owning rank by a NCCL
    all-to-all (dispatch) -> quantizable ``_grouped_linear`` -> all-to-all (combine), rather than every
    rank computing all tokens followed by an EP all-reduce. Every rank is data-parallel (sees DIFFERENT
    tokens), so calibration MUST shard the data per-rank.

    The ``dp`` dimension (present when ``ep_degree < world_size``) is additional data parallelism across
    EP groups: non-expert params stay REPLICATED per rank and each EP group calibrates on different
    data; experts stay EP-sharded (resident). No FSDP and no concurrent ``mesh_dp``/``mesh_ep``
    collectives -> no deadlock by construction. Costs more per-rank memory (non-experts + experts
    replicated across dp) but is the reliable path at scale.

    With ``fsdp_dense`` (or ``dense_strategy="sharded"``) the dense part is instead FSDP2-sharded over
    the ``ep`` group and each rank loads only its shard from the checkpoint, never the whole of it --
    for a model whose dense part does not fit one GPU.

    EP CANNOT be applied to a meta-init model (transformers' ``GroupedGemmParallel.shard_tensor``
    slices materialized tensors), so this loads real weights via ``from_pretrained`` over the ``ep``
    submesh. Requires the transformers MoE EP framework (ExpertsInterface / grouped-GEMM EP styles /
    enable_expert_parallel), first present in transformers >= 5.6. (Per-model plugins enforce a higher
    floor where their stock support landed later, e.g. qwen3_moe's stock base_model_ep_plan in >= 5.7.)

    Assumes the caller has already initialized (and device-bound) the process group -- distributed
    bring-up is the launcher's job, not this loader's (hf_ptq.main() does it via modelopt_session).
    """
    # Hard floor: the transformers MoE expert-parallel framework. Below 5.6 there is no ExpertsInterface
    # / grouped-GEMM EP styles, and enable_expert_parallel silently shards nothing (no error) -> fail loud.
    require_transformers(
        "5.6", "ModelOpt DP=EP expert-parallel PTQ (transformers MoE EP framework)"
    )
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    from .distributed import ParallelState

    if mo_session.world_size % ep_degree != 0:
        raise ValueError(
            f"world_size {mo_session.world_size} not divisible by ep_degree {ep_degree}"
        )
    dp = mo_session.world_size // ep_degree
    if hf_config is None:
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if attn_implementation is not None:
            config_kwargs["attn_implementation"] = attn_implementation
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    dtype = getattr(hf_config, "torch_dtype", None) or torch.bfloat16

    # This loader assumes the caller has ALREADY initialized (and device-bound) the process group --
    # distributed bring-up belongs to the launcher, not this library function. The example launcher does
    # it in hf_ptq.main() via modelopt_session() -> modelopt.torch.utils.distributed.setup() before
    # the load. A live, device-bound PG is required here so the mesh subgroups built below create their
    # NCCL comms eagerly (the synchronized warmup all-reduce per subgroup further down) instead of lazily
    # on first collective (which deadlocks at DP>1 because ranks reach the lazy init in different phases).
    if not mo_session.initialized:
        raise RuntimeError(
            "load_and_prepare_ep requires an initialized ModelOptSession (with a live, device-bound "
            "process group); enter modelopt.torch.utils.distributed.modelopt_session() (or pass an "
            "initialized session) before calling it."
        )
    # Two INDEPENDENT meshes (NeMo-AutoModel style), named by the model part they govern -- two
    # factorizations of the same world:
    #   sparse_moe_mesh = (edp, ep): experts are SHARDED over the ``ep`` axis and REPLICATED over ``edp``;
    #       the ``ep`` submesh also drives from_pretrained's expert sharding + the a2a transport.
    #   dense_mesh = (dp,): dense/non-expert params are data-parallel-REPLICATED over the WHOLE world.
    # This subsumes the old a2a-vs-partial split: a2a is simply dp == 1 (``edp`` size 1), and dense amax
    # always reduces over the whole world (fixing the DP>1 case, where the old dp submesh missed the ep dim).
    sparse_moe_mesh = init_device_mesh("cuda", (dp, ep_degree), mesh_dim_names=("edp", "ep"))
    dense_mesh = init_device_mesh("cuda", (mo_session.world_size,), mesh_dim_names=("dp",))
    ep_submesh = sparse_moe_mesh["ep"]

    # NO TENSOR PARALLELISM IS REQUESTED HERE, despite the name. transformers implements expert
    # parallelism THROUGH the TP application path: with enable_expert_parallel=True the model's
    # ``tp_plan`` PROPERTY returns ``_ep_plan`` (distributed/mixin.py), so what actually gets applied
    # is the expert plan. ``tp_plan="auto"`` is only the SWITCH that makes ``tp_size > 1`` in
    # DistributedConfig.__post_init__ and therefore reaches ``apply_tensor_parallelism`` at all --
    # drop it and __post_init__ early-returns with tp_size=None, which then TypeErrors on
    # ``tp_size > 1`` in prepare_distribute_model. Dense params stay REPLICATED (= our DP) because
    # ``swapped_plan`` below keeps only expert entries; attention/embeds match no plan key.
    #
    # WHERE it goes is version-dependent: transformers >=5.16 raises if tp_plan is passed alongside
    # distributed_config ("Set tensor-parallel options on DistributedConfig when using it"); older
    # releases have no tp_plan field on DistributedConfig at all. Passing tp_plan ALONE is not an
    # option either -- the >=5.16 back-compat path rebuilds the config as
    # DistributedConfig(tp_plan=..., tp_size=...) and DROPS enable_expert_parallel, silently losing
    # EP. So decide from the dataclass fields, which tell the two apart on any release.
    import dataclasses as _dc

    _dc_has_tp_plan = any(f.name == "tp_plan" for f in _dc.fields(DistributedConfig))
    if _transformers_mesh_builder() is not None:
        # This release sizes its own mesh from tp_size; _experts_sharded_over swaps in the EP mesh.
        distributed_config = DistributedConfig(
            enable_expert_parallel=True, tp_plan="auto", tp_size=ep_degree
        )
    elif _dc_has_tp_plan:
        distributed_config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto")
    else:
        distributed_config = DistributedConfig(enable_expert_parallel=True)
    from_pretrained_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_mesh": ep_submesh,
        "distributed_config": distributed_config,
        "trust_remote_code": trust_remote_code,
    }
    if not _dc_has_tp_plan:
        from_pretrained_kwargs["tp_plan"] = "auto"
    if attn_implementation is not None:
        from_pretrained_kwargs["attn_implementation"] = attn_implementation
    # Native block-FP8 checkpoints (e.g. DeepSeek-V3: quantization_config.quant_method="fp8", e4m3 with
    # weight_block_size) must be DEQUANTIZED to bf16 on load: ModelOpt PTQ inserts/exports quantizers on
    # plain bf16 Linears, and the EP grouped_gemm experts expect bf16 fused params -- not fp8 blocks with
    # weight_scale_inv. Mirror get_model's mxfp4 dequant-on-load, using FineGrainedFP8Config(dequantize=True).
    # (compressed-tensors pack-quantized experts, e.g. Kimi, are handled separately by the on-the-fly
    # decompressor plugin; here we only cover the native fp8 case.)
    _ckpt_qm = _ckpt_quant_method(hf_config)
    if _ckpt_qm == "fp8":
        from transformers import FineGrainedFP8Config

        from_pretrained_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        _patch_finegrained_fp8_dequant(
            dtype
        )  # correct transformers' block-fp8 dequant; emit model dtype
        if mo_session.rank == 0:
            print(
                "[ep] native block-fp8 checkpoint -> FineGrainedFP8Config(dequantize=True) -> bf16",
                flush=True,
            )
    elif _ckpt_qm == "mxfp4":
        # Native mxfp4 checkpoints (e.g. gpt-oss: fused `gate_up_proj_blocks`/`_scales` + biases) must be
        # DEQUANTIZED to bf16 on load so ModelOpt can insert nvfp4/fp8 quantizers on the fused bf16 experts
        # (the a2a grouped_gemm expects bf16). transformers dequantizes blocks->bf16 during from_pretrained.
        # NB: the single-process loader needed device_map="sequential" to dodge a CUDA illegal-access in the
        # mxfp4 dequant under multi-GPU device_map="auto"; the EP path uses device_mesh (no device_map), so
        # watch for an EP-shard-awareness issue in the dequant convert-op (analogous to the fp8 patch above).
        from transformers import Mxfp4Config

        from_pretrained_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        if mo_session.rank == 0:
            print(
                "[ep] native mxfp4 checkpoint (e.g. gpt-oss) -> Mxfp4Config(dequantize=True) -> bf16",
                flush=True,
            )
    ep_cfg = _ep_config(hf_config)  # for VLMs the plan lives on .text_config
    plan = dict(getattr(ep_cfg, "base_model_ep_plan", None) or {})
    if not plan:
        raise ValueError("expert-parallel PTQ requires the model to declare base_model_ep_plan")
    # DP=EP all-to-all: swap the ep_plan to our torch_a2a styles (torch_a2a_router emits GLOBAL ids
    # unmasked; torch_a2a_experts injects the EP process_group + skips the all-reduce) and select the
    # `torch_a2a` experts forward (NCCL all-to-all dispatch/combine around the quantizable
    # _grouped_linear). Tokens are routed to the expert-owning rank by all-to-all instead of every rank
    # computing all tokens + all-reduce. Requires per-rank (data-parallel) calib data. nemotron_h's
    # logits-only gate has no ep_router entry, so its nemotron_experts_ep also maps to torch_a2a_experts.
    register_torch_a2a()  # idempotent; register the experts forward + the two styles, right before use
    swap = {
        "ep_router": "torch_a2a_router",
        "moe_tp_experts": "torch_a2a_experts",
        "nemotron_experts_ep": "torch_a2a_experts",
    }
    # Keep expert-related entries (grouped_gemm expert params + the router/experts styles); DROP the TP
    # styles (a2a builds no TP group for them -> would shard attention across the wrong mesh).
    swapped_plan = {k: swap.get(v, v) for k, v in plan.items() if v in swap or v == "grouped_gemm"}
    # Ensure the routed-experts CONTAINER is tagged torch_a2a_experts (injects the EP process_group +
    # marks the routed expert). Some plans tag only the expert PARAMS grouped_gemm, not the container;
    # derive it as the parent of the grouped_gemm expert-param keys and add it if missing.
    if not any(v == "torch_a2a_experts" for v in swapped_plan.values()):
        for parent in {k.rsplit(".", 1)[0] for k, v in plan.items() if v == "grouped_gemm"}:
            swapped_plan[parent] = "torch_a2a_experts"
    from_pretrained_kwargs["experts_implementation"] = "torch_a2a"
    ep_cfg.base_model_ep_plan = swapped_plan
    from_pretrained_kwargs["config"] = hf_config
    if mo_session.rank == 0:
        print(f"[ep] ep_plan={swapped_plan} experts_implementation=torch_a2a", flush=True)
    # Pick the loader by model type, matching examples' get_model: VLM (nested text_config / a
    # *ForConditionalGeneration arch) -> AutoModel so the WHOLE checkpoint loads (vision tower + projector
    # + nested LM, which carries the lm_head); a plain causal LM -> AutoModelForCausalLM so the model
    # INCLUDES the lm_head. NB: for a plain causal LM, AutoModel returns the BASE model (e.g.
    # Qwen3MoeModel / NemotronHModel) WITHOUT lm_head -> the head is never calibrated/exported and the
    # exported checkpoint generates garbage (untied random head). The per-model plugin (models/) has
    # already installed any on-the-fly weight decompression as a transformers monkeypatch.
    _archs = getattr(hf_config, "architectures", None) or []
    _is_vlm = (
        getattr(hf_config, "text_config", None) is not None
        or getattr(hf_config, "vision_config", None) is not None
        or any("conditionalgeneration" in str(a).lower() for a in _archs)
    )
    loader = AutoModel if _is_vlm else AutoModelForCausalLM
    # A VLM's AutoModel returns the BACKBONE (vision + projector + nested LM) WITHOUT the lm_head. We now
    # DEFAULT require_lm_head=True (for ALL models, VLMs included) so the loaded model carries the lm_head
    # and .generate(), and the pre/post-PTQ generation preview works everywhere. For M3,
    # AutoModel -> MiniMaxM3VLModel (no lm_head; decoder at .language_model); the with-head load returns the
    # full image-text-to-text model instead (*ForConditionalGeneration): [model, lm_head], decoder at
    # model.language_model -- both resolvable. The LM submodel is extracted for quantize
    # (extract_and_prepare_language_model_from_vl); the head/vision are excluded from quant but kept for gen.
    if require_lm_head and _is_vlm:
        # A remote-code VLM may register the FULL *ForConditionalGeneration (WITH lm_head) under
        # AutoModel/AutoModelForCausalLM in its auto_map and provide NO AutoModelForImageTextToText entry
        # (e.g. Kimi-K3: auto_map AutoModelForCausalLM -> KimiK3ForConditionalGeneration). For those,
        # AutoModelForImageTextToText.from_pretrained raises "Unrecognized configuration class ...", so use
        # the remote AutoModelForCausalLM -- it already yields the with-head model + a .language_model
        # decoder. Native VLMs (M3: MiniMaxM3VLConfig; qwen3_5_moe: Qwen3_5MoeConfig) have no auto_map ->
        # the AutoModelForImageTextToText path below applies.
        _auto_map = getattr(hf_config, "auto_map", None) or {}
        _causal_full = (
            "conditionalgeneration" in str(_auto_map.get("AutoModelForCausalLM", "")).lower()
        )
        if _auto_map and _causal_full and "AutoModelForImageTextToText" not in _auto_map:
            loader = AutoModelForCausalLM
        else:
            try:
                from transformers import AutoModelForImageTextToText

                # Only commit to the with-head image-text-to-text class if it actually recognizes this
                # config; otherwise KEEP the headless AutoModel backbone rather than crashing at
                # from_pretrained with "Unrecognized configuration class". PTQ still runs; the generation
                # preview is skipped downstream (pre_quantize guards on hasattr(model, "generate")).
                if type(hf_config) in AutoModelForImageTextToText._model_mapping:
                    loader = AutoModelForImageTextToText
                elif mo_session.rank == 0:
                    print(
                        f"[ep] require_lm_head: {type(hf_config).__name__} is not registered under "
                        f"AutoModelForImageTextToText; keeping headless {loader.__name__} "
                        f"(no lm_head -> generation preview will be skipped).",
                        flush=True,
                    )
            except ImportError:
                pass
    if mo_session.rank == 0:
        print(
            f"[ep] loader={loader.__name__} (is_vlm={_is_vlm}, "
            f"require_lm_head={require_lm_head}, archs={_archs})",
            flush=True,
        )
    # Size the load-time caching-allocator warmup to the real per-rank (EP-sharded) footprint instead of
    # the full model -- otherwise transformers pre-allocates ~device-max on every rank and torch holds it
    # as reserved cache, starving the NCCL a2a buffers. Expert params are identified from `swapped_plan`
    # (the actual EP plan). Always on: falls back to stock sizing on any issue and a mis-size only nudges
    # load speed, never correctness. (See _ep_aware_caching_allocator_warmup.)
    is_expert_fn = _make_expert_param_matcher(swapped_plan)
    # ONE place decides how the DENSE (non-expert) modules are placed/loaded in the EP hybrid -- experts are
    # always EP-sharded by from_pretrained. Two strategies:
    #   "replicated"     -- dense stays replicated per rank (DP dense; from_pretrained default).
    #   "sharded"    -- dense left on meta during from_pretrained (get_device->meta via
    #                   _ep_dense_meta_load), then FSDP2-sharded + DCP-loaded (never replicated).
    # ``dense_strategy`` (explicit) wins; else derive from ``fsdp_dense`` (the plan).
    dense_strategy = _resolve_dense_strategy(dense_strategy, fsdp_dense, hf_config, mo_session)
    _dense_ctx = (
        _ep_dense_meta_load(is_expert_fn)
        if dense_strategy == "sharded"
        else contextlib.nullcontext()
    )
    with (
        _ep_aware_caching_allocator_warmup(
            ep_degree, swapped_plan, dense_on_meta=dense_strategy == "sharded"
        ),
        _dense_ctx,
        _experts_sharded_over(ep_submesh),
    ):
        model = loader.from_pretrained(ckpt_path, **from_pretrained_kwargs)
    model.eval()
    if dense_strategy == "sharded" and mo_session.rank == 0:
        _nmeta = [n for n, p in model.named_parameters() if p.is_meta]
        _dense_meta = [n for n in _nmeta if not is_expert_fn(n)]
        print(
            f"[ep] dense-meta load: {len(_nmeta)} meta param(s) after from_pretrained "
            f"({len(_dense_meta)} dense, {len(_nmeta) - len(_dense_meta)} expert); experts materialized. "
            f"e.g. dense-meta {_dense_meta[:3]}",
            flush=True,
        )
    # Force the experts forward to dispatch to `torch_a2a`: from_pretrained's `experts_implementation`
    # kwarg resolves non-fp8 impls back to "grouped_mm", but @use_experts_implementation reads
    # `config._experts_implementation` at FORWARD time, so set it on every (sub)config now. (Without
    # this the torch_a2a_experts style passes a 5th process_group arg to grouped_mm_experts_forward ->
    # TypeError.)
    for _m in [model, *model.modules()]:
        _cfg = getattr(_m, "config", None)
        if _cfg is not None and hasattr(_cfg, "_experts_implementation"):
            _cfg._experts_implementation = "torch_a2a"

    # DP=EP all-to-all: no FSDP. Non-experts stay replicated per rank (from_pretrained already placed
    # them on this rank's GPU); experts stay EP-sharded on the ep submesh. The model carries
    # model._device_mesh = ep_submesh (sparse_moe_mesh["ep"]), which the export uses to detect EP and the
    # per-expert distributed writer reads for the ep_rank offset + dp-replica dedup.
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if freeze:
        model.requires_grad_(False)

    # The routed-expert containers, from the SAME base_model_ep_plan that drove the sharding. A MoE model
    # loaded WITHOUT EP (dense DP) never reaches this loader, so its experts stay dense.
    expert_modules = _routed_expert_modules(model, swapped_plan)

    # "FSDP2 dense + EP MoE" hybrid: experts are EP-sharded over ``ep`` and the DENSE (non-expert) modules
    # are FSDP2-sharded over the SAME ``ep`` group, replicated over ``edp``. At EDP=1 that is the full world
    # (dense shards over every rank); at EDP>1 it is the 2D (edp, ep) HSDP layout -- dense shards over ``ep``
    # (== ``mp``) and replicates over ``edp`` (== ``dp``), byte-for-byte aligned with the experts'
    # ``sparse_moe_mesh``. Dense all-gather then expert a2a run sequentially within a forward (same
    # ``ep`` group), so there is no concurrent-collective deadlock. Weight amax is computed on the
    # FSDP-gathered full weight (identical per rank); activation amax syncs over the whole world via the
    # dense ``dp`` group registered below.
    #
    # "sharded" shards dense (see above); "replicated" leaves dense data-parallel and needs no post-load step.
    if dense_strategy == "sharded":
        # Fully-distributed dense: dense was left on meta during from_pretrained; FSDP2-shard it and
        # DCP-load each rank's shard directly (never replicated anywhere). EDP=1 -> the 1D full-world dense
        # mesh (the original, validated layout); EDP>1 -> the 2D (edp, ep) mesh (fully_shard replicates dim0
        # ``edp``, shards dim1 ``ep``), so the meta+DCP loader fills [Replicate(edp), Shard(ep)] DTensors.
        from .plugins.hf_checkpoint_utils import ensure_local_checkpoint

        dense_fsdp_mesh = dense_mesh["dp"] if dp == 1 else sparse_moe_mesh
        # Collective: rank 0 makes the checkpoint local (downloading a Hub ID once); the others wait.
        _, local_dir = ensure_local_checkpoint(ckpt_path)
        n_dense = _load_ep_dense_distributed(
            model, dense_fsdp_mesh, local_dir, is_expert_fn, mo_session, expert_modules.values()
        )
        # The keep-sharded writer reads each weight's own mesh/placements, so it dedups the ``edp``
        # replicas at EDP>1.
        if mo_session.rank == 0:
            _dims = tuple(dense_fsdp_mesh.shape)
            print(
                f"[ep] dense_strategy=sharded: FSDP2-sharded {n_dense} dense module(s) on mesh "
                f"{dense_fsdp_mesh.mesh_dim_names}={_dims} (shard over ep={ep_degree}, replicate over "
                f"edp={dp}); experts EP-sharded on ep",
                flush=True,
            )

    # Each module's ParallelState for calibration's amax sync: dense modules reduce over the whole world;
    # routed experts reduce over their expert-data-parallel replicas, and share one input amax across the
    # layer's experts over the expert-parallel group. A size-1 axis is left unused (-1): no collective.
    def _axis(mesh):
        return mesh.get_group() if mesh.size() > 1 else -1

    _seed_parallel_states(
        model,
        expert_modules,
        dense_state=ParallelState(data_parallel_group=_axis(dense_mesh["dp"])),
        expert_state=ParallelState(
            data_parallel_group=_axis(sparse_moe_mesh["edp"]),
            expert_model_parallel_group=_axis(sparse_moe_mesh["ep"]),
        ),
    )
    if mo_session.rank == 0:
        print(
            f"[ep] seeded the expert-parallel state of {len(expert_modules)} routed-expert "
            "module(s) from base_model_ep_plan",
            flush=True,
        )
    # Re-planting the a2a forwards after quantizer insertion (``_reinstall_a2a_forwards``) needs this mesh.
    mo_session.ep_mesh = ep_submesh

    if mo_session.distributed:
        # Eagerly create each amax collective's NCCL comm in a globally-consistent order (barrier around
        # each), so the sub-communicators are built in lockstep rather than lazily mid-calibration --
        # which deadlocks when submesh groups are phase-skewed at DP>1. NO forward warmup (variable
        # per-group time would desync ranks before comm creation). Skip size-1 axes (no collective). The
        # EP group is already live from from_pretrained's expert sharding; re-warming it is a harmless
        # redundant all-reduce. Same order on every rank -> lockstep.
        warm_device = next(model.parameters()).device
        for _sub in (dense_mesh["dp"], sparse_moe_mesh["ep"], sparse_moe_mesh["edp"]):
            if _sub.size() <= 1:
                continue
            torch.distributed.barrier()
            torch.distributed.all_reduce(torch.ones(1, device=warm_device), group=_sub.get_group())
            torch.cuda.synchronize()
        torch.distributed.barrier()
    return model
