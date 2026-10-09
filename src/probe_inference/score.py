"""Score activations with a loaded :class:`~probe_inference.load.TrainedProbe`.

The scoring path:

1. The read tokens of each layer are gathered to the front of the row, in order, and the rest is
   zero-padded (:func:`compact_selected`).
2. The probe's padding mask is the first stacked layer's non-zero rows.
3. The probe runs in float32, and its per-token outputs are pooled over the valid tokens with the
   probe's aggregator (:func:`aggregate`).
4. Linear and MLP probes hold one layer probe per layer, each reading one token. Their score is the mean
   of the per-layer sigmoids over ``used_layers`` (a probability). EFC and axial probes return the pooled
   logit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from probe_inference.load import TrainedProbe

Acts = dict[int, torch.Tensor] | torch.Tensor

#: The single token before the answer's end-of-turn token (linear and MLP).
LAST_ANSWER_TOKEN = "second-last-token-generation"
#: Every token from the start of the final user turn through the answer's end-of-turn token (EFC, axial).
LAST_USER_AND_ANSWER = "last-user-and-assistant-generation"
READ_WINDOWS = (LAST_ANSWER_TOKEN, LAST_USER_AND_ANSWER)


def build_read_mask(
    read_window: str,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    followup_start_positions: torch.Tensor,
) -> torch.Tensor:
    """Build the boolean read mask for ``read_window`` from token masks.

    Args:
        read_window: The probe's read window: ``second-last-token-generation`` (the answer's last token
            before its end-of-turn token) or ``last-user-and-assistant-generation`` (every real token from
            the start of the final user turn on).
        prompt_mask: ``(seq,)`` or ``(B, seq)`` bool, True on every real token that is not the answer.
        completion_mask: ``(seq,)`` or ``(B, seq)`` bool, True on the final assistant answer, including
            its end-of-turn token.
        followup_start_positions: ``()`` or ``(B,)`` int, the position where the final user turn starts.

    Returns:
        ``(seq,)`` or ``(B, seq)`` bool read mask, matching the input's batch shape.

    Raises:
        ValueError: If the masks' shapes disagree, the window is unknown, an answer has fewer than two
            tokens (``second-last-token-generation``), or a start position is negative.
    """
    single = completion_mask.ndim == 1
    prompt = (prompt_mask.reshape(1, -1) if single else prompt_mask).to(torch.bool)
    completion = (completion_mask.reshape(1, -1) if single else completion_mask).to(torch.bool)
    starts = followup_start_positions.reshape(-1)
    if prompt.shape != completion.shape or prompt.ndim != 2 or starts.shape != (completion.shape[0],):
        raise ValueError(
            f"prompt_mask {tuple(prompt_mask.shape)}, completion_mask {tuple(completion_mask.shape)} and "
            f"followup_start_positions {tuple(followup_start_positions.shape)} must be (seq,), (seq,), () or "
            "(B, seq), (B, seq), (B,)."
        )
    window = read_window.lower().replace("_", "-")
    if window == LAST_ANSWER_TOKEN:
        short = completion.sum(dim=1) < 2
        if bool(short.any()):
            rows = short.nonzero().flatten().tolist()
            raise ValueError(
                f"Rows {rows}: the answer span has fewer than 2 tokens, so it has no token before its "
                "end-of-turn token. The completion mask must include the end-of-turn token."
            )
        from_back = completion.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        mask = completion & (from_back == 2)
    elif window == LAST_USER_AND_ANSWER:
        if bool((starts < 0).any()):
            rows = (starts < 0).nonzero().flatten().tolist()
            raise ValueError(f"Rows {rows}: followup_start_positions is negative (no final user turn).")
        positions = torch.arange(completion.shape[1], device=completion.device).unsqueeze(0)
        mask = (positions >= starts.unsqueeze(1)) & (prompt | completion)
    else:
        raise ValueError(f"Unknown read window {read_window!r}; expected one of {list(READ_WINDOWS)}.")
    return mask[0] if single else mask


def compact_selected(activations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather each row's selected positions to the front, right-padded with zeros.

    The probes were trained on this layout: each row is boolean-indexed by the read mask and the batch
    right-padded, so a probe never sees unselected positions. Leaving them in place as zeros changes
    the output of any probe that mixes across tokens (the axial probe's rotary embedding counts raw
    positions).

    Args:
        activations: ``(B, S, D)`` for one layer.
        mask: ``(B, S)`` selected positions.

    Returns:
        ``(B, W, D)`` where ``W`` is the batch's largest selection count.
    """
    counts = mask.sum(dim=1)
    width = max(int(counts.max()), 1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)[:, :width]
    gathered = activations.gather(1, order.unsqueeze(-1).expand(-1, -1, activations.shape[-1]))
    within = torch.arange(width, device=mask.device).unsqueeze(0) < counts.unsqueeze(1)
    return gathered * within.unsqueeze(-1)


def aggregate(values: torch.Tensor, mask: torch.Tensor, method: str) -> torch.Tensor:
    """Pool ``(B, S, nhead)`` probe outputs over the masked positions.

    Raises:
        ValueError: On an unsupported method, or a row whose mask selects nothing.
    """
    if not bool(mask.any(dim=1).all()):
        raise ValueError("A row's readout mask selects no position; refusing to score it.")
    summed = values.sum(dim=-1)
    if method == "last":
        last_index = mask.shape[1] - 1 - mask.flip(dims=[1]).float().argmax(dim=1)
        return summed.gather(1, last_index.unsqueeze(1)).squeeze(1)
    if method == "mean":
        return (summed * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unsupported eval_sequence_aggregator {method!r}; this scorer implements 'last' and 'mean'.")


def score(probe: TrainedProbe, acts: Acts, read_mask: torch.Tensor) -> float:
    """Score one transcript.

    Args:
        probe: The loaded probe.
        acts: Either ``{layer: Tensor[seq, d_model]}`` holding at least ``probe.used_layers``, or one
            ``Tensor[len(probe.layers), seq, d_model]`` stacked in ``probe.layers`` order. Any floating
            dtype; the probe runs in float32.
        read_mask: ``(seq,)`` bool, True on the tokens the probe's read window selects (see
            :func:`build_read_mask`).

    Returns:
        A probability for linear and MLP probes (mean of per-layer sigmoids), a logit for EFC and axial.

    Raises:
        ValueError: On any shape, dtype, layer or read-window violation (see :func:`score_batch`).
    """
    if read_mask.ndim != 1:
        raise ValueError(f"score() takes one transcript: read_mask must be (seq,), got {tuple(read_mask.shape)}.")
    if isinstance(acts, torch.Tensor):
        if acts.ndim != 3:
            raise ValueError(
                f"score() takes one transcript: acts must be (layers, seq, d_model), got {tuple(acts.shape)}."
            )
        batched: Acts = acts.unsqueeze(0)
    else:
        batched = {layer: _single(layer, tensor).unsqueeze(0) for layer, tensor in acts.items()}
    return float(score_batch(probe, batched, read_mask.unsqueeze(0))[0])


def score_batch(probe: TrainedProbe, acts: Acts, read_mask: torch.Tensor) -> torch.Tensor:
    """Score a batch of transcripts.

    Args:
        probe: The loaded probe.
        acts: ``{layer: Tensor[B, seq, d_model]}`` holding at least ``probe.used_layers``, or one
            ``Tensor[B, len(probe.layers), seq, d_model]`` stacked in ``probe.layers`` order.
        read_mask: ``(B, seq)`` bool read mask.

    Returns:
        ``(B,)`` float64 scores: probabilities for linear and MLP, logits for EFC and axial.

    Raises:
        ValueError: If ``acts`` lacks a layer, has the wrong shape or a non-floating dtype, or holds a
            non-finite value at a read position; if ``read_mask`` is not a ``(B, seq)`` bool tensor; if a
            linear or MLP row does not read exactly one token; if an EFC row reads fewer than two tokens
            or only identical tokens; or if a read token's activation is exactly zero in the layer the
            padding mask is taken from (the probes treat all-zero rows as padding).
    """
    if probe.per_layer:
        logits = layer_logits(probe, acts, read_mask)
        return torch.stack([torch.sigmoid(logits[layer].double()) for layer in probe.used_layers]).mean(dim=0)
    layer_acts, mask = _prepare(probe, acts, read_mask, probe.layers)
    counts = mask.sum(dim=1)
    if probe.arch == "efc" and bool((counts < 2).any()):
        bad = (counts < 2).nonzero().flatten().tolist()
        raise ValueError(
            f"EFC needs at least 2 read tokens per row for its centred covariance; rows {bad} select "
            f"{counts[bad].tolist()}. Its window is the final user turn through end-of-turn."
        )
    stacked = torch.stack([_compact(layer_acts[layer], mask, counts, layer) for layer in probe.layers], dim=1)
    padding = _padding_mask(stacked[:, 0], counts, probe.layers[0])
    if probe.arch == "efc":
        _check_efc_distinct(stacked, padding)
    with torch.no_grad():
        values = probe.module(stacked, padding_mask=padding)
    return aggregate(values, padding, probe.aggregator).double()


def layer_logits(probe: TrainedProbe, acts: Acts, read_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    """Per-layer logits of a linear or MLP probe, before the sigmoid and the layer mean.

    Args:
        probe: A linear or MLP probe.
        acts: As for :func:`score_batch`.
        read_mask: ``(B, seq)`` bool read mask; each row must select exactly one token.

    Returns:
        ``{layer: (B,) float64 logits}`` for every layer in ``probe.used_layers``.

    Raises:
        ValueError: If the probe is not linear or MLP, or on any check :func:`score_batch` makes.
    """
    if not probe.per_layer:
        raise ValueError(f"layer_logits() needs a linear or MLP probe; this is {probe.arch}. Use score_batch().")
    layer_acts, mask = _prepare(probe, acts, read_mask, probe.used_layers)
    counts = mask.sum(dim=1)
    if not bool((counts == 1).all()):
        bad = (counts != 1).nonzero().flatten().tolist()
        raise ValueError(
            f"{probe.arch} probes read exactly one token per row (window {probe.read_window!r}); rows {bad} select "
            f"{counts[bad].tolist()} tokens. Build the mask with probe.read_mask(...)."
        )
    out: dict[int, torch.Tensor] = {}
    for layer in probe.used_layers:
        compacted = _compact(layer_acts[layer], mask, counts, layer)
        padding = _padding_mask(compacted, counts, layer)
        with torch.no_grad():
            values = probe.modules[layer](compacted, padding_mask=padding)
        out[layer] = aggregate(values, padding, probe.aggregator).double()
    return out


def _single(layer: int, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
        raise ValueError(f"score() takes one transcript: acts[{layer}] must be (seq, d_model), got {shape}.")
    return tensor


def _prepare(
    probe: TrainedProbe, acts: Acts, read_mask: torch.Tensor, needed: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Validate the inputs and return ``{layer: (B, S, D)}`` on the probe's device and a ``(B, S)`` bool mask."""
    device = next(iter(probe.modules.values())).input_scale.device
    by_layer = _split_layers(probe, acts, needed)
    shapes = {tuple(tensor.shape) for tensor in by_layer.values()}
    if len(shapes) != 1:
        raise ValueError(f"Layer activations must share one (B, seq, d_model) shape; got {sorted(shapes)}.")
    batch, seq, width = shapes.pop()
    if width != probe.d_model:
        raise ValueError(f"Activations have hidden size {width}; this probe reads d_model={probe.d_model}.")
    if not isinstance(read_mask, torch.Tensor) or read_mask.dtype != torch.bool:
        raise ValueError(f"read_mask must be a bool tensor, got {getattr(read_mask, 'dtype', type(read_mask))}.")
    if tuple(read_mask.shape) != (batch, seq):
        raise ValueError(
            f"read_mask shape {tuple(read_mask.shape)} does not match activations (B, seq)=({batch}, {seq})."
        )
    mask = read_mask.to(device)
    if not bool(mask.any(dim=1).all()):
        rows = (~mask.any(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f"read_mask selects no token in rows {rows}.")
    prepared: dict[int, torch.Tensor] = {}
    for layer, tensor in by_layer.items():
        if not tensor.is_floating_point():
            raise ValueError(f"acts[{layer}] has dtype {tensor.dtype}; expected a floating dtype (bf16 as stored).")
        prepared[layer] = tensor.to(device=device)
    return prepared, mask


def _compact(activations: torch.Tensor, mask: torch.Tensor, counts: torch.Tensor, layer: int) -> torch.Tensor:
    """Compact one layer's read tokens to the front and cast them to float32.

    Compacting before the cast gives the same values as casting first, since
    the gather is exact, and keeps float32 copies to the read window. Positions past a row's read count
    are zeroed with ``where`` rather than a multiply, so a non-finite value outside the window cannot
    leak into the padding.

    Raises:
        ValueError: If a read token's activation is not finite.
    """
    compacted = compact_selected(activations, mask)
    within = torch.arange(compacted.shape[1], device=mask.device).unsqueeze(0) < counts.unsqueeze(1)
    compacted = torch.where(within.unsqueeze(-1), compacted, torch.zeros((), dtype=compacted.dtype)).float()
    if not bool(torch.isfinite(compacted).all()):
        rows = (~torch.isfinite(compacted)).any(dim=-1).any(dim=-1).nonzero().flatten().tolist()
        raise ValueError(f"acts[{layer}] holds a non-finite value at a read position in rows {rows}.")
    return compacted


def _split_layers(probe: TrainedProbe, acts: Acts, needed: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Turn either input form into ``{layer: (B, S, D)}`` for the ``needed`` layers."""
    if isinstance(acts, torch.Tensor):
        if acts.ndim != 4 or acts.shape[1] != len(probe.layers):
            raise ValueError(
                f"Stacked acts must be (B, {len(probe.layers)}, seq, d_model) in layer order {list(probe.layers)}; "
                f"got {tuple(acts.shape)}."
            )
        return {layer: acts[:, probe.layers.index(layer)] for layer in needed}
    if not isinstance(acts, dict):
        raise ValueError(f"acts must be a dict of layer -> Tensor or a stacked Tensor, got {type(acts).__name__}.")
    non_int = [key for key in acts if not isinstance(key, int) or isinstance(key, bool)]
    if non_int:
        raise ValueError(f"acts keys must be int layer indices (decoder block k), got {non_int}.")
    missing = [layer for layer in needed if layer not in acts]
    if missing:
        raise ValueError(
            f"acts lacks layers {missing} (has {sorted(acts)}; this probe reads {list(needed)}). Layer k is the "
            "output of decoder block k, i.e. Hugging Face hidden_states[k + 1]."
        )
    for layer in needed:
        if not isinstance(acts[layer], torch.Tensor) or acts[layer].ndim != 3:
            shape = tuple(acts[layer].shape) if isinstance(acts[layer], torch.Tensor) else type(acts[layer]).__name__
            raise ValueError(f"acts[{layer}] must be (B, seq, d_model), got {shape}.")
    return {layer: acts[layer] for layer in needed}


def _padding_mask(compacted: torch.Tensor, counts: torch.Tensor, layer: int) -> torch.Tensor:
    """The probe's padding mask: the compacted layer's non-zero rows, as in training.

    Raises:
        ValueError: If a read token's activation vector is exactly zero, which would silently drop it.
    """
    padding = (compacted != 0).any(dim=-1)
    expected = torch.arange(compacted.shape[1], device=counts.device).unsqueeze(0) < counts.unsqueeze(1)
    if not torch.equal(padding, expected):
        rows = (padding != expected).any(dim=1).nonzero().flatten().tolist()
        raise ValueError(
            f"Rows {rows}: a read token's layer-{layer} activation is exactly zero. The probes treat all-zero "
            "rows as padding, so it would be dropped; refusing to score."
        )
    return padding


def _check_efc_distinct(stacked: torch.Tensor, padding: torch.Tensor) -> None:
    """EFC pools a covariance over the read tokens, so they must not all be identical.

    Args:
        stacked: ``(B, L, W, D)`` compacted read tokens.
        padding: ``(B, W)`` valid-token mask.
    """
    same_as_first = (stacked == stacked[:, :, :1]).all(dim=-1).all(dim=1)
    identical = (same_as_first | ~padding).all(dim=1)
    if bool(identical.any()):
        rows = identical.nonzero().flatten().tolist()
        raise ValueError(f"EFC rows {rows}: every read token is identical, so the covariance is degenerate.")
