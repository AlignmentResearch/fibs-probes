"""Axial probe (``AxialProbe``): attention over tokens and over layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from probe_inference.archs.base import Probe

_SDPA_CUDA_MAX_BATCH_SIZE = 65_535


def _scaled_dot_product_attention_in_batch_chunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
) -> torch.Tensor:
    """Run SDPA without exceeding the CUDA grid limit on its batch axis."""
    if q.shape[0] <= _SDPA_CUDA_MAX_BATCH_SIZE:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                q[start : start + _SDPA_CUDA_MAX_BATCH_SIZE],
                k[start : start + _SDPA_CUDA_MAX_BATCH_SIZE],
                v[start : start + _SDPA_CUDA_MAX_BATCH_SIZE],
                dropout_p=dropout_p,
            )
            for start in range(0, q.shape[0], _SDPA_CUDA_MAX_BATCH_SIZE)
        ],
        dim=0,
    )


# ==============================================================================
# Positional Encoding
# ==============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Precomputes cos/sin frequencies and applies rotary embedding to Q and K.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (seq, dim/2)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x: (..., seq, head_dim) tensor
        cos: (seq, head_dim/2) cosine frequencies
        sin: (seq, head_dim/2) sine frequencies

    Returns:
        Tensor with rotary embedding applied.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ==============================================================================
# Normalization
# ==============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not re-center activations (no mean subtraction,
    no bias). This is simpler and often more stable for small transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm).to(x.dtype) * self.weight.to(x.dtype)


def _causal_window_base_mask(
    seq_len: int,
    window_size: int | None,
    padding_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Causal, optionally windowed attention mask. True = may attend.

    The window is measured in the probe's OWN token coordinates — each valid token's rank
    within ``padding_mask`` — rather than in raw tensor indices. A probe's input is the masked
    span, so "the previous ``window_size`` tokens" means the previous
    ``window_size`` tokens it can actually see; positions outside the span are not part of its
    input and must not consume window budget. Measuring in raw indices instead lets a gap
    starve the window: with span ``{1,2,5,6}`` and ``window_size=3``, token 5 would see only
    itself because positions 3 and 4 — which it cannot read — used up the band.

    Args:
        seq_len: Sequence length of the attention operands.
        window_size: Band width in span coordinates, or None for plain causal attention.
        padding_mask: (batch, seq) bool, True = a token the probe reads. None means every
            position is read, so ranks and raw indices coincide.
        device: Device for the constructed mask.

    Returns:
        (seq, seq) when ``padding_mask`` is None, else (batch, seq, seq).
    """
    if padding_mask is None:
        idx = torch.arange(seq_len, device=device)
        q_pos, kv_pos = idx.unsqueeze(1), idx.unsqueeze(0)
    else:
        # Rank of each position within the span. Positions outside the span inherit the
        # preceding rank; their rows/columns are removed by the padding intersection below.
        rank = padding_mask.long().cumsum(dim=1) - 1  # [batch, seq]
        q_pos, kv_pos = rank.unsqueeze(2), rank.unsqueeze(1)  # [batch, seq, 1], [batch, 1, seq]
    base = q_pos >= kv_pos
    if window_size is not None:
        base = base & ((q_pos - kv_pos) < window_size)
    return base


def _combine_base_and_padding_mask(
    base_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Combine a boolean base attention mask with key padding into an additive mask.

    Args:
        base_mask: bool where True = may attend (causal and/or windowed), either (seq, seq)
            shared across the batch or (batch, seq, seq) when the band is per-row.
        padding_mask: (batch, seq) bool where True = valid token.
        dtype: Output dtype for the additive mask.

    Returns:
        (batch, 1, seq, seq) additive float mask (0 = attend, -inf = blocked).
        A query whose every allowed key is padding is let through to itself:
        an all--inf row would make softmax produce NaN, which contaminates
        even masked aggregation downstream (``NaN * 0 == NaN``). Such rows are
        padding queries whose (finite) output is never read.
    """
    seq_len = base_mask.size(-1)
    base = base_mask.unsqueeze(1) if base_mask.dim() == 3 else base_mask.unsqueeze(0).unsqueeze(0)
    combined = base & padding_mask.unsqueeze(1).unsqueeze(2)
    fully_masked = ~combined.any(dim=-1, keepdim=True)
    self_eye = torch.eye(seq_len, dtype=torch.bool, device=base_mask.device).unsqueeze(0).unsqueeze(0)
    combined = combined | (fully_masked & self_eye)
    # Built directly in the target dtype: ``torch.where(combined, 0.0, -inf)`` would
    # materialize an fp32 (batch, 1, seq, seq) intermediate before the cast.
    return torch.zeros(combined.shape, dtype=dtype, device=base_mask.device).masked_fill_(~combined, float("-inf"))


# ==============================================================================
# Axial probe (cross-layer)
# ==============================================================================


class AxialAttentionBlock(nn.Module):
    """Single axial attention block: token-attention followed by layer-attention.

    Each sub-block uses multi-head self-attention with pre-norm (LayerNorm -> Attention -> Residual)
    followed by a feed-forward network (LayerNorm -> MLP -> Residual).
    """

    def __init__(
        self,
        d_proj: int,
        nhead: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        sliding_window: int | None = None,
    ):
        super().__init__()
        assert d_proj % nhead == 0, f"d_proj ({d_proj}) must be divisible by nhead ({nhead})"
        if sliding_window is not None and sliding_window < 1:
            raise ValueError(f"sliding_window must be >= 1, got {sliding_window}")
        self.nhead = nhead
        self.head_dim = d_proj // nhead
        # Token attention window: position i attends to [max(0, i-window+1), i].
        # None = full causal. Layer attention (short axis) is never windowed.
        self.sliding_window = sliding_window

        # Token attention sub-block (with RoPE)
        self.token_norm = RMSNorm(d_proj)
        self.token_qkv = nn.Linear(d_proj, 3 * d_proj, bias=False)
        self.token_q_norm = RMSNorm(self.head_dim)
        self.token_k_norm = RMSNorm(self.head_dim)
        self.token_rope = RotaryEmbedding(self.head_dim)
        self.token_out = nn.Linear(d_proj, d_proj, bias=False)
        self.token_ff_norm = RMSNorm(d_proj)
        self.token_ff = nn.Sequential(
            nn.Linear(d_proj, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_proj, bias=False),
            nn.Dropout(dropout),
        )

        # Layer attention sub-block
        self.layer_norm = RMSNorm(d_proj)
        self.layer_qkv = nn.Linear(d_proj, 3 * d_proj, bias=False)
        self.layer_q_norm = RMSNorm(self.head_dim)
        self.layer_k_norm = RMSNorm(self.head_dim)
        self.layer_out = nn.Linear(d_proj, d_proj, bias=False)
        self.layer_ff_norm = RMSNorm(d_proj)
        self.layer_ff = nn.Sequential(
            nn.Linear(d_proj, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_proj, bias=False),
            nn.Dropout(dropout),
        )

        self.attn_dropout = dropout

    def _causal_window_mask(
        self, seq_len: int, device: torch.device, token_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Causal, banded-when-windowed mask; True = may attend.

        Returns (T, T), or (batch, T, T) when a padding mask makes the band per-row. The band
        is measured in span coordinates -- see :func:`_causal_window_base_mask`.
        """
        return _causal_window_base_mask(seq_len, self.sliding_window, token_padding_mask, device)

    def forward(
        self,
        x: torch.Tensor,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with axial attention.

        Args:
            x: (batch, num_layers, seq, d_proj)
            token_padding_mask: (batch, seq) where True = valid token.

        Returns:
            (batch, num_layers, seq, d_proj)
        """
        B, L, T, D = x.shape
        H = self.nhead
        HD = self.head_dim
        dropout = self.attn_dropout if self.training else 0.0

        # --- Token attention: causal, across tokens within each layer ---
        x_tok = x.reshape(B * L, T, D)
        residual = x_tok
        x_tok = self.token_norm(x_tok)
        # QKV projection -> (B*L, T, 3, H, HD)
        qkv = self.token_qkv(x_tok).reshape(B * L, T, 3, H, HD)
        q, k, v = qkv.unbind(2)  # each (B*L, T, H, HD)
        q = self.token_q_norm(q)
        k = self.token_k_norm(k)

        # Apply RoPE to Q and K for position-aware token attention
        # cos/sin: (T, HD//2) -> (T, 1, HD//2) to broadcast over (B*L, T, H, HD//2)
        cos, sin = self.token_rope(T, x.device, q.dtype)
        cos = cos.unsqueeze(1)  # (T, 1, HD//2)
        sin = sin.unsqueeze(1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose to (B*L, H, T, HD) for SDPA
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        if token_padding_mask is not None:
            # Combined causal(+window) + padding mask, with the fully-masked-row
            # self-attend NaN guard (see _combine_base_and_padding_mask).
            causal = self._causal_window_mask(T, x.device, token_padding_mask)
            attn_mask = _combine_base_and_padding_mask(causal, token_padding_mask, x.dtype)  # (B, 1, T, T)
            # Expand for B*L: (B, 1, T, T) -> (B*L, 1, T, T)
            attn_mask = attn_mask.unsqueeze(1).expand(B, L, 1, T, T).reshape(B * L, 1, T, T)
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=dropout,
            )
        elif self.sliding_window is not None:
            banded = torch.where(self._causal_window_mask(T, x.device), 0.0, float("-inf")).to(x.dtype)
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=banded,
                dropout_p=dropout,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                dropout_p=dropout,
            )
        attn_out = attn_out.transpose(1, 2)  # back to (B*L, T, H, HD)

        attn_out = attn_out.reshape(B * L, T, D)
        x_tok = residual + self.token_out(attn_out)
        x_tok = x_tok + self.token_ff(self.token_ff_norm(x_tok))
        x = x_tok.reshape(B, L, T, D)

        # --- Layer attention: across layers for each token position ---
        x_lay = x.permute(0, 2, 1, 3).reshape(B * T, L, D)
        residual = x_lay
        x_lay = self.layer_norm(x_lay)
        qkv = self.layer_qkv(x_lay).reshape(B * T, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B*T, H, L, HD)
        q = self.layer_q_norm(q)
        k = self.layer_k_norm(k)
        attn_out = _scaled_dot_product_attention_in_batch_chunks(q, k, v, dropout_p=dropout)
        attn_out = attn_out.transpose(1, 2).reshape(B * T, L, D)
        x_lay = residual + self.layer_out(attn_out)
        x_lay = x_lay + self.layer_ff(self.layer_ff_norm(x_lay))
        x = x_lay.reshape(B, T, L, D).permute(0, 2, 1, 3)

        return x  # (batch, num_layers, seq, d_proj)


class AxialProbe(Probe):
    """Cross-layer probe using axial attention over tokens and layers.

    Takes activations from multiple layers simultaneously and applies
    factored (axial) attention: alternating token-attention and layer-attention,
    after axial transformers (Ho et al., 2019).

    Architecture:
        1. Shared down-projection: d_model -> d_proj
        2. Learned layer embeddings
        3. N axial attention blocks (causal token-attn + layer-attn each)
        4. Pool across layers (mean or CLS token) -> (batch, seq, d_proj)
        5. Output projection -> (batch, seq, nhead)

    Input: stacked activations (batch, num_layers, seq, d_model); the layer axis is
        positional, and `layer_keys` records which model layers it corresponds to.
    Output: (batch, seq, nhead) logits
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        d_proj: int = 128,
        n_attn_heads: int = 4,
        n_blocks: int = 1,
        d_ff: int = 1024,
        dropout: float = 0.0,
        use_layer_embed: bool = True,
        pool_mode: str = "cls",
        use_checkpoint: bool = True,
        normalize_input: str = "none",
        proj_adapter_rank: int = 0,
        input_adapter_rank: int = 0,
        sliding_window: int | None = None,
        layer_keys: list[int] | list[str] | list[int | str] | None = None,
    ):
        """Initialize AxialProbe.

        Args:
            d_model: Hidden dimension of the LLM activations.
            num_layers: Number of layers this probe will receive.
            d_proj: Projection dimension (activations are projected d_model -> d_proj).
            n_attn_heads: Number of attention heads inside each axial attention block.
                Distinct from ``.nhead`` (the Probe ABC output-head count), which is
                always 1 for AxialProbe.
            n_blocks: Number of axial attention blocks.
            d_ff: Feed-forward dimension within axial attention blocks.
            dropout: Dropout rate.
            use_layer_embed: Whether to add learned layer embeddings.
            pool_mode: How to pool across layers. "mean" averages all layers,
                "cls" appends a learnable CLS token to the layer dimension and
                extracts its output after attention.
            use_checkpoint: Whether to use gradient checkpointing.
            normalize_input: Input normalization mode ("none", "l2", "unit_norm").
            proj_adapter_rank: Rank of per-layer low-rank adapters on the shared
                down projection. 0 disables (all layers share the same projection).
                When > 0, each layer gets a LoRA-style residual: output_l =
                shared(x) + A_l @ B_l @ x, where A is (d_model, rank) and B is
                (rank, d_proj). B is zero-initialized so training starts from
                the shared projection.
            input_adapter_rank: Rank of per-layer low-rank adapters on the input
                activations. 0 disables. When > 0, each layer's activations are
                pre-transformed before the shared projection: x_l = x + C_l @ D_l @ x,
                where C is (d_model, rank) and D is (rank, d_model). D is
                zero-initialized so training starts from identity.
            sliding_window: Token-attention window. Position i attends to
                [max(0, i-window+1), i] instead of the full causal prefix,
                restricting what token-attention sees (the full (T, T) mask is still
                materialized).
                The effective receptive field still grows with depth: after
                n_blocks windowed layers, position i can integrate information
                from up to ~n_blocks * (window - 1) earlier tokens. None = full
                causal. Layer attention (the short axis) is never windowed.
                The window is counted in valid-token (rank) coordinates.
            layer_keys: Identities of the model layers this probe reads, in the
                order they occupy the probe's layer axis (e.g. ``["25", "36", ...]``).
                ``forward`` does not consult it — it receives an already-stacked
                tensor — but it is stored in ``config.json`` so a reloaded probe is fed
                the layers it was trained on. A probe fed a different layer set of the
                same size would score plausibly and wrongly, since the per-position
                parameters (the layer embeddings when ``use_layer_embed``, and any
                per-layer adapters) are applied by position. An empty list means
                "not recorded".
        """
        super().__init__(normalize_input=normalize_input)
        # Re-register buffers with correct shapes so load_state_dict works
        # input_mean: (1, L, 1, D) — per-layer mean vectors
        # input_scale: (1, L, 1, 1) — per-layer scale factors
        self.register_buffer("input_mean", torch.zeros(1, num_layers, 1, d_model))
        self.register_buffer("input_scale", torch.ones(1, num_layers, 1, 1))
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_proj = d_proj
        # The Probe ABC's `nhead` property is the output-head count, a different quantity.
        self.n_attn_heads = n_attn_heads
        self.n_blocks = n_blocks
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_layer_embed = use_layer_embed
        self.pool_mode = pool_mode
        self.use_checkpoint = use_checkpoint
        self.proj_adapter_rank = proj_adapter_rank
        self.input_adapter_rank = input_adapter_rank
        self.sliding_window = sliding_window

        # An empty list means "not recorded", not a zero-length axis: a checkpoint written
        # without layer keys reloads through AxialProbe(**init_args) and must not trip the
        # length check below.
        self.layer_keys: list[str] = [str(k) for k in layer_keys or []]
        if self.layer_keys and len(self.layer_keys) != num_layers:
            raise ValueError(
                f"layer_keys has {len(self.layer_keys)} entries but num_layers={num_layers}; "
                "they describe the same axis and must agree."
            )

        # Shared down-projection
        self.down_proj = nn.Linear(d_model, d_proj)

        # Per-layer low-rank adaptation of the input activations.
        # Always register the parameters (with zero-rank dimensions for the
        # disabled case) so that load_state_dict against a checkpoint with
        # mismatched rank produces a shape-mismatch error at the parameter
        # level rather than confusing missing/unexpected-key errors. Execution
        # gates on `rank > 0` at the forward-time call site, not on parameter
        # presence.
        self._input_adapter_enabled = input_adapter_rank > 0
        if input_adapter_rank > 0:
            bound_c = math.sqrt(3.0 / d_model)
            self.input_adapter_C = nn.Parameter(
                torch.empty(num_layers, d_model, input_adapter_rank).uniform_(-bound_c, bound_c)
            )
            self.input_adapter_D = nn.Parameter(torch.zeros(num_layers, input_adapter_rank, d_model))
        else:
            # Zero-rank registration — PyTorch handles zero-dim tensors fine
            # for state_dict serialization + load_state_dict shape comparison.
            self.input_adapter_C = nn.Parameter(torch.zeros(num_layers, d_model, 0))
            self.input_adapter_D = nn.Parameter(torch.zeros(num_layers, 0, d_model))

        # Per-layer low-rank adaptation of the down projection (LoRA-style).
        # Same registration pattern as input adapters above.
        self._proj_adapter_enabled = proj_adapter_rank > 0
        if proj_adapter_rank > 0:
            bound_a = math.sqrt(3.0 / d_model)
            self.proj_adapter_A = nn.Parameter(
                torch.empty(num_layers, d_model, proj_adapter_rank).uniform_(-bound_a, bound_a)
            )
            self.proj_adapter_B = nn.Parameter(torch.zeros(num_layers, proj_adapter_rank, d_proj))
        else:
            self.proj_adapter_A = nn.Parameter(torch.zeros(num_layers, d_model, 0))
            self.proj_adapter_B = nn.Parameter(torch.zeros(num_layers, 0, d_proj))

        # Optional layer embeddings (extra slot for CLS token if needed)
        n_embed = num_layers + 1 if pool_mode == "cls" else num_layers
        if use_layer_embed:
            self.layer_embed = nn.Embedding(n_embed, d_proj)

        # Learnable CLS token for layer pooling
        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, d_proj) * 0.02)

        # Axial attention blocks
        self.blocks = nn.ModuleList(
            [
                AxialAttentionBlock(
                    d_proj=d_proj,
                    nhead=n_attn_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    sliding_window=sliding_window,
                )
                for _ in range(n_blocks)
            ]
        )

        # Final RMSNorm before output projection
        self.final_norm = RMSNorm(d_proj)

        # Single logit per token
        self.out_proj = nn.Linear(d_proj, 1)

        # Apply nanochat-style initialization
        self.apply(self._init_weights)
        # Zero-init residual path output projections (blocks start as identity)
        for block in self.blocks:
            torch.nn.init.zeros_(block.token_out.weight)
            torch.nn.init.zeros_(block.layer_out.weight)
            torch.nn.init.zeros_(block.token_ff[-2].weight)
            torch.nn.init.zeros_(block.layer_ff[-2].weight)

    def _init_weights(self, module: nn.Module) -> None:
        """Nanochat-style weight initialization.

        Uses uniform init with bound = sqrt(3/d_proj) (equivalent to Kaiming
        uniform with fan_in = d_proj). Output projections on the residual path
        are zero-initialized separately after this is applied.
        """
        if isinstance(module, nn.Linear):
            bound = math.sqrt(3.0 / self.d_proj)
            torch.nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            bound = math.sqrt(3.0 / self.d_proj)
            torch.nn.init.uniform_(module.weight, -bound, bound)

    @property
    def nhead(self) -> int:
        # Probe ABC contract: output-head count. AxialProbe always produces a
        # single output logit per token (see self.out_proj = nn.Linear(d_proj, 1)).
        # Distinct from the constructor's ``n_attn_heads`` kwarg, which controls
        # the number of attention heads inside each axial attention block.
        return 1

    def _forward_impl(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Core forward pass.

        Args:
            x: Stacked activations (batch, num_layers, seq, d_model).
            padding_mask: (batch, seq) where True = valid token.

        Returns:
            Pooled representations (batch, seq, d_proj).
        """
        B, L, T, D = x.shape
        assert L == self.num_layers, f"Expected {self.num_layers} layers, got {L}"

        # Adapt and project: input adapter -> down projection -> proj adapter
        # Checkpointed to avoid storing large d_model intermediates for backward
        def _adapt_and_project(x: torch.Tensor) -> torch.Tensor:
            # Per-layer input adaptation: pre-transform activations before projection
            if self.input_adapter_rank > 0:
                # (B,L,T,D) @ (L,D,r) -> (B,L,T,r) @ (L,r,D) -> (B,L,T,D)
                input_adapt = torch.einsum("bltd,ldr->bltr", x, self.input_adapter_C)
                input_adapt = torch.einsum("bltr,lrd->bltd", input_adapt, self.input_adapter_D)
                x = x + input_adapt

            # Down-project: (B, L, T, D) -> (B, L, T, d_proj)
            if self.proj_adapter_rank > 0:
                x_proj = self.down_proj(x)
                # (B,L,T,D) @ (L,D,r) -> (B,L,T,r) @ (L,r,P) -> (B,L,T,P)
                adapter_out = torch.einsum("bltd,ldr->bltr", x, self.proj_adapter_A)
                adapter_out = torch.einsum("bltr,lrp->bltp", adapter_out, self.proj_adapter_B)
                return x_proj + adapter_out
            else:
                return self.down_proj(x)

        if self.use_checkpoint and self.training:
            x = checkpoint(_adapt_and_project, x, use_reentrant=False)
        else:
            x = _adapt_and_project(x)

        # Append CLS token to layer dimension if using CLS pooling
        if self.pool_mode == "cls":
            cls = self.cls_token.to(x.dtype).expand(B, 1, T, -1)
            x = torch.cat([x, cls], dim=1)  # (B, L+1, T, d_proj)

        # Add layer embeddings if enabled
        if self.use_layer_embed:
            n_layers = x.shape[1]  # L or L+1 depending on CLS
            layer_ids = torch.arange(n_layers, device=x.device)
            # (n_layers, d_proj) -> (1, n_layers, 1, d_proj)
            x = x + self.layer_embed(layer_ids).unsqueeze(0).unsqueeze(2).to(x.dtype)

        # Apply axial attention blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, padding_mask, use_reentrant=False)
            else:
                x = block(x, token_padding_mask=padding_mask)

        # Pool across layers: (B, L[+1], T, d_proj) -> (B, T, d_proj)
        if self.pool_mode == "cls":
            x = x[:, -1]  # extract CLS token position
        else:
            x = x.mean(dim=1)

        x = self.final_norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns per-token scores.

        Args:
            x: Stacked activations (batch, num_layers, seq, d_model).
            padding_mask: (batch, seq) where True = valid token.

        Returns:
            logits (batch, seq, nhead).
        """
        x = self._maybe_standardize(x)
        h = self._forward_impl(x, padding_mask=padding_mask)
        logits = self.out_proj(h)
        logits = 10.0 * torch.tanh(logits / 10.0)
        return logits
