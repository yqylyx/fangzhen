from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def masked_mean(x: Tensor, mask: Tensor, dim: int | tuple[int, ...], keepdim: bool = False) -> Tensor:
    # ????????????? patch???????????????
    mask_f = mask.float()
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    numerator = (x * mask_f).sum(dim=dim, keepdim=keepdim)
    denominator = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
    pe = torch.zeros((length, dim), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class GraphNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.mean_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, node_mask: Tensor) -> Tensor:
        mask = node_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        centered = x - mean * self.mean_scale
        var = ((centered * mask) ** 2).sum(dim=1, keepdim=True) / denom
        out = centered / torch.sqrt(var + self.eps)
        return out * self.weight + self.bias


class ChannelPatchEmbed(nn.Module):
    def __init__(self, patch_len: int, d_model: int, dropout: float = 0.1, patch_valid_ratio: float = 0.25) -> None:
        super().__init__()
        self.patch_len = int(patch_len)
        self.patch_valid_ratio = float(patch_valid_ratio)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        _, seq_len, _ = x.shape
        pad_t = (self.patch_len - seq_len % self.patch_len) % self.patch_len
        if pad_t:
            x = F.pad(x, (0, 0, 0, pad_t))
            mask = F.pad(mask.float(), (0, 0, 0, pad_t)).bool()

        # ????????????? patch?????????????????????
        # ????????????????
        x_patch = rearrange(x, 'b (p l) c -> b c p l', l=self.patch_len)
        m_patch = rearrange(mask.float(), 'b (p l) c -> b c p l', l=self.patch_len)
        valid_counts = m_patch.sum(dim=-1, keepdim=True)
        valid_ratio = valid_counts / float(self.patch_len)
        patch_mask = (valid_ratio.squeeze(-1) >= self.patch_valid_ratio).bool()

        # ????????????? patch ?????? d_model ???
        x_patch = (x_patch * m_patch) / valid_counts.clamp_min(1.0)
        x_patch = x_patch * float(self.patch_len)
        token = self.proj(x_patch)
        token = self.norm(token)
        token = self.dropout(token)
        token = token * patch_mask.unsqueeze(-1)
        return token, patch_mask


class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        safe_mask = mask.clone()
        empty = safe_mask.sum(dim=-1) == 0
        if empty.any():
            safe_mask[empty, 0] = True
        # ???????????????? patch ???????
        attn_in = self.norm1(x)
        attn_out, attn = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=~safe_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x, attn if return_attn else None


class TCNResidualBlock(nn.Module):
    def __init__(self, d_model: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(1, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        residual = x
        out = self.conv1(self.norm1(x))
        out = F.gelu(out)
        out = self.dropout(out)
        out = out * mask.unsqueeze(1)
        out = self.conv2(self.norm2(out))
        out = self.dropout(F.gelu(out))
        out = (residual + out) * mask.unsqueeze(1)
        return out


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        graph_type: str = 'dense_attn',
        topk: int = 16,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.graph_type = graph_type
        self.topk = topk

        self.norm1 = GraphNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = GraphNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def _apply_topk(self, scores: Tensor, node_mask: Tensor) -> Tensor:
        if self.graph_type != 'sparse_topk':
            return scores
        _, n_heads, n_nodes, _ = scores.shape
        topk = min(self.topk, n_nodes)
        if topk >= n_nodes:
            return scores

        key_mask = node_mask[:, None, None, :].expand(-1, n_heads, n_nodes, -1)
        masked_scores = scores.masked_fill(~key_mask, float('-inf'))
        topk_vals, topk_idx = torch.topk(masked_scores, k=topk, dim=-1)
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)
        return sparse_scores

    def forward(self, x: Tensor, node_mask: Tensor, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        _, n_nodes, _ = x.shape
        safe_mask = node_mask.clone()
        empty = safe_mask.sum(dim=-1) == 0
        if empty.any():
            safe_mask[empty, 0] = True

        # ???????????????????????????????
        # ???????????????????
        x_norm = self.norm1(x, safe_mask)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_mask = safe_mask[:, None, None, :].expand(-1, self.n_heads, n_nodes, -1)
        scores = scores.masked_fill(~key_mask, float('-inf'))
        scores = self._apply_topk(scores, safe_mask)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = x + self.out_proj(out)
        x = x * safe_mask.unsqueeze(-1)
        x = x + self.ff(self.norm2(x, safe_mask))
        x = x * safe_mask.unsqueeze(-1)
        return x, attn if return_attn else None


class BoundaryHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.boundary = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # ?????????? boundary logit?
        return self.boundary(x).squeeze(-1)


class PhysHPGT(nn.Module):
    def __init__(
        self,
        patch_len: int = 16,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        temporal_encoder: str = 'transformer',
        graph_type: str = 'dense_attn',
        n_exit: int = 3,
        early_exit: bool = True,
        use_freq_branch: bool = True,
        ff_mult: int = 4,
        n_graph_layers: int = 2,
        graph_topk: int = 16,
        patch_valid_ratio: float = 0.25,
        tcn_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.temporal_encoder = temporal_encoder
        self.graph_type = graph_type
        self.n_exit = n_exit
        self.early_exit = early_exit
        self.use_freq_branch = use_freq_branch
        self.n_graph_layers = n_graph_layers
        self.freq_bins = 32

        self.patch_embed = ChannelPatchEmbed(
            patch_len=patch_len,
            d_model=d_model,
            dropout=dropout,
            patch_valid_ratio=patch_valid_ratio,
        )
        self.modality_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        self.dropout_layer = nn.Dropout(dropout)

        if temporal_encoder == 'transformer':
            self.temporal_blocks = nn.ModuleList(
                [TemporalTransformerBlock(d_model, n_heads, dropout, ff_mult=ff_mult) for _ in range(n_layers)]
            )
            self.tcn_blocks = None
        elif temporal_encoder == 'tcn':
            self.temporal_blocks = None
            self.tcn_blocks = nn.ModuleList(
                [TCNResidualBlock(d_model, dilation=2 ** idx, kernel_size=tcn_kernel_size, dropout=dropout) for idx in range(n_layers)]
            )
        else:
            raise ValueError(f'unknown temporal encoder: {temporal_encoder}')

        self.graph_blocks = nn.ModuleList(
            [
                GraphAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    graph_type=graph_type,
                    topk=graph_topk,
                )
                for _ in range(n_graph_layers)
            ]
        )

        self.task_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if use_freq_branch:
            self.freq_proj = nn.Sequential(
                nn.LayerNorm(self.freq_bins * 2),
                nn.Linear(self.freq_bins * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.freq_proj = None

        self.exit_refiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(max(n_exit, 1))
            ]
        )
        self.exit_heads = nn.ModuleList([BoundaryHead(d_model, dropout=dropout) for _ in range(n_exit)])
        self.emb_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _encode_temporal(self, tokens: Tensor, patch_mask: Tensor, return_attn: bool = False) -> tuple[Tensor, list[Tensor]]:
        bsz, n_channels, n_patches, d_model = tokens.shape
        pos = sinusoidal_positional_encoding(n_patches, d_model, tokens.device)
        tokens = tokens + pos.unsqueeze(0).unsqueeze(0)
        tokens = self.dropout_layer(tokens)

        if self.temporal_encoder == 'transformer':
            # ?? batch ??????????????????????????
            # ??????????????????
            seq = rearrange(tokens, 'b c p d -> (b c) p d')
            seq_mask = rearrange(patch_mask, 'b c p -> (b c) p')
            attn_maps: list[Tensor] = []
            for block in self.temporal_blocks or []:
                seq, attn = block(seq, seq_mask, return_attn=return_attn)
                if attn is not None:
                    attn_maps.append(attn.view(bsz, n_channels, self.n_heads, n_patches, n_patches))
            seq = rearrange(seq, '(b c) p d -> b c p d', b=bsz, c=n_channels)
            return seq, attn_maps

        seq = rearrange(tokens, 'b c p d -> (b c) d p')
        seq_mask = rearrange(patch_mask, 'b c p -> (b c) p').float()
        for block in self.tcn_blocks or []:
            seq = block(seq, seq_mask)
        seq = rearrange(seq, '(b c) d p -> b c p d', b=bsz, c=n_channels)
        return seq, []

    def _spectral_feature(self, x: Tensor, channel_mask: Tensor) -> Tensor:
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], self.freq_bins))
        # ?????????? patch ?????????????
        seq = x.transpose(1, 2)
        if torch.onnx.is_in_onnx_export():
            mag = F.adaptive_avg_pool1d(seq.abs(), self.freq_bins)
        else:
            spectrum = torch.fft.rfft(seq, dim=-1)
            mag = torch.log1p(spectrum.abs())
        mag = masked_mean(mag, channel_mask.unsqueeze(-1), dim=1)
        if mag.shape[-1] != self.freq_bins:
            mag = F.interpolate(mag.unsqueeze(1), size=self.freq_bins, mode='linear', align_corners=False).squeeze(1)
        return mag

    def _fuse_modalities(
        self,
        v_nodes: Tensor,
        d_nodes: Tensor,
        ch_mask_v: Tensor,
        ch_mask_d: Tensor,
        global_nodes: Tensor,
        node_mask: Tensor,
        V: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # ????????????????????????????????
        # ????????????????????
        pooled_v = masked_mean(v_nodes, ch_mask_v, dim=1)
        pooled_d = masked_mean(d_nodes, ch_mask_d, dim=1)
        pooled_nodes = masked_mean(global_nodes, node_mask, dim=1)
        task_context = pooled_nodes + self.task_token.expand_as(pooled_nodes)
        gate_input = torch.cat([pooled_v, pooled_d, task_context], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        fused = gate * pooled_v + (1.0 - gate) * pooled_d
        fused = self.fusion_proj(torch.cat([fused, pooled_nodes], dim=-1))
        if self.use_freq_branch and self.freq_proj is not None:
            spec = torch.cat(
                [
                    self._spectral_feature(V, ch_mask_v),
                    self._spectral_feature(delta, ch_mask_d),
                ],
                dim=-1,
            )
            fused = fused + self.freq_proj(spec)
        return fused, gate

    def _resolve_exit_states(self, states: list[Tensor]) -> list[Tensor]:
        if not states:
            raise ValueError('at least one state is required for early exits')
        resolved = list(states)
        # ???????????????????????????????
        while len(resolved) < self.n_exit:
            idx = min(len(resolved) - 1, len(self.exit_refiners) - 1)
            last = resolved[-1]
            resolved.append(last + self.exit_refiners[idx](last))
        if len(resolved) > self.n_exit:
            index = torch.linspace(0, len(resolved) - 1, steps=self.n_exit).round().long().tolist()
            resolved = [resolved[idx] for idx in index]
        return resolved

    def forward(
        self,
        V: Tensor,
        delta: Tensor,
        mask_V: Tensor,
        mask_delta: Tensor,
        ch_mask_V: Tensor | None = None,
        ch_mask_delta: Tensor | None = None,
        time_mask: Tensor | None = None,
        return_attn: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        del time_mask
        if ch_mask_V is None:
            ch_mask_V = mask_V.any(dim=1)
        if ch_mask_delta is None:
            ch_mask_delta = mask_delta.any(dim=1)

        # ??????????????????????? token?
        V = V.float()
        delta = delta.float()
        mask_V = mask_V.bool()
        mask_delta = mask_delta.bool()
        ch_mask_V = ch_mask_V.bool()
        ch_mask_delta = ch_mask_delta.bool()

        v_tokens, v_patch_mask = self.patch_embed(V, mask_V)
        d_tokens, d_patch_mask = self.patch_embed(delta, mask_delta)

        v_tokens = v_tokens + self.modality_embed[0].view(1, 1, 1, -1)
        d_tokens = d_tokens + self.modality_embed[1].view(1, 1, 1, -1)

        # ???????????????????
        v_tokens, temporal_attn_v = self._encode_temporal(v_tokens, v_patch_mask, return_attn=return_attn)
        d_tokens, temporal_attn_d = self._encode_temporal(d_tokens, d_patch_mask, return_attn=return_attn)

        # ?????????? patch ??????????????
        v_nodes = masked_mean(v_tokens, v_patch_mask, dim=2)
        d_nodes = masked_mean(d_tokens, d_patch_mask, dim=2)

        node_mask = torch.cat([ch_mask_V, ch_mask_delta], dim=1)
        all_nodes = torch.cat([v_nodes, d_nodes], dim=1)
        decision_states: list[Tensor] = []
        graph_attn: list[Tensor] = []
        fusion_gates: list[Tensor] = []

        # ?????????????????????????????
        # ??????????????
        fused, gate = self._fuse_modalities(v_nodes, d_nodes, ch_mask_V, ch_mask_delta, all_nodes, node_mask, V, delta)
        decision_states.append(fused)
        fusion_gates.append(gate)

        nv = v_nodes.shape[1]
        for block in self.graph_blocks:
            all_nodes, attn = block(all_nodes, node_mask, return_attn=return_attn)
            if attn is not None:
                graph_attn.append(attn)
            v_cur = all_nodes[:, :nv, :]
            d_cur = all_nodes[:, nv:, :]
            fused, gate = self._fuse_modalities(v_cur, d_cur, ch_mask_V, ch_mask_delta, all_nodes, node_mask, V, delta)
            decision_states.append(fused)
            fusion_gates.append(gate)

        exit_states = self._resolve_exit_states(decision_states)
        logits_list: list[Tensor] = []
        for idx, state in enumerate(exit_states):
            logits_list.append(self.exit_heads[idx](state))

        # ????????????????????????????
        emb = F.normalize(self.emb_proj(exit_states[-1]), dim=-1)
        channel_importance = all_nodes.norm(dim=-1)
        channel_importance = channel_importance * node_mask.float()

        outputs: dict[str, Any] = {
            'logits_list': logits_list,
            'emb': emb,
        }

        if return_attn:
            outputs['attn'] = {
                'temporal_V': temporal_attn_v,
                'temporal_delta': temporal_attn_d,
                'graph': graph_attn,
                'fusion_gate': fusion_gates,
                'channel_importance': channel_importance,
                'node_mask': node_mask,
            }
        elif self.temporal_encoder == 'tcn':
            outputs['attn'] = {
                'channel_importance': channel_importance,
                'fusion_gate': fusion_gates,
                'node_mask': node_mask,
            }

        if return_hidden:
            outputs['hidden'] = {
                'decision_states': exit_states,
                'v_nodes': v_nodes,
                'delta_nodes': d_nodes,
                'all_nodes': all_nodes,
            }

        return outputs
