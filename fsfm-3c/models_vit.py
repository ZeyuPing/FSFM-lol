# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class VisionTransformerWithTokenFusion(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer backbone + Forgery-Aware Token Fusion (FTF) head.

    FTF augments the standard CLS-token classification with a local forgery
    descriptor computed from the top-k most anomalous patch tokens.  The two
    descriptors are fused through a learned gating MLP before the final
    classifier, so the model can exploit both global face semantics and sparse
    local forgery traces without any extra supervision signal.

    Core pipeline (see FSFM_DiFF_FTF_Plan.md §3 for full derivation):
        1. Run the pretrained ViT backbone → (cls token c, patch tokens p_1…p_N).
        2. Patch anomaly score:  a_i = 1 − cos(p_i, c).
        3. Local descriptor:     l = Σ_{i∈TopK} softmax(a_i/τ) · p_i.
        4. Global descriptor:    g = fc_norm(mean of patch tokens)  [global_pool=True]
                                 g = norm(c)                         [global_pool=False]
        5. Fusion gate:          α = sigmoid(MLP([g ; l]))
                                 f = α·g + (1−α)·l
        6. Logits:               head(f)
    """

    def __init__(
        self,
        global_pool: bool = False,
        topk_patches: int = 16,
        fusion_tau: float = 0.5,
        fusion_gate_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.global_pool = global_pool
        embed_dim: int = kwargs['embed_dim']
        norm_layer = kwargs['norm_layer']

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

        # ── FTF hyper-parameters ──────────────────────────────────────────
        self.topk_patches = topk_patches
        self.fusion_tau = fusion_tau

        # Small projection to map patch tokens into the anomaly-scoring space.
        # Using identity (no extra params) works fine; a linear layer adds
        # slight flexibility but is optional.  We keep it lightweight.
        self.local_proj = nn.Identity()

        # Gating MLP: [g ; l] → scalar gate α ∈ (0, 1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, fusion_gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_gate_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_local_descriptor(
        self, patch_tokens: torch.Tensor, cls_token: torch.Tensor
    ) -> torch.Tensor:
        """Compute the local forgery descriptor l for a batch.

        Args:
            patch_tokens: (B, N, D) – all patch tokens after the last block.
            cls_token:    (B, D)    – the class token after the last block.

        Returns:
            l: (B, D) – weighted aggregation of the top-k anomalous patches.
        """
        # Step 1 – project patch tokens (identity by default)
        p = self.local_proj(patch_tokens)               # (B, N, D)

        # Step 2 – cosine similarity between each patch and the cls token
        #   cos_sim shape: (B, N)
        c = F.normalize(cls_token.unsqueeze(1), dim=-1)  # (B, 1, D)
        p_norm = F.normalize(p, dim=-1)                  # (B, N, D)
        cos_sim = (p_norm * c).sum(dim=-1)               # (B, N)

        # Anomaly score: high ⟹ the patch is far from the global face embedding
        anomaly = 1.0 - cos_sim                          # (B, N)

        # Step 3 – top-k selection
        k = min(self.topk_patches, anomaly.shape[1])
        topk_scores, topk_idx = anomaly.topk(k, dim=1)  # (B, k)

        # Gather the corresponding patch token vectors
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, p.shape[-1])
        topk_patches = p.gather(1, idx_exp)              # (B, k, D)

        # Soft-aggregate with temperature-scaled softmax weights
        weights = F.softmax(topk_scores / self.fusion_tau, dim=1)  # (B, k)
        l = (weights.unsqueeze(-1) * topk_patches).sum(dim=1)      # (B, D)

        return l

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Extended forward_features that returns the fused global-local descriptor."""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # ── Obtain raw cls token and patch tokens ──────────────────────
        if self.global_pool:
            # In global_pool mode there is no self.norm; use fc_norm on mean
            patch_tokens_raw = x[:, 1:, :]               # (B, N, D)
            cls_token_raw = x[:, 0, :]                   # (B, D)
            g = self.fc_norm(patch_tokens_raw.mean(dim=1))  # global descriptor
            patch_tokens = self.fc_norm(patch_tokens_raw)   # normalise for scoring
            cls_for_scoring = self.fc_norm(cls_token_raw)
        else:
            x = self.norm(x)
            cls_token_raw = x[:, 0, :]                   # (B, D)
            patch_tokens = x[:, 1:, :]                   # (B, N, D)
            g = cls_token_raw                            # global descriptor
            cls_for_scoring = cls_token_raw

        # ── Local forgery descriptor ───────────────────────────────────
        l = self._compute_local_descriptor(patch_tokens, cls_for_scoring)  # (B, D)

        # ── Fusion gate ────────────────────────────────────────────────
        gate_input = torch.cat([g, l], dim=-1)           # (B, 2D)
        alpha = torch.sigmoid(self.fusion_gate(gate_input))  # (B, 1)
        fused = alpha * g + (1.0 - alpha) * l            # (B, D)

        return fused


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,  # ViT-small config in MOCO_V3
        # patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=True,  # ViT-small config in timm
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_tokenfusion(
    topk_patches: int = 16,
    fusion_tau: float = 0.5,
    fusion_gate_hidden_dim: int = 256,
    **kwargs,
):
    """ViT-Base/16 backbone with Forgery-Aware Token Fusion (FTF) head.

    Drop-in replacement for ``vit_base_patch16`` that additionally mines the
    top-k most anomalous patch tokens and fuses them with the global descriptor
    before the final classifier.

    Extra kwargs:
        topk_patches (int):           number of top-k anomalous patches to pool (default 16).
        fusion_tau (float):           softmax temperature for patch weighting   (default 0.5).
        fusion_gate_hidden_dim (int): hidden size of the two-layer gating MLP   (default 256).
    """
    model = VisionTransformerWithTokenFusion(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        topk_patches=topk_patches,
        fusion_tau=fusion_tau,
        fusion_gate_hidden_dim=fusion_gate_hidden_dim,
        **kwargs,
    )
    return model
