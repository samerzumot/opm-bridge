"""BrainOmni wrapper and transfer/control arms (FR3, FR4, §6).

Addresses the critical upstream finding discovered in Phase 0:
In upstream `OpenTSLab/BrainOmni`, `BrainTokenizer.tokenize()` is decorated with `@torch.no_grad()`
and `self.tokenizer.encoder.neuros` is `.detach()`ed inside `BrainOmni.encode()`.
Consequently:
  1. Upstream "full fine-tuning" never updates `BrainTokenizer` or `BrainSensorModule`.
  2. Upstream `pretrained=False` leaves `BrainTokenizer` randomly initialized AND frozen (`@torch.no_grad()`).

This wrapper explicitly separates and supports:
  - `frozen`: Frozen pre-trained tokenizer + frozen pre-trained backbone (linear/MLP probe).
  - `partial_ft`: Frozen pre-trained tokenizer + last K transformer blocks fine-tuned.
  - `backbone_ft`: Frozen pre-trained tokenizer + full pre-trained backbone fine-tuned (upstream default).
  - `end_to_end_ft`: Pre-trained tokenizer (including `BrainSensorModule`) + backbone fine-tuned end-to-end
    with straight-through RVQ gradients.
  - `control_a_random_e2e`: Fully randomly initialized tokenizer + backbone trained end-to-end with
    gradients enabled through `BrainTokenizer` (primary control in §6).
  - `control_b_random_backbone`: Frozen pre-trained tokenizer + randomly initialized backbone
    (diagnostic separating tokenizer transfer from backbone transfer in §6).
"""

from __future__ import annotations

import json
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from opm_bridge.models.vendor_brainomni.brainomni.model import BrainOmni

BRAINOMNI_HF_REPO = "OpenTSLab/BrainOmni"
BRAINOMNI_HF_REVISION = "9a4d3c70495370397ccfbfd6d2496f25647545a5"

TransferMode = Literal[
    "frozen",
    "partial_ft",
    "backbone_ft",
    "end_to_end_ft",
    "control_a_random_e2e",
    "control_b_random_backbone",
]


def load_brainomni_backbone(
    size: Literal["tiny", "base"] = "tiny",
    load_pretrained_backbone: bool = True,
    load_pretrained_tokenizer: bool = True,
    cfg_override: Optional[dict] = None,
) -> BrainOmni:
    """Instantiate BrainOmni and optionally load official MIT weights from Hugging Face."""
    if cfg_override is not None:
        cfg = dict(cfg_override)
    else:
        cfg_path = hf_hub_download(
            BRAINOMNI_HF_REPO,
            f"{size}/model_cfg.json",
            revision=BRAINOMNI_HF_REVISION,
        )
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    model = BrainOmni(**cfg)

    if load_pretrained_backbone or load_pretrained_tokenizer:
        ckpt_path = hf_hub_download(
            BRAINOMNI_HF_REPO,
            f"{size}/BrainOmni.pt",
            revision=BRAINOMNI_HF_REVISION,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if load_pretrained_backbone and load_pretrained_tokenizer:
            model.load_state_dict(state, strict=True)
        elif load_pretrained_tokenizer and not load_pretrained_backbone:
            tok_state = {
                k.removeprefix("tokenizer."): v
                for k, v in state.items()
                if k.startswith("tokenizer.")
            }
            model.tokenizer.load_state_dict(tok_state, strict=True)
        elif load_pretrained_backbone and not load_pretrained_tokenizer:
            bb_state = {k: v for k, v in state.items() if not k.startswith("tokenizer.")}
            model.load_state_dict(bb_state, strict=False)

    return model


class BrainOmniClassifier(nn.Module):
    """Downstream classifier wrapping BrainOmni across all pre-registered transfer & control arms."""

    def __init__(
        self,
        backbone: BrainOmni,
        num_classes: int,
        mode: TransferMode = "backbone_ft",
        partial_ft_blocks: int = 2,
        label_smoothing: float = 0.1,
        bypass_vq_in_control_a: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.mode: TransferMode = mode
        self.bypass_vq_in_control_a = bypass_vq_in_control_a

        n_neuro = self.backbone.tokenizer.encoder.neuros.shape[0]
        in_features = n_neuro * self.backbone.lm_dim
        n_dim = self.backbone.lm_dim

        self.class_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, n_dim),
            nn.SELU(),
            nn.Linear(n_dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._configure_trainable_params(partial_ft_blocks=partial_ft_blocks)

    def _configure_trainable_params(self, partial_ft_blocks: int) -> None:
        # Never need tokenizer decoder or masked-LM predict_head during classification
        for p in self.backbone.tokenizer.decoder.parameters():
            p.requires_grad = False
        for p in self.backbone.predict_head.parameters():
            p.requires_grad = False
        self.backbone.mask_token.requires_grad = False

        if self.mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif self.mode == "partial_ft":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for block in self.backbone.blocks[:-1][-partial_ft_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif self.mode in ("backbone_ft", "control_b_random_backbone"):
            for p in self.backbone.tokenizer.parameters():
                p.requires_grad = False
            for p in self.backbone.projection.parameters():
                p.requires_grad = True
            for block in self.backbone.blocks[:-1]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.blocks[-1].parameters():
                p.requires_grad = False
        elif self.mode in ("end_to_end_ft", "control_a_random_e2e"):
            for p in self.backbone.tokenizer.encoder.parameters():
                p.requires_grad = True
            for p in self.backbone.tokenizer.sensor_embed.parameters():
                p.requires_grad = True
            for p in self.backbone.tokenizer.quantizer.parameters():
                p.requires_grad = True
            for p in self.backbone.projection.parameters():
                p.requires_grad = True
            for block in self.backbone.blocks[:-1]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.backbone.blocks[-1].parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unknown TransferMode: {self.mode}")

    def _tokenize_with_grad(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable tokenization path for `end_to_end_ft` and `control_a_random_e2e`."""
        tok = self.backbone.tokenizer
        x_unf = tok.unfold(x, overlap_ratio=self.backbone.overlap_ratio)
        sensor_embedding = tok.sensor_embed(pos, sensor_type)
        feature = tok.encoder(x_unf, sensor_embedding)
        if self.mode == "control_a_random_e2e" and self.bypass_vq_in_control_a:
            commitment_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            feature, _indices, commitment_loss = tok.quantizer(feature)
        feature = rearrange(feature, "B C N T D -> B C (N T) D")
        return feature, commitment_loss

    def encode_features(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized spatiotemporal features (B, C_latent, W, D) and auxiliary VQ loss."""
        use_tok_grad = self.mode in ("end_to_end_ft", "control_a_random_e2e") and self.training
        if use_tok_grad:
            feat, vq_loss = self._tokenize_with_grad(x, pos, sensor_type)
            c_lat = feat.shape[1]
            neuro = self.backbone.tokenizer.encoder.neuros.type_as(feat).view(1, c_lat, 1, -1)
        else:
            feat, _ = self.backbone.tokenizer.tokenize(
                x, pos, sensor_type, self.backbone.overlap_ratio
            )
            vq_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            c_lat = feat.shape[1]
            neuro = (
                self.backbone.tokenizer.encoder.neuros.type_as(feat)
                .detach()
                .view(1, c_lat, 1, -1)
            )

        h = feat + neuro
        h = self.backbone.projection(h)
        for block in self.backbone.blocks[:-1]:
            h = block(h)
        h = F.normalize(h, p=2.0, dim=-1, eps=1e-6)
        return h, vq_loss

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "frozen":
            self.backbone.eval()
        elif self.mode in ("partial_ft", "backbone_ft", "control_b_random_backbone"):
            self.backbone.tokenizer.eval()

        h, vq_loss = self.encode_features(x, pos, sensor_type)
        pooled = h.mean(dim=2).reshape(h.shape[0], -1)
        logits = self.class_head(pooled)
        out = {"logits": logits, "embeddings": pooled}
        if labels is not None:
            ce_loss = self.criterion(logits, labels.long())
            out["loss"] = ce_loss + 0.1 * vq_loss
            out["ce_loss"] = ce_loss.detach()
        return out
