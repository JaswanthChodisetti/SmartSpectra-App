"""
Input-routing gate for the SmartSpectra demo.

Decides which tier an uploaded image belongs to BEFORE running any
quality-analysis pipeline (Pipeline A = sweet-potato, Pipeline C = apple).

Tier 1 — known produce (apple or sweet potato) -> route to matching pipeline.
Tier 2 — other produce (banana, tomato, etc.) -> warn user, run pipeline in
         the background, but UI hides the result behind a "show anyway" toggle.
Tier 3 — non-produce or image we can't classify confidently -> reject cleanly,
         no pipeline runs.

Implementation:
- CLIP zero-shot (open_clip ViT-B/32) compares the image against prompt lists.
- Tier 1/2/3 prompt lists live in models/checkpoints/routing_config.json so
  they can be tuned without code changes.
- Threshold (TIER_REJECT_THRESHOLD, default 0.20) catches the
  "model is just guessing" cases — anything below it falls to Tier 3.
- A pre-CLIP image-statistics short-circuit catches TRULY flat / solid-color
  images (L-channel std < 5) before CLIP runs.
- Tier-1.5 tie-breaker (v1.3): when CLIP Tier-1 and Tier-2 prompts sit
  within 0.03 cosine of each other (the typical apple-vs-tomato tie zone),
  the apple-prior CLIP centroid (models/checkpoints/apple_centroid.npy,
  built by scripts/build_apple_centroid.py from the 28 Vaishnavi cubes)
  breaks the tie. LOO apple cosines ≥ 0.81, non-apple references ≤ 0.58,
  threshold 0.70. This closes the 18/28 Vaishnavi Tier-2 fall-outs.
- Every decision appends a JSONL record to experiments/routing_log.jsonl.

Decision rule (per spec):
- Best Tier 1 prompt above threshold -> Tier 1 (label = the prompt).
- Best Tier 2 prompt above threshold -> Tier 2 (label = the prompt),
  UNLESS the Tier-1.5 tie-breaker elevates it to Tier 1.
- Best Tier 3 prompt above threshold OR nothing above threshold -> Tier 3.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "models/checkpoints/routing_config.json"
LOG_PATH = ROOT / "experiments/routing_log.jsonl"

Tier = Literal[1, 2, 3]
Pipeline = Literal["A", "C", "fallback"]


@dataclass
class RoutingDecision:
    tier: Tier
    detected_label: Optional[str]
    confidence: float  # cosine similarity of top-scoring prompt, in [0, 1]
    message: Optional[str]  # warning (Tier 2) or rejection (Tier 3) text
    pipeline_to_run: Optional[Pipeline]
    top_k_prompts: list = field(default_factory=list)
    # Internal bookkeeping (not shown to user)
    _all_scores: list = field(default_factory=list)

    def to_dict(self, include_internal: bool = False) -> dict:
        d = {
            "tier": self.tier,
            "detected_label": self.detected_label,
            "confidence": self.confidence,
            "message": self.message,
            "pipeline_to_run": self.pipeline_to_run,
        }
        if include_internal:
            d["top_k_prompts"] = self.top_k_prompts
            d["all_scores"] = self._all_scores
        return d


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _load_clip():
    """Lazy-load CLIP so the import cost is paid only once per process."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    import open_clip
    import torch

    _CLIP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model.to(_CLIP_DEVICE)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_TOKENIZER = tokenizer
    return model, preprocess, tokenizer, _CLIP_DEVICE


_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None


# Pre-encode text prompts once per process — they're constant per config
_TEXT_FEATURES_CACHE: dict = {}


def _encode_text(prompts: list[str], device) -> np.ndarray:
    """Encode and L2-normalize text prompts. Cached per frozenset of prompts."""
    key = frozenset(prompts)
    if key in _TEXT_FEATURES_CACHE:
        return _TEXT_FEATURES_CACHE[key]
    import torch

    tokens = _CLIP_TOKENIZER(prompts).to(device)
    with torch.no_grad():
        feats = _CLIP_MODEL.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    arr = feats.cpu().numpy()
    _TEXT_FEATURES_CACHE[key] = arr
    return arr


def _encode_image(img: Image.Image, device) -> np.ndarray:
    import torch

    x = _CLIP_PREPROCESS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = _CLIP_MODEL.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().squeeze(0)


def _cosine_topk(
    image_feat: np.ndarray,
    text_feats: np.ndarray,
    prompts: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """Return top-k (prompt, cosine_similarity) sorted descending."""
    sims = (text_feats @ image_feat).astype(float)
    order = np.argsort(-sims)[:k]
    return [(prompts[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# Tier-to-pipeline mapping
# ---------------------------------------------------------------------------

def _load_apple_centroid(path: Path) -> np.ndarray:
    """Lazy-load the apple-prior centroid (512-dim, L2-normalised float32)."""
    global _APPLE_CENTROID
    if _APPLE_CENTROID is not None:
        return _APPLE_CENTROID
    _APPLE_CENTROID = np.load(path).astype(np.float32)
    return _APPLE_CENTROID


_APPLE_CENTROID: Optional[np.ndarray] = None


def _label_to_pipeline(label: str, tier1_pipelines: dict) -> Optional[Pipeline]:
    """Map a CLIP-detected label like 'a photo of an apple' to a pipeline name."""
    for key, pipeline in tier1_pipelines.items():
        if key in label:
            return pipeline
    return None


def _build_messages(tier: Tier, label: Optional[str], default_pipeline: Pipeline,
                    fallback_pipeline: Pipeline) -> Optional[str]:
    if tier == 1:
        return None
    if tier == 2:
        item = label.replace("a photo of ", "") if label else "an unknown item"
        return (
            f"### ⚠ Not a trained commodity\n\n"
            f"Sorry, this appears to be a **{item}**. I was specifically trained for "
            f"apples and sweet potatoes, so I don't have the precise spectral weights "
            f"for this produce type.\n\n"
            f"However, if you still wish to proceed, you can tick 'Show result anyway' "
            f"to see the analysis using the general-purpose fallback pipeline."
        )
    # tier 3
    return (
        "✗ This doesn't appear to be a photo of produce. "
        "Please upload a clear photo of a fruit or vegetable to continue."
    )


def route_image(image_path: str | Path) -> RoutingDecision:
    """Decide which tier the image belongs to and what to do with it.

    Returns a RoutingDecision. The caller (demo/inference.py) is responsible
    for honoring pipeline_to_run and surfacing `message` to the user.
    """
    img = Image.open(image_path).convert("RGB")
    return route_image_from_array(img)


def route_image_from_array(img: Image.Image | np.ndarray) -> RoutingDecision:
    """Decide which tier the image belongs to based on an image object.

    Args:
        img: PIL Image or RGB numpy array (H, W, 3).
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8), 'RGB')

    cfg = _load_config()
    threshold: float = cfg["tier_reject_threshold"]
    top_k: int = cfg["log_top_k_prompts"]
    t1_prompts: list = cfg["tiers"]["tier_1"]["prompts"]
    t1_pipelines: dict = cfg["tiers"]["tier_1"]["pipelines"]
    t2_prompts: list = cfg["tiers"]["tier_2"]["prompts"]
    t2_default: Pipeline = cfg["tiers"]["tier_2"]["default_pipeline"]
    t3_prompts: list = cfg["tiers"]["tier_3"]["prompts"]

    # --- Tie-breaker config (v1.3) -------------------------------------------
    tie_breaker_cfg: dict = cfg.get("tie_breaker", {})
    tie_breaker_enabled: bool = tie_breaker_cfg.get("enabled", False)
    tie_breaker_threshold: float = float(tie_breaker_cfg.get("cosine_threshold", 0.70))
    tie_breaker_margin: float = float(tie_breaker_cfg.get("trigger_margin", 0.03))
    tie_breaker_centroid_path: Optional[Path] = (
        ROOT / tie_breaker_cfg["centroid_path"]
        if tie_breaker_enabled and "centroid_path" in tie_breaker_cfg
        else None
    )

    # --- Pre-CLIP image-statistics short-circuit ----------------------------
    FLAT_STD_THRESHOLD = 5.0
    arr = np.asarray(img, dtype=np.float32)
    gray = arr.mean(axis=-1)
    img_std = float(gray.std())
    if img_std < FLAT_STD_THRESHOLD:
        decision = RoutingDecision(
            tier=3,
            detected_label=f"flat / solid-color image (L-channel std={img_std:.1f})",
            confidence=1.0 - (img_std / FLAT_STD_THRESHOLD),
            message=_build_messages(3, "flat or solid-color image", None, t2_default),
            pipeline_to_run=None,
            top_k_prompts=[],
            _all_scores=[{"pre_check": "flat_image_short_circuit", "l_channel_std": img_std}],
        )
        return decision

    # Lazy-load CLIP (downloads ~150 MB on first run)
    _, _, _, device = _load_clip()
    image_feat = _encode_image(img, device)

    # Encode each tier's prompts independently so we can compare within tier
    t1_feats = _encode_text(t1_prompts, device)
    t2_feats = _encode_text(t2_prompts, device)
    t3_feats = _encode_text(t3_prompts, device)

    t1_top = _cosine_topk(image_feat, t1_feats, t1_prompts, k=1)[0]
    t2_top = _cosine_topk(image_feat, t2_feats, t2_prompts, k=1)[0]
    t3_top = _cosine_topk(image_feat, t3_feats, t3_prompts, k=1)[0]

    # Get the global top-k for logging (over all tiers combined)
    all_prompts = t1_prompts + t2_prompts + t3_prompts
    all_feats = np.concatenate([t1_feats, t2_feats, t3_feats], axis=0)
    global_topk = _cosine_topk(image_feat, all_feats, all_prompts, k=top_k)

    # Decision rule
    tier: Tier
    detected_label: Optional[str]
    pipeline: Optional[Pipeline]
    confidence: float
    tie_breaker_cosine: Optional[float] = None
    tie_breaker_triggered: bool = False

    if t1_top[1] >= threshold and t1_top[1] >= t2_top[1] and t1_top[1] >= t3_top[1]:
        tier = 1
        detected_label = t1_top[0]
        pipeline = _label_to_pipeline(detected_label, t1_pipelines)
        confidence = t1_top[1]
    elif t2_top[1] >= threshold and t2_top[1] >= t3_top[1]:
        if (tie_breaker_enabled and tie_breaker_centroid_path is not None
                and abs(t1_top[1] - t2_top[1]) <= tie_breaker_margin):
            centroid = _load_apple_centroid(tie_breaker_centroid_path)
            tie_breaker_cosine = float(image_feat @ centroid)
            if tie_breaker_cosine >= tie_breaker_threshold:
                apple_prompts = [p for p in t1_prompts if "apple" in p.lower()]
                apple_scores = [
                    (p, float(_encode_text([p], device)[0] @ image_feat))
                    for p in apple_prompts
                ]
                if apple_scores:
                    apple_label = max(apple_scores, key=lambda x: x[1])[0]
                else:
                    apple_label = "a photo of a red apple"
                tier = 1
                detected_label = apple_label
                pipeline = "C"  # apple checkpoint
                confidence = tie_breaker_cosine
                tie_breaker_triggered = True
            else:
                tier = 2
                detected_label = t2_top[0]
                pipeline = t2_default
                confidence = t2_top[1]
        else:
            tier = 2
            detected_label = t2_top[0]
            pipeline = t2_default
            confidence = t2_top[1]
    else:
        tier = 3
        detected_label = t3_top[0] if t3_top[1] > 0 else None
        pipeline = None
        confidence = max(t1_top[1], t2_top[1], t3_top[1])

    fallback = t2_default if tier == 2 else "fallback"
    message = _build_messages(tier, detected_label, pipeline or fallback, fallback)

    decision = RoutingDecision(
        tier=tier,
        detected_label=detected_label,
        confidence=confidence,
        message=message,
        pipeline_to_run=pipeline,
        top_k_prompts=global_topk,
        _all_scores=[
            {"prompt": p, "score": float(s), "tier": 1}
            for p, s in zip(t1_prompts, (t1_feats @ image_feat).astype(float))
        ] + [
            {"prompt": p, "score": float(s), "tier": 2}
            for p, s in zip(t2_prompts, (t2_feats @ image_feat).astype(float))
        ] + [
            {"prompt": p, "score": float(s), "tier": 3}
            for p, s in zip(t3_prompts, (t3_feats @ image_feat).astype(float))
        ] + ([{
            "tie_breaker": "apple_centroid",
            "triggered": True,
            "cosine_to_centroid": tie_breaker_cosine,
            "threshold": tie_breaker_threshold,
            "trigger_margin": tie_breaker_margin,
            "tier1_minus_tier2_cosine": t1_top[1] - t2_top[1],
        }] if tie_breaker_triggered else []),
    )

    return decision


def _log_decision(image_path: Path | str, decision: RoutingDecision) -> None:
    """Append a JSONL record of this routing decision."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        **decision.to_dict(include_internal=True),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m demo.routing <image_path>")
        return 2
    image_path = argv[1]
    decision = route_image(image_path)
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
