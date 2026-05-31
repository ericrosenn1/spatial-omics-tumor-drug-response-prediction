"""
Script:
    03_build_histology_teacher.py

Purpose:
    Build histology teacher scores for Visium samples.

Role:
    Image teacher step in teacher_builder.
    Uses trained treatment-conditioned histology model.
    Scores Visium tissue_hires_image images where available.
    Skips samples without usable images.
    No fusion here.

Pipeline position:
    01_validate_teacher_inputs.py
        checks paths, model artifacts, sample table

    02_build_expression_teacher.py
        builds sample x drug expression teacher table

    03_build_histology_teacher.py
        builds sample x drug histology teacher table

    04_fuse_teacher_tables.py
        combines expression and histology teachers

Expected YAML fields:
    project_dir:
        project root

    metadata_table:
        sample metadata from spatial pipeline

    sample_col:
        sample_id

    dataset_col:
        dataset_id

    cancer_col:
        cancer_type

    slides_dir:
        optional standardized Visium slide folder

    raw_visium_dir:
        optional raw Visium file tree

    histology_model_dir:
        folder containing best_model.pt, config.json,
        response_encoder.json, treatment_encoder.json

    output_root:
        teacher_builder output root

    hires_image_name:
        tissue_hires_image.png

    hires_image_patterns:
        optional image search patterns

    tile_size:
        256

    stride:
        256

    white_threshold:
        220

    min_tissue_fraction:
        0.20

    max_tiles_per_slide:
        500

    image_size:
        224

    batch_size:
        32

    device:
        auto

    save_tile_predictions:
        false or true

Outputs:
    outputs/03_histology_teacher/
        histology_teacher_scores.tsv
        visium_histology_slide_scores.tsv
        histology_teacher_slide_manifest.tsv
        histology_teacher_treatment_summary.tsv
        skipped_histology_samples.tsv
        histology_teacher_summary.txt

Notes:
    sample_id and slide_id are both written.
    slide_id is set equal to sample_id for downstream compatibility.
    Missing hires image does not fail the script.
    Empty histology output is allowed so fusion can use expression only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import gzip
import hashlib
import json
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import yaml


# ============================================================
# CONFIG HELPERS
# ============================================================

def parse_args():
    """parse CLI args"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_config(path: Path) -> dict:
    """load yaml config"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(project_dir: Path, value: str | Path | None) -> Path | None:
    """resolve absolute or project-relative path"""
    if value in [None, ""]:
        return None

    p = Path(value)

    if p.is_absolute():
        return p

    return project_dir / p


def get_cfg(cfg: dict, key: str, default: Any = None) -> Any:
    """get config value with fallback"""
    return cfg[key] if key in cfg else default


def ensure_dir(path: Path) -> None:
    """make folder if missing"""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, lines: list[str]) -> None:
    """write small text report"""
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# TEXT / NUMERIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    """clean text value"""
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_key(x: Any) -> str:
    """normalize drug or treatment key"""
    x = clean_text(x).lower()
    return " ".join(x.split())


def safe_filename(x: Any) -> str:
    """safe filename token"""
    text = clean_text(x)

    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', " "]
    for ch in bad:
        text = text.replace(ch, "_")

    return text[:180]


def clip01(x: float) -> float:
    """clip probability-like value"""
    return float(np.clip(float(x), 0.0, 1.0))


def compute_confidence_from_prob(prob: float) -> float:
    """confidence from distance to 0.5"""
    if pd.isna(prob):
        return np.nan

    return clip01(abs(float(prob) - 0.5) * 2.0)


def compute_ci_from_confidence(score: float, confidence: float) -> tuple[float, float]:
    """simple confidence interval proxy"""
    if pd.isna(score) or pd.isna(confidence):
        return np.nan, np.nan

    half_width = 0.5 * (1.0 - float(confidence))
    low = clip01(float(score) - half_width)
    high = clip01(float(score) + half_width)

    return low, high


# ============================================================
# IO HELPERS
# ============================================================

def load_table(path: Path) -> pd.DataFrame:
    """load csv or tsv table"""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)

    return pd.read_csv(path, low_memory=False)


def load_json(path: Path) -> dict:
    """load json file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_setting: str) -> torch.device:
    """choose torch device"""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_setting)


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def default_image_patterns() -> list[str]:
    """default hires image patterns"""
    return [
        "tissue_hires_image.png",
        "*tissue_hires_image.png",
        "*tissue_hires_image.png.gz",
        "*hires_image.png",
        "*hires_image.png.gz",
    ]


def extract_gz_image(gz_path: Path, cache_dir: Path) -> Path:
    """extract png.gz image into cache"""
    ensure_dir(cache_dir)

    digest = hashlib.md5(str(gz_path).encode("utf-8")).hexdigest()[:10]
    out_name = gz_path.with_suffix("").name
    out_path = cache_dir / f"{digest}__{out_name}"

    # keep existing extracted copy
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # stream extraction
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return out_path


def build_image_index(roots: list[Path], patterns: list[str]) -> pd.DataFrame:
    """index candidate hires images"""
    rows = []
    seen = set()

    for root in roots:
        if root is None or not root.exists():
            continue

        # search each root once per pattern
        for pat in patterns:
            for path in sorted(root.rglob(pat)):
                if not path.is_file():
                    continue

                key = str(path.resolve())
                if key in seen:
                    continue

                seen.add(key)

                rows.append({
                    "image_path": str(path),
                    "image_name": path.name,
                    "image_parent": path.parent.name,
                    "image_path_lower": str(path).lower(),
                    "root": str(root),
                })

    return pd.DataFrame(rows)


def direct_standard_image_paths(sample_id: str, slides_dir: Path | None, hires_name: str) -> list[Path]:
    """standard slide image candidates"""
    paths = []

    if slides_dir is None:
        return paths

    # current sample naming
    paths.append(slides_dir / sample_id / "spatial" / hires_name)

    # old 10-slide naming support
    paths.append(slides_dir / sample_id / hires_name)

    return paths


def sample_tokens(row: pd.Series, sample_col: str) -> list[str]:
    """tokens used for image matching"""
    tokens = set()

    # always include sample id
    sid = clean_text(row.get(sample_col, ""))
    if sid:
        tokens.add(sid.lower())

    # collect likely GEO and original-name fields
    for col, value in row.items():
        text = clean_text(value)
        low = text.lower()

        if not text:
            continue

        if "gsm" in low or "gse" in low:
            tokens.add(Path(text).name.lower())
            tokens.add(Path(text).stem.lower())

        if any(k in col.lower() for k in ["original", "loaded", "file", "path", "sample", "slide"]):
            tokens.add(Path(text).name.lower())
            tokens.add(Path(text).stem.lower())

    # add split tokens from filenames
    expanded = set()
    for token in tokens:
        expanded.add(token)

        for part in token.replace("-", "_").split("_"):
            if len(part) >= 5:
                expanded.add(part)

    return sorted(expanded)

def infer_raw_prefix_by_barcode_count(row: pd.Series, raw_dataset_dir: Path) -> str | None:
    """match mtx_manual sample to raw prefix by barcode count"""
    target = pd.to_numeric(row.get("filtering__spots_before"), errors="coerce")

    if pd.isna(target):
        return None

    matches = []

    for path in sorted(raw_dataset_dir.glob("*barcodes.tsv")):
        if not path.is_file():
            continue

        count = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))

        if int(count) == int(target):
            prefix = path.name.replace("_barcodes.tsv", "")
            matches.append(prefix)

    for path in sorted(raw_dataset_dir.glob("*barcodes.tsv.gz")):
        if not path.is_file():
            continue

        import gzip
        count = sum(1 for _ in gzip.open(path, "rt", encoding="utf-8", errors="ignore"))

        if int(count) == int(target):
            prefix = path.name.replace("_barcodes.tsv.gz", "")
            matches.append(prefix)

    matches = sorted(set(matches))

    if len(matches) == 1:
        return matches[0]

    return None


def match_image_for_sample(
    row: pd.Series,
    sample_col: str,
    slides_dir: Path | None,
    hires_name: str,
    image_index: pd.DataFrame,
    raw_visium_dir: Path | None = None,
) -> tuple[Path | None, str]:
    """find best image for one sample"""
    sample_id = clean_text(row[sample_col])

    # standard direct checks
    for path in direct_standard_image_paths(sample_id, slides_dir, hires_name):
        if path.exists():
            return path, "standard_path"

    # strict raw prefix match for raw GEO folders
    dataset_id = clean_text(row.get("dataset_id", ""))
    loaded_from = clean_text(row.get("loaded_from", ""))

    if raw_visium_dir is not None and dataset_id:
        raw_dataset_dir = Path(raw_visium_dir) / f"{dataset_id}_RAW"

        if raw_dataset_dir.exists():
            prefix = None

            # h5 samples usually preserve the GSM prefix in loaded_from
            if loaded_from and loaded_from != "mtx_manual":
                prefix = loaded_from
                prefix = prefix.replace("_raw_feature_bc_matrix.h5", "")
                prefix = prefix.replace("_filtered_feature_bc_matrix.h5", "")
                prefix = prefix.replace("_matrix.mtx.gz", "")
                prefix = prefix.replace("_matrix.mtx", "")

            # mtx_manual samples need recovery by barcode count
            if not prefix:
                prefix = infer_raw_prefix_by_barcode_count(row, raw_dataset_dir)

            if prefix:
                candidates = [
                    raw_dataset_dir / f"{prefix}_tissue_hires_image.png",
                    raw_dataset_dir / f"{prefix}_tissue_hires_image.png.gz",
                    raw_dataset_dir / f"{prefix}_visium_tissue_hires_image.png",
                    raw_dataset_dir / f"{prefix}_visium_tissue_hires_image.png.gz",
                ]

                for path in candidates:
                    if path.exists():
                        return path, "raw_prefix_match"

    # indexed raw search
    if image_index.empty:
        return None, "not_found"

    tokens = sample_tokens(row, sample_col)
    scored = []

    for _, img in image_index.iterrows():
        low_path = img["image_path_lower"]
        name = img["image_name"].lower()

        score = 0
        for token in tokens:
            if token and token in low_path:
                score += 1

        if "tissue_hires" in name:
            score += 3
        elif "hires" in name:
            score += 2

        if score > 0:
            scored.append((score, len(low_path), Path(img["image_path"])))

    if not scored:
        return None, "not_found"

    # highest match, shorter path second
    scored = sorted(scored, key=lambda x: (-x[0], x[1]))
    return scored[0][2], "indexed_match"


def discover_sample_images(
    metadata_df: pd.DataFrame,
    cfg: dict,
    image_cache_dir: Path,
) -> pd.DataFrame:
    """build sample to image manifest"""
    project_dir = Path(cfg["project_dir"])
    sample_col = cfg["sample_col"]

    slides_dir = resolve_path(project_dir, get_cfg(cfg, "slides_dir", "slides"))
    raw_visium_dir = resolve_path(project_dir, get_cfg(cfg, "raw_visium_dir", "raw_visium_new"))

    hires_name = get_cfg(cfg, "hires_image_name", "tissue_hires_image.png")
    patterns = get_cfg(cfg, "hires_image_patterns", default_image_patterns())

    roots = [p for p in [slides_dir, raw_visium_dir] if p is not None]
    image_index = build_image_index(roots, patterns)

    rows = []

    for _, row in metadata_df.iterrows():
        sample_id = clean_text(row[sample_col])

        image_path, source = match_image_for_sample(
            row=row,
            sample_col=sample_col,
            slides_dir=slides_dir,
            hires_name=hires_name,
            image_index=image_index,
            raw_visium_dir=raw_visium_dir,
        )

        has_image = image_path is not None and image_path.exists()
        usable_path = None
        was_gz = False
        error = ""

        if has_image:
            try:
                if image_path.name.endswith(".gz"):
                    was_gz = True
                    usable_path = extract_gz_image(image_path, image_cache_dir)
                else:
                    usable_path = image_path
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                usable_path = None
                has_image = False

        rows.append({
            "sample_id": sample_id,
            "slide_id": sample_id,
            "dataset_id": row.get(cfg.get("dataset_col", "dataset_id"), ""),
            "cancer_type": row.get(cfg.get("cancer_col", "cancer_type"), ""),
            "hires_image_path": str(usable_path) if usable_path is not None else "",
            "source_image_path": str(image_path) if image_path is not None else "",
            "image_source": source,
            "image_was_gz": was_gz,
            "has_hires_image": bool(has_image and usable_path is not None),
            "image_error": error,
        })

    return pd.DataFrame(rows)


# ============================================================
# TILE HELPERS
# ============================================================

def is_tissue_tile(
    pil_img: Image.Image,
    white_threshold: int,
    min_tissue_fraction: float,
) -> bool:
    """detect tissue-rich tile"""
    arr = np.asarray(pil_img)

    if arr.ndim != 3 or arr.shape[2] < 3:
        return False

    rgb = arr[:, :, :3]

    # non-white pixel mask
    non_white = np.any(rgb < int(white_threshold), axis=2)
    tissue_fraction = float(non_white.mean())

    return tissue_fraction >= float(min_tissue_fraction)


def get_tile_coords(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """regular tile coordinate grid"""
    coords = []

    if width < tile_size or height < tile_size:
        return coords

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            coords.append((x, y))

    return coords


def extract_valid_tiles_from_image(image_path: Path, cfg: dict) -> list[dict[str, Any]]:
    """tile image and keep tissue tiles"""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    tile_size = int(get_cfg(cfg, "tile_size", 256))
    stride = int(get_cfg(cfg, "stride", 256))
    white_threshold = int(get_cfg(cfg, "white_threshold", 220))
    min_tissue_fraction = float(get_cfg(cfg, "min_tissue_fraction", 0.20))
    max_tiles = get_cfg(cfg, "max_tiles_per_slide", 500)

    if max_tiles in ["", None, "null"]:
        max_tiles = None
    else:
        max_tiles = int(max_tiles)

    coords = get_tile_coords(width, height, tile_size, stride)

    # random tile sampling, same seed controlled upstream
    random.shuffle(coords)

    valid_tiles = []

    for x, y in coords:
        if max_tiles is not None and len(valid_tiles) >= max_tiles:
            break

        tile = image.crop((x, y, x + tile_size, y + tile_size))

        if not is_tissue_tile(tile, white_threshold, min_tissue_fraction):
            continue

        valid_tiles.append({
            "x": x,
            "y": y,
            "tile_image": tile,
        })

    return valid_tiles


def build_eval_transform(cfg: dict) -> transforms.Compose:
    """model eval image transform"""
    image_size = int(get_cfg(cfg, "image_size", 224))

    mean = tuple(get_cfg(cfg, "normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(get_cfg(cfg, "normalize_std", [0.229, 0.224, 0.225]))

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ============================================================
# MODEL
# ============================================================

def build_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """build CNN backbone"""
    if backbone_name == "resnet18":
        backbone = models.resnet18(weights=None)
        out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, out_dim

    if backbone_name == "resnet50":
        backbone = models.resnet50(weights=None)
        out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, out_dim

    raise ValueError(f"Unsupported backbone_name: {backbone_name}")


class ConditionedHistologyModel(nn.Module):
    """image plus treatment conditioned classifier"""

    def __init__(
        self,
        n_treatments: int,
        n_classes: int,
        backbone_name: str,
        treatment_embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # image encoder
        self.backbone, image_feature_dim = build_backbone(backbone_name)

        # treatment condition
        self.treatment_embedding = nn.Embedding(
            num_embeddings=n_treatments,
            embedding_dim=treatment_embedding_dim,
        )

        fusion_dim = image_feature_dim + treatment_embedding_dim

        # response classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, image: torch.Tensor, treatment_id: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        image_features = self.backbone(image)
        treatment_features = self.treatment_embedding(treatment_id)

        fused = torch.cat([image_features, treatment_features], dim=1)
        logits = self.classifier(fused)

        return logits


def parse_encoder(raw: dict) -> tuple[dict[str, int], dict[int, str]]:
    """parse label encoder json"""
    if "item_to_idx" in raw:
        item_to_idx = {str(k): int(v) for k, v in raw["item_to_idx"].items()}
        idx_to_item = {int(k): str(v) for k, v in raw["idx_to_item"].items()}
        return item_to_idx, idx_to_item

    item_to_idx = {str(k): int(v) for k, v in raw.items()}
    idx_to_item = {int(v): str(k) for k, v in item_to_idx.items()}

    return item_to_idx, idx_to_item


def build_model_from_saved_config(
    model_dir: Path,
    treatment_item_to_idx: dict[str, int],
    response_item_to_idx: dict[str, int],
    cfg: dict,
) -> ConditionedHistologyModel:
    """rebuild model architecture"""
    config_path = model_dir / "config.json"

    saved_cfg = {}
    if config_path.exists():
        saved_cfg = load_json(config_path)

    backbone_name = saved_cfg.get("backbone_name", get_cfg(cfg, "backbone_name", "resnet18"))
    treatment_embedding_dim = int(saved_cfg.get("treatment_embedding_dim", get_cfg(cfg, "treatment_embedding_dim", 32)))
    hidden_dim = int(saved_cfg.get("hidden_dim", get_cfg(cfg, "hidden_dim", 256)))
    dropout = float(saved_cfg.get("dropout", get_cfg(cfg, "dropout", 0.30)))

    return ConditionedHistologyModel(
        n_treatments=len(treatment_item_to_idx),
        n_classes=len(response_item_to_idx),
        backbone_name=backbone_name,
        treatment_embedding_dim=treatment_embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def load_checkpoint_state(checkpoint: Any) -> dict:
    """extract model state dict"""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError("Unsupported checkpoint format")


def load_model_and_encoders(cfg: dict):
    """load model, encoders, and device"""
    project_dir = Path(cfg["project_dir"])
    model_dir = resolve_path(project_dir, cfg["histology_model_dir"])

    response_encoder_path = model_dir / "response_encoder.json"
    treatment_encoder_path = model_dir / "treatment_encoder.json"
    checkpoint_path = model_dir / "best_model.pt"

    if not response_encoder_path.exists():
        raise FileNotFoundError(response_encoder_path)

    if not treatment_encoder_path.exists():
        raise FileNotFoundError(treatment_encoder_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    # encoders from training run
    response_raw = load_json(response_encoder_path)
    treatment_raw = load_json(treatment_encoder_path)

    response_item_to_idx, response_idx_to_item = parse_encoder(response_raw)
    treatment_item_to_idx, treatment_idx_to_item = parse_encoder(treatment_raw)

    responder_idx = None
    for label, idx in response_item_to_idx.items():
        if clean_text(label).upper() == "RESPONDER":
            responder_idx = int(idx)
            break

    if responder_idx is None:
        raise ValueError(f"RESPONDER class not found: {response_item_to_idx}")

    model = build_model_from_saved_config(
        model_dir=model_dir,
        treatment_item_to_idx=treatment_item_to_idx,
        response_item_to_idx=response_item_to_idx,
        cfg=cfg,
    )

    device = get_device(get_cfg(cfg, "device", "auto"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = load_checkpoint_state(checkpoint)

    # Compatibility patch for histology_response_model_v2 checkpoints.
    # Some v2 checkpoints store the treatment embedding as treat_emb.weight,
    # while this older teacher_builder scorer expects treatment_embedding.weight.
    if isinstance(state, dict):
        if "treat_emb.weight" in state and "treatment_embedding.weight" not in state:
            state["treatment_embedding.weight"] = state["treat_emb.weight"]
        if "treatment_embedding.weight" in state and "treat_emb.weight" not in state:
            state["treat_emb.weight"] = state["treatment_embedding.weight"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return (
        model,
        device,
        response_item_to_idx,
        response_idx_to_item,
        treatment_item_to_idx,
        treatment_idx_to_item,
        responder_idx,
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def score_tiles_for_one_treatment(
    model: nn.Module,
    tiles: list[dict[str, Any]],
    treatment_id: int,
    responder_idx: int,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    """score tiles for one treatment"""
    if len(tiles) == 0:
        return pd.DataFrame(), np.nan

    tensor_list = []
    meta_rows = []

    # transform each tile once
    for item in tiles:
        tensor_list.append(transform(item["tile_image"]))
        meta_rows.append({"x": item["x"], "y": item["y"]})

    probs = []
    preds = []

    # batched inference
    for start in range(0, len(tensor_list), batch_size):
        end = start + batch_size

        batch_imgs = torch.stack(tensor_list[start:end]).to(device)
        batch_treatment = torch.full(
            (batch_imgs.shape[0],),
            fill_value=int(treatment_id),
            dtype=torch.long,
            device=device,
        )

        logits = model(batch_imgs, batch_treatment)
        batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        batch_preds = np.argmax(batch_probs, axis=1)

        responder_probs = batch_probs[:, responder_idx]

        probs.extend(responder_probs.tolist())
        preds.extend(batch_preds.tolist())

    tile_df = pd.DataFrame(meta_rows)
    tile_df["tile_prob_responder"] = probs
    tile_df["tile_pred_class"] = preds
    tile_df["tile_confidence"] = tile_df["tile_prob_responder"].apply(compute_confidence_from_prob)

    slide_prob = float(np.mean(probs))

    return tile_df, slide_prob


def score_one_sample(
    sample_row: pd.Series,
    cfg: dict,
    model: nn.Module,
    device: torch.device,
    treatment_item_to_idx: dict[str, int],
    responder_idx: int,
    transform: transforms.Compose,
    tile_pred_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """score one Visium image across treatments"""
    sample_id = clean_text(sample_row["sample_id"])
    image_path = Path(sample_row["hires_image_path"])

    batch_size = int(get_cfg(cfg, "batch_size", 32))
    save_tile_predictions = bool(get_cfg(cfg, "save_tile_predictions", False))

    print(f"\nScoring sample: {sample_id}")
    print(f"Image: {image_path}")

    valid_tiles = extract_valid_tiles_from_image(image_path, cfg)
    n_tiles = len(valid_tiles)

    print(f"Valid tiles kept: {n_tiles}")

    sample_manifest = sample_row.to_dict()
    sample_manifest["n_tiles_kept"] = n_tiles

    if n_tiles == 0:
        sample_manifest["histology_status"] = "no_valid_tiles"
        return [], sample_manifest

    treatment_rows = []

    for treatment_key, treatment_id in sorted(treatment_item_to_idx.items(), key=lambda x: x[1]):
        tile_df, slide_prob = score_tiles_for_one_treatment(
            model=model,
            tiles=valid_tiles,
            treatment_id=treatment_id,
            responder_idx=responder_idx,
            transform=transform,
            device=device,
            batch_size=batch_size,
        )

        if pd.isna(slide_prob):
            continue

        hist_conf = compute_confidence_from_prob(slide_prob)
        ci_low, ci_high = compute_ci_from_confidence(slide_prob, hist_conf)

        drug = clean_text(treatment_key)
        drug_key = normalize_key(treatment_key)

        treatment_rows.append({
            "sample_id": sample_id,
            "slide_id": sample_id,
            "dataset_id": sample_row.get("dataset_id", ""),
            "cancer_type": sample_row.get("cancer_type", ""),
            "drug": drug,
            "drug_key": drug_key,
            "histology_prob_responder": slide_prob,
            "histology_confidence": hist_conf,
            "histology_sample_confidence": hist_conf,
            "histology_ci_low": ci_low,
            "histology_ci_high": ci_high,
            "n_tiles_used": n_tiles,
            "histology_available": True,
            "histology_model_weight": float(get_cfg(cfg, "histology_model_weight", 0.50)),
            "modality_used": "histology_only",
            "hires_image_path": str(image_path),
        })

        if save_tile_predictions:
            tmp = tile_df.copy()
            tmp["sample_id"] = sample_id
            tmp["slide_id"] = sample_id
            tmp["drug"] = drug
            tmp["drug_key"] = drug_key

            out_name = f"{safe_filename(sample_id)}__{safe_filename(drug_key)}_tile_predictions.tsv"

            tmp.to_csv(
                tile_pred_dir / out_name,
                sep="\t",
                index=False,
            )

    sample_manifest["histology_status"] = "scored"

    return treatment_rows, sample_manifest


# ============================================================
# OUTPUT HELPERS
# ============================================================

def empty_score_table() -> pd.DataFrame:
    """empty histology score schema"""
    return pd.DataFrame(columns=[
        "sample_id",
        "slide_id",
        "dataset_id",
        "cancer_type",
        "drug",
        "drug_key",
        "histology_prob_responder",
        "histology_confidence",
        "histology_sample_confidence",
        "histology_ci_low",
        "histology_ci_high",
        "n_tiles_used",
        "histology_available",
        "histology_model_weight",
        "modality_used",
        "hires_image_path",
    ])


def build_treatment_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """summarize histology scores by drug"""
    if scores.empty:
        return pd.DataFrame(columns=[
            "drug_key",
            "drug",
            "n_slides",
            "mean_histology_prob_responder",
            "mean_histology_confidence",
            "mean_tiles_used",
        ])

    return (
        scores.groupby("drug_key", as_index=False)
        .agg(
            drug=("drug", "first"),
            n_slides=("slide_id", "nunique"),
            mean_histology_prob_responder=("histology_prob_responder", "mean"),
            mean_histology_confidence=("histology_confidence", "mean"),
            mean_tiles_used=("n_tiles_used", "mean"),
        )
        .sort_values(["mean_histology_confidence", "n_slides"], ascending=[False, False])
    )


def write_summary(
    out_dir: Path,
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    treatment_summary: pd.DataFrame,
    skipped: pd.DataFrame,
    cfg: dict,
) -> None:
    """write text summary"""
    lines = []
    lines.append("Histology teacher summary")
    lines.append("")
    lines.append(f"metadata_table: {cfg.get('metadata_table')}")
    lines.append(f"histology_model_dir: {cfg.get('histology_model_dir')}")
    lines.append("")
    lines.append(f"samples in metadata: {len(manifest)}")
    lines.append(f"samples with hires image: {int(manifest['has_hires_image'].sum()) if 'has_hires_image' in manifest.columns else 0}")
    lines.append(f"samples scored: {scores['sample_id'].nunique() if not scores.empty else 0}")
    lines.append(f"samples skipped: {len(skipped)}")
    lines.append(f"histology rows: {len(scores)}")
    lines.append(f"drugs scored: {scores['drug_key'].nunique() if not scores.empty else 0}")
    lines.append("")
    lines.append("Tile settings:")
    lines.append(f"  tile_size: {get_cfg(cfg, 'tile_size', 256)}")
    lines.append(f"  stride: {get_cfg(cfg, 'stride', 256)}")
    lines.append(f"  max_tiles_per_slide: {get_cfg(cfg, 'max_tiles_per_slide', 500)}")
    lines.append(f"  min_tissue_fraction: {get_cfg(cfg, 'min_tissue_fraction', 0.20)}")
    lines.append("")
    lines.append("Output role:")
    lines.append("  input to 04_fuse_teacher_tables.py")
    lines.append("  missing histology allowed")

    write_text(out_dir / "histology_teacher_summary.txt", lines)


# ============================================================
# MAIN
# ============================================================

def main():
    """run histology teacher builder"""
    args = parse_args()
    cfg = load_config(Path(args.config))

    project_dir = Path(cfg["project_dir"])
    cfg["project_dir"] = str(project_dir)

    random_seed = int(get_cfg(cfg, "random_state", 42))
    set_seed(random_seed)

    out_root = Path(cfg["output_root"])
    out_dir = out_root / "03_histology_teacher"
    ensure_dir(out_dir)

    cache_dir = out_dir / "extracted_images"
    tile_pred_dir = out_dir / "tile_predictions"

    if bool(get_cfg(cfg, "save_tile_predictions", False)):
        ensure_dir(tile_pred_dir)

    manifest_path = out_dir / "histology_teacher_slide_manifest.tsv"
    score_path = out_dir / "histology_teacher_scores.tsv"
    compat_score_path = out_dir / "visium_histology_slide_scores.tsv"
    treatment_summary_path = out_dir / "histology_teacher_treatment_summary.tsv"
    skipped_path = out_dir / "skipped_histology_samples.tsv"

    # optional skip mode
    if not bool(get_cfg(cfg, "run_histology_teacher", True)):
        scores = empty_score_table()
        scores.to_csv(score_path, sep="\t", index=False)
        scores.to_csv(compat_score_path, sep="\t", index=False)

        write_text(
            out_dir / "histology_teacher_summary.txt",
            ["Histology teacher skipped", "run_histology_teacher: false"],
        )

        print("Histology teacher skipped by config")
        return

    metadata_path = resolve_path(project_dir, cfg["metadata_table"])
    metadata_df = load_table(metadata_path)

    sample_col = cfg["sample_col"]

    if sample_col not in metadata_df.columns:
        raise ValueError(f"metadata missing sample_col: {sample_col}")

    metadata_df[sample_col] = metadata_df[sample_col].astype(str)

    # optional debug subset
    if bool(get_cfg(cfg, "test_mode", False)):
        n = int(get_cfg(cfg, "test_n_samples", 5))
        metadata_df = metadata_df.head(n).copy()

    print("Discovering histology images...")

    image_manifest = discover_sample_images(
        metadata_df=metadata_df,
        cfg=cfg,
        image_cache_dir=cache_dir,
    )

    allow_dupes = bool(get_cfg(cfg, "allow_duplicate_histology_images", False))

    if not allow_dupes and "source_image_path" in image_manifest.columns:
        dup_mask = (
            image_manifest["source_image_path"].astype(str).str.len().gt(0)
            & image_manifest.duplicated("source_image_path", keep=False)
        )

        if dup_mask.any():
            image_manifest.loc[dup_mask, "has_hires_image"] = False
            image_manifest.loc[dup_mask, "image_error"] = (
                "duplicate_source_image_path_rejected"
            )
            image_manifest.loc[dup_mask, "hires_image_path"] = ""

    image_manifest.to_csv(manifest_path, sep="\t", index=False)

    usable = image_manifest[image_manifest["has_hires_image"] == True].copy()
    skipped = image_manifest[image_manifest["has_hires_image"] != True].copy()

    skipped.to_csv(skipped_path, sep="\t", index=False)

    print(f"Samples in metadata: {len(image_manifest):,}")
    print(f"Samples with hires image: {len(usable):,}")
    print(f"Samples skipped: {len(skipped):,}")

    # no images, still write empty outputs
    if usable.empty:
        scores = empty_score_table()
        treatment_summary = build_treatment_summary(scores)

        scores.to_csv(score_path, sep="\t", index=False)
        scores.to_csv(compat_score_path, sep="\t", index=False)
        treatment_summary.to_csv(treatment_summary_path, sep="\t", index=False)

        write_summary(
            out_dir=out_dir,
            manifest=image_manifest,
            scores=scores,
            treatment_summary=treatment_summary,
            skipped=skipped,
            cfg=cfg,
        )

        print("No usable histology images found")
        print("Wrote empty histology teacher outputs")
        return

    print("Loading trained histology model and encoders...")

    (
        model,
        device,
        response_item_to_idx,
        response_idx_to_item,
        treatment_item_to_idx,
        treatment_idx_to_item,
        responder_idx,
    ) = load_model_and_encoders(cfg)

    print(f"Using device: {device}")
    print(f"Treatments in model: {len(treatment_item_to_idx):,}")
    print(f"Responder class index: {responder_idx}")

    transform = build_eval_transform(cfg)

    all_score_rows = []
    scored_manifest_rows = []

    for _, sample_row in usable.iterrows():
        try:
            score_rows, scored_manifest = score_one_sample(
                sample_row=sample_row,
                cfg=cfg,
                model=model,
                device=device,
                treatment_item_to_idx=treatment_item_to_idx,
                responder_idx=responder_idx,
                transform=transform,
                tile_pred_dir=tile_pred_dir,
            )

            all_score_rows.extend(score_rows)
            scored_manifest_rows.append(scored_manifest)

        except Exception as exc:
            failed = sample_row.to_dict()
            failed["histology_status"] = "failed"
            failed["histology_error"] = f"{type(exc).__name__}: {exc}"
            scored_manifest_rows.append(failed)

            print(f"FAILED sample {sample_row['sample_id']}: {failed['histology_error']}")

    scored_manifest_df = pd.DataFrame(scored_manifest_rows)

    # merge new status back into full manifest
    if not scored_manifest_df.empty:
        image_manifest = image_manifest.merge(
            scored_manifest_df[
                [
                    c for c in scored_manifest_df.columns
                    if c in ["sample_id", "n_tiles_kept", "histology_status", "histology_error"]
                ]
            ],
            on="sample_id",
            how="left",
        )

    scores = pd.DataFrame(all_score_rows)

    if scores.empty:
        scores = empty_score_table()

    treatment_summary = build_treatment_summary(scores)

    # write primary and compatibility names
    image_manifest.to_csv(manifest_path, sep="\t", index=False)
    scores.to_csv(score_path, sep="\t", index=False)
    scores.to_csv(compat_score_path, sep="\t", index=False)
    treatment_summary.to_csv(treatment_summary_path, sep="\t", index=False)

    write_summary(
        out_dir=out_dir,
        manifest=image_manifest,
        scores=scores,
        treatment_summary=treatment_summary,
        skipped=skipped,
        cfg=cfg,
    )

    print("\nDONE")
    print(f"Samples discovered: {len(image_manifest):,}")
    print(f"Samples with image: {len(usable):,}")
    print(f"Samples scored: {scores['sample_id'].nunique() if not scores.empty else 0:,}")
    print(f"Histology rows: {len(scores):,}")
    print(f"Drugs scored: {scores['drug_key'].nunique() if not scores.empty else 0:,}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote histology scores: {score_path}")
    print(f"Wrote compatibility scores: {compat_score_path}")
    print(f"Wrote treatment summary: {treatment_summary_path}")
    print(f"Wrote skipped samples: {skipped_path}")

    if bool(get_cfg(cfg, "save_tile_predictions", False)):
        print(f"Wrote tile predictions: {tile_pred_dir}")


if __name__ == "__main__":
    main()
