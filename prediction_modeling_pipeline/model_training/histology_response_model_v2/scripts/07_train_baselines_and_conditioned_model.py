"""
Script: 07_train_baselines_and_conditioned_model.py

Purpose:
    Train treatment_only, image_only, and image_treatment histology response models.

Pipeline role:
    Step 07 of histology_response_model_v2. This step applies slide QC,
    constructs patient/slide-balanced tile samples, trains neural response
    models, writes checkpoints and predictions, and compares image-conditioned
    performance against treatment identity and image-only baselines.

Scientific context:
    The central question is whether H&E image information adds held-out response
    signal beyond treatment identity alone. The treatment_only model is therefore
    an essential control, while image_only and image_treatment test morphology
    and morphology plus treatment context under patient-level splits.

Documentation safety:
    Documentation edits should not change executable behavior, thresholds, paths,
    schemas, model settings, or outputs.
"""


# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from histology_model_v2_lib import (
    load_yaml,
    output_root,
    ensure_dir,
    read_table,
    write_json,
    safe_auc,
    brier,
)



# =============================================================================
# Reproducibility and encoding helpers
# =============================================================================

# Reproducibility is important because tile sampling and neural training are stochastic.
def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_encoder(vals):
    """Create forward and inverse integer encoders for categorical values."""
    vals = sorted(set(str(v) for v in vals))
    return {v: i for i, v in enumerate(vals)}, {i: v for i, v in enumerate(vals)}


def load_img(path):
    """Open a tile image as RGB for model preprocessing."""
    return Image.open(path).convert("RGB")



# =============================================================================
# Slide QC loading and filtering
# =============================================================================

def load_slide_qc(cfg):
    """Load optional slide-level QC recommendations and quality weights."""
    qc_path = output_root(cfg) / "04_tiles" / "qc_fast" / "slide_tile_qc_fast.tsv"

    if not qc_path.exists():
        return pd.DataFrame(columns=["patient_id", "slide_id", "qc_recommendation", "quality_weight"])

    qc = pd.read_csv(qc_path, sep="\t", dtype=str, low_memory=False).fillna("")

    if "qc_recommendation" not in qc.columns:
        qc["qc_recommendation"] = "keep"

    if "quality_weight" not in qc.columns:
        qc["quality_weight"] = "1.0"

    keep_cols = [
        "patient_id",
        "slide_id",
        "qc_recommendation",
        "quality_weight",
        "n_tiles",
        "mean_tissue_fraction",
        "mean_sharpness",
        "qc_flag_count",
    ]

    keep_cols = [c for c in keep_cols if c in qc.columns]

    return qc[keep_cols].drop_duplicates(["patient_id", "slide_id"]).copy()


def attach_slide_qc_and_filter(df, cfg):
    """Attach slide QC fields and remove excluded or zero-weight slides."""
    qc = load_slide_qc(cfg)

    df = df.copy()

    if qc.empty:
        df["qc_recommendation"] = "keep"
        df["quality_weight"] = 1.0
        excluded = df.iloc[0:0].copy()
        return df, excluded

    df = df.merge(qc, on=["patient_id", "slide_id"], how="left")

    df["qc_recommendation"] = df["qc_recommendation"].fillna("keep").replace("", "keep")
    df["quality_weight"] = pd.to_numeric(df["quality_weight"], errors="coerce").fillna(1.0)

    excluded = df[
        (df["qc_recommendation"] == "exclude_candidate")
        | (df["quality_weight"] <= 0)
    ].copy()

    kept = df[
        ~(
            (df["qc_recommendation"] == "exclude_candidate")
            | (df["quality_weight"] <= 0)
        )
    ].copy()

    return kept, excluded



# =============================================================================
# Patient/slide-balanced tile sampling
# =============================================================================

def allocate_budget(slide_sizes, slide_weights, budget):
    """Allocate a per-patient tile budget across slides using slide quality weights."""
    slide_ids = list(slide_sizes.keys())

    budget = int(budget)

    if budget <= 0 or len(slide_ids) == 0:
        return {sid: 0 for sid in slide_ids}

    total_available = sum(int(v) for v in slide_sizes.values())

    if budget >= total_available:
        return {sid: int(slide_sizes[sid]) for sid in slide_ids}

    weights = np.array(
        [max(float(slide_weights.get(sid, 1.0)), 0.0) for sid in slide_ids],
        dtype=float,
    )

    if weights.sum() <= 0:
        weights = np.ones(len(slide_ids), dtype=float)

    weights = weights / weights.sum()

    raw = weights * budget
    alloc = np.floor(raw).astype(int)

    active = np.array([slide_sizes[sid] > 0 for sid in slide_ids], dtype=bool)

    if budget >= active.sum():
        for i, sid in enumerate(slide_ids):
            if slide_sizes[sid] > 0 and alloc[i] == 0:
                alloc[i] = 1

    for i, sid in enumerate(slide_ids):
        alloc[i] = min(int(alloc[i]), int(slide_sizes[sid]))

    remainder = budget - int(alloc.sum())

    frac_order = np.argsort(-(raw - np.floor(raw)))

    while remainder > 0:
        changed = False

        for i in frac_order:
            sid = slide_ids[i]

            if alloc[i] < slide_sizes[sid]:
                alloc[i] += 1
                remainder -= 1
                changed = True

                if remainder <= 0:
                    break

        if not changed:
            break

    return {sid: int(alloc[i]) for i, sid in enumerate(slide_ids)}


# Patient/slide-balanced sampling limits dominance by patients with many tiles or many slides.
def sample_patient_slide_balanced(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """Downsample tile rows per patient while preserving slide representation."""
    if cap is None or int(cap) <= 0:
        return df.copy()

    cap = int(cap)

    rng = np.random.default_rng(seed)
    pieces = []

    for patient_id, psub in df.groupby("patient_id", sort=False):
        if len(psub) <= cap:
            pieces.append(psub)
            continue

        slide_groups = {
            sid: sub
            for sid, sub in psub.groupby("slide_id", sort=False)
        }

        slide_sizes = {
            sid: len(sub)
            for sid, sub in slide_groups.items()
        }

        slide_weights = {}

        for sid, sub in slide_groups.items():
            if "quality_weight" in sub.columns:
                slide_weights[sid] = float(
                    pd.to_numeric(sub["quality_weight"], errors="coerce")
                    .fillna(1.0)
                    .median()
                )
            else:
                slide_weights[sid] = 1.0

        alloc = allocate_budget(slide_sizes, slide_weights, cap)

        patient_pieces = []

        for sid, n in alloc.items():
            if n <= 0:
                continue

            sub = slide_groups[sid]

            if len(sub) <= n:
                patient_pieces.append(sub)
            else:
                rs = int(rng.integers(0, 2**31 - 1))
                patient_pieces.append(sub.sample(n=n, random_state=rs))

        if patient_pieces:
            pieces.append(pd.concat(patient_pieces, ignore_index=True))

    if not pieces:
        return df.iloc[0:0].copy()

    return pd.concat(pieces, ignore_index=True)



# =============================================================================
# PyTorch dataset
# =============================================================================

class TileDS(Dataset):
    """
    PyTorch Dataset for tile-level histology response modeling.
    
    Depending on model_type, this dataset can return real image tiles or blank
    tensors for treatment-only controls, while preserving treatment IDs, response
    labels, patient IDs, slide IDs, treatment keys, and tile paths for auditing.
    """
    def __init__(self, df, transform, model_type, image_size):
        """Initialize the dataset or model object."""
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.model_type = model_type
        self.image_size = int(image_size)

    def __len__(self):
        """Return the number of rows available to the dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """Return one transformed tile example and its metadata."""
        r = self.df.iloc[i]

        if self.model_type == "treatment_only":
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        else:
            image = self.transform(load_img(r.tile_path))

        return {
            "image": image,
            "treatment_id": torch.tensor(int(r.treatment_id), dtype=torch.long),
            "label": torch.tensor(int(r.response_id), dtype=torch.long),
            "patient_id": str(r.patient_id),
            "slide_id": str(r.slide_id),
            "treatment_key": str(r.canonical_treatment_key),
            "tile_path": str(r.tile_path),
        }



# =============================================================================
# Model architecture
# =============================================================================

class HistModel(nn.Module):
    """
    Treatment-conditioned histology response neural network.
    
    The architecture supports treatment_only, image_only, and image_treatment modes
    so image information can be compared directly with treatment identity controls.
    """
    def __init__(self, model_type, n_treatments, n_classes, cfg):
        """Initialize the dataset or model object."""
        super().__init__()

        self.model_type = model_type

        hidden = int(cfg["hidden_dim"])
        dropout = float(cfg["dropout"])
        emb_dim = int(cfg["treatment_embedding_dim"])

        # Image features are disabled for treatment_only so this baseline measures treatment identity alone.
        if model_type in ["image_only", "image_treatment"]:
            name = cfg.get("backbone_name", "resnet18")

            if name == "resnet50":
                weights = models.ResNet50_Weights.DEFAULT if cfg.get("pretrained", True) else None
                self.backbone = models.resnet50(weights=weights)
            else:
                weights = models.ResNet18_Weights.DEFAULT if cfg.get("pretrained", True) else None
                self.backbone = models.resnet18(weights=weights)

            image_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            if cfg.get("freeze_backbone", False):
                for p in self.backbone.parameters():
                    p.requires_grad = False
        else:
            self.backbone = None
            image_dim = 0

        # Treatment embeddings are disabled for image_only so morphology can be evaluated separately.
        if model_type in ["treatment_only", "image_treatment"]:
            self.treat_emb = nn.Embedding(n_treatments, emb_dim)
            treat_dim = emb_dim
        else:
            self.treat_emb = None
            treat_dim = 0

        in_dim = image_dim + treat_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, image, treatment_id):
        """Run a forward pass for image, treatment, or image-treatment inputs."""
        parts = []

        if self.backbone is not None:
            parts.append(self.backbone(image))

        if self.treat_emb is not None:
            parts.append(self.treat_emb(treatment_id))

        return self.classifier(torch.cat(parts, dim=1))



# =============================================================================
# Transforms, loaders, and prediction collation
# =============================================================================

def make_transforms(cfg):
    """Create training and evaluation image transforms from the training configuration."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = int(cfg["image_size"])

    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def make_loader(df, transform, model_type, cfg, train=False):
    """Construct a PyTorch DataLoader for one split and model type."""
    ds = TileDS(df, transform, model_type, cfg["image_size"])

    num_workers = int(cfg.get("num_workers", 0))

    kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": train,
        "sampler": None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(ds, **kwargs)


def collate_predict(model, loader, device):
    """Run model inference over a DataLoader and return tile-level prediction rows."""
    model.eval()
    rows = []

    with torch.no_grad():
        for b in loader:
            image = b["image"].to(device, non_blocking=True)
            tid = b["treatment_id"].to(device, non_blocking=True)
            label = b["label"].to(device, non_blocking=True)

            logits = model(image, tid)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred = prob.argmax(axis=1)
            y = label.detach().cpu().numpy()

            for i in range(len(y)):
                rows.append({
                    "patient_id": b["patient_id"][i],
                    "slide_id": b["slide_id"][i],
                    "tile_path": b["tile_path"][i],
                    "treatment_key": b["treatment_key"][i],
                    "true_label_id": int(y[i]),
                    "pred_label_id": int(pred[i]),
                    "prob_RESPONDER": float(prob[i, 1]) if prob.shape[1] > 1 else float(prob[i, 0]),
                    "prob_class_0": float(prob[i, 0]),
                    "prob_class_1": float(prob[i, 1]) if prob.shape[1] > 1 else np.nan,
                })

    return pd.DataFrame(rows)



# =============================================================================
# Patient-level metrics and training helpers
# =============================================================================

def patient_metrics(pred):
    """Aggregate tile predictions to patient level and compute model metrics."""
    if pred.empty:
        return {}

    g = (
        pred
        .groupby(["patient_id", "treatment_key", "true_label_id"], dropna=False)
        .agg(prob_RESPONDER=("prob_RESPONDER", "mean"))
        .reset_index()
    )

    y = g["true_label_id"].astype(int).to_numpy()
    p = g["prob_RESPONDER"].astype(float).to_numpy()
    pred_label = (p >= 0.5).astype(int)

    return {
        "n_patients": int(g["patient_id"].nunique()),
        "n_rows": int(len(g)),
        "accuracy": float(accuracy_score(y, pred_label)) if len(y) else np.nan,
        "f1_macro": float(f1_score(y, pred_label, average="macro", zero_division=0)) if len(y) else np.nan,
        "auc": safe_auc(y, p),
        "brier": brier(y, p),
        "prob_mean": float(np.mean(p)) if len(p) else np.nan,
        "prob_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
    }


def make_patient_level_rows(df):
    """Return one representative row per patient for treatment-only training/evaluation."""
    return df.drop_duplicates("patient_id").copy()


def train_one_model(df, cfg, model_type, out):
    """Train one configured histology model family and write its outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_treat = int(df.treatment_id.max()) + 1
    n_classes = int(df.response_id.max()) + 1

    model = HistModel(model_type, n_treat, n_classes, cfg).to(device)

    full_train_df = df[df.split == "train"].copy()
    full_val_df = df[df.split == "val"].copy()
    full_test_df = df[df.split == "test"].copy()

    cap = cfg.get("max_tiles_per_patient_per_epoch", None)
    cap = int(cap) if cap is not None else None

    eval_cap = cfg.get("max_tiles_per_patient_for_eval", cap)
    eval_cap = int(eval_cap) if eval_cap is not None else None

    seed = int(cfg.get("random_seed", 42))

    # Treatment-only uses patient-level rows so many tiles from one patient do not inflate this baseline.
    if model_type == "treatment_only":
        train_base_df = make_patient_level_rows(full_train_df)
        val_df = make_patient_level_rows(full_val_df)
        test_df = make_patient_level_rows(full_test_df)
        train_eval_df = make_patient_level_rows(full_train_df)
    else:
        # Image models use tile rows but are capped and balanced by patient/slide to reduce overrepresentation.
        train_base_df = full_train_df
        val_df = sample_patient_slide_balanced(full_val_df, eval_cap, seed + 1000)
        test_df = sample_patient_slide_balanced(full_test_df, eval_cap, seed + 2000)
        train_eval_df = sample_patient_slide_balanced(full_train_df, eval_cap, seed + 3000)

    train_tf, eval_tf = make_transforms(cfg)

    val_loader = make_loader(val_df, eval_tf, model_type, cfg, train=False)
    test_loader = make_loader(test_df, eval_tf, model_type, cfg, train=False)
    train_eval_loader = make_loader(train_eval_df, eval_tf, model_type, cfg, train=False)

    patient_train = full_train_df.drop_duplicates("patient_id").copy()
    counts = patient_train.response_id.value_counts().to_dict()
    total = patient_train["patient_id"].nunique()

    # Class weights are computed from patient-level labels to reduce responder imbalance effects.
    class_weights = torch.tensor(
        [total / max(n_classes * counts.get(i, 1), 1) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(device)

    crit = nn.CrossEntropyLoss(
        weight=class_weights if cfg.get("use_class_weights", True) else None,
        label_smoothing=float(cfg.get("label_smoothing", 0)),
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.get("mixed_precision", True)) and torch.cuda.is_available()
    )

    best = -1e9
    bad = 0
    history = []

    model_out = ensure_dir(out / model_type)

    sampling_rows = [
        {
            "model_type": model_type,
            "split": "train_full_after_qc_filter",
            "tile_rows": len(full_train_df),
            "patients": full_train_df["patient_id"].nunique(),
            "slides": full_train_df["slide_id"].nunique(),
            "cap_per_patient": "",
        },
        {
            "model_type": model_type,
            "split": "train_eval",
            "tile_rows": len(train_eval_df),
            "patients": train_eval_df["patient_id"].nunique(),
            "slides": train_eval_df["slide_id"].nunique(),
            "cap_per_patient": eval_cap if model_type != "treatment_only" else "patient_level",
        },
        {
            "model_type": model_type,
            "split": "val_eval",
            "tile_rows": len(val_df),
            "patients": val_df["patient_id"].nunique(),
            "slides": val_df["slide_id"].nunique(),
            "cap_per_patient": eval_cap if model_type != "treatment_only" else "patient_level",
        },
        {
            "model_type": model_type,
            "split": "test_eval",
            "tile_rows": len(test_df),
            "patients": test_df["patient_id"].nunique(),
            "slides": test_df["slide_id"].nunique(),
            "cap_per_patient": eval_cap if model_type != "treatment_only" else "patient_level",
        },
    ]

    pd.DataFrame(sampling_rows).to_csv(model_out / "tile_sampling_summary.tsv", sep="\t", index=False)

    print("")
    print(f"Training model: {model_type}")
    print(f"device: {device}")
    print(f"train_full_tiles_after_qc_filter: {len(full_train_df)}")
    print(f"train_patients: {full_train_df['patient_id'].nunique()}")
    print(f"train_slides: {full_train_df['slide_id'].nunique()}")
    print(f"patient_cap_per_epoch: {cap if model_type != 'treatment_only' else 'patient_level'}")
    print(f"val_eval_rows: {len(val_df)}")
    print(f"test_eval_rows: {len(test_df)}")

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        if model_type == "treatment_only":
            train_epoch_df = train_base_df
        else:
            train_epoch_df = sample_patient_slide_balanced(train_base_df, cap, seed + epoch)

        train_loader = make_loader(train_epoch_df, train_tf, model_type, cfg, train=True)

        model.train()
        loss_sum = 0.0
        seen = 0

        for b in train_loader:
            image = b["image"].to(device, non_blocking=True)
            tid = b["treatment_id"].to(device, non_blocking=True)
            lab = b["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=bool(cfg.get("mixed_precision", True)) and torch.cuda.is_available()
            ):
                logits = model(image, tid)
                loss = crit(logits, lab)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item()) * len(lab)
            seen += len(lab)

        val_pred = collate_predict(model, val_loader, device)
        val_m = patient_metrics(val_pred)

        score = val_m.get("auc", np.nan)

        if not np.isfinite(score):
            score = val_m.get("accuracy", 0.0)

        row = {
            "epoch": epoch,
            "train_epoch_tile_rows": len(train_epoch_df),
            "train_loss": loss_sum / max(seen, 1),
        }

        row.update({f"val_patient_{k}": v for k, v in val_m.items()})
        history.append(row)

        pd.DataFrame(history).to_csv(model_out / "training_history.tsv", sep="\t", index=False)

        print(
            model_type,
            "epoch",
            epoch,
            "train_rows",
            len(train_epoch_df),
            "loss",
            round(loss_sum / max(seen, 1), 5),
            "val_auc",
            score,
            flush=True,
        )

        if score > best:
            best = score
            bad = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "n_treatments": n_treat,
                    "n_classes": n_classes,
                    "training_cfg": cfg,
                },
                model_out / "best_model.pt",
            )
        else:
            bad += 1

        if bad >= int(cfg.get("early_stopping_patience", 4)):
            print(f"early stopping: {model_type} at epoch {epoch}")
            break

    ckpt = torch.load(model_out / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    all_metrics = []

    for split_name, loader in [
        ("train", train_eval_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        pred = collate_predict(model, loader, device)
        pred.to_csv(model_out / f"{split_name}_tile_predictions.tsv", sep="\t", index=False)

        m = patient_metrics(pred)
        m["split"] = split_name
        m["model_type"] = model_type
        all_metrics.append(m)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(model_out / "metrics_by_split.tsv", sep="\t", index=False)

    print("")
    print(f"Metrics for {model_type}")
    print(metrics.to_string(index=False))

    return metrics



# =============================================================================
# Main training workflow
# =============================================================================

def main():
    """Run this command-line pipeline step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["training"]

    set_seed(int(train_cfg.get("random_seed", 42)))

    out = ensure_dir(output_root(cfg) / "07_models")

    raw_df = read_table(output_root(cfg) / "06_patient_split" / "tile_training_table_split.tsv", sep="\t")
    raw_df = raw_df[raw_df["split"].isin(["train", "val", "test"])].copy()

    df, excluded = attach_slide_qc_and_filter(raw_df, cfg)

    excluded_path = out / "excluded_slide_qc_tiles.tsv"
    excluded.to_csv(excluded_path, sep="\t", index=False)

    treat_enc, inv_treat = make_encoder(df.loc[df.split == "train", "canonical_treatment_key"].astype(str))

    resp_enc = {"NON_RESPONDER": 0, "RESPONDER": 1}
    inv_resp = {0: "NON_RESPONDER", 1: "RESPONDER"}

    df = df[df["canonical_treatment_key"].isin(treat_enc)].copy()
    df["treatment_id"] = df["canonical_treatment_key"].map(treat_enc).astype(int)
    df["response_id"] = df["binary_response_label"].map(resp_enc).astype(int)
    df["tile_path"] = df["tile_path"].astype(str)

    missing_files = df.loc[~df["tile_path"].map(lambda p: Path(str(p)).exists())]

    if len(missing_files):
        raise FileNotFoundError(f"Missing tile files detected: {len(missing_files)}")

    write_json(
        {
            "treatment_encoder": treat_enc,
            "response_encoder": resp_enc,
            "inverse_treatment_encoder": inv_treat,
            "inverse_response_encoder": inv_resp,
        },
        out / "encoders.json",
    )

    run_summary = {
        "raw_tile_rows_total": int(len(raw_df)),
        "tile_rows_after_slide_qc_filter": int(len(df)),
        "excluded_slide_qc_tile_rows": int(len(excluded)),
        "excluded_slide_qc_slides": int(excluded["slide_id"].nunique()) if len(excluded) else 0,
        "patients_total_after_slide_qc": int(df["patient_id"].nunique()),
        "slides_total_after_slide_qc": int(df["slide_id"].nunique()),
        "splits_patients_after_slide_qc": df.groupby("split")["patient_id"].nunique().to_dict(),
        "tile_rows_by_split_after_slide_qc": df["split"].value_counts().to_dict(),
        "max_tiles_per_patient_per_epoch": train_cfg.get("max_tiles_per_patient_per_epoch"),
        "max_tiles_per_patient_for_eval": train_cfg.get("max_tiles_per_patient_for_eval", train_cfg.get("max_tiles_per_patient_per_epoch")),
        "models_to_train": train_cfg.get("models_to_train"),
        "batch_size": train_cfg.get("batch_size"),
        "num_workers": train_cfg.get("num_workers"),
        "use_weighted_sampler": train_cfg.get("use_weighted_sampler"),
        "use_class_weights": train_cfg.get("use_class_weights"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }

    (out / "training_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("")
    print("Step 07 input summary")
    print(json.dumps(run_summary, indent=2))

    metrics = []

    for mt in train_cfg.get("models_to_train", ["treatment_only", "image_only", "image_treatment"]):
        metrics.append(train_one_model(df, train_cfg, mt, out))

    comp = pd.concat(metrics, ignore_index=True)
    comp.to_csv(out / "model_comparison.tsv", sep="\t", index=False)

    print("")
    print("Model comparison")
    print(comp.to_string(index=False))

    print("")
    print("DONE")
    print(out)


if __name__ == "__main__":
    main()
