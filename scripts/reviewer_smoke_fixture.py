"""Seeded, redistributable counts/coordinate fixtures; no biological observations.

The input follows the maintained manual 10x MTX loader's file schema. Each section
contains one deliberately low-count spot and two columns sharing an Ensembl ID,
so retained-spot filtering and pre-log duplicate aggregation are exercised.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


def write_visium_fixture(root, config, marker_programs):
    """Write raw MTX, barcode/gene tables and full-resolution pixel coordinates."""
    root = Path(root)
    rng = np.random.default_rng(config["seed"])
    genes = list(dict.fromkeys(g for values in marker_programs.values() for g in values))
    genes += [f"FIXTURE_GENE_{i:03d}" for i in range(config["genes"] - len(genes))]
    gene_ids = [f"ENSG{i+1:011d}.1" for i in range(len(genes))]
    gene_ids[-1] = gene_ids[0].replace(".1", ".2")
    sections = []
    total = config["training_sections"] + config["external_sections"]
    n = config["spots_per_section"]
    for i in range(total):
        sample_id = (f"FIXTURE_TRAIN_{i:03d}" if i < config["training_sections"]
                     else f"FIXTURE_EXTERNAL_{i-config['training_sections']:03d}")
        folder = root / sample_id
        (folder / "raw").mkdir(parents=True)
        (folder / "spatial").mkdir()
        latent = (i % config["training_sections"]) / (config["training_sections"] - 1)
        rates = np.full((n, len(genes)), 3.5)
        tumor = np.arange(n) % 8 < (2 + int(latent * 4))
        for name, symbols in marker_programs.items():
            indices = [genes.index(g) for g in symbols]
            program = (1 + 6 * latent) if name == "tumor_epithelial" else (2 + 4 * (1-latent))
            rates[:, indices] += program * (1 + tumor[:, None] * (name == "tumor_epithelial"))
        counts = rng.poisson(rates).astype(np.int32)
        counts[0] = 0
        barcodes = [f"FIXTURE{i:03d}SPOT{j:03d}-1" for j in range(n)]
        mmwrite(folder / "raw/matrix.mtx", sparse.coo_matrix(counts.T))
        pd.Series(barcodes).to_csv(folder / "raw/barcodes.tsv", index=False, header=False, sep="\t")
        pd.DataFrame({0: gene_ids, 1: genes, 2: "Gene Expression"}).to_csv(folder / "raw/features.tsv", index=False, header=False, sep="\t")
        positions = pd.DataFrame({"barcode": barcodes, "in_tissue": 1,
            "array_row": np.arange(n)//8, "array_col": np.arange(n)%8,
            "pxl_row_in_fullres": (np.arange(n)//8)*60 + rng.normal(0, .1, n),
            "pxl_col_in_fullres": (np.arange(n)%8)*60 + rng.normal(0, .1, n)})
        positions.to_csv(folder / "spatial/tissue_positions.csv", index=False)
        (folder / "spatial/scalefactors_json.json").write_text(json.dumps({"tissue_hires_scalef": 1., "spot_diameter_fullres": 55.}), encoding="utf-8")
        sections.append({"sample_id": sample_id, "path": folder, "is_training": i < config["training_sections"]})
    return sections
