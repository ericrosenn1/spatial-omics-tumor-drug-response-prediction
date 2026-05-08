from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# CONFIG
# =========================================================

# Source folder for the current run.
# Put the folder path here, or pass it with:
# python sort_training_dataset_intake.py --input "C:\path\to\folder"
INPUT_DATA_FOLDER = r""

# Root output folder where organized study folders will be created.
# This should point to your:
# project\prediction modeling\data for training sorting
TRAINING_SORT_ROOT = (
    r"C:\Users\Owner\OneDrive\spring 2026 NYU\Adv. Omics Fenyo\project"
    r"\prediction modeling\data for training sorting"
)

# Optional.
# If True, copy source files into the run folder.
# If False, only catalog them and write manifests.
COPY_SOURCE_FILES = False

# Optional OpenAI API use.
# The script will still run without this.
USE_OPENAI_API = True
OPENAI_MODEL_NAME = "gpt-4.1-mini"

# How many rows to preview from tabular files for schema inference.
TABULAR_PREVIEW_ROWS = 200

# How many cells per column to inspect for data type inference.
COLUMN_SAMPLE_SIZE = 100

# Size threshold for classifying very large raster images as possible WSI-like.
VERY_LARGE_IMAGE_PIXELS = 10000 * 10000

# Max characters to store from any single text document in summary fields.
# This does not limit parsing.
TEXT_SNIPPET_STORE_CHARS = 12000
# PDF parsing robustness
PDF_LOG_PAGE_COUNTS = True
PDF_MIN_EXTRACTED_CHARS_WARN = 500
PDF_PARSE_EMPTY_PAGE_WARN_THRESHOLD = 0.5

# Terminal verbosity.
VERBOSE = True


# =========================================================
# FILE TYPE REGISTRY
# =========================================================
analysis_stats = {
    "total_files": 0,
    "pdf_files": 0,
    "pdf_failed": 0,
    "pdf_low_text": 0,
    "tabular_files": 0,
    "tabular_failed": 0,
    "image_files": 0,
    "image_failed": 0,
    "text_files": 0,
    "text_failed": 0,
}

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".xml",
    ".csv",
    ".tsv",
}

TABULAR_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

WHOLE_SLIDE_EXTENSIONS = {
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".qptiff",
}

SPATIAL_OBJECT_EXTENSIONS = {
    ".h5ad",
    ".h5",
    ".loom",
    ".mtx",
    ".rds",
    ".h5seurat",
}

ARCHIVE_EXTENSIONS = {
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".tar",
    ".tgz",
    ".7z",
}

CODE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".r",
    ".R",
    ".sh",
    ".ps1",
    ".sql",
    ".jl",
}

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
}

GEO_ACCESSION_PATTERN = re.compile(r"\bGSE\d{4,}\b", re.IGNORECASE)
GSM_ACCESSION_PATTERN = re.compile(r"\bGSM\d{4,}\b", re.IGNORECASE)
GDS_ACCESSION_PATTERN = re.compile(r"\bGDS\d{3,}\b", re.IGNORECASE)
PMID_PATTERN = re.compile(r"\bPMID\s*:\s*(\d+)\b", re.IGNORECASE)
DOI_PATTERN = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE,
)

TIMEPOINT_PATTERNS = {
    "pre_treatment": [
        r"\bpre[- ]treatment\b",
        r"\bpretreatment\b",
        r"\bbaseline\b",
        r"\bpre\b",
        r"\bbefore treatment\b",
    ],
    "on_treatment": [
        r"\bon[- ]treatment\b",
        r"\bduring treatment\b",
        r"\bearly on[- ]treatment\b",
        r"\bon therapy\b",
    ],
    "post_treatment": [
        r"\bpost[- ]treatment\b",
        r"\bpost treatment\b",
        r"\bafter treatment\b",
        r"\bpost[- ]therapy\b",
        r"\bpost therapy\b",
    ],
}

CANCER_KEYWORDS = {
    "triple negative breast cancer": "triple_negative_breast_cancer",
    "tnbc": "triple_negative_breast_cancer",
    "breast cancer": "breast_cancer",
    "ovarian cancer": "ovarian_cancer",
    "eoc": "epithelial_ovarian_cancer",
    "pspc": "peritoneal_serous_papillary_carcinoma",
    "nsclc": "non_small_cell_lung_cancer",
    "lung adenocarcinoma": "lung_adenocarcinoma",
    "melanoma": "melanoma",
    "head and neck": "head_and_neck_cancer",
    "hnscc": "head_and_neck_squamous_cell_carcinoma",
    "hepatocellular carcinoma": "hepatocellular_carcinoma",
    "hcc": "hepatocellular_carcinoma",
    "retinoblastoma": "retinoblastoma",
}

MODALITY_KEYWORDS = {
    "imaging mass cytometry": "imc",
    "imc": "imc",
    "mibi": "mibi",
    "mibi-tof": "mibi",
    "vectra": "multiplex_immunofluorescence",
    "multiplex immunofluorescence": "multiplex_immunofluorescence",
    "multiplexed imaging": "multiplex_imaging",
    "whole slide image": "wsi_histology",
    "whole slide images": "wsi_histology",
    "histology": "histology",
    "h&e": "histology_he",
    "hematoxylin and eosin": "histology_he",
    "visium": "visium",
    "spatial transcriptomics": "spatial_transcriptomics",
    "scRNA-seq": "single_cell_rna",
    "scrna-seq": "single_cell_rna",
    "single-cell rna": "single_cell_rna",
    "single cell rna": "single_cell_rna",
    "geo": "geo_metadata",
    "tiff stack": "multiplex_image_stack",
}

TREATMENT_KEYWORDS = {
    "pembrolizumab": "pembrolizumab",
    "nivolumab": "nivolumab",
    "atezolizumab": "atezolizumab",
    "cabozantinib": "cabozantinib",
    "bevacizumab": "bevacizumab",
    "vorinostat": "vorinostat",
    "nab-paclitaxel": "nab_paclitaxel",
    "carboplatin": "carboplatin",
    "chemotherapy": "chemotherapy",
    "immunotherapy": "immunotherapy",
    "checkpoint blockade": "checkpoint_blockade",
    "anti-pd-1": "anti_pd_1",
    "anti pd-1": "anti_pd_1",
    "anti-pd-l1": "anti_pd_l1",
    "anti pd-l1": "anti_pd_l1",
    "avastin": "bevacizumab",
}

RESPONSE_KEYWORDS = {
    "pcr": "pathologic_complete_response",
    "pathological complete response": "pathologic_complete_response",
    "pathologic complete response": "pathologic_complete_response",
    "rd": "residual_disease",
    "residual disease": "residual_disease",
    "response": "generic_response",
    "responder": "responder",
    "non-responder": "non_responder",
    "nonresponder": "non_responder",
    "progressive disease": "pd",
    "stable disease": "sd",
    "partial response": "pr",
    "complete response": "cr",
    "effective": "effective",
    "invalid": "invalid",
    "sensitive": "sensitive",
    "resistant": "resistant",
    "recurrence": "recurrence",
    "survival": "survival",
}

UNIT_PATTERNS = {
    "u/ml": re.compile(r"\bU/ml\b", re.IGNORECASE),
    "um": re.compile(r"\bμm\b|\bum\b", re.IGNORECASE),
    "mm": re.compile(r"\bmm\b", re.IGNORECASE),
    "days": re.compile(r"\bdays?\b", re.IGNORECASE),
    "years": re.compile(r"\byears?\b", re.IGNORECASE),
    "pixels": re.compile(r"\bpixels?\b", re.IGNORECASE),
    "microns_per_pixel": re.compile(r"\bmicrons? per pixel\b|\bmicrons?/pixel\b", re.IGNORECASE),
    "x_magnification": re.compile(r"\b20x\b|\b40x\b", re.IGNORECASE),
}


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class DecisionRecord:
    step: str
    decision: str
    confidence: str
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class FileRecord:
    path: str
    rel_path: str
    file_name: str
    extension: str
    size_bytes: int
    category: str
    subtype: str
    mime_type: str
    readable_text: bool = False
    row_count_estimate: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_class: Optional[str] = None
    notes: str = ""


@dataclass
class VariableRecord:
    source_file: str
    sheet_name: str
    variable_name: str
    inferred_storage_type: str
    inferred_semantic_type: str
    units: str
    distinct_value_count_estimate: Optional[int]
    example_values: str
    treatment_related: str
    response_related: str
    notes: str


@dataclass
class StudySignals:
    geo_accessions: List[str] = field(default_factory=list)
    gsm_accessions: List[str] = field(default_factory=list)
    gds_accessions: List[str] = field(default_factory=list)
    dois: List[str] = field(default_factory=list)
    pmids: List[str] = field(default_factory=list)
    cancer_terms: List[str] = field(default_factory=list)
    modality_terms: List[str] = field(default_factory=list)
    treatment_terms: List[str] = field(default_factory=list)
    response_terms: List[str] = field(default_factory=list)
    timepoints: List[str] = field(default_factory=list)
    trial_names: List[str] = field(default_factory=list)
    title_candidates: List[str] = field(default_factory=list)
    patient_ids: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)


# =========================================================
# LOGGING
# =========================================================

class DecisionLogger:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.records: List[DecisionRecord] = []

    def log(
        self,
        step: str,
        decision: str,
        confidence: str,
        reason: str,
        evidence: Optional[List[str]] = None,
    ) -> None:
        record = DecisionRecord(
            step=step,
            decision=decision,
            confidence=confidence,
            reason=reason,
            evidence=evidence or [],
        )
        self.records.append(record)

        if self.verbose:
            print(f"[{step}] {decision} | confidence={confidence}")
            print(f"  reason: {reason}")
            if evidence:
                for item in evidence[:5]:
                    print(f"  evidence: {item}")

    def print_simplified_summary(self) -> None:
        print("\n" + "=" * 70)
        print("SIMPLIFIED DECISION SUMMARY")
        print("=" * 70)
        for rec in self.records:
            print(f"{rec.step}: {rec.decision} [{rec.confidence}]")
        print("=" * 70 + "\n")

    def save_json(self, out_path: Path) -> None:
        out_path.write_text(
            json.dumps([asdict(r) for r in self.records], indent=2),
            encoding="utf-8",
        )

    def save_text(self, out_path: Path) -> None:
        lines: List[str] = []
        for rec in self.records:
            lines.append(f"STEP: {rec.step}")
            lines.append(f"DECISION: {rec.decision}")
            lines.append(f"CONFIDENCE: {rec.confidence}")
            lines.append(f"REASON: {rec.reason}")
            if rec.evidence:
                lines.append("EVIDENCE:")
                for item in rec.evidence:
                    lines.append(f"  - {item}")
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# GENERAL HELPERS
# =========================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_slug(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "unnamed"
    return text[:max_len].strip("_")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def get_openai_client():
    if not USE_OPENAI_API:
        return None

    if OpenAI is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAI()
    except Exception:
        return None


def api_generate_study_label(client, signals: StudySignals) -> Optional[str]:
    if client is None:
        return None

    prompt = f"""
Create a short dataset label for a study intake folder.

GEO: {signals.geo_accessions}
Cancer: {signals.cancer_terms}
Modality: {signals.modality_terms}
Treatment: {signals.treatment_terms}
Timepoints: {signals.timepoints}

Return ONE concise label, lowercase, underscore separated.
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL_NAME,
            input=prompt,
        )
        text = getattr(response, "output_text", "") or ""
        return safe_slug(text.strip(), max_len=120) if text else None
    except Exception:
        return None


def api_review_dataset_summary(client, summary_text: str) -> Dict[str, Any]:
    if client is None:
        return {"api_used": False, "result": "not_run"}

    prompt = f"""
Review this dataset summary.

Answer:
1. Is classification accurate? (yes/no)
2. Main uncertainties
3. What should be manually checked next

Summary:
{summary_text}
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL_NAME,
            input=prompt,
        )
        text = getattr(response, "output_text", "") or ""
        return {"api_used": True, "result": text.strip()}
    except Exception as exc:
        return {"api_used": True, "result": f"failed: {exc}"}
    
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_extension(path: Path) -> str:
    suffixes = "".join(path.suffixes).lower()
    if suffixes in {".tar.gz"}:
        return suffixes
    return path.suffix.lower()


def guess_file_category(path: Path) -> Tuple[str, str]:
    ext = detect_extension(path)

    if ext in WHOLE_SLIDE_EXTENSIONS:
        return "image", "whole_slide_image"

    if ext in IMAGE_EXTENSIONS:
        return "image", "raster_image"

    if ext in TABULAR_EXTENSIONS:
        return "tabular", "table"

    if ext in SPATIAL_OBJECT_EXTENSIONS:
        return "omics_object", "spatial_or_single_cell_object"

    if ext in ARCHIVE_EXTENSIONS:
        return "archive", "compressed_archive"

    if ext in CODE_EXTENSIONS:
        return "code", "script_or_notebook"

    if ext == ".pdf":
        return "document", "pdf"

    if ext == ".docx":
        return "document", "docx"

    if ext in TEXT_EXTENSIONS:
        return "text", "text_like"

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        if mime_type.startswith("image/"):
            return "image", "unknown_image"
        if mime_type.startswith("text/"):
            return "text", "unknown_text"

    return "other", "unknown"


def is_probably_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return True
    except Exception:
        return True
    return False


def trim_references_section(text: str) -> str:
    """
    Keep the full parsed text, but remove the bibliography-like tail from the working text
    used for metadata extraction so that reference titles do not confuse keyword detection.

    This is intentionally conservative:
    it trims only when a strong references heading appears late in the document.
    """
    lower = text.lower()

    candidate_patterns = [
        r"\nreferences\s*\n",
        r"\nreference\s*\n",
        r"\nbibliography\s*\n",
        r"\nliterature cited\s*\n",
        r"\nworks cited\s*\n",
    ]

    match_positions = []
    for pat in candidate_patterns:
        for m in re.finditer(pat, lower):
            match_positions.append(m.start())

    if not match_positions:
        return text

    # Prefer a references heading in the later half of the text.
    text_len = len(text)
    late_positions = [p for p in match_positions if p > text_len * 0.45]
    if not late_positions:
        return text

    cut = min(late_positions)
    return text[:cut]


def find_keywords(text: str, mapping: Dict[str, str]) -> List[str]:
    found = []
    lower = text.lower()
    for key, label in mapping.items():
        if key.lower() in lower:
            found.append(label)
    return sorted(set(found))


def find_timepoints(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for label, patterns in TIMEPOINT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower, flags=re.IGNORECASE):
                found.append(label)
                break
    return sorted(set(found))


def find_trial_names(text: str) -> List[str]:
    patterns = [
        r"\bNeoTRIP\b",
        r"\bNCT\d{8}\b",
        r"\bGOG-?\d+\b",
        r"\bICON-?\d+\b",
        r"\bKEYNOTE-?\d+\b",
    ]
    hits = []
    for pat in patterns:
        hits.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return sorted(set(hits))


def find_patient_like_ids(text: str) -> List[str]:
    patterns = [
        r"\bpatient\s*\d+\b",
        r"\bP\d{1,4}\b",
        r"\bCID\d+[A-Z]?\b",
    ]
    hits = []
    for pat in patterns:
        hits.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return sorted(set(hits))[:200]


def find_sample_like_ids(text: str) -> List[str]:
    patterns = [
        r"\bGSM\d{4,}\b",
        r"\bS\d{1,5}\b",
        r"\bSample\s*\d+\b",
        r"\bROI\s*\d+\b",
    ]
    hits = []
    for pat in patterns:
        hits.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return sorted(set(hits))[:200]


def find_units_in_text(text: str) -> List[str]:
    hits = []
    for label, pat in UNIT_PATTERNS.items():
        if pat.search(text):
            hits.append(label)
    return sorted(set(hits))


def pick_title_candidates(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = []

    for line in lines[:40]:
        if 15 <= len(line) <= 220:
            if line.lower() not in {"abstract", "introduction", "summary", "article"}:
                candidates.append(line)

    return candidates[:10]


# =========================================================
# TEXT EXTRACTION
# =========================================================

def read_plain_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def read_pdf_text(path: Path, logger: Optional[DecisionLogger] = None) -> str:
    global analysis_stats
    analysis_stats["pdf_files"] += 1
    if PdfReader is None:
        if logger is not None:
            logger.log(
                step="pdf_read",
                decision=f"could not parse PDF {path.name}",
                confidence="low",
                reason="pypdf is not available",
            )
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        analysis_stats["pdf_failed"] += 1
        if logger is not None:
            logger.log(
                step="pdf_read",
                decision=f"failed to open PDF {path.name}",
                confidence="low",
                reason=f"PdfReader open failed: {exc}",
            )
        return ""

    page_count = 0
    empty_pages = 0
    chunks: List[str] = []

    try:
        page_count = len(reader.pages)
    except Exception:
        page_count = 0

    if logger is not None and PDF_LOG_PAGE_COUNTS:
        logger.log(
            step="pdf_info",
            decision=f"{path.name} page_count={page_count}",
            confidence="high",
            reason="Counted PDF pages before extraction",
        )

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            txt = ""
            if logger is not None:
                logger.log(
                    step="pdf_page_extract",
                    decision=f"page {page_index} extraction issue in {path.name}",
                    confidence="low",
                    reason=f"page extraction failed: {exc}",
                )

        if txt and txt.strip():
            chunks.append(txt)
        else:
            empty_pages += 1

    full_text = "\n".join(chunks).strip()

    if logger is not None:
        if not full_text:
            analysis_stats["pdf_failed"] += 1
            logger.log(
                step="pdf_read",
                decision=f"parsed zero text from PDF {path.name}",
                confidence="low",
                reason="All pages were empty or extraction failed",
                evidence=[
                    f"page_count={page_count}",
                    f"empty_pages={empty_pages}",
                ],
            )
        else:
            evidence = [
                f"page_count={page_count}",
                f"empty_pages={empty_pages}",
                f"extracted_chars={len(full_text)}",
            ]

            empty_frac = (empty_pages / page_count) if page_count else 0.0

            if len(full_text) < PDF_MIN_EXTRACTED_CHARS_WARN:
                analysis_stats["pdf_low_text"] += 1
                logger.log(
                    step="pdf_read",
                    decision=f"parsed unusually little text from PDF {path.name}",
                    confidence="medium",
                    reason="Extraction succeeded but text length is very small",
                    evidence=evidence,
                )
            elif empty_frac >= PDF_PARSE_EMPTY_PAGE_WARN_THRESHOLD:
                logger.log(
                    step="pdf_read",
                    decision=f"parsed PDF {path.name} with many empty pages",
                    confidence="medium",
                    reason="A large fraction of pages produced no text",
                    evidence=evidence,
                )
            else:
                logger.log(
                    step="pdf_read",
                    decision=f"parsed PDF text from {path.name}",
                    confidence="high",
                    reason="pypdf extracted text successfully",
                    evidence=evidence,
                )

    return full_text


def read_docx_text(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception:
        return ""

    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        return ""


def read_text_from_document(path: Path, logger: DecisionLogger) -> str:
    ext = detect_extension(path)

    if ext == ".pdf":
        return read_pdf_text(path, logger=logger)

    if ext == ".docx":
        text = read_docx_text(path)
        logger.log(
            step="document_read",
            decision=f"parsed DOCX text from {path.name}",
            confidence="high" if text else "low",
            reason="Used python-docx paragraph extraction",
        )
        return text

    if ext in TEXT_EXTENSIONS or not is_probably_binary(path):
        text = read_plain_text(path)
        logger.log(
            step="document_read",
            decision=f"read plain text from {path.name}",
            confidence="high" if text else "low",
            reason="Used text read with fallback encodings",
        )
        return text

    logger.log(
        step="document_read",
        decision=f"could not read text from {path.name}",
        confidence="low",
        reason="Unsupported or binary file",
    )
    analysis_stats["text_failed"] += 1
    return ""


# =========================================================
# TABULAR HELPERS
# =========================================================

def try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if pd is None:
        return None

    seps = [",", "\t", ";", "|"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, nrows=TABULAR_PREVIEW_ROWS)
            if df is not None and len(df.columns) > 0:
                return df
        except Exception:
            continue
    return None


def try_read_excel_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    if pd is None:
        return {}
    try:
        xls = pd.ExcelFile(path)
        out = {}
        for sheet in xls.sheet_names:
            try:
                out[sheet] = pd.read_excel(path, sheet_name=sheet, nrows=TABULAR_PREVIEW_ROWS)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def infer_storage_type_from_series(series: "pd.Series") -> str:
    if pd is None:
        return "unknown"

    s = series.dropna()
    if s.empty:
        return "unknown"

    try:
        if pd.api.types.is_bool_dtype(s):
            return "bool"
        if pd.api.types.is_integer_dtype(s):
            return "int"
        if pd.api.types.is_float_dtype(s):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
    except Exception:
        pass

    sample = s.astype(str).head(COLUMN_SAMPLE_SIZE)

    bool_like = {"true", "false", "yes", "no", "y", "n", "0", "1"}
    if all(v.strip().lower() in bool_like for v in sample):
        return "bool_as_str"

    numeric_count = 0
    int_count = 0
    date_like_count = 0

    for v in sample:
        vv = v.strip()
        if re.fullmatch(r"-?\d+", vv):
            numeric_count += 1
            int_count += 1
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", vv):
            numeric_count += 1
            continue
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", vv):
            date_like_count += 1

    frac_num = numeric_count / max(len(sample), 1)
    frac_int = int_count / max(len(sample), 1)
    frac_date = date_like_count / max(len(sample), 1)

    if frac_int > 0.9:
        return "int_as_str"
    if frac_num > 0.9:
        return "float_as_str"
    if frac_date > 0.7:
        return "date_as_str"

    return "str"


def infer_semantic_type(column_name: str, example_values: List[str]) -> str:
    name = column_name.lower()

    if any(k in name for k in ["patient", "sample", "slide", "image", "roi", "barcode", "id"]):
        return "identifier"

    if any(k in name for k in ["response", "effect", "responder", "survival", "recurrence", "progression", "pcr", "rd"]):
        return "response_or_outcome"

    if any(k in name for k in ["treatment", "therapy", "drug", "arm", "avastin", "bevacizumab", "nivolumab", "pembrolizumab", "atezolizumab"]):
        return "treatment"

    if any(k in name for k in ["date", "day", "timepoint", "baseline", "pre", "post", "on_treatment"]):
        return "time_or_timepoint"

    if any(k in name for k in ["age", "bmi", "figo", "stage", "diagnosis", "subtype"]):
        return "clinical"

    if any(k in name for k in ["ca-125", "ca125", "counts", "density", "fraction", "score", "expression", "intensity"]):
        return "measurement"

    vals = " ".join(example_values).lower()
    if any(k in vals for k in ["responder", "non-responder", "effective", "invalid", "pcr", "rd", "sd", "pd", "pr", "cr"]):
        return "response_or_outcome"

    return "generic"


def infer_units_for_column(column_name: str, example_values: List[str], full_text_context: str) -> str:
    name = column_name.lower()
    vals = " ".join(example_values).lower()
    text = f"{name} {vals} {full_text_context[:5000].lower()}"

    if "ca-125" in name or "ca125" in name:
        return "U/ml"
    if "pixel" in name or "pixels" in name:
        return "pixels"
    if "day" in name:
        return "days"
    if "year" in name:
        return "years"
    if "mm" in name:
        return "mm"
    if "um" in name or "μm" in name:
        return "um"

    found_units = find_units_in_text(text)
    if found_units:
        return ";".join(found_units)

    return ""

def series_example_values(series):
    try:
        vals = series.dropna().astype(str).head(5).tolist()
        return "; ".join(vals)
    except:
        return ""

def infer_semantic_type_from_name_and_values(name, series):
    try:
        vals = series.dropna().astype(str).head(10).tolist()
    except:
        vals = []
    return infer_semantic_type(name, vals)

def extract_units_from_series(name, series):
    try:
        vals = series.dropna().astype(str).head(10).tolist()
    except:
        vals = []
    return infer_units_for_column(name, vals, "")

def extract_variable_records_from_dataframe(
    df: "pd.DataFrame",
    source_file: str,
    sheet_name: str = "",
) -> List[VariableRecord]:
    if pd is None or df is None or df.empty:
        return []

    records: List[VariableRecord] = []

    for column in df.columns:
        try:
            series = df[column]
            storage_type = infer_storage_type_from_series(series)
            semantic_type = infer_semantic_type_from_name_and_values(str(column), series)
            units = extract_units_from_series(str(column), series)

            distinct_value_count_estimate: Optional[int] = None
            try:
                distinct_value_count_estimate = int(series.dropna().nunique())
            except Exception:
                distinct_value_count_estimate = None

            examples = series_example_values(series)

            text_probe = f"{str(column)} {examples}".lower()
            treatment_related = "yes" if any(k in text_probe for k in TREATMENT_KEYWORDS.keys()) else "no"
            response_related = "yes" if any(k in text_probe for k in RESPONSE_KEYWORDS.keys()) else "no"

            notes_parts = []
            if storage_type in {"str", "bool", "bool_as_str"} and distinct_value_count_estimate is not None:
                if distinct_value_count_estimate <= 10:
                    notes_parts.append("low-cardinality")
            if semantic_type != "generic":
                notes_parts.append(f"semantic={semantic_type}")

            records.append(
                VariableRecord(
                    source_file=source_file,
                    sheet_name=sheet_name,
                    variable_name=str(column),
                    inferred_storage_type=storage_type,
                    inferred_semantic_type=semantic_type,
                    units=units,
                    distinct_value_count_estimate=distinct_value_count_estimate,
                    example_values=examples,
                    treatment_related=treatment_related,
                    response_related=response_related,
                    notes="; ".join(notes_parts),
                )
            )
        except Exception:
            continue

    return records


def extract_variable_records_from_table_file(
    path: Path,
    logger: DecisionLogger,
) -> List[VariableRecord]:
    if pd is None:
        logger.log(
            step="variable_extraction",
            decision=f"skipped variable extraction for {path.name}",
            confidence="low",
            reason="pandas not available",
        )
        return []

    ext = detect_extension(path)
    records: List[VariableRecord] = []

    if ext in {".csv", ".tsv"}:
        df = try_read_csv(path)
        if df is not None:
            records.extend(
                extract_variable_records_from_dataframe(
                    df=df,
                    source_file=path.name,
                    sheet_name="",
                )
            )
            logger.log(
                step="variable_extraction",
                decision=f"extracted variables from {path.name}",
                confidence="high",
                reason="Read delimited table preview with pandas",
                evidence=[f"{len(df.columns)} columns previewed"],
            )
        else:
            analysis_stats["tabular_failed"] += 1
        return records

    if ext in {".xlsx", ".xls"}:
        sheets = try_read_excel_sheets(path)
        for sheet_name, df in sheets.items():
            records.extend(
                extract_variable_records_from_dataframe(
                    df=df,
                    source_file=path.name,
                    sheet_name=sheet_name,
                )
            )
        logger.log(
            step="variable_extraction",
            decision=f"extracted variables from {path.name}",
            confidence="high" if sheets else "low",
            reason="Read workbook sheets with pandas",
            evidence=[f"{len(sheets)} sheets previewed"],
        )
        if not sheets:
            analysis_stats["tabular_failed"] += 1
        return records
    return records


# =========================================================
# IMAGE HELPERS
# =========================================================

def inspect_image(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if Image is None:
        return None, None, None

    try:
        with Image.open(path) as img:
            width, height = img.size
            pixels = width * height

            if detect_extension(path) in WHOLE_SLIDE_EXTENSIONS:
                image_class = "whole_slide_image"

            elif pixels >= VERY_LARGE_IMAGE_PIXELS:
                image_class = "very_large_raster"

            elif max(width, height) >= 4000:
                image_class = "large_image"

            else:
                image_class = "standard_raster"

            return width, height, image_class
    except Exception:
        return None, None, None


# =========================================================
# STUDY SIGNAL EXTRACTION
# =========================================================

def extract_study_signals_from_text(text: str) -> StudySignals:
    if not text:
        return StudySignals()

    working = trim_references_section(text)

    geo = sorted(set(GEO_ACCESSION_PATTERN.findall(working)))
    gsm = sorted(set(GSM_ACCESSION_PATTERN.findall(working)))
    gds = sorted(set(GDS_ACCESSION_PATTERN.findall(working)))
    pmids = sorted(set(PMID_PATTERN.findall(working)))
    dois = sorted(set(DOI_PATTERN.findall(working)))
    cancers = find_keywords(working, CANCER_KEYWORDS)
    modalities = find_keywords(working, MODALITY_KEYWORDS)
    treatments = find_keywords(working, TREATMENT_KEYWORDS)
    responses = find_keywords(working, RESPONSE_KEYWORDS)
    timepoints = find_timepoints(working)
    trials = find_trial_names(working)
    titles = pick_title_candidates(working)

    return StudySignals(
        geo_accessions=geo,
        gsm_accessions=gsm,
        gds_accessions=gds,
        dois=dois,
        pmids=pmids,
        cancer_terms=cancers,
        modality_terms=modalities,
        treatment_terms=treatments,
        response_terms=responses,
        timepoints=timepoints,
        trial_names=trials,
        title_candidates=titles,
        patient_ids=find_patient_like_ids(working),
        sample_ids=find_sample_like_ids(working),
    )


def merge_study_signals(signals_list: Iterable[StudySignals]) -> StudySignals:
    merged = StudySignals()

    for s in signals_list:
        merged.geo_accessions.extend(s.geo_accessions)
        merged.gsm_accessions.extend(s.gsm_accessions)
        merged.gds_accessions.extend(s.gds_accessions)
        merged.dois.extend(s.dois)
        merged.pmids.extend(s.pmids)
        merged.cancer_terms.extend(s.cancer_terms)
        merged.modality_terms.extend(s.modality_terms)
        merged.treatment_terms.extend(s.treatment_terms)
        merged.response_terms.extend(s.response_terms)
        merged.timepoints.extend(s.timepoints)
        merged.trial_names.extend(s.trial_names)
        merged.title_candidates.extend(s.title_candidates)
        merged.patient_ids.extend(s.patient_ids)
        merged.sample_ids.extend(s.sample_ids)

    for field_name in merged.__dataclass_fields__.keys():
        values = getattr(merged, field_name)
        setattr(merged, field_name, sorted(set(values)))

    return merged


def choose_primary_study_id(signals: StudySignals, file_paths: List[Path]) -> str:
    if signals.geo_accessions:
        return signals.geo_accessions[0].upper()

    if signals.dois:
        return safe_slug(signals.dois[0], max_len=60)

    if signals.trial_names:
        return safe_slug(signals.trial_names[0], max_len=60)

    if signals.title_candidates:
        return safe_slug(signals.title_candidates[0], max_len=60)

    seed = "|".join(sorted(str(p.name) for p in file_paths))
    return f"study_{sha1_text(seed)[:10]}"


def choose_study_folder_name(signals: StudySignals, file_paths: List[Path]) -> str:
    parts: List[str] = []

    primary_id = choose_primary_study_id(signals, file_paths)
    parts.append(primary_id)

    if signals.cancer_terms:
        parts.append(signals.cancer_terms[0])

    if signals.modality_terms:
        parts.append(signals.modality_terms[0])

    if signals.timepoints:
        if "pre_treatment" in signals.timepoints and "post_treatment" in signals.timepoints:
            parts.append("paired_treatment_timepoints")
        else:
            parts.append(signals.timepoints[0])

    return safe_slug("_".join(parts), max_len=120)


# =========================================================
# FILE DISCOVERY AND CATALOGING
# =========================================================

def collect_input_files(input_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in input_root.rglob("*"):
        if path.is_file():
            files.append(path)
    return sorted(files)


def build_file_record(path: Path, input_root: Path, logger: DecisionLogger) -> FileRecord:
    global analysis_stats
    category, subtype = guess_file_category(path)
    # Track file category counts
    if category == "image":
        analysis_stats["image_files"] += 1

    if category == "tabular":
        analysis_stats["tabular_files"] += 1

    if category in {"text", "document"}:
        analysis_stats["text_files"] += 1
    
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or ""

    size_bytes = 0
    try:
        size_bytes = path.stat().st_size
    except Exception:
        pass

    file_record = FileRecord(
        path=str(path),
        rel_path=str(path.relative_to(input_root)),
        file_name=path.name,
        extension=detect_extension(path),
        size_bytes=size_bytes,
        category=category,
        subtype=subtype,
        mime_type=mime_type,
    )

    if category in {"text", "document"}:
        file_record.readable_text = True

    if category == "tabular" and pd is not None:
        try:
            if detect_extension(path) in {".csv", ".tsv"}:
                df = try_read_csv(path)
                if df is not None:
                    file_record.row_count_estimate = len(df)
            elif detect_extension(path) in {".xlsx", ".xls"}:
                sheets = try_read_excel_sheets(path)
                first_df = next(iter(sheets.values()), None)
                if first_df is not None:
                    file_record.row_count_estimate = len(first_df)
        except Exception:
            pass

    if category == "image":
        width, height, image_class = inspect_image(path)
        file_record.image_width = width
        file_record.image_height = height
        file_record.image_class = image_class

        if width is None and height is None and image_class is None:
            analysis_stats["image_failed"] += 1

    logger.log(
        step="file_catalog",
        decision=f"cataloged {path.name} as {category}/{subtype}",
        confidence="high",
        reason="Category inferred from extension and quick inspection",
        evidence=[
            f"extension={file_record.extension}",
            f"size_bytes={file_record.size_bytes}",
        ],
    )

    return file_record


# =========================================================
# STUDY GROUPING
# =========================================================

def analyze_files_for_study_grouping(
    files: List[Path],
    input_root: Path,
    logger: DecisionLogger,
) -> Tuple[List[FileRecord], List[VariableRecord], StudySignals, Dict[str, str]]:
    file_records: List[FileRecord] = []
    variable_records: List[VariableRecord] = []
    text_signals: List[StudySignals] = []
    file_text_cache: Dict[str, str] = {}

    for path in files:
        file_record = build_file_record(path, input_root, logger)
        file_records.append(file_record)

        text = ""
        if file_record.category in {"document", "text"}:
            text = read_text_from_document(path, logger)
            if text:
                file_text_cache[str(path)] = text[:TEXT_SNIPPET_STORE_CHARS]
                text_signals.append(extract_study_signals_from_text(text))

        if file_record.category == "tabular":
            variable_records.extend(extract_variable_records_from_table_file(path, logger))

            # Also inspect column names and head rows as text signals
            if pd is not None:
                try:
                    if detect_extension(path) in {".csv", ".tsv"}:
                        df = try_read_csv(path)
                        if df is not None:
                            probe = " ".join(map(str, df.columns.tolist()))
                            probe += " " + " ".join(df.astype(str).head(10).fillna("").values.flatten().tolist())
                            text_signals.append(extract_study_signals_from_text(probe))
                    elif detect_extension(path) in {".xlsx", ".xls"}:
                        sheets = try_read_excel_sheets(path)
                        for _, df in sheets.items():
                            probe = " ".join(map(str, df.columns.tolist()))
                            probe += " " + " ".join(df.astype(str).head(10).fillna("").values.flatten().tolist())
                            text_signals.append(extract_study_signals_from_text(probe))
                except Exception:
                    pass

    merged_signals = merge_study_signals(text_signals)

    logger.log(
        step="study_signal_merge",
        decision="merged study-level signals from parsed files",
        confidence="high" if text_signals else "low",
        reason="Combined accessions, disease terms, modality terms, treatment terms, and timepoint clues",
        evidence=[
            f"GEO={','.join(merged_signals.geo_accessions[:5]) or 'none'}",
            f"cancer_terms={','.join(merged_signals.cancer_terms[:5]) or 'none'}",
            f"modality_terms={','.join(merged_signals.modality_terms[:5]) or 'none'}",
            f"timepoints={','.join(merged_signals.timepoints[:5]) or 'none'}",
        ],
    )

    return file_records, variable_records, merged_signals, file_text_cache


# =========================================================
# OUTPUT WRITERS
# =========================================================

def write_file_manifest(records: List[FileRecord], out_csv: Path) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_variable_manifest(records: List[VariableRecord], out_csv: Path) -> None:
    rows = [asdict(r) for r in records]
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_study_summary(
    out_path: Path,
    study_folder_name: str,
    signals: StudySignals,
    file_records: List[FileRecord],
    variable_records: List[VariableRecord],
) -> None:
    category_counts = Counter(r.category for r in file_records)
    subtype_counts = Counter(r.subtype for r in file_records)

    lines = []
    lines.append(f"study_folder_name: {study_folder_name}")
    lines.append("")
    lines.append("primary_signals:")
    lines.append(f"  geo_accessions: {', '.join(signals.geo_accessions) if signals.geo_accessions else 'none'}")
    lines.append(f"  dois: {', '.join(signals.dois) if signals.dois else 'none'}")
    lines.append(f"  pmids: {', '.join(signals.pmids) if signals.pmids else 'none'}")
    lines.append(f"  cancer_terms: {', '.join(signals.cancer_terms) if signals.cancer_terms else 'none'}")
    lines.append(f"  modality_terms: {', '.join(signals.modality_terms) if signals.modality_terms else 'none'}")
    lines.append(f"  treatment_terms: {', '.join(signals.treatment_terms) if signals.treatment_terms else 'none'}")
    lines.append(f"  response_terms: {', '.join(signals.response_terms) if signals.response_terms else 'none'}")
    lines.append(f"  timepoints: {', '.join(signals.timepoints) if signals.timepoints else 'none'}")
    lines.append(f"  trial_names: {', '.join(signals.trial_names) if signals.trial_names else 'none'}")
    lines.append("")
    lines.append("file_counts_by_category:")
    for k, v in sorted(category_counts.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("file_counts_by_subtype:")
    for k, v in sorted(subtype_counts.items()):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"tabular_variables_extracted: {len(variable_records)}")
    lines.append("")
    if signals.title_candidates:
        lines.append("title_candidates:")
        for item in signals.title_candidates[:10]:
            lines.append(f"  - {item}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def copy_files_into_study_folder(
    files: List[Path],
    input_root: Path,
    study_root: Path,
    logger: DecisionLogger,
) -> None:
    import shutil

    for path in files:
        rel = path.relative_to(input_root)
        dst = study_root / "source_files" / rel
        ensure_dir(dst.parent)
        try:
            shutil.copy2(path, dst)
        except Exception as exc:
            logger.log(
                step="copy_files",
                decision=f"failed to copy {path.name}",
                confidence="low",
                reason=str(exc),
            )


def write_text_snippets(
    snippets: Dict[str, str],
    input_root: Path,
    study_root: Path,
) -> None:
    snippets_root = study_root / "parsed_text_snippets"
    ensure_dir(snippets_root)

    for abs_path, text in snippets.items():
        path = Path(abs_path)
        rel = path.relative_to(input_root)
        out_name = safe_slug(str(rel).replace("\\", "_").replace("/", "_"), max_len=120) + ".txt"
        (snippets_root / out_name).write_text(text, encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess and organize a mixed intake folder for training data studies."
    )
    parser.add_argument(
        "--input",
        dest="input_folder",
        default=INPUT_DATA_FOLDER,
        help="Folder containing all PDFs and data to be assessed in this run.",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        default=TRAINING_SORT_ROOT,
        help="Root folder where organized study folders will be created.",
    )
    return parser.parse_args()


def main() -> None:
    for k in analysis_stats:
        analysis_stats[k] = 0
    args = parse_args()

    input_folder = Path(args.input_folder).expanduser()
    output_root = Path(args.output_root).expanduser()

    if not str(input_folder).strip():
        print("ERROR: No input folder provided.")
        print('Set INPUT_DATA_FOLDER in the config, or pass --input "path_to_folder"')
        sys.exit(1)

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"ERROR: Input folder does not exist or is not a directory: {input_folder}")
        sys.exit(1)

    ensure_dir(output_root)

    logger = DecisionLogger(verbose=VERBOSE)
    
    client = get_openai_client()

    if USE_OPENAI_API:
        if client is None:
            print("⚠ API enabled but client not initialized (missing key or package)")
        else:
            print("✔ OpenAI API client initialized")

    logger.log(
        step="run_start",
        decision="started intake assessment",
        confidence="high",
        reason="Input folder accepted",
        evidence=[str(input_folder)],
    )

    files = collect_input_files(input_folder)
    analysis_stats["total_files"] = len(files)

    logger.log(
        step="file_discovery",
        decision=f"discovered {len(files)} files",
        confidence="high",
        reason="Recursive directory scan complete",
    )

    if not files:
        print("No files found.")
        sys.exit(0)

    file_records, variable_records, merged_signals, file_text_cache = analyze_files_for_study_grouping(
        files=files,
        input_root=input_folder,
        logger=logger,
    )

    study_folder_name = choose_study_folder_name(merged_signals, files)

    if USE_OPENAI_API and client is not None:
        api_label = api_generate_study_label(client, merged_signals)
        if api_label:
            logger.log(
                step="api_label",
                decision=f"api suggested label: {api_label}",
                confidence="medium",
                reason="LLM generated label from study signals",
            )
            
    study_root = output_root / study_folder_name
    ensure_dir(study_root)

    logger.log(
        step="study_grouping",
        decision=f"grouped run into study folder: {study_folder_name}",
        confidence="medium" if merged_signals.geo_accessions or merged_signals.title_candidates else "low",
        reason="Folder name inferred from strongest available accessions and study descriptors",
        evidence=[
            f"geo={','.join(merged_signals.geo_accessions[:3]) or 'none'}",
            f"title={merged_signals.title_candidates[0] if merged_signals.title_candidates else 'none'}",
        ],
    )

    write_file_manifest(file_records, study_root / "file_manifest.csv")
    write_variable_manifest(variable_records, study_root / "variable_manifest.csv")
    write_study_summary(
        out_path=study_root / "study_summary.txt",
        study_folder_name=study_folder_name,
        signals=merged_signals,
        file_records=file_records,
        variable_records=variable_records,
    )

    # API review (correct scope)
    review_payload = {"api_used": False, "result": "not_run"}

    if USE_OPENAI_API and client is not None:
        summary_text = (study_root / "study_summary.txt").read_text(encoding="utf-8")
        review_payload = api_review_dataset_summary(client, summary_text)

    (study_root / "api_review.json").write_text(
        json.dumps(review_payload, indent=2),
        encoding="utf-8",
    )

    write_text_snippets(file_text_cache, input_folder, study_root)

    logger.save_json(study_root / "decision_log.json")
    logger.save_text(study_root / "decision_log.txt")

    if COPY_SOURCE_FILES:
        copy_files_into_study_folder(
            files=files,
            input_root=input_folder,
            study_root=study_root,
            logger=logger,
        )
        logger.log(
            step="copy_files",
            decision="copied source files into study folder",
            confidence="high",
            reason="COPY_SOURCE_FILES is enabled",
        )
    else:
        logger.log(
            step="copy_files",
            decision="did not copy source files",
            confidence="high",
            reason="COPY_SOURCE_FILES is disabled",
        )

    logger.log(
        step="run_complete",
        decision="finished intake assessment",
        confidence="high",
        reason="All outputs written",
        evidence=[str(study_root)],
    )

    logger.print_simplified_summary()

    helper_patch_ok = all(
        name in globals()
        for name in [
            "series_example_values",
            "infer_semantic_type_from_name_and_values",
            "extract_units_from_series",
        ]
    )

    print("Run complete.")
    print(f"Study folder: {study_root}")
    print(f"File manifest: {study_root / 'file_manifest.csv'}")
    print(f"Variable manifest: {study_root / 'variable_manifest.csv'}")
    print(f"Decision log: {study_root / 'decision_log.txt'}")
    print(f"Helper patch detected at runtime: {'YES' if helper_patch_ok else 'NO'}")

    pdf_files_seen = sum(1 for r in file_records if r.extension == ".pdf")
    print(f"PDF robustness patch active: YES")
    print(f"PDF files detected in this run: {pdf_files_seen}")
    print("PDF parsing now logs page counts and warns on low extracted text or many empty pages")
    print("\n" + "="*70)
    print("DATA COVERAGE SUMMARY")
    print("="*70)

    print(f"Total files processed: {analysis_stats['total_files']}")

    print("\nPDF files:")
    print(f"  total: {analysis_stats['pdf_files']}")
    print(f"  failed to parse: {analysis_stats['pdf_failed']}")
    print(f"  low text extracted: {analysis_stats['pdf_low_text']}")

    print("\nTabular files:")
    print(f"  total: {analysis_stats['tabular_files']}")
    print(f"  failed to parse: {analysis_stats['tabular_failed']}")

    print("\nImage files:")
    print(f"  total: {analysis_stats['image_files']}")
    print(f"  failed to inspect: {analysis_stats['image_failed']}")

    print("\nText/document files:")
    print(f"  total: {analysis_stats['text_files']}")
    print(f"  failed to read: {analysis_stats['text_failed']}")

    print("\nInterpretation:")
    if analysis_stats["pdf_failed"] > 0 or analysis_stats["tabular_failed"] > 0:
        print("⚠ Some files were not fully processed. Review decision_log.txt")
    else:
        print("✔ All files processed without major failure")

    print("="*70)

    print(f"API review executed: {review_payload['api_used']}")

if __name__ == "__main__":
    main()