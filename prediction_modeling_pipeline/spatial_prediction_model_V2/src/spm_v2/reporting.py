from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_table, write_text_report


def terminal_block(title: str, lines: list[str]) -> str:
    rule = "=" * max(70, len(title) + 4)
    return "\n".join([rule, title, rule] + lines)


def write_output_manifest(output_root: str | Path) -> pd.DataFrame:
    output_root = Path(output_root)
    rows = []
    if output_root.exists():
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                rows.append({
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                })
    df = pd.DataFrame(rows)
    write_table(df, output_root / "output_manifest.tsv")
    return df


def write_summary_report(output_root: str | Path, title: str, summary: dict, sections: dict[str, str] | None = None) -> Path:
    output_root = ensure_dir(output_root)
    lines = [title, ""]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    if sections:
        for section, body in sections.items():
            lines.extend(["", section, body])
    return write_text_report(output_root / "summary_report.txt", "\n".join(lines))
