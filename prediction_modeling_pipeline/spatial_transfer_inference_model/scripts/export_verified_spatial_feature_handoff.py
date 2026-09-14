"""Export verified cumulative Step09 measurements with external sample identities.

Inputs are a JSON list of extraction groups and a TSV identity audit. Each group
binds the Step07/Step09 tables and internal-to-external sample map by SHA256.
The audit binds each external identity to its original matrix hash and retained
barcode check. Relative source paths resolve beside their containing manifest.

This command preserves measurement values and explicit NA. Distances remain in
the extraction's coordinate units (normally full-resolution image pixels); no
normalization, biological-absence zero fill, teacher construction or fitting is
performed. Apply a saved feature reference and estimator/atlas after export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from _stim_utils import read_table


def sha(path: Path) -> str:
    """Hash a source without loading a potentially large matrix into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_path(value: str, authority: Path) -> Path:
    """Resolve an explicit source path relative to its manifest when needed."""
    path = Path(value)
    return path if path.is_absolute() else authority.parent / path


def unique_keys(frame: pd.DataFrame, column: str, description: str) -> None:
    """Reject missing, duplicate or whitespace-ambiguous sample keys."""
    if column not in frame or frame[column].isna().any():
        raise ValueError(f'Missing {description}: {column}')
    values = frame[column].astype(str)
    if values.str.strip().eq('').any() or values.ne(values.str.strip()).any() or values.duplicated().any():
        raise ValueError(f'Duplicate, blank or ambiguous {description}: {column}')


def verified_true(value: object) -> bool:
    """Parse an audit Boolean explicitly; the string False is never truthy."""
    return str(value).strip().lower() in {'true', '1'}


def export(manifest_path: str | Path, identity_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Verify every source/key before exporting one row per external section."""
    manifest_path, identity_path, output = map(Path, [manifest_path, identity_path, output_path])
    if output.exists() or output.with_suffix('.manifest.json').exists():
        raise FileExistsError('Use a new feature handoff output path')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Source manifest must contain at least one extraction group')
    identities = read_table(identity_path)
    unique_keys(identities, 'sample_id', 'canonical external identities')
    required = {'source_matrix_path', 'source_matrix_sha256', 'retained_barcode_subset', 'duplicate_retained_barcodes'}
    if not required.issubset(identities.columns):
        raise ValueError(f'Identity audit missing columns: {sorted(required - set(identities.columns))}')
    parts, sources = [], []
    for group in manifest:
        paths = {}
        for key in ['mapping', 'source07', 'source09']:
            path = source_path(group[key + '_path'], manifest_path)
            if sha(path) != group[key + '_sha256']:
                raise ValueError(f'Protected feature source hash mismatch: {path}')
            paths[key] = path
            sources.append({'path': str(path), 'sha256': sha(path)})
        mapping = read_table(paths['mapping'])
        unique_keys(mapping, 'internal_sample_id', 'internal sample mapping')
        unique_keys(mapping, 'transfer_sample_id', 'external sample mapping')
        frame = read_table(paths['source09'])
        unique_keys(frame, 'sample_id', 'raw measurement identities')
        if set(frame.sample_id) != set(mapping.internal_sample_id):
            raise ValueError('Raw measurement identities do not match the verified mapping')
        mapped = frame.sample_id.map(mapping.set_index('internal_sample_id').transfer_sample_id)
        expected = group['sample_ids']
        if len(set(expected)) != len(expected) or set(mapped) != set(expected):
            raise ValueError('External identity set differs from source manifest')
        if not set(mapped).issubset(set(identities.sample_id)):
            raise ValueError('Mapped external sample is absent from the identity audit')
        subset = identities.set_index('sample_id').loc[mapped]
        for row in subset.to_dict('records'):
            matrix = source_path(row['source_matrix_path'], identity_path)
            if sha(matrix) != row['source_matrix_sha256']:
                raise ValueError('Canonical external source matrix hash mismatch')
            if not verified_true(row['retained_barcode_subset']) or pd.isna(row['duplicate_retained_barcodes']) or float(row['duplicate_retained_barcodes']) != 0:
                raise ValueError('Retained barcode identity gate failed')
        frame.insert(1, 'original_internal_sample_id', frame.sample_id)
        frame['sample_id'] = mapped
        frame['external_source_identity_status'] = frame.sample_id.map(
            lambda value: 'ACCESSION_AND_CANONICAL_SOURCE_VERIFIED' if str(value).startswith('GSM') else 'CANONICAL_LOCAL_PATH_AND_LIBRARY_VERIFIED')
        frame['source_spatial_output_root'] = str(paths['source09'].parents[1])
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    unique_keys(result, 'sample_id', 'combined external sample identities')
    if set(result.sample_id) != set(identities.sample_id):
        raise ValueError('Combined handoff identity set differs from the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    report = dict(status='PASS', samples=result.sample_id.tolist(), rows=len(result), columns=len(result.columns),
                  source_manifests={'feature_manifest_sha256': sha(manifest_path), 'identity_audit_sha256': sha(identity_path)},
                  sources=sources, output_sha256=sha(output), raw_feature_reextraction=False,
                  expression_or_histology_retraining=False, zero_filling=False,
                  definition='Export of hash-verified cumulative spatial measurements with canonical external identities; no teacher labels attached')
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'rows', 'columns', 'output_sha256']}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--identity-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.source_manifest, args.identity_audit, args.output)


if __name__ == '__main__':
    main()
