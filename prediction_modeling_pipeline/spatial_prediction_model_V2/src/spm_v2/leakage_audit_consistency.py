"""Independent relational checks for the saved grouped evaluation.

This audit does not fit or select models. It connects numerical summaries to
their source prediction rows so mutually consistent derived errors cannot pass.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def unique_keys(frame, keys, context):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise AssertionError(f'{context}: missing or duplicate keys {keys}')


def same_table(actual, expected, keys, context):
    unique_keys(actual, keys, context)
    unique_keys(expected, keys, context + ' expected')
    missing = set(expected.columns) - set(actual.columns)
    if missing:
        raise AssertionError(f'{context}: missing columns {sorted(missing)}')
    a = actual[expected.columns].sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False,
                                      rtol=1e-10, atol=1e-12)
    except AssertionError as exc:
        raise AssertionError(f'{context}: source-to-derived table mismatch') from exc


def bind_input_arguments(config, metadata, teacher):
    for key, path in [('metadata', metadata), ('teacher', teacher)]:
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if h != config['inputs'][key]['sha256']:
            raise AssertionError(f'{key}: loaded argument differs from recorded input hash')


def bind_reviewed_feature_classes(saved, source, raw_columns):
    expected=source.rename(columns={'strict_class':'feature_class'})
    expected=expected[expected.feature_name.ne('sample_id')]
    if set(expected.feature_name)!=set(raw_columns)-{'sample_id'}:
        raise AssertionError('Reviewed class source does not match exact raw feature universe')
    same_table(saved,expected,['feature_name'],'reviewed feature classes')


def audit_partition_keys(parts, meta):
    unique_keys(meta, ['sample_id'], 'metadata')
    unique_keys(parts, ['evaluation', 'sample_id'], 'partitions')
    names = {f'outer_{i}' for i in range(5)} | {'independent_selection'}
    if set(parts.evaluation) != names or set(parts.partition) != {'train', 'test'}:
        raise AssertionError('Incomplete evaluation or invalid partition names')
    outer_test = parts[parts.evaluation.str.startswith('outer_') & parts.partition.eq('test')]
    unique_keys(outer_test, ['sample_id'], 'outer test membership')
    if set(outer_test.sample_id) != set(meta.sample_id):
        raise AssertionError('Every cohort sample must appear in one outer test fold')


def audit_screen_keys_targets(sm, sp, ss, teacher, eligible, repeats):
    unique_keys(sm, ['drug_key', 'repeat'], 'screen metrics')
    unique_keys(sp, ['drug_key', 'repeat', 'sample_id'], 'screen predictions')
    unique_keys(ss, ['drug_key'], 'screen summaries')
    expected = {(d, r) for d in eligible for r in range(repeats)}
    if set(zip(sm.drug_key, sm.repeat)) != expected:
        raise AssertionError('Incomplete screen metric treatment/repeat universe')
    if set(zip(sp.drug_key, sp.repeat)) != expected or set(ss.drug_key) != set(eligible):
        raise AssertionError('Incomplete screen prediction or summary treatment universe')
    source = teacher[['sample_id', 'drug_key', 'fused_residual_vs_prior']]
    joined = sp.merge(source, on=['sample_id', 'drug_key'], how='left', validate='many_to_one')
    if not np.isfinite(joined.fused_residual_vs_prior).all():
        raise AssertionError('Screen target has no finite teacher source')
    if not np.allclose(joined.target, joined.fused_residual_vs_prior, rtol=0, atol=1e-12):
        raise AssertionError('Screen target differs from corrected teacher')


def verify_prediction_metadata(pred, meta):
    keys = ['sample_id', 'evaluation_group', 'metadata_dataset_id']
    expected = pred[['sample_id']].merge(meta[keys], on='sample_id', how='left', validate='many_to_one')
    if expected[keys[1:]].isna().any().any():
        raise AssertionError('Prediction sample lacks verified grouping/source metadata')
    for c in keys[1:]:
        if not np.array_equal(pred[c].astype(str).to_numpy(), expected[c].astype(str).to_numpy()):
            raise AssertionError(f'Prediction {c} differs from verified metadata')


def expected_block_predictions(pred, meta, family=None):
    base = pred.drop(columns=['evaluation_group', 'metadata_dataset_id'], errors='ignore')
    base = base.merge(meta[['sample_id', 'evaluation_group', 'metadata_dataset_id']],
                      on='sample_id', how='left', validate='many_to_one')
    if base[['evaluation_group', 'metadata_dataset_id']].isna().any().any():
        raise AssertionError('Block prediction has an unknown sample identity')
    if family is not None:
        base = base[base.drug_key.isin(family)]
    keys = ['evaluation_group', 'metadata_dataset_id', 'drug_key']
    if 'arm' in base:
        keys = ['arm'] + keys
    universe = base[keys].drop_duplicates()
    finite = base[base.target_available]
    grouped = finite.groupby(keys, as_index=False).agg(
        target=('target', 'mean'), prediction=('prediction', 'mean'),
        baseline=('baseline', 'mean'), sections=('sample_id', 'count'))
    return universe.merge(grouped, on=keys, how='left', validate='one_to_one')


def expected_matched(metrics):
    unique_keys(metrics, ['arm', 'drug_key'], 'matched source metrics')
    a = metrics[metrics.arm.eq('composition')]
    b = metrics[metrics.arm.eq('composition_plus_spatial')]
    pair = a.merge(b, on='drug_key', suffixes=('_composition', '_spatial'), validate='one_to_one')
    for c in ['pearson', 'spearman', 'r2', 'mae', 'rmse', 'rmse_improvement']:
        pair['delta_' + c] = pair[c + '_spatial'] - pair[c + '_composition']
    if not pair.n_composition.eq(pair.n_spatial).all():
        raise AssertionError('Matched arms have different observed sample counts')
    return pair


def expected_source_influence(pred):
    from .leakage_audit import recompute
    rows = []
    for (arm, drug), s in pred[pred.target_available].groupby(['arm', 'drug_key']):
        for source in sorted(s.metadata_dataset_id.unique()):
            kept = s[s.metadata_dataset_id.ne(source)]
            rows.append(dict(arm=arm, drug_key=drug, removed_source=source,
                             **recompute(kept.target, kept.prediction, kept.baseline)))
    return pd.DataFrame(rows)


def audit_null_structure(grouped, mappings, summary):
    unique_keys(grouped, ['evaluation_group', 'drug_key'], 'independent block predictions')
    if (grouped.groupby('evaluation_group').metadata_dataset_id.nunique() != 1).any():
        raise AssertionError('An evaluation block spans multiple recorded sources')
    y = grouped.pivot(index='evaluation_group', columns='drug_key', values='target').sort_index()
    source = grouped.drop_duplicates('evaluation_group').set_index('evaluation_group').metadata_dataset_id
    strata = {}
    for g, row in y.iterrows():
        strata.setdefault((source[g], tuple(np.isfinite(row))), []).append(g)
    movable = sum(len(s) for s in strata.values() if len(s) > 1)
    unique_keys(mappings, ['permutation_id', 'evaluation_group'], 'null mappings')
    if set(mappings.permutation_id) != set(range(1000)):
        raise AssertionError('Null must contain all 1000 permutation IDs')
    seen = set()
    for _, m in mappings.groupby('permutation_id'):
        if set(m.evaluation_group) != set(y.index) or set(m.source_group) != set(y.index):
            raise AssertionError('Null mapping is not a complete block bijection')
        seen.add(tuple(m.set_index('evaluation_group').source_group.reindex(y.index)))
    expected = dict(tested_family=y.shape[1], permutation_ids=1000,
                    effective_unique_mappings=len(seen), movable_groups=movable,
                    null_denominator=1000)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(f'Null summary {key} differs from saved numerical structure')
    if sorted(summary.get('stratum_sizes', [])) != sorted(map(len, strata.values())):
        raise AssertionError('Null stratum sizes differ from verified source/missingness groups')


def audit_relational_consistency(root, metadata, teacher):
    root = Path(root)
    config = json.loads((root / 'configuration.json').read_text())
    bind_input_arguments(config, metadata, teacher)
    meta = pd.read_csv(metadata, sep='\t')
    t = pd.read_csv(teacher, sep='\t', usecols=['sample_id', 'drug_key', 'fused_residual_vs_prior'])
    unique_keys(t, ['sample_id', 'drug_key'], 'teacher')
    import joblib
    from .feature_reference import numeric
    raw = pd.read_csv(config['inputs']['raw_features']['path'])
    unique_keys(raw, ['sample_id'], 'raw feature source')
    classes = pd.read_csv(root / 'feature_classes.tsv', sep='\t')
    if 'feature_classes' in config['inputs']:
        reviewed=pd.read_csv(config['inputs']['feature_classes']['path'],sep='\t')
        bind_reviewed_feature_classes(classes,reviewed,raw.columns)
    parts = pd.read_csv(root / 'partitions.tsv', sep='\t')
    audit_partition_keys(parts, meta)
    for name, rows in parts.groupby('evaluation'):
        train = set(rows.loc[rows.partition.eq('train'), 'sample_id'])
        deps = json.loads((root / name / 'reference_dependencies.json').read_text())
        if len(deps['training_ids']) != len(train) or set(deps['training_ids']) != train:
            raise AssertionError(f'{name}: reference dependency training IDs differ')
        reference = joblib.load(root / name / 'feature_reference.joblib')
        if len(reference.training_ids) != len(train) or set(reference.training_ids) != train:
            raise AssertionError(f'{name}: fitted reference training identities differ')
        # Reproduce the recorded transform row order before exact unique-value
        # counting. Pandas row reductions can differ by one float64 ULP when
        # batch shape changes, splitting a numerical tie without changing the
        # fitted reference or the nonconstant filter decision. Independently
        # check train-only transformation with the existing numerical tolerance.
        train_ids = rows.loc[rows.partition.eq('train'), 'sample_id'].tolist()
        test_ids = rows.loc[rows.partition.eq('test'), 'sample_id'].tolist()
        indexed = raw.set_index('sample_id', drop=False)
        transformed = reference.transform(indexed.loc[train_ids + test_ids]).set_index('sample_id', drop=False).loc[train_ids]
        train_only = reference.transform(indexed.loc[train_ids]).set_index('sample_id', drop=False).loc[train_ids]
        allowed = classes.loc[classes.feature_class.ne('excluded'), 'feature_name'].tolist()
        x = transformed[allowed].apply(numeric)
        independent_x = train_only[allowed].apply(numeric)
        if not np.array_equal(x.isna(), independent_x.isna()):
            raise AssertionError(f'{name}: reference missingness depends on batch')
        if not np.allclose(x, independent_x, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise AssertionError(f'{name}: reference values depend on batch beyond numerical tolerance')
        fraction = x.notna().mean(); counts = x.nunique(dropna=True)
        independent_keep = (independent_x.notna().mean() >= .2) & (independent_x.nunique(dropna=True) > 1)
        if not np.array_equal((fraction >= .2) & (counts > 1), independent_keep):
            raise AssertionError(f'{name}: training feature filter depends on transform batch')
        expected_filter = pd.DataFrame(dict(feature_name=allowed,
                                           train_nonmissing_fraction=fraction.to_numpy(),
                                           train_unique=counts.to_numpy(),
                                           available_training_feature=((fraction >= .2) & (counts > 1)).to_numpy()))
        same_table(pd.read_csv(root / name / 'training_feature_filter.tsv', sep='\t'),
                   expected_filter, ['feature_name'], name + ' raw-to-training feature filter')
        eligibility = pd.read_csv(root / name / 'training_eligibility.tsv', sep='\t')
        unique_keys(eligibility, ['drug_key'], name + ' eligibility')
        eligible = set(eligibility.loc[eligibility.eligible, 'drug_key'])
        for arm in ['composition', 'composition_plus_spatial']:
            ad = root / name / arm
            audit_screen_keys_targets(pd.read_parquet(ad / 'screen_metrics.parquet'),
                                      pd.read_parquet(ad / 'screen_predictions.parquet'),
                                      pd.read_csv(ad / 'training_screen_summary.tsv', sep='\t'),
                                      t, eligible, config['settings']['screen_repeats'])
            splits = pd.read_csv(ad / 'pooled_splits.tsv', sep='\t')
            unique_keys(splits, ['repeat', 'sample_id'], name + '/' + arm + ' pooled splits')
            evidence = pd.read_parquet(ad / 'pooled_feature_evidence.parquet')
            unique_keys(evidence, ['repeat', 'feature_name'], name + '/' + arm + ' pooled evidence')
            if set(evidence.repeat) != set(range(config['settings']['pooled_repeats'])):
                raise AssertionError('Incomplete pooled feature evidence repetition universe')
            selected = pd.read_csv(ad / 'final_selected_features.tsv', sep='\t')
            unique_keys(selected, ['drug_key', 'selection_order'], 'final feature order')
            if set(selected.drug_key) != eligible:
                raise AssertionError('Final selected feature treatment universe differs from eligible set')
            for _, s in selected.groupby('drug_key'):
                if set(s.selection_order) != set(range(len(s))) or len(s) > config['settings']['max_features']:
                    raise AssertionError('Final feature ordering/count differs from documented contract')
    pred = pd.read_parquet(root / 'outer_heldout_predictions.parquet')
    verify_prediction_metadata(pred, meta)
    observed = pred[pred.target_available]
    for filename, keys in [('outer_fold_metrics.tsv', ['fold', 'arm', 'drug_key']),
                           ('outer_pooled_metrics.tsv', ['arm', 'drug_key']),
                           ('outer_block_metrics.tsv', ['arm', 'drug_key'])]:
        saved = pd.read_csv(root / filename, sep='\t')
        same_table(saved[keys], observed[keys].drop_duplicates(), keys, filename + ' keys')
    for metrics_name, pair_name in [('outer_pooled_metrics.tsv', 'matched_spatial_comparison.tsv'),
                                    ('outer_block_metrics.tsv', 'matched_spatial_comparison_block.tsv')]:
        metrics = pd.read_csv(root / metrics_name, sep='\t')
        same_table(pd.read_csv(root / pair_name, sep='\t'), expected_matched(metrics), ['drug_key'], pair_name)
    influence = pd.read_csv(root / 'source_influence_metrics.tsv', sep='\t')
    same_table(influence, expected_source_influence(pred), ['arm', 'drug_key', 'removed_source'], 'source influence')
    blocks = expected_block_predictions(pred, meta)
    # Outer summaries contain only blocks with a measured teacher target.
    blocks = blocks[blocks.target.notna()]
    same_table(pd.read_parquet(root / 'outer_block_predictions.parquet'), blocks,
               ['arm', 'drug_key', 'evaluation_group'], 'outer block predictions')
    family = pd.read_csv(root / 'independent_tested_family.tsv', sep='\t').drug_key.tolist()
    report = pd.read_csv(root / 'independent_validation_summary.tsv', sep='\t')
    unique_keys(report, ['drug_key'], 'independent validation summary')
    if family:
        independent = pd.read_parquet(root / 'independent_selection/composition_plus_spatial/heldout_predictions.parquet')
        grouped = pd.read_parquet(root / 'independent_block_predictions.parquet')
        same_table(grouped, expected_block_predictions(independent, meta, family),
                   ['evaluation_group', 'drug_key'], 'independent block predictions')
        audit_null_structure(grouped, pd.read_parquet(root / 'independent_null_mappings.parquet'),
                             json.loads((root / 'independent_null_audit.json').read_text()))
    result = dict(status='PASS', input_identity=config['identity'], partitions=int(parts.evaluation.nunique()),
                  outer_rows=len(pred), independent_family=len(family),
                  scope='source-to-derived relational completeness, no model fitting')
    destination = root / 'independent_relational_audit.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(destination)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--teacher', required=True)
    a = p.parse_args()
    print(json.dumps(audit_relational_consistency(a.root, a.metadata, a.teacher), indent=2))
