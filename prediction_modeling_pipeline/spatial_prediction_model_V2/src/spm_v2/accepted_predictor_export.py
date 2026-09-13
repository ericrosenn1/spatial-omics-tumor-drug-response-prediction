"""Export exact independently evaluated predictors after numerical audit.

Optional all-cohort fitting retains the discovery-selected ordered features.
That derivative is a deployment fit and does not inherit held-out predictions.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from .predictor_bundle import SpatialPredictorBundle, save_bundle, load_bundle, read_feature_table


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_keys(summary):
    if summary.drug_key.isna().any() or summary.drug_key.duplicated().any():
        raise ValueError('Duplicate or missing independent validation treatment keys')
    if not summary.accepted.isin([True, False]).all():
        raise ValueError('Independent accepted flag must be explicit Boolean')
    return sorted(summary.loc[summary.accepted.eq(True), 'drug_key'].astype(str))


def verify_frozen_discovery(source, selected, training_ids, settings):
    expected = selected.sort_values('selection_order').feature_name.astype(str).tolist()
    if source['features'] != expected or len(expected) != len(set(expected)):
        raise ValueError('Frozen discovery feature order differs from saved final selection')
    if set(source['training_samples']) != set(training_ids) or len(source['training_samples']) != len(training_ids):
        raise ValueError('Frozen discovery estimator training samples differ from teacher/partition')
    if selected.seed.nunique() != 1 or int(selected.seed.iloc[0]) != source['seed']:
        raise ValueError('Frozen discovery seed differs from feature-selection evidence')
    if source['target'] != 'fused_residual_vs_prior':
        raise ValueError('Unexpected discovery estimator target')
    params = source['estimator'].named_steps['model'].get_params()
    for name, expected_value in [('n_estimators', settings['estimators']),
                                 ('max_depth', settings['max_depth']),
                                 ('learning_rate', settings['learning_rate']),
                                 ('random_state', source['seed']), ('n_jobs', 1)]:
        if params[name] != expected_value:
            raise ValueError(f'Frozen discovery setting differs: {name}')
    return expected


def verify_bundle_reload(bundle, path, raw, expected=None, mode='heldout_evaluation_reproduction'):
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    observed = loaded.predict(raw, 'raw_reference', mode)
    original = bundle.predict(raw, 'raw_reference', mode)
    if not observed.equals(original):
        raise ValueError('Accepted bundle save/reload changes predictions')
    reverse = loaded.predict(raw.iloc[::-1, ::-1], 'raw_reference', mode).iloc[::-1].reset_index(drop=True)
    singles = pd.concat([loaded.predict(raw.iloc[[i]], 'raw_reference', mode) for i in range(len(raw))], ignore_index=True)
    if not observed.equals(reverse) or not observed.equals(singles):
        raise ValueError('Accepted bundle row/column or single/batch invariance failure')
    if expected is not None:
        exp = expected.set_index('sample_id').loc[observed.sample_id, 'prediction'].to_numpy(float)
        if not np.array_equal(observed.predicted_residual_vs_prior.to_numpy(float), exp):
            raise ValueError('Exact discovery bundle does not reproduce saved held-out estimator predictions')
    return observed


def export_accepted(root, output, fit_all_data_deployment=False):
    from .leakage_audit import audit
    from .feature_reference import FeatureReference
    root = Path(root); output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new versioned export directory; existing output is preserved')
    config = json.loads((root / 'configuration.json').read_text())
    if config.get('benchmark'):
        raise ValueError('Benchmark output cannot define an accepted model endpoint')
    prior_audit = json.loads((root / 'independent_numerical_audit.json').read_text())
    if prior_audit.get('status') != 'PASS' or prior_audit.get('input_identity') != config['identity']:
        raise ValueError('A compatible completed independent numerical audit is required before export')
    teacher_path = Path(config['inputs']['teacher']['path'])
    metadata_path = Path(config['inputs']['metadata']['path'])
    raw_path = Path(config['inputs']['raw_features']['path'])
    # Recheck actual numerical/relational outputs now, not only a stale PASS file.
    audit(root, metadata_path, teacher_path)
    source_reference = Path(__file__).with_name('feature_reference.py')
    if sha(source_reference) != config['code_hashes']['feature_reference.py']:
        raise ValueError('Current feature reference implementation differs from evaluated code snapshot')
    summary_path = root / 'independent_validation_summary.tsv'
    summary = pd.read_csv(summary_path, sep='\t'); accepted = accepted_keys(summary)
    output.mkdir(parents=True, exist_ok=True)
    columns = ['drug_key', 'bundle_file', 'sha256', 'n_features', 'n_fitted_rows', 'treatment_prior', 'support_status']
    if not accepted:
        pd.DataFrame(columns=columns).to_csv(output / 'bundle_manifest.tsv', sep='\t', index=False)
        result = dict(status='COMPLETE_NO_INDEPENDENTLY_SUPPORTED_MODELS', bundles=0,
                      fitted_estimator_refits=0, input_identity=config['identity'],
                      acceptance_source=str(summary_path.resolve()), acceptance_sha256=sha(summary_path))
        (output / 'bundle_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    raw = read_feature_table(raw_path)
    teacher = pd.read_csv(teacher_path, sep='\t', usecols=['sample_id', 'drug_key', 'fused_residual_vs_prior', 'treatment_prior'])
    parts = pd.read_csv(root / 'partitions.tsv', sep='\t')
    partition = parts[parts.evaluation.eq('independent_selection')]
    train_ids = partition.loc[partition.partition.eq('train'), 'sample_id'].tolist()
    test_ids = partition.loc[partition.partition.eq('test'), 'sample_id'].tolist()
    reference_path = root / 'independent_selection/feature_reference.joblib'
    reference = joblib.load(reference_path)
    if set(reference.training_ids) != set(train_ids):
        raise ValueError('Discovery reference includes incorrect training identities')
    arm = root / 'independent_selection/composition_plus_spatial'
    evidence = pd.read_csv(arm / 'final_selected_features.tsv', sep='\t')
    heldout = pd.read_parquet(arm / 'heldout_predictions.parquet')
    sources = {}
    for path in sorted((arm / 'discovery_models').glob('*.joblib')):
        artifact = joblib.load(path)
        key = artifact['drug_key']
        if key in sources:
            raise ValueError('Duplicate frozen discovery estimator treatment identity')
        sources[key] = (path, artifact)
    if not set(accepted) <= set(sources):
        raise ValueError('Accepted treatment lacks its exact frozen discovery estimator')
    reference_all = FeatureReference().fit(raw) if fit_all_data_deployment else None
    rows = []; deployment_rows = []; selections = []; checks = []; predictions = []; deployment_predictions = []
    for i, drug in enumerate(accepted, 1):
        path, artifact = sources[drug]
        sub = teacher[teacher.drug_key.eq(drug)]
        train = sub[sub.sample_id.isin(train_ids)]
        selected = evidence[evidence.drug_key.eq(drug)]
        features = verify_frozen_discovery(artifact, selected, train.sample_id.tolist(), config['settings'])
        priors = sub.treatment_prior.dropna().unique()
        if len(priors) != 1:
            raise ValueError('Accepted model prior reconstruction is ambiguous')
        provenance = dict(input_identity=config['identity'], acceptance_source=str(summary_path.resolve()),
                          acceptance_sha256=sha(summary_path), accepted_evidence=summary[summary.drug_key.eq(drug)].iloc[0].to_dict(),
                          source_estimator=str(path.resolve()), source_estimator_sha256=sha(path),
                          source_reference=str(reference_path.resolve()), source_reference_sha256=sha(reference_path),
                          evaluated_code_hashes=config['code_hashes'], environment=config['environment'],
                          evaluation_holdout_sample_ids=test_ids, reserved_cohort_sample_ids=raw.sample_id.astype(str).tolist(),
                          estimator_training_sample_ids=train.sample_id.astype(str).tolist(),
                          training_prediction_partition='DISCOVERY_FITTED_TRAINING_REPRODUCTION',
                          fit_scope='EXACT_FROZEN_DISCOVERY_ESTIMATOR; held-out estimates belong to this fitted route',
                          original_artifact_reference_hint=artifact.get('reference'),
                          reference_resolution='explicit partition reference path verified by source manifest')
        bundle = SpatialPredictorBundle(drug, features, artifact['estimator'], float(priors[0]),
                                        train.sample_id.astype(str).tolist(), reference, provenance,
                                        support_status='INDEPENDENTLY_SUPPORTED_EXACT_DISCOVERY_PREDICTOR')
        filename = f'treatment_{i:02d}_{hashlib.sha256(drug.encode()).hexdigest()[:12]}.joblib'
        test_raw = raw.set_index('sample_id', drop=False).loc[test_ids].reset_index(drop=True)
        pred = verify_bundle_reload(bundle, output / filename, test_raw, heldout[heldout.drug_key.eq(drug)])
        predictions.append(pred)
        rows.append(dict(drug_key=drug, bundle_file=filename, sha256=sha(output / filename),
                         n_features=len(features), n_fitted_rows=len(train), treatment_prior=float(priors[0]), support_status=bundle.support_status))
        selections += [dict(drug_key=drug, feature_position=j, feature_name=f) for j, f in enumerate(features)]
        checks.append(dict(drug_key=drug, exact_saved_heldout_predictions=True, reload=True, single_batch=True, row_column_reorder=True))
        if fit_all_data_deployment:
            transformed = reference_all.transform(raw).set_index('sample_id')
            x = transformed.loc[sub.sample_id, features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
            if x.isna().all(axis=0).any():
                raise ValueError('Fixed discovery feature is all-missing in all-data deployment fitting')
            pipeline = clone(artifact['estimator'])
            pipeline.fit(x, sub.fused_residual_vs_prior.to_numpy(float))
            dep_provenance = dict(provenance, fit_scope='ALL_COHORT_DEPLOYMENT_REFIT_FIXED_DISCOVERY_FEATURES',
                                  evaluation_holdout_sample_ids=[], estimator_training_sample_ids=sub.sample_id.tolist(),
                                  training_prediction_partition='ALL_DATA_FITTED_TRAINING_REPRODUCTION',
                                  performance_scope='Held-out estimates describe the source discovery estimator; this all-data derivative was not held-out evaluated')
            dep = SpatialPredictorBundle(drug, features, pipeline, float(priors[0]), sub.sample_id.astype(str).tolist(),
                                         reference_all, dep_provenance,
                                         support_status='DEPLOYMENT_REFIT_OF_INDEPENDENTLY_SUPPORTED_DISCOVERY_FEATURE_SET')
            dep_path = output / 'all_data_deployment' / filename
            deployment_predictions.append(verify_bundle_reload(dep, dep_path, raw, mode='training_reproduction'))
            deployment_rows.append(dict(rows[-1], bundle_file=filename, sha256=sha(dep_path),
                                        n_fitted_rows=len(sub), support_status=dep.support_status))
        print(f'[{i}/{len(accepted)}] exported exact independently supported discovery predictor: {drug}', flush=True)
    pd.DataFrame(rows).to_csv(output / 'bundle_manifest.tsv', sep='\t', index=False)
    pd.DataFrame(selections).to_csv(output / 'ordered_feature_manifest.tsv', sep='\t', index=False)
    pd.DataFrame(checks).to_csv(output / 'bundle_reproducibility_checks.tsv', sep='\t', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'heldout_evaluation_reproduction.tsv', sep='\t', index=False)
    if deployment_rows:
        pd.DataFrame(deployment_rows).to_csv(output / 'all_data_deployment/bundle_manifest.tsv', sep='\t', index=False)
        pd.concat(deployment_predictions, ignore_index=True).to_csv(output / 'all_data_deployment/all_data_fitted_predictions.tsv', sep='\t', index=False)
    result = dict(status='PASS', bundles=len(rows), fitted_estimator_refits=len(deployment_rows),
                  discovery_estimator_refits=0, input_identity=config['identity'],
                  acceptance_source=str(summary_path.resolve()), acceptance_sha256=sha(summary_path),
                  heldout_reproduction_rows=sum(map(len, predictions)), all_data_deployment_bundles=len(deployment_rows))
    (output / 'bundle_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
