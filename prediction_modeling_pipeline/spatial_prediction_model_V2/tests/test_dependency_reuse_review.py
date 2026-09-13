import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spm_v2.dependency_reuse import INVARIANT_FUNCTIONS, reuse_spatial_partition, sha, verify_invariant_code

HOOK = """        if arm=='composition_plus_spatial' and reuse_spatial_from and reuse_spatial_partition(reuse_spatial_from,root,name,configuration,pool):
            print('REUSED verified unchanged spatial arm',name,flush=True)
            continue
"""


def code_pair(folder):
    old = folder / 'old'; new = folder / 'new'; old.mkdir(); new.mkdir()
    base = '\n'.join(f'def {name}():\n    return 1\n' for name in INVARIANT_FUNCTIONS)
    before = "def partition_run(root,name,pool):\n    for arm in ['composition','composition_plus_spatial']:\n        score = 1\n    return score\n"
    after = before.replace('root,name,pool):', 'root,name,pool,configuration=None,reuse_spatial_from=None):').replace('        score = 1\n', HOOK + '        score = 1\n')
    for p, text in [(old, base + before), (new, base + after)]:
        for filename in ['feature_reference.py', 'model_training.py', 'target_building.py', 'validation.py']:
            (p / filename).write_text('UNCHANGED', encoding='utf-8')
        (p / 'leakage_evaluation.py').write_text(text, encoding='utf-8')
    return old, new


class DependencyReuseReviewTests(unittest.TestCase):
    def test_exact_guard_and_skip_preserve_original_computation(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = code_pair(Path(folder)); verify_invariant_code(old, new)

    def test_cannot_hide_computations_in_removed_hook(self):
        changes = [
            lambda s: s.replace('            continue', '            score = 999\n            continue'),
            lambda s: s.replace('            continue', '            continue\n        else:\n            score = 999'),
            lambda s: s.replace("arm=='composition_plus_spatial' and reuse_spatial_from", "arm=='composition' and reuse_spatial_from"),
            lambda s: s.replace(HOOK, HOOK + HOOK),
            lambda s: s.replace('configuration=None,reuse_spatial_from=None', 'configuration=99,reuse_spatial_from=None'),
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                old, new = code_pair(Path(folder)); path = new / 'leakage_evaluation.py'
                path.write_text(change(path.read_text()), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'reuse hook'):
                    verify_invariant_code(old, new)

    def fixture(self, folder):
        old_code, new_code = code_pair(folder)
        source = folder / 'source'; target = folder / 'target'
        for root, code in [(source, old_code), (target, new_code)]:
            (root / 'code_snapshot').mkdir(parents=True)
            shutil.copytree(code, root / 'code_snapshot/spm_v2')
        source_config = dict(identity='source', settings=dict(seed=42), environment=dict(python='fixture'),
                             inputs={k: dict(path=k, sha256=k) for k in ['teacher', 'raw_features', 'metadata']})
        target_config = dict(source_config, identity='derivative')
        (source / 'configuration.json').write_text(json.dumps(source_config))
        partition = dict(name='outer_0', train_ids=['a', 'b'], test_ids=['c'])
        for root in [source, target]:
            p = root / 'outer_0'; p.mkdir()
            (p / 'partition.json').write_text(json.dumps(partition))
            for name in ['feature_reference.joblib', 'training_feature_filter.tsv', 'reference_dependencies.json', 'training_eligibility.tsv']:
                (p / name).write_bytes(b'frozen dependency')
        arm = source / 'outer_0/composition_plus_spatial'; arm.mkdir()
        (arm / 'permissible_pool.tsv').write_text('feature_name\nf1\nf2\n')
        (arm / 'heldout_predictions.parquet').write_bytes(b'frozen numerical checkpoint')
        p = source / 'outer_0'
        files = {str(f.relative_to(p)): sha(f) for f in p.rglob('*') if f.is_file()}
        (p / 'COMPLETE.json').write_text(json.dumps(dict(identity='source', files=files)))
        return source, target, target_config, arm

    def test_unmanifested_source_file_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, config, arm = self.fixture(Path(folder))
            (arm / 'unreviewed_predictions.tsv').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                reuse_spatial_partition(source, target, 'outer_0', config, ['f1', 'f2'])
            self.assertFalse((target / 'outer_0/composition_plus_spatial').exists())

    def test_manifested_files_copy_identically_and_order_change_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, config, arm = self.fixture(Path(folder))
            self.assertTrue(reuse_spatial_partition(source, target, 'outer_0', config, ['f1', 'f2']))
            for filename in ['permissible_pool.tsv', 'heldout_predictions.parquet']:
                self.assertEqual(sha(arm / filename), sha(target / 'outer_0/composition_plus_spatial' / filename))
            with self.assertRaisesRegex(ValueError, 'ordered spatial-arm'):
                reuse_spatial_partition(source, target, 'outer_0', config, ['f2', 'f1'])


if __name__ == '__main__':
    unittest.main()
