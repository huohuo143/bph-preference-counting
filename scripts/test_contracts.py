#!/usr/bin/env python3
"""Check source preservation and acceptance boundaries with temporary fixtures.

These tests use no experimental counts as fabricated observations and perform
no full-image detection. The numerical integration fixture is synthetic.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import workflow


class WorkflowBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bph-skill-test-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_unowned_output_preserved(self):
        output = self.root / 'existing'
        output.mkdir()
        sentinel = output / 'keep.txt'
        sentinel.write_text('original content')
        with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
            workflow.create_output(output, 'statistics-replay')
        self.assertEqual(sentinel.read_text(), 'original content')
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_no_results_inside_source_or_skill(self):
        source = self.root / 'source'
        source.mkdir()
        with self.assertRaisesRegex(ValueError, '原始照片目录'):
            workflow.create_output(source / 'results', 'raw-replay', [source])
        with self.assertRaisesRegex(ValueError, 'Skill 自身'):
            workflow.create_output(workflow.SKILL / 'forbidden-output', 'raw-replay')

    def test_new_study_cannot_inherit_historical_acceptance(self):
        config = self.root / 'new-study.json'
        config.write_text(json.dumps({
            'profile': 'new-study',
            'validation_status': 'passed',
            'historical_acceptance_inherited': True,
            'groups': [{'id': 'E1', 'source_folder': str(self.root / 'raw')}],
        }))
        output = self.root / 'new-project'
        workflow.init_new(config, output)
        actual = workflow.read_json(output / 'data/new_study_config.json')
        self.assertEqual(actual['validation_status'], 'pending')
        self.assertFalse(actual['historical_acceptance_inherited'])
        for command in ['run', 'verify', 'export']:
            result = subprocess.run(
                [sys.executable, str(workflow.SKILL / 'scripts/workflow.py'), command,
                 '--workspace', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, (command, result.stdout, result.stderr))
            self.assertIn('不能给新照片套用历史模型', result.stderr)
        self.assertFalse((output / 'data/逐图计数与质量.csv').exists())

    def make_runtime(self):
        output = self.root / 'runtime'
        output.mkdir()
        shutil.copytree(workflow.FROZEN / 'scripts', output / 'scripts')
        # Read-only hashes through this link; never run a detector or edit cache.
        (output / 'cache').symlink_to(workflow.FROZEN / 'cache', target_is_directory=True)
        return output

    def test_modified_detector_does_not_inherit_fingerprint(self):
        output = self.make_runtime()
        workflow.check_runtime(output)
        script = output / 'scripts/hybrid_detector.py'
        script.write_text(script.read_text() + '\n# synthetic test-only edit\n')
        with self.assertRaisesRegex(ValueError, '检测器已变化'):
            workflow.check_runtime(output)

    def test_missing_full_records_blocks_export(self):
        output = self.make_runtime()
        with self.assertRaisesRegex(ValueError, '尚无全量逐图记录'):
            workflow.verify_records(output)
        self.assertIsNone(workflow.verify_records(output, require_complete=False))

    def test_partial_checkpoint_not_complete(self):
        output = self.make_runtime()
        checkpoint = output / 'data' / ('run_' + workflow.FINGERPRINT) / 'full_checkpoint.jsonl'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('{"interrupted":')
        partial = workflow.verify_records(output, require_complete=False)
        self.assertFalse(partial['all_records_present'])
        self.assertEqual(partial['n_records'], 0)
        with self.assertRaisesRegex(ValueError, '损坏行'):
            workflow.verify_records(output)

    def test_wrong_source_hash_blocks_resume(self):
        output = self.make_runtime()
        checkpoint = output / 'data' / ('run_' + workflow.FINGERPRINT) / 'full_checkpoint.jsonl'
        checkpoint.parent.mkdir(parents=True)
        reference = workflow.baseline().iloc[0]
        record = {'frame_id': reference.frame_id, 'status': 'processed', 'sha256': '0' * 64}
        checkpoint.write_text(json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, '逐图复现不匹配'):
            workflow.verify_records(output, require_complete=False)
        audit = workflow.read_json(output / 'qa/原图复现一致性.json')
        self.assertEqual(audit['mismatch_count'], 1)
        self.assertFalse(audit['all_records_present'])

    def test_duplicate_frame_ids_block_resume(self):
        output = self.make_runtime()
        checkpoint = output / 'data' / ('run_' + workflow.FINGERPRINT) / 'full_checkpoint.jsonl'
        checkpoint.parent.mkdir(parents=True)
        reference = workflow.baseline().iloc[0]
        record = {'frame_id': reference.frame_id, 'status': 'processed', 'sha256': '0' * 64}
        checkpoint.write_text((json.dumps(record) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, '逐图记录重复'):
            workflow.verify_records(output, require_complete=False)

    def test_synthetic_time_integral_contract(self):
        subprocess.run([sys.executable, str(workflow.FROZEN / 'scripts/test_statistics.py')], check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
