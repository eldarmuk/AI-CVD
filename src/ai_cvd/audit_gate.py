"""Verify a completed blocked-source audit without rescanning or building a dataset."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import tomllib

from .cli import file_hash, read_jsonl
from .dataset import assert_patient_isolation, patient_split
from .features import FEATURE_NAMES, validate_source_contract
from .task import load_task


def verify_blocked_audit(path, source, task):
    path, source = Path(path), Path(source)
    evidence = json.loads((path / 'source_evidence.json').read_text(encoding='utf-8'))
    if evidence['status'] != 'blocked_source_declarations' or evidence['task_identifier'] != task.identifier:
        raise ValueError('Wrong audit state/task')
    if tuple(evidence['feature_names']) != FEATURE_NAMES:
        raise ValueError('Feature order mismatch')
    if not evidence['source_unchanged_during_inspection']:
        raise ValueError('Source changed during the recorded audit')
    for name in (source.name, source.name + '-wal'):
        recorded = evidence['source_file_stats_after'].get(name)
        if recorded is None or evidence['source_file_stats_before'].get(name) != recorded:
            raise ValueError('Incomplete or unstable fingerprint record')
        if not re.fullmatch('[0-9a-f]{64}', evidence['source_sha256'].get(name, '')):
            raise ValueError('Incomplete SHA-256 fingerprint')
        current = source.with_name(name).stat()
        if (current.st_size, current.st_mtime_ns) != (recorded['bytes'], recorded['mtime_ns']):
            raise ValueError('Source file metadata changed since fingerprinting; reverify snapshot')
    contract = tomllib.loads((path / 'source-contract.toml').read_text(encoding='utf-8'))
    if contract['clinical_history_policy'] != 'exclude':
        raise ValueError('Undated clinical history must remain excluded')
    try:
        validate_source_contract(contract)
    except ValueError:
        pass
    else:
        raise ValueError('Blocked audit must not contain a runnable source contract')
    with open(path / 'coverage.csv', newline='', encoding='utf-8') as handle:
        if list(csv.DictReader(handle)):
            raise ValueError('Blocked audit must not fabricate confirmed coverage')
    rows = list(read_jsonl(path / 'source_patient_splits.jsonl'))
    assert_patient_isolation(rows)
    if len({r['senior_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source patient ID')
    if len(rows) != evidence['counts']['source_patients']:
        raise ValueError('Source count mismatch')
    if dict(Counter(r['split'] for r in rows)) != evidence['counts']['preeligibility_split_patients']:
        raise ValueError('Source partition count mismatch')
    if any(r['split'] != patient_split(r['senior_id'], task) for r in rows):
        raise ValueError('Source partition differs from canonical assignment')
    checksums = json.loads((path / 'artifact_checksums.json').read_text(encoding='utf-8'))
    required = {'source_evidence.json', 'source-contract.toml', 'coverage.csv',
                'coverage_requests.csv', 'source_patient_splits.jsonl',
                'dataset_integrity_report.md', 'data_owner_questions.md'}
    if not required <= checksums.keys():
        raise ValueError('Incomplete audit artifact manifest')
    for name, expected in checksums.items():
        if Path(name).name != name or file_hash(path / name) != expected:
            raise ValueError('Audit artifact checksum mismatch')
    return {'status': 'blocked_audit_verified', 'dataset_generated': False,
            'fingerprint_check': 'completed recorded hash; current size/mtime match; no new full-file hash'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--raw-db', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_blocked_audit(args.audit, args.raw_db, load_task())))


if __name__ == '__main__':
    main()
