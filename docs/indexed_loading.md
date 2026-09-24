# Loading verified indexed datasets

Use the immutable run directory from the private integrity report. No model is
trained by this example. Install `requirements-data.txt` in the chosen environment.
Run from the repository root in PowerShell:

```powershell
$env:AI_CVD_RUN = 'ABSOLUTE_PATH_TO_VERIFIED_RUN'
@'
import os, json
from pathlib import Path
import numpy as np
from src.ai_cvd.task import load_task
from src.ai_cvd.cli import file_hash
from src.ai_cvd.compact import sequence_batches
run = Path(os.environ['AI_CVD_RUN'])
task = load_task()
meta = json.loads((run / 'run_metadata.json').read_text())
assert file_hash(run / 'normalization.json') == meta['artifacts_sha256']['normalization.json']
norm = json.loads((run / 'normalization.json').read_text())
assert norm['fit_split'] == 'train' and norm['task_identifier'] == task.identifier
assert norm['feature_names'] == meta['feature_names']
for split in ('train', 'validation', 'test'):
    batches = sequence_batches(run, split, task, training_only=(split == 'train'), batch_size=64)
    batch = next(batches)  # Loading smoke check only; full training would iterate.
    normalized = (batch['X'] - np.asarray(norm['mean'], dtype=np.float32)) / np.asarray(norm['scale'], dtype=np.float32)
    assert np.array_equal(np.isnan(normalized), np.isnan(batch['X']))
    print(split, normalized.shape, len(batch['sample_ids']))
    batches.close()
'@ | python -
```

The loader requires a successful integrity report bound to unchanged run metadata,
checks the export and each consumed shard, reconstructs 96-by-59 arrays, and verifies
sample identities. Batch metadata includes prediction times, patient IDs, linked
episode IDs, labels, lead times, task identity and the export fingerprint. Keep
these IDs attached to predictions; never join by positional array order alone.

Training defaults in this example use the explicitly retrospective event-free
reference selection. Set `training_only=False` for the complete train stream.
Validation/test always retain natural supported-stream prevalence. Never refit
normalization on them. NaNs remain missing; downstream masked loss/imputation must
be explicitly implemented and validated before model training. All undated clinical
channels are unavailable, not known negative clinical histories.

For interrupted builds only, `compact --resume-interrupted` verifies/reuses existing
shards, recomputes their labels/IDs and aggregates training statistics, and preserves
partial audit files plus original shard checksums in an interruption subdirectory.
It refuses completed runs and contract changes. A narrowly scoped, explicitly
recorded v2.1.0-to-v2.1.1 amendment can exclude ambiguous Steps timestamps; verification
requires all previously completed shards to remain byte-identical and validates the
amendment's exact configuration differences. Other experiment changes require a new
run. Immutable completed outputs are never overwritten.
