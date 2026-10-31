# Reproducibility and verification

## Public foundation

The initial foundation needs Python 3.11+ and only the standard library. It does
not need the repository's historical requirements, a GPU, network access, API
keys or human data. From the repository root:

```sh
python -m unittest discover -s studies/physiological_change/tests -v
python studies/physiological_change/scripts/run_synthetic_demo.py --output runs/physiological-change/synthetic-001
python studies/physiological_change/scripts/verify_run.py --run runs/physiological-change/synthetic-001
```

Choose a new directory for another run; existing runs must not be overwritten.
Inspect the generated report and manifest. Determinism means identical synthetic
inputs/configuration produce identical scientific metrics; environment metadata
or source revisions can legitimately differ between runs.

The [CI workflow](../../../.github/workflows/physiological-change.yml) executes
these checks on supported Python versions. It only uses synthetic data and does
not upload patient artifacts. Passing it establishes software behavior, not a
successful physiological model or completed scientific replication.

## Required research invariants

The synthetic suite is intended to expose important mistakes, including future
inputs, participant overlap, bad timestamps/values, denominator changes caused
by abstention, and conflating absent references with model uncertainty. Verify
the exact current test names before attributing a check to the implementation.
Hash checking detects changed files; it does not establish that a dataset or
method is scientifically valid.

Future tests must additionally cover adapter unit/timestamp correctness against
source records, duplicate handling, causal feature construction, fit-only-on-
development preprocessing, nonoverlapping episode boundaries, reference quality
rules, and frozen evaluation access. Add these when their components exist; do
not claim the initial scaffold already enforces a complete analysis lifecycle.

## Scientific run record

A full run must preserve:

1. Commit, working-tree state, adapter/configuration version and exact command.
2. Python/platform and dependency versions; model/device/software versions where
   applicable. Freeze an environment once dataset-dependent libraries are chosen.
3. Source identifiers, access terms, source file hashes, preprocessing and
   reference-quality rules; keep identifying source details local.
4. Person/split/episode definition hashes, seeds, calibration scope and an exposure
   log showing when each evaluation set was inspected.
5. Baselines and tuning budget, all primary/sensitivity results including failures,
   exclusions and denominators, plus generated artifact hashes.
6. A verification command that rebuilds aggregate results from frozen predictions,
   and a documented rerun path from authorized original sources.

Do not replace an earlier result when a bug is found. Write a new run, explain the
affected comparison, and retain both versions. A source-hash match proves identity
of bytes, not accuracy of physiological values.

## External evaluation release

Before accessing external outcome performance, commit a protocol and freeze the
model, calibration, thresholds and expected analyses. Record access separately
from the public repository if it contains subject keys. A locked dataset is one
not used in this project's method decisions, not a claim that nobody has ever
published results on that public dataset.

After the declared evaluation, publish permitted aggregate tables and figures,
commands, environment information, model/measurement assumptions and limitations.
Include person-level uncertainty, all-reference-eligible denominators and reasons
for abstention. A fixed-threshold transfer result and a result recalibrated with
target reference labels must be reported separately.

Public access to code does not settle the license of source datasets or imported
code. Review their separate terms before reuse/redistribution. Do not assign a
repository-wide license to historical or third-party work as a routine setup step.
