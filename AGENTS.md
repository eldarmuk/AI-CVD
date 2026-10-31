# Working on physiological change reliability

For the new study, begin with
[`studies/physiological_change/RESEARCH_PLAN.md`](studies/physiological_change/RESEARCH_PLAN.md).
It is the authoritative continuation guide. Read its status and first unfinished
milestone before running experiments. The previous studies remain historical,
separately versioned work; their protocols do not define this new task.

## Scope and preservation

- Work on `research/hr-change-reliability` or an explicitly requested
  descendant branch. Do not alter other branch references, overwrite frozen
  results, or clean unrelated untracked files.
- Keep new implementation, tests, configuration and study documents under
  `studies/physiological_change/`. Root changes are limited to discoverability,
  repository instructions, the new study's CI workflow, and the reviewed public
  release manifest/checks needed to include this study.
- Treat source datasets as immutable. Never use historical processed telecare
  values as a measurement-accuracy reference.
- The active task is reference-validated heart-rate change inference. A future
  same-schema telecare batch is an optional engineering evaluation; it does not
  supply physiological or clinical truth on its own.

## Scientific decisions

- Distinguish implemented code, synthetic checks, exploratory results, frozen
  analyses and confirmed findings. Do not promote a proposed method to a novel
  or effective method before the stated gates pass.
- Preserve negative findings and protocol changes in the experiment log. Log
  evaluation exposure before looking at external outcomes. Re-splitting data
  already used in decisions does not restore an untouched evaluation.
- Fit transformations and select thresholds using development data only. Count
  people and independent change episodes, not just rows or overlapping windows.
- Keep model abstention in the denominator of reference-eligible true changes.
  Distinguish absent/indeterminate reference from a model refusing to decide.
- No clinical deployment, recruitment, third-party messaging, new data purchase,
  or major change of research question is implied by routine implementation.

## Reproducibility and publication

- Use Python 3.11+ and the study-specific commands in its README. The initial
  synthetic foundation uses only the standard library; historical requirements
  are not this study's environment.
- Keep dataset downloads, identifiers, detailed traces, split assignments,
  calibration records and generated runs out of Git. Inspect the exact staged
  file list before each commit; do not stage the entire workspace.
- Public documentation describes the scientific question, methods, limitations
  and results. Personal administrative discussions and private research notes
  are not publication artifacts.
- A future release needs dependency versions, dataset versions/access terms,
  source hashes, split/protocol hashes, seeds, commands and independent checks.
  Synthetic CI passing is evidence about software behavior, not model validity.

After a task, update the status table and append a dated experiment-log entry:
what changed, evidence, tests, limitations, and the next bounded action.
