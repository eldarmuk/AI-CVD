# Experiment and decision log

Append entries; retain failed hypotheses and superseded decisions. Do not record
patient identifiers, restricted source snippets or individual trajectories here.
Detailed local run records are linked by a nonidentifying run label/hash when
appropriate for release.

## 2026-10-31 — Initial research branch

- Prepared a separate research scaffold; preserved previous studies.
- Defined a reference-based HR-change question, separate from alarm prediction.
- Core work uses public references. Later company involvement may consist only
  of a same-schema batch after model development; that supports engineering
  evaluation unless independent reference evidence is supplied.
- Documented prior art: scalar reliability warnings, selective correction,
  uncertainty and external testing already exist. The proposed change-focused
  contribution remains a hypothesis.
- Added a synthetic software foundation, separate study documentation and CI.
  Exact local verification results are appended after the branch setup checks.
- No human dataset downloaded, model trained, new cohort recruited or company
  contacted in this setup task.
- Next action: M1 data/reference/chronology audit and development-only baseline
  feasibility, with the external outcome stream reserved from method selection.

## 2026-11-01 — Isolated public branch

- Created `research/hr-change-reliability` from the published curated base
  `a81239ac4419f0ea0c5d36a8850e0063c6e2bc91` in a separate checkout.
- Copied only the reviewed new study files. Historical unpublished commits and
  local research notes are not part of this branch's ancestry or release.
- Preserved the published pipeline and its documentation; the root README directs
  new study work to this plan.
- Independent review checked decision accounting for all combinations of
  reference increase/stable/decrease and prediction increase/stable/decrease/
  abstain. No blocking accounting defect was found.
- Local verification: all 31 synthetic unit tests passed on Python 3.11.14 and
  3.12.14. Synthetic generation and exact replay verification passed. Ruff 0.11.13
  lint and format checks passed across the public checkout (69 Python files).
- The manifest checks cover artifact consistency and replay against current
  source; they do not authenticate a claimed Git revision or establish sensor
  accuracy. No scientific dataset result is inferred from these checks.
- The CI workflow adds Python 3.11/3.12/3.13 synthetic checks. CI execution is
  separate from the local results above and should be inspected on the commit.

## Entry template

```text
Date / question:
Protocol and source revisions:
Data role and prior exposure:
Method and allowed information:
Run label and verification:
Independent people / eligible episodes / exclusions:
Results, uncertainty and negative findings:
Decision and deviation from the previous plan:
First next bounded action:
```
