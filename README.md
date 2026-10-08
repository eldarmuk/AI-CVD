# AI-CVD · Wearable time-series research

Research on **older-adult telecare**: anomaly detection, continuous early warning,
and prediction of recorded outcomes after an SOS alarm. The wearable channels include
heart rate, systolic/diastolic blood pressure, oxygen saturation, temperature and activity
(steps). These are retrospective research tasks, not a validated cardiovascular detector.

**Published:** Eldar Mukhtarov and Krzysztof Grudzień,
*Unsupervised Anomaly Detection of Acute Health Events in Older Adults Using
Longitudinal Wearable Sensor Data*, AIME 2026.
[Springer / DOI](https://doi.org/10.1007/978-3-032-30710-1_17)

| Study | Question | Status |
| --- | --- | --- |
| [AIME 2026](docs/studies/aime-2026.md) | Can an LSTM-VAE provide an unsupervised feasibility baseline? | Published; AUROC 0.68 |
| [Study A](docs/studies/study-a.md) | Can preceding wearable history forecast an alarm later classified Level 3? | Repaired study; internally completed |
| [Study B](docs/studies/study-b.md) | Given an SOS episode, can strictly earlier history help prioritize recorded outcomes? | Completed; manuscript in preparation |

These tasks have different populations, labels and evaluation designs. Their AUROCs
are **not a leaderboard**. Study B's held-out AUROC is 0.67102 and AP is 0.05391;
85.19% sensitivity requires prioritizing 56.63% of episodes, with only 27 test L3 episodes.

[Start here](docs/START_HERE.md) · [Research overview](docs/RESEARCH_OVERVIEW.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) · [Limitations](docs/LIMITATIONS.md) ·
[Publications](docs/PUBLICATIONS.md)

## Quick start

Use **64-bit Python 3.12** on Windows or Linux (CPU).
From this checkout, in a new virtual environment:

```shell
python -m venv .venv
# Activate: Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Activate: Linux: source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m ai_cvd.demo --seed 17 --output outputs/synthetic
python -m pytest
```

The demo creates fictional subjects, wearable arrays and SOS episodes; isolates
subjects across splits; constructs strictly pre-alarm inputs; fits a tiny fictional
tabular baseline; exercises untrained GRU-D and mTAN models; and saves toy scores,
metrics and a validation-selected priority threshold. It uses no clinical database,
private notes, fitted scientific checkpoint or private prediction file.
**It does not reproduce clinical results.** Choose a new output directory for each run.

## Scope and status

This curated software candidate contains source, documentation and tests.
Generated outputs are ignored. Private-data extraction, frozen training pipelines and
scientific provenance remain outside this distribution. [Code map](docs/architecture/code-map.md)
and [public/private boundary](docs/PUBLIC_PRIVATE_BOUNDARY.md) explain the separation.

Retrospective recorded telecare outcomes do not establish independently adjudicated
clinical events, prospective effectiveness, clinical utility or deployment readiness.
See [all limitations](docs/LIMITATIONS.md).

Version **1.0.0rc1** is a curated release candidate. See the
[release preparation notes](docs/releases/v1.0.0.md) for verification and release status.
No software license is granted by this repository; a license decision remains separate.
