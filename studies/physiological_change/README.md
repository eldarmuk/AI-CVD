# Physiological change reliability

Can cleaning wearable heart-rate measurements improve average accuracy while
removing real physiological changes? This study investigates that possibility
against independent ECG references, then tests whether decisions about change
can be improved without hiding difficult cases through exclusion.

**Start with [RESEARCH_PLAN.md](RESEARCH_PLAN.md).** It contains the research
question, continuation instructions, milestones, stop conditions and status.
The current release is a research scaffold and synthetic verification package.
It contains no trained model or reference-backed performance result.

## Run the public software checks

From the repository root, with Python 3.11 or later:

```sh
python -m unittest discover -s studies/physiological_change/tests -v
python studies/physiological_change/scripts/run_synthetic_demo.py --output runs/physiological-change/synthetic-001
python studies/physiological_change/scripts/verify_run.py --run runs/physiological-change/synthetic-001
```

Use a new output directory for each demo. The synthetic example tests accounting,
temporal rules and reproducibility; its numbers are constructed, not estimates
of performance on people. No dataset download, patient record, network access or
third-party Python dependency is needed for these checks.

## Find the right component

| Location | Purpose |
|---|---|
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Authoritative entry point for researchers and future agents |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | Population, reference, baselines, metrics, selection and validation |
| [docs/EVIDENCE_AND_DATASETS.md](docs/EVIDENCE_AND_DATASETS.md) | Primary literature, dataset candidates and reuse/access limits |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Data flow, implemented components and planned interfaces |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | Timing, units, independent reference and provenance requirements |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Commands, manifests, CI and release requirements |
| [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md) | Interpretable outputs and how their explanations will be tested |
| [docs/COMPANY_EVALUATION_REQUEST.md](docs/COMPANY_EVALUATION_REQUEST.md) | Specific request for an optional later company evaluation |
| [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) | Results, decisions, amendments and continuation state |
| `hr_change/`, `scripts/`, `tests/`, `configs/` | Executable foundation, commands, tests and study configuration |
| [data/README.md](data/README.md), [artifacts/README.md](artifacts/README.md) | Local data/run conventions; generated contents are ignored |

The existing source tree, manuscripts and studies are preserved. This project
has a separate endpoint, reference standard, implementation and evaluation.
