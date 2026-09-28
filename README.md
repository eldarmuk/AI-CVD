# AI-CVD

I study acute health events in older adults using wearable sensor data. This repository brings together data preparation, model comparisons and work on interpretability.

[AIME 2026 paper](https://doi.org/10.1007/978-3-032-30710-1_17) · [Feature dictionary](docs/FEATURE_DICTIONARY.md)

Research code, with later experiments alongside the published study. Reproducing it requires the relevant data and experiment settings.

<details>
<summary>Setup and technical notes</summary>

# AI-CVD

Research on detecting acute health events in older adults from longitudinal wearable data. I’m interested in what makes a model useful beyond its score: the data it sees, the patterns it relies on, and the number of alarms a chosen threshold might produce.

My work at Łódź University of Technology includes data preparation, physiological and circadian features, model comparisons, and interpretation, supervised by Prof. Krzysztof Grudzień.

## Start here

- [AIME 2026 paper](https://doi.org/10.1007/978-3-032-30710-1_17): **Unsupervised Anomaly Detection of Acute Health Events in Older Adults Using Longitudinal Wearable Sensor Data**, Eldar Mukhtarov and Krzysztof Grudzień.
- [Feature dictionary](docs/FEATURE_DICTIONARY.md): how the features are defined.
- [`src/models`](src/models): sequence and circadian-model implementations.
- [`src/pipelines`](src/pipelines): preprocessing, training, evaluation and comparisons.

The conference paper concerns the earlier anomaly-detection study. The repository also contains later supervised, circadian and deep/tree fusion experiments. Their results should not be treated as results from the same experiment or as clinical validation.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/components/` | Data loading, processing and shared model components |
| `src/pipelines/00*`–`01*` | Feature preparation, cohort filtering and dataset generation |
| `src/pipelines/02*`–`08*` | VAE training/evaluation, tabular baselines and comparison reports |
| `src/pipelines/09*`–`14*` | Supervised sequences, interpretation, circadian models and fusion |
| `src/archive/` | Earlier experimental scripts; not the main entry point |
| `notebooks/` | Database exploration and preprocessing notebooks |
| `models/`, `reports/` | Saved experimental outputs; read with the corresponding configuration |

## Reproducibility and setup

This is a research workspace, not a self-contained demo. A fresh clone does **not** contain everything needed to reproduce the full study: the source database, some generated arrays and checkpoints are external to the checked-in pipeline. Several scripts use fixed paths and experiment-specific settings.

For source inspection, no dataset is required. To prepare a separate Python environment:

```sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
```

The current requirements file is unpinned and incomplete for all experiments. In particular, the source also imports NumPy, PyTorch and SHAP; install compatible versions for the experiment and hardware you intend to use. An exact, locked reproduction environment is not supplied.

With an authorized dataset and those dependencies, start by reviewing the arguments to the feature builder:

```sh
python src/pipelines/00_build_multimodal_features_duckdb.py --help
```

The builder accepts `--db`, `--output`, `--elite-cohort`, `--threads` and memory/chunk limits. Dataset generation subsequently reads `data/processed/multimodal_features.parquet` and writes arrays under `data/processed/anomaly_detection/`. Inspect each script’s input paths before running it; the numeric filenames are an orientation guide, not a guarantee that every experiment runs as one uninterrupted sequence.

## Interpretation and limitations

- Measurements, source patients and a filtered cohort are different denominators. Report the actual cohort for each experiment.
- Model scores depend on the target, split, normalization and threshold selection. Check these together before comparing saved metrics.
- These are research models, not a diagnostic service or a clinically validated decision system.
- Existing data-derived files are not a blanket grant to redistribute the underlying study data. Use only material for which you have permission.
- There is no end-to-end reproduction claim from this documentation update. Source files were syntax-checked; training and patient-data pipelines were not rerun.

[About my work](https://eldarmukhtar.ovh/) · [Contact](mailto:eldar.mukhtarov.tech@gmail.com)

</details>
