# Fictional telecare example

Install using [the reproducibility guide](../../docs/REPRODUCIBILITY.md), then run:

```shell
python -m ai_cvd.demo --seed 17 --output outputs/synthetic
```

Source: [synthetic generation](../../src/ai_cvd/data/synthetic.py),
[episode grouping](../../src/ai_cvd/data/episodes.py) and
[complete demo](../../src/ai_cvd/demo.py).

All subjects and events are invented. The year 2099 and explicit synthetic labels
make them distinguishable from source records. The distributions are arithmetic
fixtures plus seeded noise, not a clinical simulator or a fit to private data.

Outputs contain only generated fictional records and toy scores. The tabular model
learns from fictional training subjects; GRU-D/mTAN remain randomly initialized.
Do not compare the output AUROC/AP with the research papers.
