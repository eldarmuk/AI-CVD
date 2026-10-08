# Code map and migration scope

The distribution uses a conventional src/ai_cvd package. Imports begin with ai_cvd,
not src. Package resources are located through importlib.resources.

| Public module | Role and internal origin |
| --- | --- |
| data/episodes.py | Note-free fictional grouping and subject isolation; replaces demo-only grouping with UTC-normalized sorting |
| data/synthetic.py | Seeded arithmetic fixture derived from the existing safe demo, with no clinical samples |
| features/episode.py | Pure Study B episode statistics and PartitionPreprocessor |
| features/temporal.py | Study B GRU-D input preparation |
| features/attention.py | Study B mTAN closure-time input adapter |
| features/continuous.py | New in-memory tensor boundary for retained Study A models |
| models/lstm_vae.py | Historical AIME-associated LSTM-VAE definition; debug entry point removed |
| models/continuous.py | Retained Study A R0/R1 encoders, masked loss and frozen risk head |
| models/grud.py; models/mtan.py | Retained Study B temporal model definitions |
| models/tabular.py | Small public fictional-data logistic baseline; not the frozen model-search runner |
| evaluation/metrics.py | Public binary metric/threshold adapter with explicit argument validation |
| demo.py | Full fictional workflow and ignored output writer |
| models/legacy/ | Earlier-task CGTA, hybrid CGTA and supervised sequence architectures |

Original paths included src/components/lstm_vae_model.py, src/architecture_study/models.py,
src/study_b_b1/core.py, src/study_b_b2/{inputs,model}.py,
src/study_b_b3/{inputs,model}.py and src/models/*.py.
Copies were adapted for package imports and consistent formatting. Frozen original
implementations, historical hashes and exact copy provenance remain in the preserved
research checkout, not in a redistributed provenance manifest.

Release-only behavior changes include input validation, UTC-normalized episode
sorting and refusing existing demo output directories. The legacy supervised model's
checkpoint-loading convenience method was removed; no fitted checkpoint is shipped.
The original legacy regression files remain preserved privately. Public tests retain
the relevant decay/cache, masking, causal-grid, placeholder and threshold behaviors
without importing private runners/configurations.

## Legacy scope

CGTA/fusion models explored an earlier task. They are kept to make architectural
history understandable, not as the preferred public demo or validated improvements.
Historical training scripts, database connectors, source note lexicons, optimizer
state, plotting exports and mixed private/public reports are deliberately excluded.
The exact scientific pipelines remain private rather than being silently rewritten.
