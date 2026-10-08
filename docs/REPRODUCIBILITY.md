# Public reproducibility

## Supported environment

Use 64-bit CPython **3.12** on Windows or Linux, CPU only. The package pins the
numerical libraries; [requirements.lock.txt](../requirements.lock.txt) also pins the
transitive/runtime and development dependencies. The lock selects PyTorch's CPU
wheel from its official index. This candidate does not promise macOS or GPU support.

From a fresh checkout, create and activate a virtual environment, then run:

```shell
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m ai_cvd.demo --seed 17 --output outputs/synthetic
python -m pytest
python -m ruff check .
python -m ruff format --check .
python tools/release_checks.py
cffconvert --validate
```

The [README](../README.md) includes environment activation commands.
The package is installed normally, not through a personal path or private runtime.
The installed command also works from a directory outside the checkout:

```shell
ai-cvd-demo --seed 17 --output outputs/another-fictional-run
```

## What the demo does

1. Generates 12 unmistakably fictional subjects, 48 SOS alerts and seeded wearable
   arrays in the year 2099. No source records, notes or fitted artifacts are inputs.
2. Groups alerts into 24 episodes using adjacent gaps of at most ten minutes.
3. Splits subjects into 8 training, 2 validation and 2 test subjects.
4. Extracts a seven-day personal baseline before the recent 24-hour window.
   Recent five-minute buckets must close **strictly before** the alarm.
5. Builds 90 episode summaries, a 56-component context and temporal inputs with
   287 admitted slots plus one padding slot; retains observation masks and elapsed time.
6. Fits a small tabular logistic model on fictional training subjects. Preprocessing
   is fitted on that partition only. GRU-D and mTAN forward paths use seeded,
   untrained parameters.
7. Calculates ranking/operating metrics and selects each model's threshold using
   fictional validation labels. The threshold maximizes specificity while meeting
   target sensitivity; ties stay whole, with the largest qualifying threshold preferred.
8. Saves fictional alerts, episodes, split assignments, wearable arrays, scores and
   metrics under the ignored output directory. Existing directories are refused.

Outputs include metrics.json, predictions.csv and fictional_wearables.npz.
These are generated examples, not tracked research artifacts. Numerical repeatability
is tested within the pinned environment; bitwise equality across operating systems
or hardware is not promised.

## What this cannot reproduce

The demo does not reproduce AIME or Study A/B clinical metrics, confidence intervals,
patient membership, calibration, thresholds or private scientific provenance.
Those require separately authorized data and frozen internal execution records.
The [public/private boundary](PUBLIC_PRIVATE_BOUNDARY.md) is part of the reproducibility
contract, not an invitation to substitute unapproved data.

The [CI workflow](../.github/workflows/ci.yml) runs public-only checks on Windows and
Linux. Consult the run for the exact commit being reviewed; workflow configuration
alone does not establish a passing result.
