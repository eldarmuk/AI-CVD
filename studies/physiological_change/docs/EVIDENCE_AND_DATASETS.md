# Evidence and dataset decisions

Reviewed 31 October 2026. This is a study-selection record, not evidence that a new method works. The linked primary sources and repository metadata were inspected; datasets and model weights were not downloaded, and published results were not reproduced during this review. Dataset selection remains conditional on the intake checks below.

## The question worth testing

The primary question is whether methods that improve individual heart-rate readings also preserve reference-confirmed changes between two periods. A secondary, connected question is whether explicitly estimating uncertainty in the **change** improves that trade-off when observations are intermittent or the device/population differs from development data.

The contribution must be established empirically. Numerical-only inputs, personalization, uncertainty, rejection of unreliable readings, external testing, and synthetic gaps are established ideas. Combining them does not itself establish originality.

## Closest work and required comparisons

| Primary source | What it establishes | Consequence for this study |
|---|---|---|
| [Islambouli et al., 2025 preprint](https://arxiv.org/html/2508.19818v1), *Towards a Real-Time Warning System for Detecting Inaccuracies in Photoplethysmography-Based Heart Rate Measurements in Wearable Devices* | A numerical-HR-only error predictor using short histories, evaluated on Polar/Garmin measurements from the ValOpti cohort. The methods use a participant-separated development/test allocation. | Numerical-only reliability detection is an essential comparator, not a new claim. The paper's formula labelled accuracy is precision; reconstruct metrics from definitions rather than copying names. Missing paired observations were removed, limiting direct support for irregular monitoring. Exact code, weights and their reuse terms have not been verified. |
| [Wang et al., October 2026 preprint](https://arxiv.org/abs/2610.30688), *An Agent Architecture with Tools and Memory for Selective Correction of Wrist PPG Heart Rate Estimates* | Retain/correct/reject decisions, helpful and harmful corrections, and transfer from DaLiA to other datasets. It explicitly separates correction from selection on accepted windows. | Generic selective correction and matched-coverage analysis are already occupied. Its pulse/motion inputs differ from a numerical-only deployment. Use an applicable same-information implementation where feasible, and a richer-input version as a separate information benchmark. The abstract was inspected; complete implementation, weights and runtime remain an audit task. |
| [Neudorfer et al., 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/sms.14756), *Validity of Four Consumer-Grade Optical Heart Rate Sensors for Assessing Volume and Intensity Distribution of Physical Activity* | Paired consumer HR and medical-grade ECG across laboratory and daily-life conditions. Validation includes activity-intensity summaries as well as pointwise agreement. | Claims must distinguish the proposed change target from already-studied activity-intensity agreement. Sensor accuracy and useful interpretation are related but different estimands. |

Baseline requirements are unchanged recording, causal median/last observation, a simple robust state-space smoother, numerical-HR error gating, and **direct change regression with ordinary calibrated intervals**. The latter is particularly important: choosing the correct prediction target is not automatically a new method. Compare methods with the same available inputs and tuning opportunity. Include both matched-coverage curves and thresholds fixed on development/calibration data; selecting a test-cohort coverage quantile is not a deployable fixed policy.

## Dataset candidates and roles

No cohort is selected by which gives the strongest result. Decide development and external roles after a metadata/reference feasibility audit, before comparing model outcomes. Repeated devices and sessions from one participant remain in one partition.

| Candidate | Verified evidence and reuse status | Proposed role and boundary |
|---|---|---|
| [PPG-DaLiA, UCI release](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia) | 15 participants; wrist optical BVP and acceleration with chest ECG reference. BVP 64 Hz, acceleration 32 Hz, ECG 700 Hz. Archive 2.7 GB; dataset CC BY 4.0. | Manageable development candidate. Derived HR from these waveforms is not a proprietary band's exported HR. Audit chronology, ECG reference generation and window overlap; row count is not participant count. |
| [Multi-site PPG paper](https://arxiv.org/html/2605.17859v2) and [data card](https://huggingface.co/datasets/snowballlab/Multisite-PPG) | 20 participants, ages 20–51, 19 women; several wearable sites and ECG during daytime activities. Data CC BY-NC 4.0. [Baseline/preparation code](https://github.com/jiayimaggieshao/wearable-ppg-dataset) is GPL-3.0. | External-device/environment candidate. Multiple sites from the same people are not independent cohorts. Reference filtering and discarded irregular rhythms constrain clinical claims. Prefer a single predeclared wearable site; the full multimodal benchmark would expand the task substantially. |
| [ValOpti released HR measurements](https://zenodo.org/records/13767292) | Four SPSS files, 496.1 MB, containing 1 Hz consumer-device HR paired with medical-grade ECG HR. [Zenodo API metadata](https://zenodo.org/api/records/13767292), checked directly on the review date, declares `license.id = cc-by-4.0` and `access_right = open`. | Priority additional candidate because it contains actual numerical device outputs rather than requiring a new waveform estimator. Audit timestamp/session schema and preprocessing before acquisition. The dataset license was verified separately from the article license. |
| [Auricle 1.0.0](https://physionet.org/content/auricle/1.0.0/) | Open Data Commons Attribution v1.0; 42 healthy participants and 28 with heart failure. Three-lead ECG and finger/ear PPG; roughly 30-minute recorded sessions with activity annotations. Healthy participants may have two sessions. 15.4 GB uncompressed. | Reserve clinical-cohort challenge, not a default extra study. This is supervised measurement with different sensor locations, not wrist-band home monitoring. Cohorts differ in age, sex and protocol, so between-cohort performance cannot isolate a disease effect. ECG can reference HR; CNAP-derived measurements are not interchangeable with invasive blood-pressure truth. |

### Why ValOpti deserves an early intake check

The [primary validation paper](https://onlinelibrary.wiley.com/doi/full/10.1111/sms.14756) reports 32 participants aged 38–74, including 14 with cardiovascular conditions; 11 used HR-limiting medication. That broadens the clinical context relative to young-adult activity datasets, without making it representative of frail telecare users. Atrial fibrillation was excluded.

Two preprocessing details matter. ECG values outside participant-specific full-record extrema were removed along with corresponding sensor values. Devices were aligned using whole-record cross-correlation. Determine whether the released files preserve timestamps and removed intervals, and whether alignment has already been applied. Do not present retrospective reference-assisted alignment as an online capability. These are published processing choices; their effect on our change target has not been measured.

ValOpti is also the source cohort used in the [2025 warning-system paper](https://arxiv.org/html/2508.19818v1). The dataset and those papers are not independent replications. If used for external testing here, do not reuse a model trained on its participants as an allegedly untouched comparator. Published preprocessing alone does not establish that usable raw, unscreened references are released.

The preferred intake decision is therefore DaLiA for development and ValOpti for external evaluation **if** the released numerical series preserve usable chronology and reference eligibility. Multi-site remains an alternative external-device cohort if that audit fails, or a prespecified later replication. Do not switch between them after comparing favorable model results. Previously published benchmark results should be disclosed; an external cohort held out from this project is not globally unexamined data.

### Software and compute decisions

No pretrained model is required for the first feasibility gate. The numerical baselines should run on a workstation; verify memory and runtime on a bounded development subset before committing to a full run. SPSS ingestion may require an additional parsing library; record its version, reuse terms and conversion checks. Do not label a conversion successful until row counts, participant IDs, missingness and timestamp units agree.

The Multi-site repository provides preparation and baseline scripts, but successful local reproduction and downloadable pretrained weights have not been established. Its code license is separate from the data license. Auricle's [PulseKit repository](https://github.com/chai-toronto/pulsekit) supplies a signal-processing example, but no root license was visible in the inspected listing; do not copy its implementation under an assumed license. Independently implementing a documented adapter is an alternative. No large neural-model training or paid compute is a prerequisite for the core question.

## A possible technical distinction to falsify

Let a recorded value be a true value plus error. The error in a change is the error at the recent period minus the error at baseline. A constant offset can cancel, whereas changing bias does not. Equivalently,

```text
Var(error_recent - error_baseline)
  = Var(error_recent) + Var(error_baseline)
    - 2 Cov(error_recent, error_baseline)
```

This is an elementary identity, not a novel theorem. It motivates a controlled question: does modelling paired errors or directly calibrating change uncertainty retain more genuine changes than judging both readings separately? A pointwise estimator can have substantial bias but useful change tracking; smoothing can also have good mean error but attenuate a true transition. Neither pattern is assumed to occur in the chosen data.

The first experiment should keep the HR estimator fixed and compare:

1. Point-error gating and ordinary propagation of pointwise uncertainty.
2. Direct change regression with calibrated intervals.
3. A proposed dependence-aware change method, only if the first two reveal a specific unresolved failure.

Evaluate all on identical reference-eligible change intervals. Report point error, change error, false change conclusions, wrong directions, and reference-confirmed changes suppressed by smoothing or abstention. Keep every eligible genuine change in the recall denominator, including rejected intervals; separately report missing/indeterminate reference labels. Compare numerical corrections on identical accepted sets to separate correction from selection. Explainable outputs should state the estimated change, uncertainty, baseline/recent observation support and reason for withholding a conclusion. Do not attribute a failure to motion unless motion information supports that explanation.

## Intake checks before choosing the final cohorts

- Confirm exact data version, file hashes, license, distribution rules, reference construction, reference-quality exclusions and source participant identity across releases.
- Verify preserved chronological ordering, timestamp units, clock alignment and sensor averaging/latency. Establish the attainable time scale before defining change intervals.
- Quantify independently usable participants, sessions and nonoverlapping reference-supported transitions. Do not choose thresholds from external outcomes.
- Preserve raw measurements and masks. Audit original versus released missingness; never silently bridge deleted intervals or treat an absent record as no change.
- Separate waveform-derived HR, actual device numerics and ECG reference. An unavailable input cannot be supplied to the deployed method through its evaluator.
- Record which datasets and outcomes have been inspected. Keep the chosen external outcomes sealed until the method, thresholds, preprocessing and report are frozen.
- Examine performance and coverage by available age, activity, device and clinical context, with participant-level uncertainty. Small strata are descriptive; missing medication or disease fields cannot be imputed into verified histories.

## Role of later telecare exports

A later batch with the same schema can test whether parsing, support, output distributions and abstention behaviour remain stable over time. It can support a frozen temporal feasibility evaluation and, if recorded, operator assessment of explanation clarity. It supplies physiological accuracy evidence only if it adds independent reference measurements. Operational escalation labels remain a different target. Do not report later same-provider data as an independent clinical reference merely because its calendar period is new.

Observed acquisition gaps may motivate controlled resampling experiments on paired public data. Such experiments establish sensitivity to those imposed gaps; they do not prove recovery of genuinely missing telecare values or transfer of unknown device biases. Clinical deployment, future-event prediction and automatic alarm suppression are outside the present measurement-method claim.

## Continue or stop

Continue only if reference coverage is adequate, the closest-method comparison reveals a substantive failure, and a specific intervention improves the change/coverage trade-off on development data without hiding difficult people. If direct change regression or a conventional filter already solves the problem, report that result and revise the contribution rather than inventing a more complex architecture. A reserve dataset is added to answer a prespecified transfer question, not to search for a favorable test result.
