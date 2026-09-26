# Bounded R0/R1 development study

## Frozen analysis declaration (before private fitting)

Run: `runs/model_studies/r0-r1-bounded-3fold-v1`.
The approved `configs/model_studies/r0_r1_v1.toml` is unchanged: CPU/four threads,
batch cap 64, 2,000 SSL draws per recipe/fold, three canonical-training-only folds.
The data task, projection, patients, sample plans, masks, widths, learning rates,
head supervision and checkpoint criterion remain frozen. Added telemetry observes
training without changing its random generators or optimization.

R0 and R1 share scaler, sample plans and masking generators. Checkpoint selection
remains minimum mean loss on the fixed inner-held SSL batches at scheduled checkpoints
(earliest tie); architecture interpretation instead uses downstream paired results.
The initial held loss is diagnostic only and cannot become a new checkpoint candidate.

Predeclared inexpensive controls reuse the existing frozen-head fitting implementation:
random-initialized **frozen R0** plus logistic head (not end-to-end supervised training),
process-only logistic on R0's fixed process summary and evidence flags, and R0 value
summary plus evidence flags. These use the exact same head/assessment IDs and weights.
The value summary already uses validity/recency, so it is not observation-process-free.
No additional deep encoder is trained.

Descriptive metrics: inclusion-probability-weighted AP, AUROC, log loss, Brier score,
mean risk versus weighted prevalence, and precision/recall at a fixed 0.5 probability.
The 0.5 threshold is an engineering diagnostic, not a clinical operating point or
an optimized threshold. Stratify by any of the five vital-sign channels observed
inside the window versus none; report primitive-evidence absence separately because
Steps alone are activity evidence. Undefined class/stratum metrics are unavailable.

Compare fold-wise differences and use 500 paired patient-cluster bootstrap draws
with fixed analysis seed 20260926 for pooled held-fold AP/AUROC differences. Carry
each selected patient's windows and inclusion weights together; omit and count
resamples with missing classes. This accounts for patient clustering conditional
on the fixed fitted models and sampled windows, not model-refitting or within-patient
sampling uncertainty. These are development estimates, not final generalization CIs.

No operational alert policy exists in this implementation. The sparse assessment
sample cannot recover refractory-period alert emission or supported-time burden.
Episode sensitivity and false alerts per patient-day are therefore unavailable;
linked positive episodes are counted separately from positive windows. No new policy
or full-stream evaluation will be invented after seeing results.

Canonical validation/test outcomes remain untouched. No automatic superiority
claim or progression follows a numerically higher point estimate; fold consistency,
uncertainty, calibration and controls determine the conservative recommendation.

## Verified results from saved artifacts

**R1 did not demonstrate reproducible downstream benefit. Retain R0 as the simpler
provisional reference; neither R2 nor R3 is warranted by this screen.** Both models
learn the masked reconstruction objective, but SSL's benefit over the random-encoder
control and the incremental value of physiology remain unresolved because the
predeclared control heads have not been fitted/saved. This report is complete for
the saved R0/R1 comparison, not for all planned control analyses.

The user completed the remaining fits manually. All six `complete_verified` model
markers and root `complete.json` passed the saved-artifact audit. The audit checked
checkpoint/manifest hashes, frozen encoder equality, exact reproduction of recorded
scores by exported heads, IDs against head/assessment manifests, episode-group labels,
complete draw histories, original/resume provenance, and canonical task/run bindings.
R0/R1 share the same per-fold sampling plans and recorded sample/corruption digests.
Every SSL/scaler/head reference belongs to its fit-patient set; checkpoint/assessment
references belong to its training-fold holdout. Fold assignments were recomputed
from the canonical **train** export and matched. Canonical validation/test patients
did not enter these manifests.

The config, dataset and numerical training implementation were not changed. Audit
scope is saved model artifacts, canonical metadata and train-export fingerprints;
the raw database and patient-shard contents were not reread. Mask draws were checked
through matching saved digests, not regenerated through another private-data pass.
No private model was retrained. Saved-only verification/analysis took 8.69 seconds.
The private result is `saved_artifact_analysis.json` in the study directory; public
code is `src/architecture_study/saved_analysis.py`.

## Assessment population and metrics

| Fold | Fit patients | Held patients | Head samples | Assessment windows | Positive windows | Event patients | Linked episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5,802 | 2,902 | 6,136 | 3,082 | 180 | 36 | 43 |
| 1 | 5,802 | 2,902 | 6,150 | 3,068 | 166 | 36 | 40 |
| 2 | 5,804 | 2,900 | 6,150 | 3,068 | 168 | 35 | 41 |

The pooled assessment contains 9,218 selected windows from 8,704 distinct held
patients, with 514 positive windows linked to **124 distinct episodes in 107 event
patients**. Positive windows are not independent clinical events. Patients occur
in only one holdout fold, although each is a fit patient in the other two folds.

Metrics use inverse window-inclusion probabilities. The weighted training-stream
prevalence estimate is 0.0000310223 (0.00310223%); the unweighted enriched sample
fraction is not prevalence. This is not a full natural-stream or test-set evaluation.
One sampled negative per patient limits characterization of temporal false alarms;
patient bootstrap does not eliminate that within-patient sampling uncertainty.

In the tables, **AP × 10⁻⁵** means the displayed number must be multiplied by 0.00001.
AP is the stepwise average-precision estimator, not trapezoidal PR area.

| Fold | R0 AP × 10⁻⁵ | R1 AP × 10⁻⁵ | R0 AUROC | R1 AUROC | R0 log loss | R1 log loss |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 5.4709 | 4.4745 | 0.6161 | 0.6006 | 0.00036532 | 0.00036691 |
| 1 | 4.1823 | 3.8095 | 0.6226 | 0.5913 | 0.00033850 | 0.00034105 |
| 2 | 5.4952 | 6.0029 | 0.6623 | 0.6634 | 0.00035127 | 0.00035024 |
| Pooled out-of-fold | 4.6750 | 4.4913 | 0.6307 | 0.6149 | 0.00035163 | 0.00035267 |

R1 loses AP and AUROC in folds 0 and 1. Its fold-2 AUROC advantage is only 0.00114.
Unweighted fold mean ± sample SD: AP is (5.0495 ± 0.7510) × 10⁻⁵ for R0 and
(4.7623 ± 1.1247) × 10⁻⁵ for R1; AUROC is 0.6337 ± 0.0250 versus 0.6184 ± 0.0393.
These three-fold SDs are descriptive variability, not confidence intervals.

The prespecified 500 paired patient-cluster bootstrap draws (seed 20260926) gave:

| R1 minus R0, pooled | Observed difference | 95% percentile interval |
|---|---:|---:|
| AP | −0.000001837 | [−0.000010847, +0.000003043] |
| AUROC | −0.01581 | [−0.04056, +0.00758] |

No resample lacked a class. Both intervals include zero. This is no evidence of
R1 superiority; it is also not a definitive statistical proof of R0 superiority.
The intervals condition on these fitted models and selected assessment windows,
without refitting or estimating all within-patient sampling variability.

## Calibration and clinical usefulness

At the frozen diagnostic threshold 0.5, both models emit **zero positive predictions**
in every fold: window recall is zero and precision is undefined, not zero. This
unoptimized threshold is unsuitable for claiming useful operating performance at
this prevalence. No threshold was retuned after seeing results.

Pooled mean risk is 0.0000314285 for R0 and 0.0000317249 for R1, compared with the
weighted prevalence estimate 0.0000310223. Pooled Brier scores are respectively
0.00003102184 and 0.00003102162. Their small absolute size largely reflects extreme
imbalance; similar overall means do not establish individual-risk calibration.
R1's tiny Brier advantage does not offset inconsistent ranking or establish utility.

Episode-level sensitivity and false alerts per supported patient-day are **unavailable**:
there is no implemented predeclared alert/refractory policy, and sparse assessment
samples cannot reconstruct alert sequences. Clinical utility has not been demonstrated.

## Evidence-availability strata

Vital evidence means at least one temperature, HR, SBP, DBP or saturation observation
within the window. No-vital windows may still contain Steps. No-primitive windows
contain none of the six primitive channels. These strata were recovered exactly
from R0's saved fixed natural-mask summaries and checked against its evidence flag;
no learned latent coordinate was treated as an observation mask.

| Fold | Stratum | Windows / positives | R0 AP × 10⁻⁵ | R1 AP × 10⁻⁵ | R0 AUROC | R1 AUROC |
|---|---|---:|---:|---:|---:|---:|
| 0 | Vital evidence | 2,330 / 163 | 5.7065 | 4.6317 | 0.5708 | 0.5606 |
| 0 | No vital evidence | 752 / 17 | 2.3100 | 1.4735 | 0.6180 | 0.3951 |
| 0 | No primitive evidence | 735 / 17 | 2.3185 | 1.6673 | 0.6100 | 0.4039 |
| 1 | Vital evidence | 2,301 / 145 | 4.3793 | 3.9459 | 0.5997 | 0.5639 |
| 1 | No vital evidence | 767 / 21 | 2.0633 | 2.0061 | 0.5415 | 0.4488 |
| 1 | No primitive evidence | 737 / 21 | 2.2647 | 2.2769 | 0.5525 | 0.4506 |
| 2 | Vital evidence | 2,325 / 158 | 5.8592 | 6.2522 | 0.6227 | 0.6255 |
| 2 | No vital evidence | 743 / 10 | 1.6914 | 1.0487 | 0.6989 | 0.4721 |
| 2 | No primitive evidence | 720 / 10 | 1.9338 | 1.1956 | 0.7137 | 0.4777 |

All these windows were retained. R1's no-vital AUROC is below 0.5 in all folds;
R0's is above 0.5, although subgroup positive counts are small and dependent.
No subgroup significance claim is made. At threshold 0.5, recall remains zero
and precision undefined in every listed stratum.

On no-primitive windows R0's value pool is deterministically zero, so any ranking
comes from observation/recency/clock context. Its above-chance point estimates in
these windows show that context can carry signal without observed values. This
does **not** establish that full-cohort prediction is predominantly process-driven:
the process-only versus joint head comparison remains necessary. More complex
process encoding in R1 was particularly unhelpful in the empty-window subgroup.

## Reconstruction and representation learning

| Fold/model | Initial held loss | Selected held loss | Final held loss | Selected draw | Train loss first/last 100 mean |
|---|---:|---:|---:|---:|---:|
| 0/R0 | 0.5103 | 0.1950 | 0.1978 | 1,800 | 0.3235 / 0.1993 |
| 0/R1 | 0.4637 | 0.1942 | 0.1944 | 1,800 | 0.3105 / 0.1988 |
| 1/R0 | 0.3955 | 0.2019 | 0.2020 | 1,800 | 0.3427 / 0.2163 |
| 1/R1 | 0.4014 | 0.2003 | 0.2003 | 2,000 | 0.3431 / 0.2159 |
| 2/R0 | 0.5511 | 0.2381 | 0.2395 | 1,800 | 0.3601 / 0.2207 |
| 2/R1 | 0.5236 | 0.2372 | 0.2415 | 1,800 | 0.3582 / 0.2201 |

Both models learn the masked standardized Huber objective, with finite, nonconstant
representations. R1 achieves slightly lower selected held loss in every fold,
yet improves downstream AP/AUROC only in fold 2. Better reconstruction therefore
does not establish better Level-3 prediction. Without the unfitted random-encoder
head control, SSL's incremental supervised benefit is **unknown**, not positive.

For reference, held-loss curves at draws 200, 400, ..., 2,000 are:

| Fit | Fixed held-loss sequence |
|---|---|
| 0/R0 | .2535, .2144, .2087, .2056, .1996, .1989, .1969, .1979, .1950, .1978 |
| 0/R1 | .2296, .2075, .2037, .2003, .1964, .1960, .1951, .1958, .1942, .1944 |
| 1/R0 | .2528, .2150, .2120, .2046, .2176, .2053, .2038, .2036, .2019, .2020 |
| 1/R1 | .2422, .2104, .2070, .2041, .2042, .2034, .2022, .2015, .2011, .2003 |
| 2/R0 | .2791, .2552, .2454, .2447, .2424, .2428, .2400, .2397, .2381, .2395 |
| 2/R1 | .2762, .2584, .2441, .2433, .2415, .2394, .2372, .2374, .2372, .2415 |

Full per-draw training curves and head fitting objectives are preserved privately.
All heads retained unchanged frozen encoders and reduced their training objective;
the optimizer cap was 50 iterations, not an independently recorded convergence certificate.
Held representation effective ranks were 2.36/2.48/2.39 for R0 and 7.07/8.07/6.31
for R1. These scale-sensitive whole-vector diagnostics include process summaries
and flags; higher rank is not proof of more useful physiology or avoidance of collapse.

| Fold, each recipe | Exposed patients | Window exposures | Unique sampled windows | Masked targets | Windows without targets |
|---|---:|---:|---:|---:|---:|
| 0 | 4,375 | 127,948 | 127,930 | 2,468,466 | 24.73% |
| 1 | 4,343 | 127,980 | 127,905 | 2,576,248 | 23.86% |
| 2 | 4,352 | 127,982 | 127,849 | 2,484,190 | 24.20% |

Each fit made exactly 2,000 optimizer updates; no entire batch lacked targets.
Natural empty-primitive exposure percentages were 24.42%, 23.56%, 23.90% respectively.
Total exposures across the six fits were 767,820 and total contributing masked
targets 15,057,808. R0/R1 exposure/target counts match within every fold. Short
patient pools explain totals slightly below the 768,000 maximum; no budget changed.

## Runtime/resources

| Fit | SSL minutes, including loading/checks | Windows/second | Head/export minutes, approximate |
|---|---:|---:|---:|
| 0/R0 | 18.41 | 115.9 | 4.41 |
| 0/R1 | 9.76 | 218.4 | 2.81 |
| 1/R0 | 9.03 | 236.2 | 1.87 |
| 1/R1 | 7.21 | 295.9 | 1.89 |
| 2/R0 | 9.21 | 231.7 | 2.39 |
| 2/R1 | 11.62 | 183.5 | 2.19 |

Summed SSL time was 65.24 minutes; head/export timestamp intervals totaled 15.55
minutes. The manual resumed session lasted approximately 67.77 minutes, including
remaining preparation. The original session lasted approximately 29.82 minutes
through its deliberate pause, giving about 97.6 active session minutes excluding
the idle gap. Head/export estimates are file timestamp differences, not isolated
optimizer timings. Peak recorded process RSS reached 1,533,620,224 bytes (~1.43 GiB).
RSS is cumulative per process, and the run crossed two sessions; these are not
isolated architecture memory measurements. I/O/cache/system variation also prevents
using these timings as a controlled R0/R1 speed comparison. Both sessions used CPU,
four threads, batch cap 64. No GPU study fitting was performed.

## Remaining controls and decision

| Question | Current answer |
|---|---|
| SSL R0 versus frozen random R0 + head | Unavailable: random encoder checkpoint exists, but its fitted control head/scores do not. |
| Process-only versus joint | Unavailable: the required process-only logistic head/scores were not generated. |
| Value-access versus joint | Unavailable: the reduced-representation head/scores were not generated. |
| Reproducible R1 advantage | Not demonstrated; two folds favor R0, pooled paired intervals include zero. |
| R2 progression earned | **No.** Do not add cross-attention to compensate for an unproven separation benefit. |
| R3 justified now | **No empirical justification from this screen.** Adding a variational objective does not resolve missing control evidence or absent clinical-utility evaluation. |

**Single next experiment:** complete the already declared frozen-head controls on
the same folds, IDs, weights and label budget. This entails no encoder retraining,
new architecture, threshold tuning or canonical test access. It is needed before
claiming SSL benefit or predominantly physiological prediction. If no additional
head fitting is permitted, those comparisons must remain unavailable.

The existing `analyze` command performs these previously unfitted logistic-head
controls as well as the full analysis; it is **not a purely read-only report command**.
It was not executed because it reads selected training-patient shards and is likely
to exceed five minutes. Run manually only to complete those declared controls:

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$studyLog = 'runs/model_studies/r0-r1-bounded-3fold-v1/analysis-progress.log'
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.architecture_study.analyze --run runs/model_studies/r0-r1-bounded-3fold-v1 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $studyLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw "Analysis failed; preserve its output directory and log." }
```

Budget **15–30 minutes**, with substantial I/O variability. This command does not
resume partial analysis: it rejects an existing `analysis/` directory. Preserve
partial outputs and request recovery rather than delete/restart blindly. Success
requires `analysis/comparison.json`, a completed `analysis/artifacts_sha256.json`,
the console message `Training-fold analysis complete`, and a successful exit code.
The existing root `complete.json` proves only the six original fits are complete.
Console `.log` files are excluded from the artifact checksum manifest because
PowerShell can append the completion message after hashing. Preserve them, but do
not interpret an in-progress console-log hash as a frozen provenance check.

Monitor from Git Bash (WSL uses `/mnt/c` instead of `/c`):

```bash
cd /c/Users/eldar/Projects/AI-CVD
tail -F runs/model_studies/r0-r1-bounded-3fold-v1/analysis-progress.log
```

The analysis command logs each fold and predefined control; it can be quiet while
loading shards or extracting embeddings. File creation under `analysis/` also shows
progress. Preserve the whole study directory, including
all completion/provenance files, both progress logs, saved-artifact analysis and
eventual control outputs. Nothing under this private run is committed to Git.
