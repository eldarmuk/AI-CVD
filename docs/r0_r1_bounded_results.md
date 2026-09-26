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

Results pending execution. All patient-level artifacts remain Git-ignored.
