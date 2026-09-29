# Problem 10: definitive held-out R0 results

**Valid final evaluation; Problem 10 scientifically complete.** The full test audit and fixed uncertainty analysis passed. No rescoring, fitting, recalibration, threshold/policy change or new model experiment occurred during closure. Comparisons are descriptive; no post-hoc significance test was introduced.

## Verification and immutable provenance

The test run, independent full audit and uncertainty analysis all have successful completion markers and the expected terminal messages. Closure verified their seals and bindings before interpreting results. The full audit verified every canonical sample ID, label, source-shard identity, pooled-array entry, artifact checksum, episode link, calibrated score, policy replay and aggregate metric calculation across 39,284,268 windows. Closure verified the saved audit rather than repeating that full private-data pass.

The model/head/scaler/projection, scientific specification, engineering supplement and scoring sources match the frozen versions. The approved guarded CPU batch-256 scorer was used. The recovered 391-patient/8,837,958-window prefix remains exactly preserved. Original receipt, interruption provenance and the user's no-test-driven-changes attestation remain intact. There is no fitting/tuning path in the evaluation or uncertainty procedures. Saved scores are authenticated by sealed hashes and frozen scorer provenance, not recomputed at closure.

The uncertainty procedure used exactly 200 ordinary patient-cluster bootstrap replicates, seed 101027, and exact weighted tied-score ranking. Closure checked all replicate checksums, the exact seeded patient draw matrix, interval arithmetic and source bindings. All 200 replicates were defined for all four reported statistics. Saved frozen predictions alone were used; no model, head, calibration or threshold was refitted.

Frozen test completion SHA256: `463e1e24b64b92b944af4570a37a037c39f62faf6430dcd1ef0d377c1bfa79ea`.
Test-results SHA256: `6b5adcfd98f5791e77515a72016cb3050fe54764fbdd788a7ad27b1a11d97771`.
Scientific-specification SHA256: `0ad2c76b2f8ec65ead77c66067e3760d3581e6d762b47a1ed2e3cb5a6986899c`.
Engineering-supplement SHA256: `d284a32cbf27a315f72cd8a8da8dd7bd4cd622cadf3ccae19d831221e3e5be83`.

## Population and support

| Quantity | Test result |
|---|---:|
| Supported patients | 1,839 |
| Eligible five-minute windows | 39,284,268 |
| Positive windows | 1,349 |
| Patients with eligible Level-3 episodes | 27 |
| Unique eligible Level-3 episodes | 29 |
| Supported patient-days | 136,403.7083 |
| Eligible prediction-time hours | 3,273,689 |
| Eligible supported prediction-time coverage | 100% |

The 1,349 positive windows are overlapping opportunities for **29 unique episodes in 27 patients**, not 1,349 independent events. Repeated episodes within a patient may also correlate. Coverage is complete over canonical eligible support, not verified enrollment or continuous clinical surveillance. The within-patient eligible-span coverage statistic is also 1.0. Researcher-attested alert completeness and measurement availability, and retrospective activity-support censoring, remain limitations.

## Discrimination and uncertainty

| Metric | Estimate | 95% patient-cluster percentile interval |
|---|---:|---:|
| AUROC | **0.606845** | **0.473888–0.728772** |
| Average precision | **0.0000847131** | **0.0000351220–0.0002343944** |
| Natural prevalence | 0.0000343394 (0.00343394%) | Not separately estimated |
| AP / prevalence | **2.46693×** | Not separately estimated |
| Calibrated log loss | 0.0003870721 | Not estimated |
| Calibrated Brier score | 0.000034337954 | Not estimated |

AP exceeds prevalence in the point estimate, but absolute precision is very small. The AUROC interval includes 0.5. Comparing the AP interval with a fixed prevalence is not a paired test of lift: prevalence also varies under patient resampling, and no lift interval was prespecified. These results support a modest observed ranking signal with substantial uncertainty, not robustly established or clinically sufficient discrimination.

## Calibration transport

The unchanged validation-fitted intercept is **−0.26813403062419017**. Calibration is intercept-only, with no test fitting; AUROC and AP remain unchanged.

| Diagnostic | Raw | Calibrated |
|---|---:|---:|
| Mean predicted risk | 0.0000308349 | 0.0000235830 |
| Mean risk / observed prevalence | 0.89794 | 0.68676 |
| Log loss | 0.0003851168 | 0.0003870721 |
| Brier score | 0.000034337828 | 0.000034337954 |

Observed prevalence was **0.0000343394**. Raw mean risk underestimated it by 10.2%; calibrated mean risk underestimated it by **31.3%**. The fixed intercept slightly worsened test log loss and Brier score. Calibration did not transport well in the mean. Tiny absolute Brier scores are dominated by rarity and do not establish good tail calibration. Calibration slopes and reliability-bin uncertainty were not reported by the frozen analysis and are not invented here.

## Frozen operational policy

The calibrated threshold remains **0.00007816438698653833**, with first crossing, four-hour cooldown and one-to-one matching to the earliest unmatched episode in `(t,t+4h]`. No point adjustment or test-dependent policy change was used.

| Quantity | Test result |
|---|---:|
| Emitted alerts | 12,814 |
| Matched alerts / detected episodes | 4 |
| Unmatched alerts | 12,810 |
| Unmatched alerts per supported patient-day | **0.0939124** |
| 95% patient-cluster interval for burden | **0.0862240–0.1039739** |
| Alert precision | **0.0312159%** (4/12,814) |
| Detected / eligible episodes | **4/29** |
| Episode sensitivity | **13.7931%** |
| 95% patient-cluster interval for sensitivity | **3.8426–25.9300%** |
| Supplementary exact binomial interval | **3.8895–31.6641%** |

The burden point estimate meets the 0.1 unmatched-alerts/day cap; its interval crosses 0.1, so a guaranteed population bound is not established. About 86% of eligible episodes were missed, with approximately 3,204 emitted alerts per matched episode. Unmatched means unmatched to the Level-3 endpoint; other possible clinical relevance was not adjudicated.

All four detected lead times, sorted without patient/event identifiers:

| Detection | Minutes | Time |
|---|---:|---:|
| 1 | 45.05 | 45m03s |
| 2 | 50.20 | 50m12s |
| 3 | 75.45 | 1h15m27s |
| 4 | 210.75 | 3h30m45s |

Median lead time was **62.825 minutes**; Q1–Q3 was **48.9125–109.275 minutes** (IQR width 60.3625 minutes), with range **45.05–210.75 minutes**. Quartiles interpolate only four detections. These conditional warning times do not describe reliability across all episodes. The endpoint is **alarm initiation retrospectively classified Level 3**, not physiological crisis onset, ambulance dispatch or confirmed cardiovascular deterioration.

## Evidence-availability strata

| Stratum | Windows (% of all) | Positive windows | Eligible episodes / event patients | AUROC | AP | AP/prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Physiology observed | 29,444,680 (74.95%) | 1,106 | 26 / 24 | 0.614527 | 0.0000956941 | 2.54763× |
| No physiology | 9,839,588 (25.05%) | 243 | 6 / 6 | 0.551606 | 0.0000275575 | 1.11586× |
| No primitive evidence, nested | 9,511,771 (24.21%) | 242 | 6 / 6 | 0.564152 | 0.0000470992 | 1.85123× |

| Stratum | Emitted / matched / unmatched alerts | Detected / eligible episodes | Sensitivity | Precision | Unmatched alerts/day |
|---|---:|---:|---:|---:|---:|
| Physiology observed | 12,755 / 4 / 12,751 | 4/26 | 15.38% | 0.03136% | 0.124718 |
| No physiology | 59 / 0 / 59 | 0/6 | 0% | 0% | 0.00172690 |
| No primitive evidence | 0 / 0 / 0 | 0/6 | 0% | Undefined: no alerts | 0 |

Alerts are attributed by evidence at emission under the global policy; crossing/cooldown was not rerun within strata. Episode opportunity sets and event-patient counts overlap and must not be added. No-primitive is nested within no-physiology. Subgroup intervals were not prespecified or produced by the fixed overall bootstrap; they are unavailable, particularly important for the six-event strata.

Validation's striking empty-window discrimination **did not reproduce**: no-physiology AUROC fell from 0.8170 to 0.5516, and no-primitive AUROC from 0.8126 to 0.5642. Physiology-observed AUROC increased descriptively from 0.5487 to 0.6145. The test does not justify repeating a claim that prediction is predominantly observation-process driven. Nevertheless, some ranking remains without observed primitive values, prior controls demonstrated process signal, and joint representations retain masks/recency/context. Observation-process dependence remains a limitation, not a resolved causal explanation. Stratum differences do not isolate the incremental contribution of physiology.

## Development → validation → test

| Metric | Training-fold development | Canonical validation | Held-out test |
|---|---:|---:|---:|
| AUROC | ≈0.631 | 0.60574 | 0.60685 |
| AP | ≈0.0000468 | 0.000049405 | 0.000084713 |
| Natural prevalence | Not restated here | 0.000026568 | 0.000034339 |
| AP/prevalence | Not restated here | 1.86× | 2.47× |
| Detected / eligible episodes | Policy unavailable | 1/23 | 4/29 |
| Episode sensitivity | Not available | 4.35% | 13.79% |
| Unmatched alerts/patient-day | Not available | 0.10429 | 0.09391 |
| Emitted alerts | Not available | 14,558 | 12,814 |
| Alert precision | Not available | 0.00687% | 0.03122% |
| Detected lead time | Not available | 51m55s, n=1 | Median 62.825 min, n=4 |

1. **Discrimination:** modest point-estimate ranking weakened from development to validation and reproduced almost exactly on test. The wide test interval prevents a strong above-chance generalization claim.
2. **AP:** remained above natural prevalence in the point estimate (2.47×), but absolute precision is low. AP is prevalence-sensitive; no significance claim or lift interval is available.
3. **Calibration:** did not transport well; the fixed correction increased mean-risk underestimation and slightly worsened proper scores.
4. **Episode sensitivity:** numerically higher on test, but still low and uncertain. Poor operational coverage persists.
5. **Burden:** broadly similar and below the cap in the test point estimate; its interval extends above the cap.
6. **Process dependence:** the dramatic validation empty-window pattern did not reproduce. Retain the limitation without asserting dominance.
7. **Validation representativeness:** broadly representative for overall AUROC and low utility; numerically pessimistic for test AP, sensitivity and precision, but optimistic for empty-window ranking and calibration transport. These descriptions are not statistically established differences.
8. **Most important scientific result:** modest ranking did not translate into useful episode detection at the frozen alert-burden target. Rigorous auditing establishes confidence in that finding, not model success.

## Final scientific judgment

R0 demonstrably produces reproducible, patient-isolated scores and shows modest point-estimate ranking of this retrospective endpoint. It detected four eligible alarms before initiation. It did **not demonstrate** reliable episode coverage, adequate alert precision, transported probability calibration, physiological-deterioration detection or deployment readiness.

Only 29 unique episodes in 27 patients underpin outcome inference. Intervals are wide and the AUROC interval includes chance. Two hundred patient-cluster replicates give limited tail precision. The supplementary binomial interval assumes independent episodes and may miss within-patient correlation; patient-cluster intervals are primary. No uncertainty was estimated for precision, AP lift or subgroups.

The strongest substantive result is the independently tested separation between modest ranking and poor operational detection. The weakest result is low sensitivity and extremely low precision, compounded by uncertain discrimination and calibration transport. Findings concern this retrospective telecare cohort and researcher-attested source semantics; prospective and external generalization remain unknown.

## Claims we can make

- Under frozen patient-isolated retrospective evaluation, R0 achieved AUROC 0.6068 and AP 0.00008471 (2.47× observed prevalence), with wide patient-cluster uncertainty. This supports a limited, qualified retrospective risk-ranking claim.
- The fixed policy detected 4/29 eligible Level-3 alarms at 0.09391 unmatched alerts per supported patient-day, with 0.0312% precision.
- Median lead time among four detections was 62.83 minutes to alarm initiation retrospectively classified Level 3; most eligible episodes were missed.
- The validation-derived intercept underestimated mean test risk and did not improve test log loss or Brier score.
- Performance differed by evidence availability; unusually strong empty-window validation ranking did not reproduce.
- Causal timing, explicit masks, patient isolation, episode accounting and natural-stream evaluation enable an auditable assessment of these limitations; they do not establish model superiority.

## Claims we must avoid

- Reliable early warning, demonstrated clinical utility/benefit, safe deployment or deployment readiness.
- Cardiovascular crisis prediction, confirmed deterioration detection, warning before physiological crisis onset or prediction of ambulance dispatch.
- Strong calibration or reliable individual probabilities on new patients.
- Architectural superiority, validated benefit of cross-attention/VAE components, or an established improvement from a longer SSL budget.
- Millions of independent clinical observations, or warning-time summaries applying to all episodes.
- A demonstrated physiological mechanism or definitive observation-process dominance on test from stratification alone.
- Generalization beyond this retrospective telecare cohort, prospective safety, external validity or a guaranteed alert-burden cap.

## Paper-ready Results paragraph

In the held-out cohort of 1,839 patients, 39,284,268 eligible five-minute prediction windows included 1,349 positive windows corresponding to 29 unique Level-3 alarm episodes in 27 patients. The frozen R0 model achieved an AUROC of 0.6068 (95% patient-cluster bootstrap interval, 0.4739–0.7288) and average precision of 8.471×10⁻⁵ (3.512×10⁻⁵–2.344×10⁻⁴), compared with a natural prevalence of 3.434×10⁻⁵ (AP/prevalence, 2.47). The frozen first-crossing policy with a four-hour cooldown detected 4/29 episodes (13.79%; 95% patient-cluster interval, 3.84–25.93%) and emitted 12,814 alerts, of which 12,810 were unmatched to the endpoint. Unmatched-alert burden was 0.09391 per supported patient-day (95% interval, 0.08622–0.10397), and alert precision was 0.0312%. Among four detections, median lead time to alarm initiation retrospectively classified Level 3 was 62.83 minutes (Q1–Q3, 48.91–109.28; range, 45.05–210.75). Calibrated mean risk was 2.358×10⁻⁵, approximately 31.3% below observed prevalence. AUROC was 0.6145 in windows with physiological evidence, 0.5516 without physiological evidence and 0.5642 in the nested no-primitive-evidence subgroup.

## Paper-ready Discussion paragraph

Overall discrimination was similar on validation and test, but modest retrospective ranking did not translate into reliable episode detection. The fixed policy missed 25 of 29 eligible episodes and had very low alert precision despite an unmatched-alert rate below the prespecified cap in the test point estimate. Warning times among four detected episodes should therefore not be interpreted as reliable clinical early warning. The validation-derived calibration intercept underestimated test risk, and strong empty-window discrimination observed in validation did not reproduce, indicating uncertainty about the stability and source of the learned signal. Wide patient-cluster intervals, only 27 event patients, researcher-attested outcome coverage, retrospective support restrictions and an endpoint based on subsequently classified alarm initiation limit interpretation. These findings support a cautious retrospective risk-ranking claim within this cohort, but do not establish physiological deterioration detection, clinical utility or deployment readiness.

## Closure and Problem 11

Problem 10 is scientifically complete; uncertainty is complete. A separate immutable closure summary under `runs/final_studies/r0-final-v1-paper-summary-v1` binds the final test, full audit, uncertainty and this report. Original artifacts remain unchanged, including historical text saying uncertainty was pending at initial scoring completion; the separately sealed uncertainty result and closure summary supersede that status.

No further canonical test-set access, model scoring or experiment is needed for this paper. No test-informed model development or scientific retuning is allowed within this frozen confirmatory analysis. Future model research requires a separately declared exploratory study and a new untouched evaluation cohort; it must not be folded back into this paper's confirmatory result.

Problem 11 should assemble the manuscript and reproducibility package: reconciled cohort flow, frozen methods, final tables/figures from saved aggregate results, claims and limitations, NDA-safe synthetic fixtures, code/config availability and reporting-consistency checks. It should not be another model-selection stage.
