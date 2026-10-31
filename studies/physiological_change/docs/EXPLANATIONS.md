# Interpretable results and explanation checks

The useful explanation is why the available measurements do or do not support
a change. It must be reconstructable from the model's real inputs and decision
rule. It is not a diagnosis or a story generated after seeing the reference.

## Planned output for each decision

| Output | Meaning |
|---|---|
| Decision time and measurement support | Age of latest reading, number and spread of observations, gap information |
| Prior and recent HR summaries | Exactly the values used by the model, with units and interval boundaries |
| Estimated change and interval | Magnitude/direction and the model's uncertainty, with calibration assumptions |
| Decision | Supported increase, decrease, stable, or insufficient evidence |
| Reason codes | For example insufficient history, stale readings or interval spanning incompatible conclusions |
| Information setting | Scalar-only or additional sensor/reference information, explicitly declared |
| Provenance | Model/configuration version and source-adapter version |

Labels such as "motion artifact" require actual supporting sensor evidence.
Missing data alone cannot explain non-wear or a person's behavior. Do not use
"safe", "healthy", "deteriorating" or "corrected true HR" for these outputs.

## Example, not a clinical result

An illustrative explanation might say: "The recent summary is 18 beats/min above
the previous interval. The estimated change interval spans -2 to 31 beats/min,
so the direction cannot be established. Two long gaps reduce available support."
The numerical range is invented for illustration and is not a calibrated output
of the current scaffold. The explanation must be generated from actual fields
once the relevant model exists; do not fabricate a confidence interval.

## How explanations will be evaluated

1. **Faithfulness:** regenerate every displayed value/reason from saved inputs
   and intermediate calculations; fail on any inconsistent decision or interval.
2. **Causality of information:** explanations must not read evaluation-only ECG,
   future measurements or later operator actions.
3. **Stability:** perturb timestamps/readings within documented tolerances and
   assess whether explanations change without a material change in evidence.
4. **Completeness:** show exclusions and abstentions, and identify which people
   or conditions lose usable output. A polished selected-case example is not enough.
5. **Usability:** if company feedback becomes available, use a structured task to
   check whether operators understand the decision and limitation. This assesses
   usability, not accuracy or clinical benefit.

For a learned model, a feature-attribution tool is optional. It must explain the
specified output and be compared with the transparent baseline's explanation;
correlated features and missingness can make attributions unstable. No causal
interpretation follows from feature importance. A generative explanation layer
is not part of the core study.

The initial package provides accounting and elementary estimates only. Model
intervals, the structured renderer and the above explanation experiments are
planned M4 work, with model-side leakage checks also needed earlier in development.
