# Problem 7: source evidence gate

The source-readiness audit is complete; canonical private dataset generation remains
blocked pending evidence. This document contains the reusable decision procedure,
not private counts, patient identifiers, source fingerprints or clinical records.
The private `dataset_integrity_report.md`, declarations, coverage request sheet,
partition inventory and artifact checksums remain under ignored `data/private/`.

## Four questions for the data owner

1. **Coverage and completeness:** For the supplied patient IDs, provide documented monitoring/enrollment start, continuous measurement-coverage end, and continuous alert/outcome-coverage start and end, using timezone-aware timestamps. Identify withdrawals, outages and missing exports. Confirm whether absence of a qualifying alert during the declared outcome interval means no **recorded escalation**, and identify the frozen export to which this confirmation applies. Return `senior_id,enrollment_time,measurement_coverage_end,outcome_coverage_start,outcome_coverage_end`. If the same dates truly apply to everyone, explicitly confirm each field for every supplied ID and list all exceptions; a global export date range alone is insufficient.
2. **Measurement time and timezone:** Does `measurements.date` represent acquisition time, server receipt/availability time, or something else? State when each reading could first be used for prediction, including delayed uploads and backfills. If it differs from `date`, supply a per-reading `available_at` timestamp keyed by measurement ID. Specify the timezone of measurement and alert timestamps and the rule for ambiguous/nonexistent DST times. State whether the answer applies to every device, patient and export period; list exceptions.
3. **Escalation time:** What exactly does `alert_date` represent? Was the qualifying severity already recorded at that timestamp, or was `sos_note` written/edited later? Either confirm that qualifying severity was recorded at `alert_date` for all relevant records, or provide the first qualifying dispatch/escalation-recording timestamp keyed by alert ID, including note revisions. Identify exceptions and whether the supplied feed captures all such recorded escalations during the coverage intervals in question 1.
4. **Steps:** Are `Steps` interval increments or cumulative counters? Specify units, reset rule, reset timezone, and behavior after restart, replacement or rollover. Confirm whether the same rules apply to all devices and periods; otherwise identify the affected records/devices and their rules. Provide the device/export specification or an accountable source-system confirmation. The observed cumulative pattern alone is not authoritative documentation.

## Global versus patient-level answers

Time semantics, timezone, alert-note timing and step-counter rules can be confirmed
globally if the owner explicitly attests uniformity for the entire identified
snapshot and identifies exceptions. Patient-level answers are needed where these
rules differ; the present single-contract runner cannot silently mix such rules.

Coverage is a patient-level scientific fact. One global attestation can establish
it only if it explicitly covers every included patient, the actual enrollment
start, each required interval and all exceptions. Such an attestation may be expanded
into one identical declaration row per covered patient, with provenance retained.
A file's export range, its first/last readings, or import completion cannot do so.
Different starts, withdrawals or coverage gaps require patient-specific evidence.
The current runner supports one continuous declared interval per patient; gaps or
multiple episodes of enrollment require explicit handling before extraction.

## Blockers, assumptions and usable features

| Item | Classification | Consequence |
|---|---|---|
| Enrollment and continuous outcome coverage | Required external fact, unresolved | Blocks labeled cohort generation and negative-label validity |
| Measurement availability and timezone | Required external facts, unresolved | Blocks claims that private sequences contain only prediction-time information |
| First qualifying escalation recording time | Required external fact, unresolved | Blocks the four-hour endpoint and lead-time interpretation |
| Cumulative Steps | Strong empirical support, provisional interpretation | Does not establish reset/device semantics; no private step-derived features are certified |
| Undated clinical history | Temporal availability unresolved; exclusion is an explicit design choice | Non-blocking while excluded; all static values remain unknown, with zero known masks |
| Keyword Level-3 endpoint | Explicit study proxy | Generation may use the prespecified proxy once timing/coverage are justified; clinical adjudication remains necessary for stronger clinical claims |

No assumption equating acquisition with receipt, alert creation with later note
content, or first/last measurements with coverage has been adopted.

Raw HR, SBP, DBP, temperature and SpO2 can be inspected for descriptive quality
checks now. Once measurement timing/timezone and coverage are established, these
physiological channels, paired pulse pressure, co-bucket HR/SBP, valid-observation
masks, recencies and causal rolling descriptors can be used without clinical history.
If alert timing remains unknown, no canonical supervised four-hour labels are valid.
If measurement availability remains unknown, none of the time-series channels is
certified as a prospective model input merely because its value is plausible.
UTC clock features also require correct source-time conversion.

If only Steps remains unresolved, other physiological channels may be scientifically
usable, but the current fixed feature schema still requires a Steps declaration.
Dropping Steps would require an explicitly versioned feature/task change, not a
silent bypass. No such change is made in Problem 7.

## Minimum sufficient answers and next command

For the existing runner without further code changes, obtain:

- Verified continuous coverage/enrollment declarations for the included patients,
  with an identified complete frozen export and no unhandled gaps.
- A confirmed source timezone/DST interpretation and evidence that the stored
  measurement timestamp is the availability time. If separate receipt timestamps
  are supplied, the current source lacks the `available_at` field: an audited
  read-only join or separately versioned compatible export is required first.
  Do not add fields to or overwrite the original raw database.
- Evidence that the alert timestamp is the first qualifying severity recording
  time. If it is not, supplying separate escalation timestamps is necessary but
  requires an audited adapter change before running the current command.
- Confirmed Steps semantics and a supported reset rule (`daily`, with the proper
  timezone, or justified `decrease_only`). Keep clinical history excluded.

An accountable, specific source-system attestation can supply evidence; device
documentation is not the only acceptable form. Merely agreeing to an unsupported
assumption does not retrospectively establish coverage or clinical timing.

Preserve the blocked audit. Put the new reviewed files at
`data/private/confirmed/source-contract.toml` and
`data/private/confirmed/coverage.csv`. These paths are for new declarations, not
renamed copies of the unresolved placeholders. After confirming that all answers
fit the existing adapter and contract, run from the repository root:

```powershell
$runPath = "runs/level3-4h-private-" + [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
python -m src.ai_cvd.cli build --raw-db db/hrp_data.db --source-contract data/private/confirmed/source-contract.toml --coverage data/private/confirmed/coverage.csv --output $runPath --source-snapshot-id (Read-Host "Owner-confirmed immutable snapshot ID")
```

This resumes Problem 7 only. Validate the completed dataset and source fingerprints
before exporting sequences or fitting training-only statistics. It is not a
Problem 8 command. No model training or normalization is authorized by this audit.

## Validation and privacy boundary

The resumed complete suite passed 49 tests without skips, including four new
audit-gate tests; the earlier suite passed 45 tests. The
private report separately verifies the blocked contract rejection, empty confirmed
coverage, unique/disjoint prespecified source partitions, completed fingerprint
records and artifact checksums. Source counts are not eligible cohort sizes.
Prevalence, lead-time distributions, eligible counts and window exclusions remain
**unavailable**, not zero. The database was not rebuilt or relabeled.

Only this procedure and generic verification code/tests belong in the audit commit.
Keep patient-derived reports, metadata, hashes, IDs and coverage sheets out of Git.

To verify a completed blocked audit without repeating the source scan:

```powershell
python -m src.ai_cvd.audit_gate --audit PATH_TO_PRIVATE_AUDIT --raw-db db/hrp_data.db
```

This checks saved fingerprint completion and current file size/mtime, not a new
full-file content hash. Changed metadata requires snapshot reverification.
