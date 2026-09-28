# Explicit continuation after accidental interruption

This is an engineering recovery amendment, not a scientific amendment. The user
attests that no test performance was used to change model, calibration, threshold,
policy, architecture or evaluation specification. The original frozen artifacts
and original preflight receipt remain unchanged. No recovery execution was started
by the assistant.

## Read-only audit findings

The original runner commit was `7794c5ddf92848c1eb379b4c4f1456090d544d35`.
No Python evaluation process remained running when process inventory was checked.
The original console log ended at patient **391/1,839**, with **8,837,958** of
**39,284,268** windows complete. The directory contained:

- 391 prediction NPZ files and 391 corresponding patient JSON records;
- original `started.json` and five preallocated pooled NPY arrays;
- no `patients.json`, `test_results.json`, `final_result_manifest.json` or
  `complete.json`.

The first read-only audit independently verified all 391 completed patient pairs
in approximately 98 seconds. Their combined NPZ/JSON size was 592,121,885 bytes.
Their filename-to-hash mapping fingerprint was
`14f980f8856f4a70184a1e2038ae5f60f6fb6a72acfd9c1259c472e63e35ffcb`.
Checks covered original receipt and scientific/source bindings, each saved
prediction checksum, canonical source shard checksum, patient/order/sample IDs,
prediction timestamps, labels and event index ranges, evidence masks, calibrated
scores, saved policy-record replay and exact pooled-prefix alignment. It did not
rescore any completed prediction or aggregate/interpret test performance.

The final post-commit recovery audit also checks cache dtypes/shapes, untouched
zero-initialized tails, absence of unexpected/orphan files and snapshots every
partial artifact by SHA256. Its private report is stored outside the interrupted
directory at `runs/final_studies/r0-final-v1-test-recovery-audit-v1.json`.

**No aggregate test performance report was produced or printed.** The original
runner had maintained in-memory counters and saved per-patient policy records,
as designed. Those records were replayed only for integrity. Full-stream AP,
AUROC, calibration diagnostics and aggregate operational reports are computed
only after all patients finish. There was no fitting/tuning code path, and the
source and scientific artifact checks show no change to the frozen protocol.

## Decision and safeguards

Use **deterministic continuation from patient 392**, preserving the existing
directory. There is no integrity reason to discard or regenerate the completed
work. Remaining work is 1,448 patients and 30,446,310 windows.

`scripts/test_recovery.py` provides an explicit audit and continuation entry point.
The ordinary one-shot `run` command still refuses an existing directory. Recovery:

1. Rechecks every frozen scientific binding and the original preflight receipt.
   The original runner source is authenticated against its original Git commit.
   Only this reviewed recovery amendment is exempt from byte identity with that
   earlier runner; the original scorer, metric helper and all scientific modules
   remain unchanged. The recovery source and commit are bound separately.
2. Requires an exact fresh audit snapshot and a clean working tree before writing.
   It rejects stale reports, changed artifacts, gaps, orphan/partial files,
   duplicate IDs, cache mismatch, unexpected files or aggregate-stage output.
3. Acquires a Windows named kernel mutex, preventing concurrent recovery processes.
   The OS releases the mutex after termination; no stale lock file is deleted.
4. Reuses complete patient NPZ/JSON files byte-for-byte. Completed pooled prefixes
   are verified and left untouched. It restores in-memory accounting from verified
   patient records, opens pooled arrays in update mode, and scores only the
   remaining patients with the original model and approved guarded CPU256 scorer.
5. Flushes new pooled entries before committing each new patient JSON record.
   An interruption mid-file is not silently salvaged: it fails the next audit and
   requires further explicit review.
6. Appends `recovery-0001.json` (and numbered records for subsequent authorized
   recoveries), preserving `started.json`, original receipt and original log.
   Recovery provenance records the interruption explanation, user attestation,
   complete audit snapshot, old/new code bindings and exact reused prefix.
7. Produces the original final result structure and checksum/completion seal.
   Scientific outputs/order are unchanged. Recovery provenance and timestamps
   necessarily differ from a hypothetical uninterrupted run. Final runtime/resource
   fields explicitly describe the current continuation, not invented cumulative
   active time; prior start/progress records remain available.

The recovery auditor reads only the completed patients' event records from the
canonical export; it does not decode remaining event metadata or open remaining
patient shards. No score forward pass or aggregate discrimination is performed
by the audit command. As with the original seal, checksums establish integrity
relative to saved provenance; they are not an external cryptographic attestation.

## Quick tests

Six new recovery tests exercise byte-preserved reuse, equality with uninterrupted
fictional evaluation, repeated interruption and continuation, stale-snapshot
rejection before writes, orphan/tampered chunk rejection, cache-prefix/tail
rejection, completed-run refusal and concurrent-mutex rejection. Existing
evaluation-only AST checks now also cover recovery code. The nine evaluation
tests and two scorer-equivalence tests are rerun. All data in these tests are
fictional; no private prediction is recomputed.

## Manual continuation

The post-commit audit receipt is prepared once by this command (about two minutes;
read-only with respect to the interrupted directory). Do not rerun it at the same
path if the prepared receipt already exists:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_recovery audit --receipt runs/final_studies/r0-final-v1-test-preflight-v1.json --audit runs/final_studies/r0-final-v1-test-recovery-audit-v1.json
```

From `C:\Users\eldar\Projects\AI-CVD`, run the following manually after review:

```powershell
& C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe -u -m scripts.test_recovery continue --receipt runs/final_studies/r0-final-v1-test-preflight-v1.json --audit runs/final_studies/r0-final-v1-test-recovery-audit-v1.json --attest-no-test-driven-changes 2>&1 | Tee-Object -FilePath runs/final_studies/r0-final-v1-test-recovery-v1.console.log
if ($LASTEXITCODE -ne 0) { throw 'Recovery failed. Preserve every artifact and log; do not restart automatically.' }
```

**Expected remaining runtime: approximately 10–13 hours**, plus a short re-audit.
The actual interrupted run took about three hours for 8.84 million windows, much
slower than the earlier small-subset estimate. This estimate uses observed progress,
not test performance. Do not change threads, batch size or scoring settings to
chase the estimate. Avoid closing the terminal or allowing sleep.

Monitor from another Bash window:

```bash
cd /c/Users/eldar/Projects/AI-CVD
tail -f runs/final_studies/r0-final-v1-test-recovery-v1.console.log
```

Success still requires `runs/final_studies/r0-final-v1-test/complete.json` with
`status: complete` and valid hashes, plus `TEST EVALUATION COMPLETE`. Individual
patient progress is not a completion marker. The uncertainty job remains separate
and must not start until the final seal is verified.

If interrupted again, do not reuse the old audit receipt or truncate an existing
log. First check process state, then create a new immutable `...recovery-audit-v2.json`
using the same audit command with that new path. Review its result before using
`continue` with that receipt and a new `...recovery-v2.console.log`. A clean,
complete prefix can continue; partial files or inconsistent caches stop for
explicit recovery review. No automatic restart or artifact deletion is provided.

Preserve the entire interrupted/recovered directory, original console log,
original preflight receipt, every recovery audit and console log, numbered
recovery records, and frozen model/specification/engineering supplement. Private
artifacts remain Git-ignored. Do not delete partial files to make the audit pass.
