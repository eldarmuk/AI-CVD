# Continue the existing bounded study manually

The existing run is `runs/model_studies/r0-r1-bounded-3fold-v1`.
Fold 0 / R0 is complete and verified; five model/fold fits remain. No private
training was launched during the resume-repair task. The frozen configuration,
data task, model/input definitions and study budget are unchanged.

The original runner rejected existing output directories and had no resume support.
The repaired runner requires **explicit `--resume`**. Repeating the original command
without this flag still fails safely; it does not overwrite or restart the run.

With `--resume`, the runner verifies completed model files and skips them, retains
the saved fold/scaler/sample plans, and checks R0/R1 corruption pairing. New folds
use the original deterministic definitions and seeds. It records a new resume
provenance receipt without rewriting the original study metadata. Windows manifest
path separators are normalized for the orchestration compatibility check; numerical
implementation hashes remain enforced.

## Manual PowerShell command

Allow approximately **2–3 hours remaining** on the same CPU, with variation from
disk access and other workloads. Fold 0/R0's SSL stage alone took approximately
18.4 minutes; plan construction and head/export work take additional time.

This version-compatible logging command appends UTF-8 lines and works with Bash
`tail`, including when launched from Windows PowerShell:

```powershell
Set-Location 'C:\Users\eldar\Projects\AI-CVD'
$studyLog = 'runs/model_studies/r0-r1-bounded-3fold-v1/progress.log'
& 'C:/Users/eldar/anaconda3/envs/ai-cvd-gpu/python.exe' -u -m src.architecture_study.cli run --config configs/model_studies/r0_r1_v1.toml --device cpu --output runs/model_studies/r0-r1-bounded-3fold-v1 --resume 2>&1 | ForEach-Object {
    $line = "$_"
    Write-Host $line
    Add-Content -LiteralPath $studyLog -Value $line -Encoding utf8
}
if ($LASTEXITCODE -ne 0) { throw "Study exited with code $LASTEXITCODE; preserve the run and progress.log." }
```

Expect `Skipping verified fold 0 / R0`, then Fold 0/R1 and the two remaining folds.
Do not run a second copy simultaneously or change the config between invocations.
There can be quiet periods while plans, heads or exports are being prepared.
SSL progress is printed every 200 draws.

## Monitor from another Bash window

In Git Bash:

```bash
cd /c/Users/eldar/Projects/AI-CVD
tail -F runs/model_studies/r0-r1-bounded-3fold-v1/progress.log
```

For WSL, use `/mnt/c/Users/eldar/Projects/AI-CVD` instead. Stop `tail` with Ctrl+C
without affecting the PowerShell training process. To inspect completed fits:

```bash
find runs/model_studies/r0-r1-bounded-3fold-v1 -maxdepth 2 -name '*-complete.json' -print
cat runs/model_studies/r0-r1-bounded-3fold-v1/complete.json
```

The final `cat` fails harmlessly until whole-study completion. Success requires:

1. All six `fold-{0,1,2}/{R0,R1}-complete.json` markers with `complete_verified`.
2. Root `complete.json` with `status: complete`.
3. Console message `Bounded study complete: complete.json and six verified model markers are present.`
4. A successful Python exit code.

The historical `pause_receipt.json` remains as audit evidence; it is not an active
pause switch. Root completion covers the six model/head fits, not the separate
cheap-control/aggregate analysis. Request analysis after these completion checks.

## Interruption and preservation

Resume is safe **between completed fits**, not from arbitrary optimizer steps.
If interrupted mid-fit, completed earlier models remain protected, but partial
artifacts are rejected. Do not delete, overwrite or force-restart them to bypass
that guard; request a recovery review. An interruption between final export and
completion-marker creation may need explicit verified adoption, as performed for
the original Fold 0/R0. Partial fold preparation is likewise not silently redone.

Preserve the entire run directory: config, original study/source provenance,
resume/pause receipts, fold assignments, all sampling plans and hashes, checkpoints
and sidecars, completion markers, telemetry, embeddings, predictions, metrics and
`progress.log`. These remain private and Git-ignored.

For future work, commands expected to exceed about five minutes, consume substantial
resources, or process the full private dataset must be supplied for manual execution
with runtime, completion, resume and log-preservation guidance. They are not to be
launched autonomously.
