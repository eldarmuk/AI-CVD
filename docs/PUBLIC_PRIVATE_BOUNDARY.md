# Public/private content boundary

This candidate contains maintained pipeline algorithms, historical source runners,
source-neutral storage/ingestion adapters, reviewed model copies, synthetic-generator source,
documentation, tests, configuration, CI and dependency specifications.
[public-release-manifest.json](../public-release-manifest.json) lists every candidate
file and its digest (the manifest itself is self-listed without a recursive digest).

No generated dataset, embedding, prediction file, fitted checkpoint, optimizer
history, log, private notebook, clinical figure, report export, source connector,
note lexicon, institution-specific schema, private manifest or machine path belongs
in the candidate. Generated fictional examples also stay untracked under outputs/.

This curated tree is being reviewed in the existing AI-CVD repository. Updating its
current files does not sanitize earlier Git objects, branches, PR references or
downloads. Historical exposure and access control require separate remediation.
The release contribution must contain only the approved tree and must not import
local experimental branches or their artifacts.

The original scientific checkout, branches, frozen files, failed experiments,
calibration, thresholds and provenance are preserved. Public adapters do not replace
those records. Aggregate results included here are the specifically supplied Study B
summary and independently verified published AIME summary.

The repository owner confirmed the required clearance and authorized publication
of the curated candidate. Software licensing and historical remediation remain
separate decisions. Permission is not inferred from GitHub repository ownership.
