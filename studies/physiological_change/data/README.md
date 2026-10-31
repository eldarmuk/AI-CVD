# Local datasets

Only this README is tracked here. Keep downloaded data, identifiers, detailed
reference traces and private split rosters local. Suggested local layout:

```text
data/
  sources/<dataset>/<version>/     # immutable originals
  canonical/<adapter-version>/    # reproducible derived records
  registries/                    # source hashes, splits and access/exposure logs
```

Review the provider's current access and reuse terms before download. Publicly
accessible human data are not automatically redistributable. Register version,
source URL, license, reference method, known cohort overlap and file hashes.
Never join or upload patient identifiers. Keep old telecare files at their
existing approved local location and access them read-only if specifically needed.
