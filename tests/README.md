# Public test scope

- test_data.py: deterministic fictional generation, grouping, timezone normalization,
  duplicate rejection, subject isolation, causal boundaries, elapsed time and fitting scope.
- test_models.py: package imports, retained model forward interfaces, missing-placeholder
  invariance, GRU-D decay/cache equations, empty attention and Study A masking.
- test_evaluation.py: known metrics, undefined cases, input validation and whole-tie thresholds.
- test_demo.py: complete saved-output workflow, repeatability and installed CLI outside checkout.
- test_release_checks.py: release-gate regression cases.

These tests read no private records, original scientific configurations or fitted artifacts.
Legacy shape checks are not comprehensive validation of every historical experiment.
