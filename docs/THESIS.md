# Thesis: automated processing of remote patient telemetry

**Design and Implementation of an Automated Data Processing Pipeline for Remote Patient Telemetry**

The pipeline is the thesis's central engineering contribution. The downstream models
are consumers that help verify whether processed datasets are usable. AIME and
Studies A/B provide research context and different task requirements; their AUROCs
are not a benchmark of storage or processing architecture.

## Suggested thesis structure and implementation evidence

| Chapter | Questions and repository evidence |
| --- | --- |
| Problem and requirements | Heterogeneous telemetry, missing/late observations, artifact handling, availability and outcome coverage; [input contract](architecture/source-contract.md) |
| Technology analysis | SQLite ingestion history, pandas reference processing, DuckDB analytical SQL and compressed Parquet; compare deployment and workload tradeoffs |
| Architecture | Source adapter, transactional ingestion, quarantine, canonical feature layer, compact datasets and downstream interfaces; [pipeline](PIPELINE.md) |
| Implementation | Paired BP validity, steps semantics, masks/recency, time alignment, subject splits, run manifests and resumable processing |
| Verification | Hand-constructed boundary tests, reference/vector parity, ingestion rollback/replay, integrity failures and full fictional execution |
| Performance evaluation | Workload specification, repeated timings, throughput, output size and memory limitations; distinguish measured observations from intended scalability |
| Research applications | AIME feasibility, repaired continuous windows and SOS-conditioned inputs; [research overview](RESEARCH_OVERVIEW.md) |
| Limitations and development | Private source adapters, retrospective labels, largest-subject memory, single-writer operation, live-stream/distributed extensions and rights boundaries |

DuckDB supports analytical queries and Parquet I/O in the same local process;
Parquet provides a portable columnar artifact for downstream processing. These are
architectural reasons for the choice, not measured proof that one engine wins on
every workload. See the official [DuckDB Parquet documentation](https://duckdb.org/docs/current/data/parquet/overview)
and [configuration documentation](https://duckdb.org/docs/current/configuration/overview).

## Reproducible evaluation plan

1. Define source rates, subject counts, duration, missingness, invalid values,
   duplicate frequency, delayed availability and step semantics before measuring.
2. Use correctness tests before timing. Compare the retained reference and vectorized
   implementations only on contracts both support.
3. Run repeated synthetic processing benchmarks at increasing subject counts and
   durations, with fixed seeds and explicit thread/memory settings.
4. Report individual repeats, environment, warm/cold cache policy, median/spread,
   throughput, generated size and measured memory if instrumented. The current
   report records timing and throughput; it does not measure process peak RSS.
5. Evaluate failure/recovery deliberately: malformed sources, duplicate IDs, interrupted
   ingestion, stale artifacts, changed input hashes and simultaneous-writer attempts.
6. If institution-authorized real-data processing benchmarks are included in the thesis,
   disclose only approved aggregates and retain their private provenance separately.

```shell
python -m ai_cvd.pipeline benchmark --subjects 2 4 --days 2 --repeats 2 --output outputs/thesis-benchmark
```

The benchmark generates local evidence rather than tracking benchmark dumps in Git.
Larger benchmark sizes can be selected explicitly. Small fictional runs establish
functionality and provide a measurement method; they do not establish hospital-scale
capacity, prospective reliability or clinical benefit.

The [historical mapping](architecture/pipeline-history.md) distinguishes retained
research algorithms, release adapters and frozen internal stages. Cite actual tested
commits in the thesis. The public example configuration is not the frozen scientific
configuration, and the source-neutral adapter is not the institution's proprietary
extraction implementation. Describe both roles accurately.
