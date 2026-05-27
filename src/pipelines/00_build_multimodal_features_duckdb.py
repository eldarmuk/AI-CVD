"""
High-speed DuckDB feature extraction from hrp_processed.db.

This pipeline keeps the expensive work inside DuckDB:
- 5-minute time_bucket() aggregation of measurements
- 10-minute burst-alert compression, keeping only the first alert per burst
- feature typing/downcasting before Parquet materialization
- native ORDER BY senior_id, timestamp for sequential batching
"""

from pathlib import Path
import argparse
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "hrp_processed.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "multimodal_features.parquet"

BUCKET_SIZE = "5 minutes"
BURST_WINDOW = "10 minutes"
LOOKAHEAD_WINDOW = "24 hours"
RECENT_ALERT_WINDOW = "48 hours"

RISK_COLUMNS = [
    "cardiovascular",
    "metabolic_endocrine",
    "neurological",
    "psychiatric_cognitive",
    "musculoskeletal",
    "respiratory",
    "gastro_renal_urologic",
    "oncological",
    "sensory",
    "other_functional_risk",
    "unclassified",
]


def build_feature_sql() -> str:
    risk_selects = ",\n            ".join(
        f"CAST(coalesce(r.{col}, 0) AS TINYINT) AS "
        f"{'other' if col == 'unclassified' else col}"
        for col in RISK_COLUMNS
    )

    return f"""
        WITH typed_measurements AS (
            SELECT
                CAST(senior_id AS INTEGER) AS senior_id,
                CAST(date AS TIMESTAMP) AS meas_ts,
                type,
                CAST(value AS FLOAT) AS value,
                CAST(sbp AS FLOAT) AS sbp,
                CAST(dbp AS FLOAT) AS dbp,
                CAST(pulse_pressure AS FLOAT) AS pulse_pressure
            FROM hrp.measurements
            WHERE date IS NOT NULL
        ),
        measurement_buckets AS (
            SELECT
                senior_id,
                time_bucket(INTERVAL '{BUCKET_SIZE}', meas_ts) AS bucket_ts,
                CAST(avg(CASE WHEN type = 'Temperature' THEN value END) AS FLOAT) AS temperature,
                CAST(avg(CASE WHEN type = 'Heartrate' THEN value END) AS FLOAT) AS heartrate,
                CAST(avg(CASE WHEN type = 'BloodPressure' THEN sbp END) AS FLOAT) AS sbp,
                CAST(avg(CASE WHEN type = 'BloodPressure' THEN dbp END) AS FLOAT) AS dbp,
                CAST(avg(CASE WHEN type = 'BloodPressure' THEN pulse_pressure END) AS FLOAT) AS pulse_pressure,
                CAST(sum(CASE WHEN type = 'Steps' THEN value ELSE 0 END) AS FLOAT) AS steps,
                CAST(avg(CASE WHEN type = 'Saturation' THEN value END) AS FLOAT) AS saturation
            FROM typed_measurements
            GROUP BY senior_id, time_bucket(INTERVAL '{BUCKET_SIZE}', meas_ts)
        ),
        senior_bounds AS (
            SELECT senior_id, min(bucket_ts) AS min_ts, max(bucket_ts) AS max_ts
            FROM measurement_buckets
            GROUP BY senior_id
        ),
        time_grid AS (
            SELECT b.senior_id, gs.bucket_ts
            FROM senior_bounds b,
            generate_series(b.min_ts, b.max_ts, INTERVAL '{BUCKET_SIZE}') AS gs(bucket_ts)
        ),
        alert_ordered AS (
            SELECT
                CAST(alert_id AS INTEGER) AS alert_id,
                CAST(senior_id AS INTEGER) AS senior_id,
                CAST(alert_date AS TIMESTAMP) AS alert_ts,
                CAST(severity AS TINYINT) AS severity,
                lag(CAST(alert_date AS TIMESTAMP)) OVER (
                    PARTITION BY senior_id
                    ORDER BY CAST(alert_date AS TIMESTAMP), alert_id
                ) AS prev_alert_ts
            FROM hrp.alerts
            WHERE alert_date IS NOT NULL
        ),
        compressed_alerts AS (
            SELECT alert_id, senior_id, alert_ts, severity
            FROM alert_ordered
            WHERE prev_alert_ts IS NULL
               OR alert_ts > prev_alert_ts + INTERVAL '{BURST_WINDOW}'
        ),
        base_features AS (
            SELECT
                g.senior_id,
                g.bucket_ts AS timestamp,
                mb.temperature,
                mb.heartrate,
                mb.sbp,
                mb.dbp,
                mb.pulse_pressure,
                coalesce(mb.steps, 0)::FLOAT AS steps,
                mb.saturation,
                CAST(stddev_samp(mb.heartrate) OVER (
                    PARTITION BY g.senior_id
                    ORDER BY g.bucket_ts
                    RANGE BETWEEN INTERVAL '4 hours' PRECEDING AND CURRENT ROW
                ) AS FLOAT) AS hr_volatility,
                CAST(regr_slope(
                    mb.sbp,
                    epoch(g.bucket_ts)
                ) OVER (
                    PARTITION BY g.senior_id
                    ORDER BY g.bucket_ts
                    RANGE BETWEEN INTERVAL '3 hours' PRECEDING AND CURRENT ROW
                ) AS FLOAT) AS bp_trend,
                CAST(extract(hour FROM g.bucket_ts) AS TINYINT) AS hour,
                CAST(CASE
                    WHEN extract(hour FROM g.bucket_ts) BETWEEN 0 AND 5 THEN 1
                    ELSE 0
                END AS TINYINT) AS is_night,
                CAST(sum(coalesce(mb.steps, 0)) OVER (
                    PARTITION BY g.senior_id
                    ORDER BY g.bucket_ts
                    RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND CURRENT ROW
                ) AS FLOAT) AS steps_rolling_sum_6h,
                CAST(CASE
                    WHEN mb.dbp IS NOT NULL AND mb.dbp <> 0 AND mb.heartrate IS NOT NULL
                    THEN mb.heartrate / mb.dbp
                    ELSE NULL
                END AS FLOAT) AS shock_index,
                CAST(extract(dow FROM g.bucket_ts) AS TINYINT) AS day_of_week
            FROM time_grid g
            LEFT JOIN measurement_buckets mb
              ON g.senior_id = mb.senior_id
             AND g.bucket_ts = mb.bucket_ts
        ),
        alert_features AS (
            SELECT
                bf.senior_id,
                bf.timestamp,
                CAST(count(ca.alert_id) FILTER (
                    WHERE ca.severity IN (1, 2, 3)
                      AND ca.alert_ts >= bf.timestamp - INTERVAL '{RECENT_ALERT_WINDOW}'
                      AND ca.alert_ts < bf.timestamp
                ) AS FLOAT) AS recent_event_burden,
                CAST(max(CASE
                    WHEN ca.severity = 1
                     AND ca.alert_ts > bf.timestamp
                     AND ca.alert_ts <= bf.timestamp + INTERVAL '{LOOKAHEAD_WINDOW}'
                    THEN 1 ELSE 0
                END) AS TINYINT) AS label_1,
                CAST(max(CASE
                    WHEN ca.severity = 2
                     AND ca.alert_ts > bf.timestamp
                     AND ca.alert_ts <= bf.timestamp + INTERVAL '{LOOKAHEAD_WINDOW}'
                    THEN 1 ELSE 0
                END) AS TINYINT) AS label_2,
                CAST(max(CASE
                    WHEN ca.severity = 3
                     AND ca.alert_ts > bf.timestamp
                     AND ca.alert_ts <= bf.timestamp + INTERVAL '{LOOKAHEAD_WINDOW}'
                    THEN 1 ELSE 0
                END) AS TINYINT) AS label_3
            FROM base_features bf
            LEFT JOIN compressed_alerts ca
              ON ca.senior_id = bf.senior_id
             AND ca.alert_ts >= bf.timestamp - INTERVAL '{RECENT_ALERT_WINDOW}'
             AND ca.alert_ts <= bf.timestamp + INTERVAL '{LOOKAHEAD_WINDOW}'
            GROUP BY bf.senior_id, bf.timestamp
        )
        SELECT
            bf.timestamp,
            bf.temperature,
            bf.heartrate,
            bf.sbp,
            bf.dbp,
            bf.pulse_pressure,
            bf.steps,
            bf.saturation,
            CAST(bf.senior_id AS INTEGER) AS senior_id,
            bf.hr_volatility,
            bf.bp_trend,
            bf.hour,
            bf.is_night,
            bf.steps_rolling_sum_6h,
            {risk_selects},
            CAST(s.age AS SMALLINT) AS age,
            CAST(CASE
                WHEN lower(s.gender) LIKE 'k%' OR lower(s.gender) LIKE 'f%' THEN 1
                WHEN lower(s.gender) LIKE 'm%' THEN 0
                ELSE NULL
            END AS TINYINT) AS gender,
            bf.shock_index,
            bf.day_of_week,
            af.recent_event_burden,
            af.label_1,
            af.label_2,
            af.label_3
        FROM base_features bf
        LEFT JOIN alert_features af
          ON bf.senior_id = af.senior_id
         AND bf.timestamp = af.timestamp
        LEFT JOIN hrp.senior_risk_profiles r
          ON bf.senior_id = r.senior_id
        LEFT JOIN hrp.seniors s
          ON bf.senior_id = s.id
        ORDER BY bf.senior_id, bf.timestamp
    """


def export_multimodal_features(db_path: Path, output_path: Path, threads: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET threads={threads};")
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS hrp (TYPE sqlite, READ_ONLY true);")

    feature_sql = build_feature_sql()
    copy_sql = f"""
        COPY (
            {feature_sql}
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
    """

    print(f"Exporting 5-minute multimodal features from {db_path}")
    print(f"Output: {output_path}")
    con.execute(copy_sql)
    con.close()
    print("DuckDB Parquet export complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sorted, typed multimodal_features.parquet with DuckDB pushdown."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_multimodal_features(args.db, args.output, args.threads)


if __name__ == "__main__":
    main()
