"""
Filter seniors to a high-density "Elite" measurement cohort.

The active timeline for each senior is measured from first to last measurement.
Seniors are retained when their average daily measurement density reaches both:
- at least 4 Heartrate readings per active day
- at least 2 BloodPressure readings per active day
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "hrp_processed.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gold_seniors" / "elite_cohort_ids.csv"

MIN_HEARTRATE_PER_DAY = 4.0
MIN_BLOODPRESSURE_PER_DAY = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_elite_cohort(
    db_path: Path,
    min_heartrate_per_day: float,
    min_bloodpressure_per_day: float,
) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Processed database not found: {db_path}")

    query = """
        WITH senior_counts AS (
            SELECT
                CAST(senior_id AS INTEGER) AS senior_id,
                MIN(date) AS first_measurement_at,
                MAX(date) AS last_measurement_at,
                SUM(CASE WHEN type = 'Heartrate' THEN 1 ELSE 0 END) AS heartrate_readings,
                SUM(CASE WHEN type = 'BloodPressure' THEN 1 ELSE 0 END) AS bloodpressure_readings
            FROM hrp.measurements
            WHERE date IS NOT NULL
            GROUP BY CAST(senior_id AS INTEGER)
        ),
        senior_density AS (
            SELECT
                senior_id,
                first_measurement_at,
                last_measurement_at,
                GREATEST(
                    date_diff(
                        'second',
                        CAST(first_measurement_at AS TIMESTAMP),
                        CAST(last_measurement_at AS TIMESTAMP)
                    ) / 86400.0 + 1.0,
                    1.0
                ) AS active_days,
                heartrate_readings,
                bloodpressure_readings
            FROM senior_counts
        )
        SELECT
            senior_id,
            first_measurement_at,
            last_measurement_at,
            active_days,
            heartrate_readings,
            bloodpressure_readings,
            heartrate_readings / active_days AS heartrate_per_day,
            bloodpressure_readings / active_days AS bloodpressure_per_day
        FROM senior_density
        ORDER BY senior_id
    """
    con = duckdb.connect()
    try:
        con.execute("SET preserve_insertion_order=false;")
        con.execute("INSTALL sqlite;")
        con.execute("LOAD sqlite;")
        con.execute(f"ATTACH '{db_path}' AS hrp (TYPE sqlite, READ_ONLY true);")
        density = con.execute(query).fetchdf()
    finally:
        con.close()

    elite = density[
        (density["heartrate_per_day"] >= min_heartrate_per_day)
        & (density["bloodpressure_per_day"] >= min_bloodpressure_per_day)
    ].copy()
    return density, elite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-density elite senior cohort.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--min-heartrate-per-day", type=float, default=MIN_HEARTRATE_PER_DAY)
    parser.add_argument("--min-bloodpressure-per-day", type=float, default=MIN_BLOODPRESSURE_PER_DAY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    density, elite = compute_elite_cohort(
        args.db,
        args.min_heartrate_per_day,
        args.min_bloodpressure_per_day,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    elite.to_csv(args.output, index=False)

    retained = len(elite)
    total = len(density)
    logger.info(
        "Retained %s out of %s seniors (%.2f%%) with >= %.1f HR/day and >= %.1f BP/day.",
        f"{retained:,}",
        f"{total:,}",
        retained / total * 100 if total else 0.0,
        args.min_heartrate_per_day,
        args.min_bloodpressure_per_day,
    )
    logger.info("Saved elite cohort IDs and density audit columns to %s", args.output)


if __name__ == "__main__":
    main()
