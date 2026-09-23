"""Read-only aggregate evidence audit. Outputs are PRIVATE, never commit them."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3
import statistics


def audit(path, n_patients=40, max_rows=20000, seed=42):
    connection = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    ids = [r[0] for r in connection.execute("SELECT id FROM seniors ORDER BY id")]
    selected = random.Random(seed).sample(ids, min(n_patients, len(ids)))
    counts, ratios, per_day, crossday_last_ratios = Counter(), [], [], []
    try:
        tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        schema = {t: [r[1] for r in connection.execute('PRAGMA table_info("'+t.replace('"','""')+'")')] for t in tables if not t.startswith("sqlite_")}
        for sid in selected:
            rows = connection.execute("SELECT date,value FROM measurements WHERE senior_id=? AND type='Steps' ORDER BY date LIMIT ?", (sid, max_rows)).fetchall()
            if not rows:
                continue
            counts["patients_with_steps"] += 1
            counts["rows"] += len(rows)
            counts["patients_reaching_row_cap"] += len(rows) == max_rows
            days = defaultdict(list)
            for date, value in rows:
                if value is None:
                    counts["nulls"] += 1
                    continue
                days[str(date)[:10]].append((str(date), float(value)))
                counts["zeros"] += value == 0
                counts["negatives"] += value < 0
            ordered = sorted(days.items())
            for i, (day, items) in enumerate(ordered):
                values = [v for _, v in items]
                if i:
                    previous = ordered[i-1][1][-1][1]
                    first = values[0]
                    counts["cross_day_pairs"] += 1
                    counts["cross_day_decreases"] += first < previous
                    if previous > 0:
                        crossday_last_ratios.append(first/previous)
                if len(items) < 2:
                    continue
                delta = [b-a for a,b in zip(values, values[1:])]
                counts["days_with_multiple_readings"] += 1
                counts["within_day_transitions"] += len(delta)
                counts["equal"] += sum(x == 0 for x in delta)
                counts["increase"] += sum(x > 0 for x in delta)
                counts["decrease"] += sum(x < 0 for x in delta)
                counts["nondecreasing_days"] += all(x >= 0 for x in delta)
                counts["duplicate_timestamps"] += len(items)-len({d for d, _ in items})
                if max(values) > 0:
                    ratios.append(sum(values)/max(values))
                per_day.append(len(values))
    finally:
        connection.close()
    return {"sampling": {"patient_count": len(selected), "seed": seed, "per_patient_row_cap": max_rows, "design": "uniform patients; first step rows by time; not uniform patient-days"},
            "counts": dict(counts), "median_daily_sum_over_max": statistics.median(ratios) if ratios else None,
            "median_records_per_day": statistics.median(per_day) if per_day else None,
            "median_next_day_first_over_previous_last": statistics.median(crossday_last_ratios) if crossday_last_ratios else None,
            "source_schema": schema,
            "limits": "Patterns can support cumulative-counter interpretation, but do not prove device units, reset rules, delivery times, enrollment, outcome ascertainment or baseline-history availability."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--patients", type=int, default=40)
    args = p.parse_args()
    result = audit(args.raw_db, args.patients)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: result[k] for k in ("sampling", "counts", "median_daily_sum_over_max", "median_next_day_first_over_previous_last")}, indent=2))


if __name__ == "__main__":
    main()
