"""
Data loading pipeline: Extract data from Excel files and load into SQLite database.
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging
from typing import Iterable, List
from database import initialize_database, normalize_from_sources, bulk_upsert_seniors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "HRP_new"


def _load_single_measurement_file(excel_path: Path) -> pd.DataFrame:
    """Load and normalize one measurement Excel file (multi-sheet)."""
    logger.info(f"Loading measurements from {excel_path}")

    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")

    all_measurements = []

    for sheet_name in sheet_names:
        if sheet_name.startswith("_"):
            continue

        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="pyarrow")
        logger.info(f"  {sheet_name}: {len(df)} rows, {df.shape[1]} columns")

        df_normalized = pd.DataFrame({
            "senior_id": df.iloc[:, 0],
            "value": df.iloc[:, 1],
            "sbp": df.iloc[:, 2],
            "dbp": df.iloc[:, 3],
            "date": df.iloc[:, 4],
            "type": df.iloc[:, 5]
        })

        # Type conversions
        df_normalized["senior_id"] = pd.to_numeric(df_normalized["senior_id"], errors="coerce")
        df_normalized["value"] = pd.to_numeric(df_normalized["value"], errors="coerce")
        df_normalized["sbp"] = pd.to_numeric(df_normalized["sbp"], errors="coerce")
        df_normalized["dbp"] = pd.to_numeric(df_normalized["dbp"], errors="coerce")
        df_normalized["type"] = df_normalized["type"].astype(str).str.strip()

        # Drop rows missing senior_id
        df_normalized = df_normalized.dropna(subset=["senior_id"])

        all_measurements.append(df_normalized)

    return pd.concat(all_measurements, ignore_index=True)


def load_measurements_data(excel_path: Path | List[Path]) -> pd.DataFrame:
    """Load measurement data from one or many Excel files and normalize."""
    paths: List[Path] = [excel_path] if isinstance(excel_path, Path) else list(excel_path)
    frames = []
    for path in paths:
        frames.append(_load_single_measurement_file(path))
    df_all = pd.concat(frames, ignore_index=True)
    logger.info(f"Total measurements loaded across files: {len(df_all)}")
    return df_all


def load_seniors_demographics(excel_path: Path) -> pd.DataFrame:
    """Load seniors demographics (gender, birthdate, age)."""
    logger.info(f"Loading seniors demographics from {excel_path}")
    df = pd.read_excel(excel_path, engine="pyarrow")
    df = df.rename(columns={
        "seniorID": "senior_id",
        "gender": "gender",
        "birthDate": "birthdate",
        "age": "age"
    })
    if "senior_id" not in df.columns:
        raise ValueError("seniors demographics file must contain 'seniorID' column")

    df["senior_id"] = pd.to_numeric(df["senior_id"], errors="coerce")
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    if "birthdate" in df.columns:
        df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce").dt.date

    df = df.dropna(subset=["senior_id"])
    logger.info(f"Loaded {len(df)} senior demographic rows")
    return df[["senior_id", "gender", "birthdate", "age"]]


def insert_seniors_demographics(df: pd.DataFrame, conn: sqlite3.Connection):
    """Upsert seniors with demographic info."""
    if df.empty:
        logger.info("No senior demographics to insert")
        return

    records = df.to_dict(orient="records")
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO seniors (id, gender, birthdate, age)
            VALUES (:senior_id, :gender, :birthdate, :age)
            """,
            records,
        )
    logger.info(f"Upserted {len(df)} seniors with demographics")


def load_medical_info(excel_path: Path) -> pd.DataFrame:
    """Load medical information (diseases and medicines)."""
    logger.info(f"Loading medical info from {excel_path}")
    df = pd.read_excel(excel_path, engine="pyarrow")
    
    # Rename columns to match database schema
    df = df.rename(columns={
        'seniorID': 'senior_id',
        'diseaseNames': 'disease_names',
        'medicineNames': 'medicine_names'
    })
    
    # Handle duplicates: keep last entry for each senior_id
    if df['senior_id'].duplicated().any():
        duplicate_count = df['senior_id'].duplicated().sum()
        logger.warning(f"  Found {duplicate_count} duplicate senior_ids - keeping last entry for each")
        df = df.drop_duplicates(subset=['senior_id'], keep='last')
    
    logger.info(f"  Loaded {len(df)} records")
    return df


def load_alerts(excel_path: Path) -> pd.DataFrame:
    """Load SOS alerts data."""
    logger.info(f"Loading alerts from {excel_path}")
    df = pd.read_excel(excel_path, engine="pyarrow")
    
    # Rename columns to match database schema
    df = df.rename(columns={
        'seniorID': 'senior_id',
        'alertDate': 'alert_date',
        'sosNote': 'sos_note'
    })
    
    logger.info(f"  Loaded {len(df)} records")
    return df


def insert_measurements(df: pd.DataFrame, conn: sqlite3.Connection, batch_size: int = 50000):
    """Insert measurements into database with batch processing."""
    logger.info(f"Inserting {len(df)} measurements...")
    
    # Insert measurements in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch.to_sql("measurements", conn, if_exists="append", index=False)
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"    Inserted {i + batch_size:,} measurements...")
    
    conn.commit()
    logger.info("✓ Measurements inserted successfully")


def insert_medical_info(df: pd.DataFrame, conn: sqlite3.Connection):
    """Insert medical information."""
    logger.info(f"Inserting {len(df)} medical records...")
    df.to_sql("medical_info", conn, if_exists="append", index=False)
    conn.commit()
    logger.info("✓ Medical info inserted successfully")


def insert_alerts(df: pd.DataFrame, conn: sqlite3.Connection):
    """Insert alert records."""
    logger.info(f"Inserting {len(df)} alert records...")
    df.to_sql("alerts", conn, if_exists="append", index=False)
    conn.commit()
    logger.info("✓ Alerts inserted successfully")


def load_all_data(data_dir: Path = RAW_DATA_PATH, fresh_start: bool = False):
    """
    Main pipeline: Load all data from Excel files into SQLite.
    
    Args:
        data_dir: Directory containing the Excel files
        fresh_start: If True, delete existing database and start fresh
    """
    measurement_files = [
        data_dir / "data_202512221122-01-09.xlsx",
        data_dir / "data_202512221231-16-23.xlsx",
        data_dir / "data_202512221344-24-30.xlsx",
    ]
    medical_file = data_dir / "Med&Diseases_202512221410.xlsx"
    alerts_file = data_dir / "SOS_202512221411.xlsx"
    seniors_demo_file = data_dir / "SeniorGenderAge_202512221409.xlsx"

    # Initialize database
    db_path = Path(__file__).parent.parent / "db" / "hrp_data.db"
    if fresh_start and db_path.exists():
        db_path.unlink()
        logger.info("Deleted existing database")
    
    conn = initialize_database(db_path)
    
    try:
        # Load seniors demographics first (to satisfy FK during measurements insert)
        if seniors_demo_file.exists():
            seniors_df = load_seniors_demographics(seniors_demo_file)
            insert_seniors_demographics(seniors_df, conn)
        else:
            logger.warning(f"Seniors demographic file not found: {seniors_demo_file}")

        # Load measurements (largest dataset)
        measurements = load_measurements_data(measurement_files)
        # Ensure seniors exist for FK before insert
        bulk_upsert_seniors(conn, measurements["senior_id"].dropna().unique())
        insert_measurements(measurements, conn)
        
        # Load medical info
        medical_info = load_medical_info(medical_file)
        insert_medical_info(medical_info, conn)
        
        # Load alerts
        alerts = load_alerts(alerts_file)
        insert_alerts(alerts, conn)
        
        # Normalize and sync derived tables from raw imports
        normalize_from_sources(conn)
        
        logger.info("\n✓✓✓ All data loaded and normalized successfully! ✓✓✓")
        
        # Print summary statistics
        print_summary_stats(conn)
        
    finally:
        conn.close()


def print_summary_stats(conn: sqlite3.Connection):
    """Print database summary statistics."""
    cursor = conn.cursor()
    
    def _count(table: str) -> int:
        try:
            return int(cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0
    
    stats = {
        "seniors": _count("seniors"),
        "measurements": _count("measurements"),
        "alerts": _count("alerts"),
        "medical_info (raw)": _count("medical_info"),
        "diseases": _count("diseases"),
        "medicines": _count("medicines"),
        "senior_diseases": _count("senior_diseases"),
        "senior_medicines": _count("senior_medicines"),
    }
    
    print("\n" + "="*50)
    print("DATABASE SUMMARY")
    print("="*50)
    for key, value in stats.items():
        print(f"{key:.<30} {value:>15,}")
    print("="*50)


if __name__ == "__main__":
    load_all_data(fresh_start=True)
