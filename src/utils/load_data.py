"""
Data loading pipeline: Extract data from Excel files and load into SQLite database.
"""

import pandas as pd
import sqlite3
from pathlib import Path
import logging
from database import initialize_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "HRP"


def load_measurements_data(excel_path: Path) -> pd.DataFrame:
    """
    Load all measurement sheets from Excel file and normalize them.
    
    File structure for each sheet:
    - Column 0: seniorID
    - Column 1: value (measurement value)
    - Column 2: sbp (systolic blood pressure - may be NULL for non-BP)
    - Column 3: dbp (diastolic blood pressure - may be NULL for non-BP)
    - Column 4: date (timestamp)
    - Column 5: type (parameter type e.g., Steps, Heart Rate)
    
    Args:
        excel_path: Path to data_202511181045.xlsx
        
    Returns:
        DataFrame with columns: senior_id, value, sbp, dbp, date, type
    """
    logger.info(f"Loading measurements from {excel_path}")
    
    # Get all sheet names
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
    
    all_measurements = []
    
    for sheet_name in sheet_names:
        # Skip non-data sheets if needed
        if sheet_name.startswith("_"):
            continue
            
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="pyarrow")
        logger.info(f"  {sheet_name}: {len(df)} rows, {df.shape[1]} columns")
        
        # Extract columns: seniorID, value, sbp, dbp, date, type
        df_normalized = pd.DataFrame({
            "senior_id": df.iloc[:, 0],
            "value": df.iloc[:, 1],
            "sbp": df.iloc[:, 2],
            "dbp": df.iloc[:, 3],
            "date": df.iloc[:, 4],
            "type": df.iloc[:, 5]
        })
        
        # Convert data types
        df_normalized["senior_id"] = pd.to_numeric(df_normalized["senior_id"], errors="coerce")
        df_normalized["value"] = pd.to_numeric(df_normalized["value"], errors="coerce")
        df_normalized["sbp"] = pd.to_numeric(df_normalized["sbp"], errors="coerce")
        df_normalized["dbp"] = pd.to_numeric(df_normalized["dbp"], errors="coerce")
        df_normalized["type"] = df_normalized["type"].astype(str).str.strip()
        
        # Remove rows with NULL senior_id
        df_normalized = df_normalized.dropna(subset=["senior_id"])
        
        all_measurements.append(df_normalized)
    
    df_all = pd.concat(all_measurements, ignore_index=True)
    logger.info(f"Total measurements loaded: {len(df_all)}")
    return df_all


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
    # Initialize database
    db_path = Path(__file__).parent.parent / "db" / "hrp_data.db"
    if fresh_start and db_path.exists():
        db_path.unlink()
        logger.info("Deleted existing database")
    
    conn = initialize_database(db_path)
    
    try:
        # Load measurements (largest dataset)
        measurements = load_measurements_data(data_dir / "data_202511181045.xlsx")
        insert_measurements(measurements, conn)
        
        # Load medical info
        medical_info = load_medical_info(data_dir / "Med&Dis_202511181011.xlsx")
        insert_medical_info(medical_info, conn)
        
        # Load alerts
        alerts = load_alerts(data_dir / "SOS_202511181012.xlsx")
        insert_alerts(alerts, conn)
        
        logger.info("\n✓✓✓ All data loaded successfully! ✓✓✓")
        
        # Print summary statistics
        print_summary_stats(conn)
        
    finally:
        conn.close()


def print_summary_stats(conn: sqlite3.Connection):
    """Print database summary statistics."""
    cursor = conn.cursor()
    
    stats = {
        "seniors": cursor.execute("SELECT COUNT(*) FROM seniors").fetchone()[0],
        "measurements": cursor.execute("SELECT COUNT(*) FROM measurements").fetchone()[0],
        "medical_records": cursor.execute("SELECT COUNT(*) FROM medical_info").fetchone()[0],
        "alerts": cursor.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
    }
    
    print("\n" + "="*50)
    print("DATABASE SUMMARY")
    print("="*50)
    for key, value in stats.items():
        print(f"{key:.<30} {value:>15,}")
    print("="*50)


if __name__ == "__main__":
    load_all_data(fresh_start=True)
