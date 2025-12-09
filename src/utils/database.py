"""
Database initialization and schema setup for HRP health monitoring data.
Creates SQLite database with optimized schema for time-series health measurements.
"""

import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "db" / "hrp_data.db"


def initialize_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Initialize SQLite database with optimized schema for health data.
    
    Args:
        db_path: Path to store the SQLite database file
        
    Returns:
        sqlite3.Connection object
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
    -- Measurements table (fact table - largest)
    -- Each row represents a single health measurement for a senior
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        senior_id INTEGER NOT NULL,
        value REAL,
        sbp REAL,
        dbp REAL,
        date TIMESTAMP NOT NULL,
        type TEXT NOT NULL
    );
    
    -- Create indices for faster queries (critical for time-series analysis)
    CREATE INDEX IF NOT EXISTS idx_measurements_senior_id 
        ON measurements(senior_id);
    CREATE INDEX IF NOT EXISTS idx_measurements_date 
        ON measurements(date);
    CREATE INDEX IF NOT EXISTS idx_measurements_type 
        ON measurements(type);
    CREATE INDEX IF NOT EXISTS idx_measurements_senior_date 
        ON measurements(senior_id, date);
    CREATE INDEX IF NOT EXISTS idx_measurements_type_date 
        ON measurements(type, date);
    
    -- Medical information (dimension)
    -- Maps seniors to their diseases and medications
    CREATE TABLE IF NOT EXISTS medical_info (
        senior_id INTEGER PRIMARY KEY,
        disease_names TEXT,
        medicine_names TEXT
    );
    
    -- Alerts/SOS table
    -- Records of alerts triggered for seniors
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        senior_id INTEGER NOT NULL,
        alert_date TIMESTAMP NOT NULL,
        sos_note TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_alerts_senior_date 
        ON alerts(senior_id, alert_date);
    CREATE INDEX IF NOT EXISTS idx_alerts_alert_date 
        ON alerts(alert_date);
    """)
    
    conn.commit()
    logger.info(f"Database initialized at {db_path}")
    return conn


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get connection to existing database."""
    return sqlite3.connect(db_path)


if __name__ == "__main__":
    initialize_database()
    logger.info("Database schema created successfully!")
