"""
Database initialization and schema setup for HRP health monitoring data.
Creates SQLite database with optimized schema for time-series health measurements.
"""

import sqlite3
from typing import Iterable
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
    # Ensure foreign keys are enforced
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
    -- Core seniors table (central entity)
    CREATE TABLE IF NOT EXISTS seniors (
        id INTEGER PRIMARY KEY,
        gender TEXT,
        birthdate DATE,
        age INTEGER
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_seniors_id ON seniors(id);
    
    -- Measurements table (fact table - largest)
    -- Each row represents a single health measurement for a senior
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        senior_id INTEGER NOT NULL,
        value REAL,
        sbp REAL,
        dbp REAL,
        date TIMESTAMP NOT NULL,
        type TEXT NOT NULL,
        FOREIGN KEY (senior_id) REFERENCES seniors(id)
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
    
    -- Medical information (raw import; denormalized source)
    -- disease_names and medicine_names are comma-separated lists
    CREATE TABLE IF NOT EXISTS medical_info (
        senior_id INTEGER PRIMARY KEY,
        disease_names TEXT,
        medicine_names TEXT
    );
    
    -- Normalized reference tables
    CREATE TABLE IF NOT EXISTS diseases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_name TEXT NOT NULL
    );
    -- Guarantee case-insensitive uniqueness for names
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_disease_name 
        ON diseases(disease_name COLLATE NOCASE);
    
    CREATE TABLE IF NOT EXISTS medicines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        medicine_name TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_medicine_name 
        ON medicines(medicine_name COLLATE NOCASE);
    
    -- Link tables (many-to-many)
    CREATE TABLE IF NOT EXISTS senior_diseases (
        senior_id INTEGER NOT NULL,
        disease_id INTEGER NOT NULL,
        PRIMARY KEY (senior_id, disease_id),
        FOREIGN KEY (senior_id) REFERENCES seniors(id),
        FOREIGN KEY (disease_id) REFERENCES diseases(id)
    );
    CREATE INDEX IF NOT EXISTS idx_sd_senior ON senior_diseases(senior_id);
    CREATE INDEX IF NOT EXISTS idx_sd_disease ON senior_diseases(disease_id);
    
    CREATE TABLE IF NOT EXISTS senior_medicines (
        senior_id INTEGER NOT NULL,
        medicine_id INTEGER NOT NULL,
        PRIMARY KEY (senior_id, medicine_id),
        FOREIGN KEY (senior_id) REFERENCES seniors(id),
        FOREIGN KEY (medicine_id) REFERENCES medicines(id)
    );
    CREATE INDEX IF NOT EXISTS idx_sm_senior ON senior_medicines(senior_id);
    CREATE INDEX IF NOT EXISTS idx_sm_medicine ON senior_medicines(medicine_id);
    
    -- Alerts/SOS table
    -- Records of alerts triggered for seniors
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        senior_id INTEGER NOT NULL,
        alert_date TIMESTAMP NOT NULL,
        sos_note TEXT,
        FOREIGN KEY (senior_id) REFERENCES seniors(id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_alerts_senior_date 
        ON alerts(senior_id, alert_date);
    CREATE INDEX IF NOT EXISTS idx_alerts_alert_date 
        ON alerts(alert_date);
    """)
    
    _ensure_seniors_demographics_columns(conn)
    # Create index for gender (safe if column already existed)
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_seniors_gender ON seniors(gender);")
    except sqlite3.OperationalError as exc:
        logger.warning(f"Skipping gender index creation: {exc}")

    conn.commit()
    
    # Backfill and migrate from existing data into normalized tables
    try:
        _ensure_seniors_populated(conn)
        _normalize_medical_info(conn)
    except Exception as e:
        logger.warning(f"Normalization step skipped/failed: {e}")
    
    logger.info(f"Database initialized at {db_path}")
    return conn


def _ensure_seniors_demographics_columns(conn: sqlite3.Connection) -> None:
    """Ensure seniors table has demographic columns even on existing databases."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(seniors)")
    existing_cols = {row[1] for row in cur.fetchall()}

    additions = [
        ("gender", "ALTER TABLE seniors ADD COLUMN gender TEXT"),
        ("birthdate", "ALTER TABLE seniors ADD COLUMN birthdate DATE"),
        ("age", "ALTER TABLE seniors ADD COLUMN age INTEGER"),
    ]

    for col_name, ddl in additions:
        if col_name not in existing_cols:
            cur.execute(ddl)


def _ensure_seniors_populated(conn: sqlite3.Connection) -> None:
    """Ensure all known senior IDs exist in seniors table.
    Collects IDs from measurements, alerts, and medical_info.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    # Collect distinct ids from existing tables (if they exist)
    sources = [
        ("measurements", "senior_id"),
        ("alerts", "senior_id"),
        ("medical_info", "senior_id"),
    ]
    senior_ids = set()
    for table, col in sources:
        try:
            cur.execute(f"SELECT DISTINCT {col} FROM {table}")
            senior_ids.update(r[0] for r in cur.fetchall() if r[0] is not None)
        except sqlite3.OperationalError:
            # Table may not exist yet; skip
            continue
    if not senior_ids:
        return
    cur.executemany(
        "INSERT OR IGNORE INTO seniors(id) VALUES (?)",
        [(int(sid),) for sid in senior_ids]
    )
    conn.commit()


def _normalize_medical_info(conn: sqlite3.Connection) -> None:
    """Populate diseases/medicines and link tables from medical_info.
    Assumes medical_info has comma-separated names per the spec.
    Idempotent: safe to run multiple times.
    """
    cur = conn.cursor()
    # Fetch all medical_info rows
    try:
        rows = cur.execute(
            "SELECT senior_id, disease_names, medicine_names FROM medical_info"
        ).fetchall()
    except sqlite3.OperationalError:
        return
    if not rows:
        return

    def _split_names(raw: str) -> list[str]:
        if raw is None:
            return []
        # Split by comma, strip whitespace, drop empties
        parts = [p.strip() for p in str(raw).split(',')]
        return [p for p in parts if p]

    # Insert or get id helpers
    def _get_or_create(table: str, col: str, value: str) -> int:
        cur.execute(f"SELECT id FROM {table} WHERE {col} = ? COLLATE NOCASE", (value,))
        r = cur.fetchone()
        if r:
            return int(r[0])
        cur.execute(f"INSERT OR IGNORE INTO {table}({col}) VALUES (?)", (value,))
        # Now retrieve id
        cur.execute(f"SELECT id FROM {table} WHERE {col} = ? COLLATE NOCASE", (value,))
        rid = cur.fetchone()
        return int(rid[0])

    # Ensure senior ids exist
    cur.executemany(
        "INSERT OR IGNORE INTO seniors(id) VALUES (?)",
        [(int(r[0]),) for r in rows if r[0] is not None]
    )

    # Populate normalized tables and links
    for senior_id, disease_names, medicine_names in rows:
        if senior_id is None:
            continue
        sid = int(senior_id)
        for d in _split_names(disease_names):
            did = _get_or_create("diseases", "disease_name", d)
            cur.execute(
                "INSERT OR IGNORE INTO senior_diseases(senior_id, disease_id) VALUES (?, ?)",
                (sid, did)
            )
        for m in _split_names(medicine_names):
            mid = _get_or_create("medicines", "medicine_name", m)
            cur.execute(
                "INSERT OR IGNORE INTO senior_medicines(senior_id, medicine_id) VALUES (?, ?)",
                (sid, mid)
            )
    conn.commit()


def normalize_from_sources(conn: sqlite3.Connection) -> None:
    """Public: ensure seniors table is populated and normalize medical_info.
    Safe to call after loading new data.
    """
    _ensure_seniors_populated(conn)
    _normalize_medical_info(conn)


def bulk_upsert_seniors(conn: sqlite3.Connection, senior_ids: Iterable[int]) -> int:
    """Insert senior ids into `seniors` if missing.
    Returns number of rows affected (inserted or ignored by SQLite reports 0 for ignore).
    """
    ids = [int(s) for s in senior_ids if s is not None]
    if not ids:
        return 0
    cur = conn.cursor()
    cur.executemany("INSERT OR IGNORE INTO seniors(id) VALUES (?)", [(sid,) for sid in ids])
    conn.commit()
    # Changes returns number of rows modified by the last operation in this connection
    return cur.rowcount


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get connection to existing database."""
    return sqlite3.connect(db_path)


if __name__ == "__main__":
    initialize_database()
    logger.info("Database schema created successfully!")
