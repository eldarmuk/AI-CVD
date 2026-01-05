#!/usr/bin/env python
# coding: utf-8

"""
Memory-Efficient Data Processing Pipeline for HRP Database

This script transforms the raw HRP database (db/hrp_data.db) into a cleaned
and processed database (data/processed/hrp_processed.db) suitable for ML modeling.

Key Features:
- Chunked processing for large measurements table
- Advanced deduplication with conflict resolution
- Medical outlier clipping
- Temporal alert burst merging
- Disease-to-clinical domain mapping
- Optimized with SQL indexes
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENT_YEAR = 2026
CHUNK_SIZE = 5_000_000  # Process measurements in batches
BURST_WINDOW_MINUTES = 10  # Merge alerts within this window

# Outlier clipping ranges
OUTLIER_RANGES = {
    'Heartrate': (30, 220),
    'SBP': (70, 250),
    'DBP': (40, 140),
    'Temperature': (30, 45),
    'Saturation': (50, 100),
}

# Disease domain mapping (from 02_comprehensive_eda.ipynb Section 10.2)
DISEASE_CATEGORIES = {
    'cardiovascular': [
        'serc', 'nadciśnien', 'ciśnien', 'arytmi', 'migotani',
        'trzepotani', 'niewydolność serca', 'zawał',
        'wieńc', 'miażdżyc', 'aort', 'tętniak',
        'dusznica', 'zakrzep', 'zator', 'płucn',
        'kołatanie', 'hipotensj'
    ],
    'metabolic_endocrine': [
        'cukrzyc', 'insulin', 'lipid', 'cholesterol',
        'hiperlipid', 'otyło', 'nadwag',
        'tarczyc', 'hashimoto', 'metabolic',
        'endokryn', 'hipoglik', 'stan przedcukrzyc'
    ],
    'neurological': [
        'udar', 'neurolog', 'padaczk',
        'parkinson', 'stwardnieni',
        'neuropat', 'polineurop',
        'rwa', 'zawrot', 'omdleni'
    ],
    'psychiatric_cognitive': [
        'depresj', 'nerwic', 'lękow',
        'schizofren', 'psychicz',
        'demencj', 'otępieni',
        'alzheimer', 'zanik pamięci',
        'bezsenność'
    ],
    'musculoskeletal': [
        'staw', 'kostn', 'reumaty',
        'zwyrodnieni', 'kręgosłup',
        'dyskopati', 'skolioz',
        'osteoporoz', 'osteopeni',
        'endoprotez', 'lasce', 'chodzeni'
    ],
    'respiratory': [
        'astm', 'pochp', 'rozedm',
        'oddech', 'bezdech',
        'duszno', 'płuc'
    ],
    'gastro_renal_urologic': [
        'żołądk', 'jelit', 'refluks',
        'wrzod', 'uchyłk', 'przepuklin',
        'wątroba', 'trzustk', 'żółci',
        'nerk', 'kamica', 'mocz',
        'prostat'
    ],
    'oncological': [
        'nowotwór', 'rak',
        'onkolog', 'guzy',
        'mastektomia'
    ],
    'sensory': [
        'wzrok', 'widzeni', 'zaćm',
        'jaskr', 'plamki',
        'słuch', 'niedosłuch',
        'głuchot', 'szumy uszne'
    ],
    'other_functional_risk': [
        'niepełnosprawność', 'porusz',
        'chodzeni', 'laska',
        'nikotyn', 'palenie',
        'autoimmun', 'łuszczyc',
        'borelioz', 'anemi',
        'implant', 'stent',
        'rozrusznik', 'bajpas'
    ]
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_severity(note):
    if pd.isna(note):
        return 0

    note_l = str(note).lower()

    if any(k in note_l for k in ['alarm przypadkowy', 'alarm testowy', 'alert techniczny']):
        return 0

    no_zrm = 'brak wskazań do interwencji zrm' in note_l

    if (not no_zrm and (
            'zdecydowano się na interwencję zrm' in note_l or
            'zdecydowano wezwać zrm' in note_l or
            'wezwanie zrm' in note_l or 
            (
                'zagrożenia życia i zdrowia' in note_l and 
                'nawiązano kontakt' in note_l
            ) 
            or
            (
                'na podstawie odczytów z systemu' in note_l and
                'zrm' in note_l
            )
        )
    ):
        return 3

    if 'nie nawiązano kontaktu' in note_l:
        return 2

    if ('nawiązano kontakt' in note_l or'opiekun powiadomiony' in note_l):
        return 1

    return 0


def categorize_disease(disease_name):
    """Map disease name to clinical domain."""
    disease_lower = str(disease_name).lower()
    
    for category, keywords in DISEASE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in disease_lower:
                return category
    
    return 'other'


def round_measurements(df):
    """Round measurement values to appropriate decimal places."""
    # Ensure types are numeric before rounding
    for col in ['value', 'sbp', 'dbp']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Round Value column based on type
    # We use np.where for speed instead of .loc masks in a loop
    is_temp = (df['type'] == 'Temperature')
    
    # Round Temperature to 1 decimal, others to 0
    df['value'] = np.where(is_temp, 
                           np.round(df['value'], 1), 
                           np.round(df['value'], 0))
    
    # Round Blood Pressure to 0 decimals
    if 'sbp' in df.columns:
        df['sbp'] = np.round(df['sbp'], 0)
    if 'dbp' in df.columns:
        df['dbp'] = np.round(df['dbp'], 0)
    
    return df


def clip_outliers(df):
    """Apply outlier clipping to measurement values."""
    for mtype, (min_val, max_val) in OUTLIER_RANGES.items():
        if mtype == 'Heartrate':
            mask = df['type'] == 'Heartrate'
            df.loc[mask, 'value'] = np.clip(df.loc[mask, 'value'], min_val, max_val)
        
        elif mtype == 'Temperature':
            mask = df['type'] == 'Temperature'
            # Values > 45°C set to NaN
            df.loc[mask & (df['value'] > max_val), 'value'] = np.nan
            # Clip lower bound
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lower=min_val)
        
        elif mtype == 'Saturation':
            mask = df['type'] == 'Saturation'
            # Minimum 50%
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lower=min_val)
        
        elif mtype == 'SBP':
            if 'sbp' in df.columns:
                df['sbp'] = np.clip(df['sbp'], min_val, max_val)
        
        elif mtype == 'DBP':
            if 'dbp' in df.columns:
                df['dbp'] = np.clip(df['dbp'], min_val, max_val)
    
    return df


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_seniors(conn_in, conn_out):
    """
    Clean and process seniors table.
    RESUMABLE: Skipped if already processed.
    
    Steps:
    1. Filter age between 40-120
    2. Replace 'Unknown' gender with NaN
    3. Drop seniors with both birthdate AND age as NULL
    4. Calculate missing age or birthdate using 2026 as current year
    """
    print("\n" + "="*80)
    print("PROCESSING SENIORS TABLE (RESUMABLE)")
    print("="*80)
    
    # Check if already completed
    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM seniors").fetchone()[0]
        if count > 0:
            print(f"✓ Seniors table already processed: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM seniors", conn_out)
    except Exception:
        pass
    
    # Load seniors
    df = pd.read_sql_query("SELECT * FROM seniors", conn_in)
    print(f"Initial seniors: {len(df):,}")
    
    # 1. Filter age
    age_mask = (df['age'] >= 40) & (df['age'] <= 120)
    df = df[age_mask]
    print(f"After age filter (40-120): {len(df):,}")
    
    # 2. Replace 'Unknown' gender with NaN
    df.loc[df['gender'] == 'Unknown', 'gender'] = np.nan
    
    # 3. Drop seniors with both birthdate AND age as NULL
    both_null = df['birthdate'].isna() & df['age'].isna()
    df = df[~both_null]
    print(f"After dropping both NULL: {len(df):,}")
    
    # 4. Calculate missing values
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
    
    # Calculate missing age from birthdate
    missing_age = df['age'].isna() & df['birthdate'].notna()
    if missing_age.any():
        df.loc[missing_age, 'age'] = CURRENT_YEAR - df.loc[missing_age, 'birthdate'].dt.year
    
    # Calculate missing birthdate from age
    missing_birthdate = df['birthdate'].isna() & df['age'].notna()
    if missing_birthdate.any():
        birth_years = CURRENT_YEAR - df.loc[missing_birthdate, 'age'].astype(int)
        df.loc[missing_birthdate, 'birthdate'] = pd.to_datetime(
            birth_years.astype(str) + '-01-01'
        )
    
    # Convert birthdate back to ISO string
    df['birthdate'] = df['birthdate'].dt.strftime('%Y-%m-%d')
    
    print(f"Final seniors: {len(df):,}")
    print(f"  - With valid gender: {df['gender'].notna().sum():,}")
    print(f"  - With age: {df['age'].notna().sum():,}")
    print(f"  - With birthdate: {df['birthdate'].notna().sum():,}")
    
    # Write to database
    df.to_sql('seniors', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'seniors', 0, completed=True)
    print("✓ Seniors table written")
    
    return df


def init_checkpoint_table(conn_out):
    """Initialize checkpoint table for resumable processing."""
    cursor = conn_out.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_checkpoints (
            table_name TEXT PRIMARY KEY,
            last_offset INTEGER DEFAULT 0,
            completed BOOLEAN DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn_out.commit()


def get_checkpoint(conn_out, table_name):
    """Retrieve the last processed offset for a table."""
    cursor = conn_out.cursor()
    try:
        result = cursor.execute(
            "SELECT last_offset FROM processing_checkpoints WHERE table_name = ?",
            (table_name,)
        ).fetchone()
        if result:
            offset = result[0]
            # Ensure it's an integer, not bytes
            return int(offset) if offset is not None else 0
        return 0
    except Exception:
        return 0


def update_checkpoint(conn_out, table_name, offset, completed=False):
    """Update the checkpoint for a table."""
    cursor = conn_out.cursor()
    # Ensure offset is an integer
    offset = int(offset) if offset is not None else 0
    cursor.execute("""
        INSERT OR REPLACE INTO processing_checkpoints (table_name, last_offset, completed)
        VALUES (?, ?, ?)
    """, (table_name, offset, 1 if completed else 0))
    conn_out.commit()


def process_measurements_chunked(conn_in, conn_out):
    """
    Process measurements table with chunking for memory efficiency.
    RESUMABLE: Can resume from last checkpoint if interrupted.
    
    Steps:
    1. Sort by senior_id and date
    2. Identify duplicates (same senior_id, date, type)
    3. Resolve conflicts by averaging values
    4. Round values appropriately
    5. Clip outliers
    6. Add pulse_pressure feature
    """
    print("\n" + "="*80)
    print("PROCESSING MEASUREMENTS TABLE (CHUNKED & RESUMABLE)")
    print("="*80)
    # Get total row count
    total_rows = pd.read_sql_query("SELECT COUNT(*) as cnt FROM measurements", conn_in).iloc[0]['cnt']
    print(f"Total measurements: {total_rows:,}")
    
    # Create output table
    cursor_out = conn_out.cursor()
    cursor_out.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            senior_id INTEGER NOT NULL,
            value REAL,
            sbp REAL,
            dbp REAL,
            pulse_pressure REAL,
            date TEXT NOT NULL,
            type TEXT NOT NULL
        )
    """)
    conn_out.commit()
    
    # Check for checkpoint - now stores last_id instead of offset
    last_id = get_checkpoint(conn_out, 'measurements')
    last_id = int(last_id)  # Ensure it's an integer
    
    if last_id > 0:
        print(f"\n⚠ Resuming from checkpoint: last_id {last_id:,}")
        existing_count = cursor_out.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        print(f"  - Already processed: {existing_count:,} rows")
    else:
        print("\n✓ Starting fresh processing")
        existing_count = 0
    
    # Process in chunks
    total_processed = existing_count
    total_written = existing_count
    
    pbar = tqdm(total=total_rows, initial=existing_count, desc="Processing measurements", unit="rows")
    
    while True:  # CHANGE 1: Use infinite loop with break condition
        # CHANGE 2: Use WHERE id > last_id instead of OFFSET
        query = f"""
            SELECT id, senior_id, value, sbp, dbp, date, type
            FROM measurements
            WHERE id > {last_id}
            ORDER BY id
            LIMIT {CHUNK_SIZE}
        """
        chunk = pd.read_sql_query(query, conn_in)
        
        if chunk.empty:
            break
        
        chunk_size = len(chunk)
        total_processed += chunk_size
        
        # CHANGE 3: Update last_id to max id in chunk
        last_id = chunk['id'].max()
        
        # Convert date to datetime (handle microseconds)
        chunk['date'] = pd.to_datetime(chunk['date'], format='ISO8601')
        
        # Identify duplicates
        dup_cols = ['senior_id', 'date', 'type']
        
        # CHANGE 4: Use .agg() but preserve groupby columns
        chunk_dedup = chunk.groupby(dup_cols, as_index=False).agg({
            'value': 'mean',
            'sbp': 'mean',
            'dbp': 'mean',
            'id': 'first'
        })
        
        # Round values
        chunk_dedup = round_measurements(chunk_dedup)
        
        # Clip outliers
        chunk_dedup = clip_outliers(chunk_dedup)
        
        # Add pulse pressure feature
        chunk_dedup['pulse_pressure'] = chunk_dedup['sbp'] - chunk_dedup['dbp']
        
        # Convert date back to ISO string
        chunk_dedup['date'] = chunk_dedup['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Drop id column - let SQLite auto-generate new ids
        chunk_dedup = chunk_dedup.drop(columns=['id'])
        
        # Write to output database
        chunk_dedup.to_sql('measurements', conn_out, if_exists='append', index=False)
        total_written += len(chunk_dedup)
        
        # CHANGE 5: Update checkpoint with last_id
        update_checkpoint(conn_out, 'measurements', last_id)
        
        pbar.update(chunk_size)
    
    pbar.close()
    
    print("\n✓ Measurements processing complete")
    print(f"  - Total processed: {total_processed:,}")
    print(f"  - Total written: {total_written:,}")
    print(f"  - Duplicates removed: {total_processed - total_written:,}")
    
    # Mark as completed
    update_checkpoint(conn_out, 'measurements', last_id, completed=True)
    
    # Create indexes
    print("\nCreating indexes...")
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_measurements_senior ON measurements(senior_id)")
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_measurements_date ON measurements(date)")
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_measurements_senior_date ON measurements(senior_id, date)")
    conn_out.commit()
    print("✓ Indexes created")


def process_alerts(conn_in, conn_out):
    """
    Process alerts with temporal burst merging.
    RESUMABLE: Skipped if already processed.
    
    Steps:
    1. Apply severity classification
    2. Sort by senior_id and alert_date
    3. Merge alerts within 10-minute windows into events
    4. Keep max severity and add event_size feature
    """
    print("\n" + "="*80)
    print("PROCESSING ALERTS TABLE (RESUMABLE)")
    print("="*80)
    
    # Check if already completed
    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        if count > 0:
            print(f"✓ Alerts table already processed: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM alerts", conn_out)
    except Exception:
        pass
    
    # Load alerts
    df = pd.read_sql_query("SELECT * FROM alerts", conn_in)
    print(f"Initial alerts: {len(df):,}")
    
    # Apply severity classification
    df['severity'] = df['sos_note'].apply(classify_severity)
    
    # Convert alert_date to datetime (handle microseconds)
    df['alert_date'] = pd.to_datetime(df['alert_date'], format='ISO8601')
    
    # Sort by senior_id and alert_date
    df = df.sort_values(['senior_id', 'alert_date']).reset_index(drop=True)
    
    # Calculate time difference between consecutive alerts
    df['time_diff'] = df.groupby('senior_id')['alert_date'].diff()
    df['time_diff_minutes'] = df['time_diff'].dt.total_seconds() / 60
    
    # Identify burst events (within 10 minutes)
    df['is_burst'] = (df['time_diff_minutes'] <= BURST_WINDOW_MINUTES) & (df['time_diff_minutes'].notna())
    
    # Create event groups
    df['new_event'] = (~df['is_burst']).astype(int)
    df['event_group'] = df.groupby('senior_id')['new_event'].cumsum()
    
    # Merge burst events
    def merge_event(group):
        result = group.iloc[0].copy()
        result['event_id'] = group['alert_id'].iloc[0]  # First alert ID
        result['alert_date'] = group['alert_date'].iloc[0]  # First timestamp
        result['severity'] = group['severity'].max()  # Maximum severity
        result['event_size'] = len(group)  # Number of alerts in burst
        
        # Concatenate unique notes if multiple
        if len(group) > 1:
            notes = group['sos_note'].dropna().unique().tolist()
            result['sos_note'] = ' | '.join([str(n) for n in notes])
        
        return result
    
    df_events = df.groupby(['senior_id', 'event_group'], as_index=False).apply(merge_event)
    if isinstance(df_events, pd.DataFrame) and df_events.index.nlevels > 1:
        df_events = df_events.droplevel([0, 1])
    
    print(f"After burst merging: {len(df_events):,} events")
    print(f"  - Compression ratio: {len(df_events)/len(df)*100:.1f}%")
    print(f"  - Single-alert events: {(df_events['event_size'] == 1).sum():,}")
    print(f"  - Multi-alert events: {(df_events['event_size'] > 1).sum():,}")
    
    # Severity distribution
    print("\nSeverity distribution:")
    severity_map = {0: 'Noise/Tech', 1: 'Low', 2: 'Potential', 3: 'Acute'}
    for sev, count in df_events['severity'].value_counts().sort_index().items():
        print(f"  - Level {sev} ({severity_map[sev]}): {count:,}")
    
    # Convert alert_date back to ISO string
    df_events['alert_date'] = df_events['alert_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Select final columns
    columns = ['event_id', 'senior_id', 'alert_date', 'severity', 'event_size', 'sos_note']
    df_events = df_events[columns]
    
    # Write to database
    df_events.to_sql('alerts', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'alerts', 0, completed=True)
    print("✓ Alerts table written")
    
    # Create index
    cursor_out = conn_out.cursor()
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_alerts_senior ON alerts(senior_id)")
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(alert_date)")
    conn_out.commit()
    
    return df_events


def create_risk_profiles(conn_in, conn_out):
    """
    Create senior risk profiles based on disease-to-domain mapping.
    RESUMABLE: Skipped if already processed.
    
    Creates a binary matrix with 11 columns representing clinical domains.
    """
    print("\n" + "="*80)
    print("CREATING MEDICAL RISK PROFILES (RESUMABLE)")
    print("="*80)
    
    # Check if already completed
    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM senior_risk_profiles").fetchone()[0]
        if count > 0:
            print(f"✓ Risk profiles already created: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM senior_risk_profiles", conn_out)
    except Exception:
        pass
    
    # Load disease data
    diseases_df = pd.read_sql_query("SELECT * FROM senior_diseases", conn_in)
    diseases_lookup = pd.read_sql_query("SELECT * FROM diseases", conn_in)
    
    print(f"Total disease records: {len(diseases_df):,}")
    print(f"Unique diseases: {len(diseases_lookup):,}")
    
    # Map disease IDs to names
    disease_map = dict(zip(diseases_lookup['id'], diseases_lookup['disease_name']))
    diseases_df['disease_name'] = diseases_df['disease_id'].map(disease_map)
    
    # Categorize diseases
    diseases_df['domain'] = diseases_df['disease_name'].apply(categorize_disease)
    
    # Create binary matrix
    domain_columns = list(DISEASE_CATEGORIES.keys()) + ['other']
    
    # Get unique seniors
    unique_seniors = diseases_df['senior_id'].unique()
    print(f"Seniors with disease data: {len(unique_seniors):,}")
    
    # Initialize binary matrix
    risk_matrix = pd.DataFrame(0, index=unique_seniors, columns=domain_columns)
    
    # Fill matrix
    for _, row in diseases_df.iterrows():
        senior_id = row['senior_id']
        domain = row['domain']
        risk_matrix.loc[senior_id, domain] = 1
    
    # Reset index to make senior_id a column
    risk_matrix = risk_matrix.reset_index().rename(columns={'index': 'senior_id'})
    
    # Domain distribution
    print("\nDomain distribution:")
    for domain in domain_columns:
        count = risk_matrix[domain].sum()
        pct = count / len(risk_matrix) * 100
        print(f"  - {domain}: {count:,} seniors ({pct:.1f}%)")
    
    # Write to database
    risk_matrix.to_sql('senior_risk_profiles', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'senior_risk_profiles', 0, completed=True)
    print("\n✓ Risk profiles table written")
    
    # Create index
    cursor_out = conn_out.cursor()
    cursor_out.execute("CREATE INDEX IF NOT EXISTS idx_risk_profiles_senior ON senior_risk_profiles(senior_id)")
    conn_out.commit()
    
    return risk_matrix


def copy_auxiliary_tables(conn_in, conn_out):
    """Copy auxiliary tables that don't need processing."""
    print("\n" + "="*80)
    print("COPYING AUXILIARY TABLES")
    print("="*80)
    
    tables_to_copy = [
        'diseases',
        'medicines',
        'senior_diseases',
        'senior_medicines'
    ]
    
    for table in tables_to_copy:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn_in)
            df.to_sql(table, conn_out, if_exists='replace', index=False)
            print(f"✓ Copied {table}: {len(df):,} rows")
        except Exception as e:
            print(f"⚠ Warning: Could not copy {table}: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main data processing pipeline."""
    print("="*80)
    print("HRP DATA PROCESSING PIPELINE")
    print("="*80)
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print(f"Burst window: {BURST_WINDOW_MINUTES} minutes")
    print(f"Current year: {CURRENT_YEAR}")
    
    # Define paths (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    raw_db_path = project_root / "db" / "hrp_data.db"
    processed_db_path = project_root / "data" / "processed" / "hrp_processed.db"
    
    # Create output directory
    processed_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we're resuming
    is_resuming = processed_db_path.exists()
    if is_resuming:
        print("\n⚠ Database exists: RESUMING from checkpoint")
        print(f"   Location: {processed_db_path}")
    else:
        print("\n✓ Starting fresh processing")
        print(f"   Output: {processed_db_path}")
    
    # Connect to databases
    print(f"\nConnecting to raw database: {raw_db_path}")
    conn_in = sqlite3.connect(raw_db_path)
    
    print(f"{'Resuming' if is_resuming else 'Creating'} processed database: {processed_db_path}")
    conn_out = sqlite3.connect(processed_db_path)
    
    # Initialize checkpoint table
    init_checkpoint_table(conn_out)
    
    try:
        # Process tables
        df_seniors = process_seniors(conn_in, conn_out)
        process_measurements_chunked(conn_in, conn_out)
        df_alerts = process_alerts(conn_in, conn_out)
        df_risk_profiles = create_risk_profiles(conn_in, conn_out)
        copy_auxiliary_tables(conn_in, conn_out)
        
        # Final summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        
        cursor_out = conn_out.cursor()
        
        # Get table sizes
        tables = ['seniors', 'measurements', 'alerts', 'senior_risk_profiles']
        for table in tables:
            count = cursor_out.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  - {table}: {count:,} rows")
        
        # Database file size
        db_size_mb = processed_db_path.stat().st_size / (1024 * 1024)
        print(f"\nProcessed database size: {db_size_mb:.1f} MB")
        print(f"Location: {processed_db_path}")
        
        print("\n✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        raise
    
    finally:
        # Close connections
        conn_in.close()
        conn_out.close()


if __name__ == "__main__":
    main()
