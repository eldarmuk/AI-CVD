import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import duckdb

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENT_YEAR = 2026
CHUNK_SIZE = 5_000_000  # Process measurements in batches (DuckDB path doesn't use chunks tho)
BURST_WINDOW_MINUTES = 10  # Drop repeated alerts within this window (keep first only)
PULSE_PRESSURE_MIN = 10  # Minimum valid pulse pressure

OUTLIER_RANGES = {
    'Heartrate':   (30, 220),
    'SBP':         (70, 250),
    'DBP':         (40, 140),
    'Temperature': (30, 45),
    'Saturation':  (50, 100),
    'Steps':       (0, 50_000),
}

DISEASE_CATEGORIES = {
    'cardiovascular': [
        'serc', 'nadciśnien', 'ciśnien', 'arytmi', 'migotani',
        'trzepotani', 'niewydolność serca', 'zawał',
        'wieńc', 'miażdżyc', 'aort', 'tętniak',
        'dusznica', 'zakrzep', 'zator', 'płucn',
        'kołatanie', 'hipotensj',
    ],
    'metabolic_endocrine': [
        'cukrzyc', 'insulin', 'lipid', 'cholesterol',
        'hiperlipid', 'otyło', 'nadwag',
        'tarczyc', 'hashimoto', 'metabolic',
        'endokryn', 'hipoglik', 'stan przedcukrzyc',
    ],
    'neurological': [
        'udar', 'neurolog', 'padaczk',
        'parkinson', 'stwardnieni',
        'neuropat', 'polineurop',
        'rwa', 'zawrot', 'omdleni',
    ],
    'psychiatric_cognitive': [
        'depresj', 'nerwic', 'lękow',
        'schizofren', 'psychicz',
        'demencj', 'otępieni',
        'alzheimer', 'zanik pamięci',
        'bezsenność',
    ],
    'musculoskeletal': [
        'staw', 'kostn', 'reumaty',
        'zwyrodnieni', 'kręgosłup',
        'dyskopati', 'skolioz',
        'osteoporoz', 'osteopeni',
        'endoprotez', 'lasce', 'chodzeni',
    ],
    'respiratory': [
        'astm', 'pochp', 'rozedm',
        'oddech', 'bezdech',
        'duszno', 'płuc',
    ],
    'gastro_renal_urologic': [
        'żołądk', 'jelit', 'refluks',
        'wrzod', 'uchyłk', 'przepuklin',
        'wątroba', 'trzustk', 'żółci',
        'nerk', 'kamica', 'mocz',
        'prostat',
    ],
    'oncological': [
        'nowotwór', 'rak',
        'onkolog', 'guzy',
        'mastektomia',
    ],
    'sensory': [
        'wzrok', 'widzeni', 'zaćm',
        'jaskr', 'plamki',
        'słuch', 'niedosłuch',
        'głuchot', 'szumy uszne',
    ],
    'other_functional_risk': [
        'niepełnosprawność', 'porusz',
        'chodzeni', 'laska',
        'nikotyn', 'palenie',
        'autoimmun', 'łuszczyc',
        'borelioz', 'anemi',
        'implant', 'stent',
        'rozrusznik', 'bajpas',
    ],
}

DOMAIN_COLUMNS = list(DISEASE_CATEGORIES.keys()) + ['unclassified']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_valid_senior_ids(conn_out):
    """Retrieve valid senior IDs from the processed seniors table."""
    cursor = conn_out.cursor()
    cols_info = cursor.execute("PRAGMA table_info(seniors)").fetchall()
    cols = [c[1] for c in cols_info] if cols_info else []

    id_col = None
    if 'id' in cols:
        id_col = 'id'
    elif 'senior_id' in cols:
        id_col = 'senior_id'

    if not id_col:
        raise RuntimeError(
            "Could not determine senior identifier column in 'seniors' table "
            "(expected 'id' or 'senior_id')."
        )

    df_ids = pd.read_sql_query(f"SELECT {id_col} AS senior_id FROM seniors", conn_out)
    sids = pd.to_numeric(df_ids['senior_id'], errors='coerce').dropna().astype(int).tolist()
    return set(sids)


def classify_severity(note):
    if pd.isna(note):
        return -1

    note_l = str(note).lower()

    if any(k in note_l for k in ['alarm przypadkowy', 'alarm testowy', 'alert techniczny']):
        return 0

    no_zrm = 'brak wskazań do interwencji zrm' in note_l

    if not no_zrm and (
        'zdecydowano się na interwencję zrm' in note_l
        or 'zdecydowano wezwać zrm' in note_l
        or 'wezwanie zrm' in note_l
        or ('zagrożenia życia i zdrowia' in note_l and 'nawiązano kontakt' in note_l)
        or ('na podstawie odczytów z systemu' in note_l and 'zrm' in note_l)
    ):
        return 3

    if 'nie nawiązano kontaktu' in note_l:
        return 2

    if 'nawiązano kontakt' in note_l or 'opiekun powiadomiony' in note_l:
        return 1

    return -1


def categorize_disease(disease_name):
    """Map disease name to clinical domain."""
    disease_lower = str(disease_name).lower()
    for category, keywords in DISEASE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in disease_lower:
                return category
    return 'unclassified'


def round_measurements(df):
    """Round measurement values to appropriate decimal places (pandas path)."""
    for col in ['value', 'sbp', 'dbp']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    is_temp = (df['type'] == 'Temperature')
    df['value'] = np.where(is_temp, np.round(df['value'], 1), np.round(df['value'], 0))

    if 'sbp' in df.columns:
        df['sbp'] = np.round(df['sbp'], 0)
    if 'dbp' in df.columns:
        df['dbp'] = np.round(df['dbp'], 0)
    return df


def clip_outliers(df):
    """Apply per-type outlier clipping (pandas path)."""
    for mtype, (lo, hi) in OUTLIER_RANGES.items():
        if mtype == 'Heartrate':
            mask = df['type'] == 'Heartrate'
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lo, hi)

        elif mtype == 'Temperature':
            mask = df['type'] == 'Temperature'
            df.loc[mask & (df['value'] > hi), 'value'] = np.nan
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lower=lo)

        elif mtype == 'Saturation':
            mask = df['type'] == 'Saturation'
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lower=lo)

        elif mtype == 'Steps':
            mask = df['type'] == 'Steps'
            df.loc[mask, 'value'] = df.loc[mask, 'value'].clip(lo, hi)

        elif mtype == 'SBP':
            if 'sbp' in df.columns:
                df['sbp'] = df['sbp'].clip(lo, hi)

        elif mtype == 'DBP':
            if 'dbp' in df.columns:
                df['dbp'] = df['dbp'].clip(lo, hi)

    return df


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def init_checkpoint_table(conn_out):
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
    cursor = conn_out.cursor()
    try:
        result = cursor.execute(
            "SELECT last_offset FROM processing_checkpoints WHERE table_name = ?",
            (table_name,)
        ).fetchone()
        if result:
            return int(result[0]) if result[0] is not None else 0
        return 0
    except Exception:
        return 0


def update_checkpoint(conn_out, table_name, offset, completed=False):
    cursor = conn_out.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO processing_checkpoints (table_name, last_offset, completed)
        VALUES (?, ?, ?)
    """, (table_name, int(offset) if offset is not None else 0, 1 if completed else 0))
    conn_out.commit()


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def process_seniors(conn_in, conn_out):
    """
    Clean and process seniors table.

    Steps
    ─────
    1. Filter age to [60, 110].
    2. Replace 'Unknown' gender with NaN.
    3. Drop rows with both birthdate AND age NULL.
    4. Cross-validate stored age vs. birthdate-derived age; trust birthdate on conflict.
    5. Impute missing age from birthdate (or vice-versa) using CURRENT_YEAR.
    """
    print("\n" + "="*80)
    print("PROCESSING SENIORS TABLE (RESUMABLE)")
    print("="*80)

    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM seniors").fetchone()[0]
        if count > 0:
            print(f"OK: Seniors table already processed: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM seniors", conn_out)
    except Exception:
        pass

    df = pd.read_sql_query("SELECT * FROM seniors", conn_in)
    print(f"Initial seniors: {len(df):,}")

    df = df[(df['age'] >= 60) & (df['age'] <= 110)]
    print(f"After age filter (60-110): {len(df):,}")

    df.loc[df['gender'] == 'Unknown', 'gender'] = np.nan

    both_null = df['birthdate'].isna() & df['age'].isna()
    df = df[~both_null]
    print(f"After dropping both NULL: {len(df):,}")

    df['birthdate_tmp'] = pd.to_datetime(df['birthdate'], errors='coerce')
    both_present = df['age'].notna() & df['birthdate_tmp'].notna()
    if both_present.any():
        computed = CURRENT_YEAR - df.loc[both_present, 'birthdate_tmp'].dt.year
        bad = (computed - df.loc[both_present, 'age']).abs() > 1
        print(f"  - Age/birthdate discrepancy (>1yr): {bad.sum()} seniors - trusting birthdate")
        df.loc[both_present & bad, 'age'] = computed[bad]
    df.drop(columns=['birthdate_tmp'], inplace=True)

    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')

    missing_age = df['age'].isna() & df['birthdate'].notna()
    if missing_age.any():
        df.loc[missing_age, 'age'] = CURRENT_YEAR - df.loc[missing_age, 'birthdate'].dt.year

    missing_bd = df['birthdate'].isna() & df['age'].notna()
    if missing_bd.any():
        years = CURRENT_YEAR - df.loc[missing_bd, 'age'].astype(int)
        df.loc[missing_bd, 'birthdate'] = pd.to_datetime(years.astype(str) + '-01-01')

    df['birthdate'] = df['birthdate'].dt.strftime('%Y-%m-%d')

    print(f"Final seniors: {len(df):,}")
    print(f"  - With valid gender: {df['gender'].notna().sum():,}")
    print(f"  - With age: {df['age'].notna().sum():,}")
    print(f"  - With birthdate: {df['birthdate'].notna().sum():,}")

    df.to_sql('seniors', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'seniors', 0, completed=True)
    print("OK: Seniors table written")
    return df


def process_measurements_duckdb(raw_db_path, processed_db_path, conn_out):
    """
    Process measurements table using DuckDB.

    Pipeline stages (all in SQL)
    ────────────────────────────
    1. valid_seniors  — inner-join against already-processed seniors.
    2. deduped        — group by (senior_id, date, type) exact-second;
                        average conflicting values (rare duplicate device pings).
    3. rounded        — Temperature → 1 dp; everything else → 0 dp.
    4. clipped        — per-type physiological bounds applied;
                        Temperature > 45 °C → NULL (device error, not a clamp);
                        Steps clipped to [0, 50 000].
    5. bp_logic       — nullify BOTH sbp and dbp when dbp >= sbp after clipping
                        (inverted pair is physiologically impossible).
    6. final select   — compute pulse_pressure; NULL when < PULSE_PRESSURE_MIN
                        OR when either BP component was nullified.
                        Individual sbp/dbp are preserved in the narrow-PP case
                        because a 5-mmHg difference still carries model signal.
    """
    print("\n" + "="*80, flush=True)
    print("PROCESSING MEASUREMENTS TABLE", flush=True)
    print("="*80, flush=True)

    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        if count > 0:
            print(f"OK: Measurements table already processed: {count:,} rows", flush=True)
            return
    except Exception:
        pass

    print("Creating schema via SQLite connection...", flush=True)
    conn_out.execute("DROP TABLE IF EXISTS measurements;")
    conn_out.execute("""
        CREATE TABLE measurements (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            senior_id     INTEGER NOT NULL,
            value         REAL,
            sbp           REAL,
            dbp           REAL,
            pulse_pressure REAL,
            date          TEXT NOT NULL,
            type          TEXT NOT NULL
        );
    """)
    conn_out.commit()
    conn_out.close()
    print("SQLite connection released - handing file to DuckDB.", flush=True)

    print("Connecting DuckDB to SQLite databases...", flush=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false;")
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{raw_db_path}' AS raw_db  (TYPE sqlite, READ_ONLY true);")
    con.execute(f"ATTACH '{processed_db_path}' AS proc_db (TYPE sqlite);")

    print("Executing deduplication and transformation...", flush=True)

    hr_lo,   hr_hi   = OUTLIER_RANGES['Heartrate']
    t_lo,    t_hi    = OUTLIER_RANGES['Temperature']
    spo2_lo, _       = OUTLIER_RANGES['Saturation']
    st_lo,   st_hi   = OUTLIER_RANGES['Steps']
    sbp_lo,  sbp_hi  = OUTLIER_RANGES['SBP']
    dbp_lo,  dbp_hi  = OUTLIER_RANGES['DBP']

    transform_query = f"""
        INSERT INTO proc_db.measurements
            (senior_id, value, sbp, dbp, pulse_pressure, date, type)
        WITH valid_seniors AS (
            SELECT id AS senior_id FROM proc_db.seniors
        ),
        deduped AS (
            -- Exact-second deduplication: average values that collide on the
            -- same (senior_id, date, type) key.
            SELECT
                m.senior_id,
                m.date,
                m.type,
                AVG(m.value) AS val,
                AVG(m.sbp)   AS sbp,
                AVG(m.dbp)   AS dbp
            FROM raw_db.measurements m
            INNER JOIN valid_seniors v ON m.senior_id = v.senior_id
            GROUP BY m.senior_id, m.date, m.type
        ),
        rounded AS (
            SELECT
                senior_id, date, type,
                CASE WHEN type = 'Temperature'
                     THEN ROUND(val, 1)
                     ELSE ROUND(val, 0)
                END AS r_val,
                ROUND(sbp, 0) AS r_sbp,
                ROUND(dbp, 0) AS r_dbp
            FROM deduped
        ),
        clipped AS (
            -- Per-type physiological outlier clipping.
            -- Temperature > {t_hi} → NULL (device error; not clamped to boundary).
            -- Steps negative → 0; Steps > {st_hi} → {st_hi} (sensor error).
            SELECT
                senior_id, date, type,
                CASE
                    WHEN type = 'Heartrate'   AND r_val > {hr_hi}   THEN {hr_hi}
                    WHEN type = 'Heartrate'   AND r_val < {hr_lo}   THEN {hr_lo}
                    WHEN type = 'Temperature' AND r_val > {t_hi}    THEN NULL
                    WHEN type = 'Temperature' AND r_val < {t_lo}    THEN {t_lo}
                    WHEN type = 'Saturation'  AND r_val < {spo2_lo} THEN {spo2_lo}
                    WHEN type = 'Steps'       AND r_val < {st_lo}   THEN {st_lo}
                    WHEN type = 'Steps'       AND r_val > {st_hi}   THEN {st_hi}
                    ELSE r_val
                END AS c_val,
                CASE
                    WHEN type = 'BloodPressure' AND r_sbp > {sbp_hi} THEN {sbp_hi}
                    WHEN type = 'BloodPressure' AND r_sbp < {sbp_lo} THEN {sbp_lo}
                    ELSE r_sbp
                END AS c_sbp,
                CASE
                    WHEN type = 'BloodPressure' AND r_dbp > {dbp_hi} THEN {dbp_hi}
                    WHEN type = 'BloodPressure' AND r_dbp < {dbp_lo} THEN {dbp_lo}
                    ELSE r_dbp
                END AS c_dbp
            FROM rounded
        ),
        bp_logic AS (
            -- Nullify BOTH components of an inverted blood-pressure pair.
            -- DBP >= SBP after clipping is physiologically impossible.
            -- Pairs where SBP - DBP > 0 but < PULSE_PRESSURE_MIN are kept here;
            -- pulse_pressure is set to NULL in the final select.
            SELECT
                senior_id,
                c_val AS value,
                CASE WHEN type = 'BloodPressure' AND c_dbp >= c_sbp
                     THEN NULL ELSE c_sbp END AS sbp,
                CASE WHEN type = 'BloodPressure' AND c_dbp >= c_sbp
                     THEN NULL ELSE c_dbp END AS dbp,
                date, type
            FROM clipped
        )
        -- Final: pulse_pressure is NULL when:
        --   (a) SBP or DBP was nullified (inverted pair), or
        --   (b) their difference falls below {PULSE_PRESSURE_MIN} mmHg.
        -- sbp and dbp are preserved in case (b) — a narrow but positive PP
        -- may still be meaningful for the model.
        SELECT
            senior_id, value, sbp, dbp,
            CASE
                WHEN type = 'BloodPressure'
                     AND sbp IS NOT NULL
                     AND dbp IS NOT NULL
                     AND (sbp - dbp) >= {PULSE_PRESSURE_MIN}
                THEN (sbp - dbp)
                ELSE NULL
            END AS pulse_pressure,
            strftime(CAST(date AS TIMESTAMP), '%Y-%m-%d %H:%M:%S') AS date,
            type
        FROM bp_logic;
    """

    con.execute(transform_query)

    print("Building database indexes...", flush=True)
    con.execute("CREATE INDEX IF NOT EXISTS idx_meas_senior      ON proc_db.measurements(senior_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_meas_date        ON proc_db.measurements(date);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_meas_type        ON proc_db.measurements(type);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_meas_senior_date ON proc_db.measurements(senior_id, date);")
    con.close()
    print("DuckDB connection closed.", flush=True)

    conn_out = sqlite3.connect(str(processed_db_path))

    update_checkpoint(conn_out, 'measurements', 0, completed=True)
    final_count = conn_out.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    print(f"\nOK: Measurements processing complete! Wrote {final_count:,} unique rows.", flush=True)


def process_alerts(conn_in, conn_out):
    """
    Process alerts: classify severity and drop burst duplicates.

    Severity scale
    ──────────────
      -1  Undocumented / unknown
       0  Noise (known false / test / technical alarm)
       1  Low  (contact established, no ZRM)
       2  Potential (no contact)
       3  Acute (ZRM dispatched)
    """
    print("\n" + "="*80)
    print("PROCESSING ALERTS TABLE (RESUMABLE)")
    print("="*80)

    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        if count > 0:
            print(f"OK: Alerts table already processed: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM alerts", conn_out)
    except Exception:
        pass

    df = pd.read_sql_query("SELECT * FROM alerts", conn_in)
    print(f"Initial alerts: {len(df):,}")

    valid_seniors = get_valid_senior_ids(conn_out)
    before = len(df)
    df = df[df['senior_id'].isin(valid_seniors)]
    print(f"After filtering unknown seniors: {len(df):,} (dropped {before - len(df):,})")

    df['severity'] = df['sos_note'].apply(classify_severity)

    df['alert_date'] = pd.to_datetime(df['alert_date'], format='ISO8601')
    df = df.sort_values(['senior_id', 'alert_date']).reset_index(drop=True)

    df['_time_diff_min'] = (
        df.groupby('senior_id')['alert_date']
          .diff()
          .dt.total_seconds()
          .div(60)
    )
    df['_is_burst_dup'] = df['_time_diff_min'] <= BURST_WINDOW_MINUTES

    n_total   = len(df)
    n_dropped = int(df['_is_burst_dup'].sum())
    df_clean  = df[~df['_is_burst_dup']].copy()

    print(f"Burst duplicates dropped : {n_dropped:,} / {n_total:,} "
          f"({n_dropped / n_total * 100:.1f}%)")
    print(f"Canonical alerts retained: {len(df_clean):,}  "
          f"(one per event, exact original timestamp)")

    severity_map = {-1: 'Undocumented', 0: 'Noise/Tech', 1: 'Low',
                    2: 'Potential', 3: 'Acute'}
    print("\nSeverity distribution (post-dedup):")
    for sev, cnt in df_clean['severity'].value_counts().sort_index().items():
        pct = cnt / len(df_clean) * 100
        label = severity_map.get(sev, '?')
        print(f"  Level {sev:+d}  ({label:14s}): {cnt:,}  ({pct:.1f}%)")

    df_clean['alert_date'] = df_clean['alert_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_out = df_clean[['alert_id', 'senior_id', 'alert_date', 'severity', 'sos_note']].copy()

    df_out.to_sql('alerts', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'alerts', 0, completed=True)
    print("OK: Alerts table written")

    c = conn_out.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_senior   ON alerts(senior_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_date     ON alerts(alert_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
    conn_out.commit()

    return df_out


def create_risk_profiles(conn_in, conn_out):
    """
    Create senior risk profiles based on disease-to-domain mapping.
    """
    print("\n" + "="*80)
    print("CREATING MEDICAL RISK PROFILES (RESUMABLE)")
    print("="*80)

    cursor = conn_out.cursor()
    try:
        count = cursor.execute("SELECT COUNT(*) FROM senior_risk_profiles").fetchone()[0]
        if count > 0:
            print(f"OK: Risk profiles already created: {count:,} rows")
            return pd.read_sql_query("SELECT * FROM senior_risk_profiles", conn_out)
    except Exception:
        pass

    diseases_df    = pd.read_sql_query("SELECT * FROM senior_diseases", conn_in)
    diseases_lookup = pd.read_sql_query("SELECT * FROM diseases", conn_in)

    print(f"Total disease records : {len(diseases_df):,}")
    print(f"Unique diseases       : {len(diseases_lookup):,}")

    valid_seniors = get_valid_senior_ids(conn_out)
    before = len(diseases_df)
    diseases_df = diseases_df[diseases_df['senior_id'].isin(valid_seniors)].copy()
    print(f"After filtering unknown seniors: {len(diseases_df):,} (dropped {before - len(diseases_df):,})")

    disease_map = dict(zip(diseases_lookup['id'], diseases_lookup['disease_name']))
    diseases_df['disease_name'] = diseases_df['disease_id'].map(disease_map)
    diseases_df['domain']       = diseases_df['disease_name'].apply(categorize_disease)

    print("\nDomain distribution (all disease records):")
    for domain, cnt in diseases_df['domain'].value_counts().items():
        pct = cnt / len(diseases_df) * 100
        print(f"  {domain:30s}: {cnt:,}  ({pct:.1f}%)")

    diseases_df['_flag'] = 1
    risk_matrix = (
        diseases_df
        .drop_duplicates(subset=['senior_id', 'domain'])
        .pivot(index='senior_id', columns='domain', values='_flag')
        .reindex(columns=DOMAIN_COLUMNS, fill_value=0)
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    print(f"\nSeniors with disease data: {len(risk_matrix):,}")
    print("\nBinary matrix column coverage:")
    for col in DOMAIN_COLUMNS:
        if col in risk_matrix.columns:
            cnt = int(risk_matrix[col].sum())
            pct = cnt / len(risk_matrix) * 100
            print(f"  {col:30s}: {cnt:,} seniors  ({pct:.1f}%)")

    risk_matrix.to_sql('senior_risk_profiles', conn_out, if_exists='replace', index=False)
    update_checkpoint(conn_out, 'senior_risk_profiles', 0, completed=True)
    print("\nOK: Risk profiles table written")

    conn_out.cursor().execute(
        "CREATE INDEX IF NOT EXISTS idx_risk_profiles_senior ON senior_risk_profiles(senior_id)"
    )
    conn_out.commit()

    return risk_matrix


def copy_auxiliary_tables(conn_in, conn_out):
    """Copy auxiliary tables that require no transformation."""
    print("\n" + "="*80)
    print("COPYING AUXILIARY TABLES")
    print("="*80)

    tables_to_copy = ['diseases', 'medicines', 'senior_diseases', 'senior_medicines']
    valid_seniors  = get_valid_senior_ids(conn_out)

    for table in tables_to_copy:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn_in)
            if table in ('senior_diseases', 'senior_medicines') and 'senior_id' in df.columns:
                before = len(df)
                df = df[df['senior_id'].isin(valid_seniors)]
                print(f"OK: Filtered {table}: kept {len(df):,}, dropped {before - len(df):,}")
            else:
                print(f"OK: Copied  {table}: {len(df):,} rows")
            df.to_sql(table, conn_out, if_exists='replace', index=False)
        except Exception as e:
            print(f"WARNING: Could not copy {table}: {e}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main data processing pipeline."""
    print("="*80)
    print("HRP DATA PROCESSING PIPELINE")
    print("="*80)
    print(f"Chunk size        : {CHUNK_SIZE:,} rows")
    print(f"Burst window      : {BURST_WINDOW_MINUTES} min  (drop duplicates, keep first)")
    print(f"Current year      : {CURRENT_YEAR}")
    print(f"Domain columns    : {DOMAIN_COLUMNS}")

    script_dir     = Path(__file__).parent
    project_root   = script_dir.parent.parent
    raw_db_path    = project_root / "db" / "hrp_data.db"
    processed_db_path = project_root / "db" / "hrp_processed.db"

    processed_db_path.parent.mkdir(parents=True, exist_ok=True)

    is_resuming = processed_db_path.exists()
    print(f"\n{'- RESUMING from checkpoint' if is_resuming else 'OK: Starting fresh'}")
    print(f"   Output: {processed_db_path}")

    conn_in  = sqlite3.connect(raw_db_path)
    conn_out = sqlite3.connect(processed_db_path)
    init_checkpoint_table(conn_out)

    try:
        process_seniors(conn_in, conn_out)
        process_measurements_duckdb(raw_db_path, processed_db_path, conn_out)
        process_alerts(conn_in, conn_out)
        create_risk_profiles(conn_in, conn_out)
        copy_auxiliary_tables(conn_in, conn_out)

        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)

        c = conn_out.cursor()
        for table in ['seniors', 'measurements', 'alerts', 'senior_risk_profiles']:
            n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:30s}: {n:,} rows")

        seniors_ids  = set(pd.read_sql_query("SELECT id FROM seniors", conn_out)['id'])
        measured_ids = set(pd.read_sql_query("SELECT DISTINCT senior_id FROM measurements", conn_out)['senior_id'])
        zero_meas    = seniors_ids - measured_ids
        print(f"  Seniors with zero measurements  : {len(zero_meas):,}  "
              f"({100*len(zero_meas)/len(seniors_ids):.1f}%)")

        db_mb = processed_db_path.stat().st_size / (1024 * 1024)
        print(f"\nProcessed DB size : {db_mb:.1f} MB")
        print(f"Location          : {processed_db_path}")
        print("\nOK: Pipeline completed successfully!")

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

    finally:
        conn_in.close()
        conn_out.close()


if __name__ == "__main__":
    main()