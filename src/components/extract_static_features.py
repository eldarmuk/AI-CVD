import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/processed/multimodal_features_original.parquet")
OUTPUT_PATH = Path("data/processed/static_features.csv")

STATIC_COLS = [
    'senior_id',
    'age', 'gender',
    'cardiovascular', 'metabolic_endocrine', 'neurological', 
    'psychiatric_cognitive', 'musculoskeletal', 'respiratory', 
    'gastro_renal_urologic', 'oncological', 'sensory', 
    'other_functional_risk', 'other'
]

def main():
    print(f"Loading {PARQUET_PATH}...")
    try:
        df = pd.read_parquet(PARQUET_PATH, columns=STATIC_COLS)
    except Exception as e:
        print(f"Error loading columns: {e}")
        print("Tip: If you already overwrote the parquet with the 12-feature version,")
        print("you will need to regenerate it from the database (db/hrp_processed.db).")
        return

    print(f"Original shape: {df.shape}")

    static_df = df.drop_duplicates(subset=['senior_id'])
    
    print(f"Unique Seniors extracted: {len(static_df)}")
    
    static_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved static features to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()