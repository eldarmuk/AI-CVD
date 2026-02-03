import pandas as pd

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Centralized logic to extract feature columns.
    Excludes metadata/labels to return only model input features.
    """
    EXCLUDE_COLS = {
        'senior_id', 'timestamp', 
        'label_1', 'label_2', 'label_3', 
        'hour', 'is_night', 'day_of_week'
    }
    
    return [col for col in df.columns if col not in EXCLUDE_COLS]