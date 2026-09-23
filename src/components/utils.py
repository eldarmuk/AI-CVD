from src.ai_cvd.features import FEATURE_NAMES


def get_feature_columns(df) -> list:
    """Fixed v2 allowlist; reject legacy/incomplete schemas instead of guessing."""
    columns = set(df.columns if hasattr(df, "columns") else df)
    missing = set(FEATURE_NAMES) - columns
    if missing:
        raise ValueError(f"Not the canonical v2 feature schema; missing: {sorted(missing)}")
    return list(FEATURE_NAMES)
