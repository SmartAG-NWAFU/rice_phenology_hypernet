from __future__ import annotations

import pandas as pd

from rice_phenology_hypernet.models.m0 import THRESHOLD_COLUMNS


THRESHOLD_DECIMALS = 2


def round_threshold_value(value: float) -> float:
    return round(float(value), THRESHOLD_DECIMALS)


def merge_by_occurrence(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    key_cols = ["SID", "year"]
    left_keyed = left.copy()
    right_keyed = right[key_cols + columns].copy()
    left_keyed["_occ"] = left_keyed.groupby(key_cols).cumcount()
    right_keyed["_occ"] = right_keyed.groupby(key_cols).cumcount()
    merged = left_keyed.merge(right_keyed, on=key_cols + ["_occ"], how="left")
    return merged.drop(columns="_occ")


def prepare_prior_map(train_df: pd.DataFrame, threshold_df: pd.DataFrame) -> dict[str, float]:
    merged = merge_by_occurrence(train_df, threshold_df, THRESHOLD_COLUMNS)
    return {col: round_threshold_value(merged[col].median()) for col in THRESHOLD_COLUMNS}
