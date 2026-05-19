from __future__ import annotations

import numpy as np
import pandas as pd

from rice_phenology_hypernet.config import YearFoldConfig, get_project_config


def _array_split_groups(values: list[int], n_splits: int) -> list[np.ndarray]:
    return [np.asarray(group, dtype=int) for group in np.array_split(np.asarray(values, dtype=int), n_splits)]


def _greedy_partition(weight_by_sid: pd.Series, n_splits: int) -> list[np.ndarray]:
    folds = [{"weight": 0, "sids": []} for _ in range(n_splits)]
    ordered = weight_by_sid.sort_values(ascending=False)
    for sid, weight in ordered.items():
        target_index = min(range(n_splits), key=lambda idx: (folds[idx]["weight"], len(folds[idx]["sids"]), idx))
        folds[target_index]["sids"].append(int(sid))
        folds[target_index]["weight"] += int(weight)
    return [np.asarray(fold["sids"], dtype=int) for fold in folds]


def _site_fold_sids(df: pd.DataFrame, n_splits: int) -> list[np.ndarray]:
    site_counts = df.groupby("SID").size()
    if len(site_counts) < n_splits:
        raise ValueError(f"Not enough sites for {n_splits} site folds: {len(site_counts)}")
    return _greedy_partition(site_counts, n_splits)


def _is_valid_fold(test: pd.DataFrame, *, min_test_samples: int, min_test_sites: int | None = None) -> tuple[bool, str | None]:
    if len(test) < min_test_samples:
        return False, f"filtered_test_n<{min_test_samples}"
    if min_test_sites is not None and test["SID"].nunique() < min_test_sites:
        return False, f"filtered_test_sites<{min_test_sites}"
    return True, None


def _summarize_filtered_test(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int | float]:
    return {
        "train_n": int(len(train)),
        "filtered_test_n": int(len(test)),
        "train_sites": int(train["SID"].nunique()) if not train.empty else 0,
        "filtered_test_sites": int(test["SID"].nunique()) if not test.empty else 0,
        "train_year_min": int(train["year"].min()) if not train.empty else np.nan,
        "train_year_max": int(train["year"].max()) if not train.empty else np.nan,
        "filtered_test_year_min": int(test["year"].min()) if not test.empty else np.nan,
        "filtered_test_year_max": int(test["year"].max()) if not test.empty else np.nan,
    }


def _site_protocol_records(df: pd.DataFrame, n_splits: int | None = None) -> list[dict[str, object]]:
    config = get_project_config().experiment.site
    n_splits = n_splits or config.n_splits
    records: list[dict[str, object]] = []
    for fold, fold_sids in enumerate(_site_fold_sids(df, n_splits), start=1):
        raw_test_mask = df["SID"].isin(fold_sids)
        train_idx = np.where(~raw_test_mask)[0]
        raw_test_idx = np.where(raw_test_mask)[0]
        train = df.iloc[train_idx]
        raw_test = df.iloc[raw_test_idx]
        seen_years = set(train["year"].tolist())
        filtered_test = raw_test[raw_test["year"].isin(seen_years)]
        filtered_test_idx = filtered_test.index.to_numpy(dtype=int)
        valid_flag, invalid_reason = _is_valid_fold(
            filtered_test,
            min_test_samples=config.min_test_samples,
            min_test_sites=config.min_test_sites,
        )
        records.append(
            {
                "task": "site",
                "protocol": "grouped_site_greedy",
                "fold": int(fold),
                "train_idx": train_idx,
                "raw_test_idx": raw_test_idx,
                "filtered_test_idx": filtered_test_idx,
                "raw_test_n": int(len(raw_test)),
                "raw_test_sites": int(raw_test["SID"].nunique()) if not raw_test.empty else 0,
                "dropped_n": int(len(raw_test_idx) - len(filtered_test_idx)),
                "dropped_years": tuple(sorted(set(raw_test["year"]) - seen_years)),
                "valid_flag": bool(valid_flag),
                "invalid_reason": invalid_reason,
                **_summarize_filtered_test(train, filtered_test),
            }
        )
    return records


def _sample_protocol_records(df: pd.DataFrame, n_splits: int | None = None, seed: int | None = None) -> list[dict[str, object]]:
    config = get_project_config().experiment.sample
    n_splits = n_splits or config.n_splits
    rng = np.random.default_rng(config.seed if seed is None else seed)
    indices = np.arange(len(df), dtype=int)
    shuffled = rng.permutation(indices)
    groups = [group.astype(int) for group in np.array_split(shuffled, n_splits)]

    records: list[dict[str, object]] = []
    for fold, test_idx in enumerate(groups, start=1):
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        valid_flag, invalid_reason = _is_valid_fold(test, min_test_samples=config.min_test_samples)
        records.append(
            {
                "task": "sample",
                "protocol": "sample_random_kfold",
                "fold": int(fold),
                "train_idx": train_idx,
                "raw_test_idx": test_idx,
                "filtered_test_idx": test_idx,
                "raw_test_n": int(len(test)),
                "raw_test_sites": int(test["SID"].nunique()) if not test.empty else 0,
                "dropped_n": 0,
                "valid_flag": bool(valid_flag),
                "invalid_reason": invalid_reason,
                **_summarize_filtered_test(train, test),
            }
        )
    return records


def _year_protocol_records(df: pd.DataFrame, folds: tuple[YearFoldConfig, ...] | None = None) -> list[dict[str, object]]:
    config = get_project_config().experiment.year
    folds = folds or config.folds
    years = df["year"].to_numpy()
    records: list[dict[str, object]] = []
    for spec in folds:
        train_idx = np.where((years >= spec.train_start) & (years <= spec.train_end))[0]
        raw_test_idx = np.where((years >= spec.test_start) & (years <= spec.test_end))[0]
        train = df.iloc[train_idx]
        raw_test = df.iloc[raw_test_idx]
        seen_sites = set(train["SID"].tolist())
        filtered_test = raw_test[raw_test["SID"].isin(seen_sites)]
        filtered_test_idx = filtered_test.index.to_numpy(dtype=int)
        valid_flag, invalid_reason = _is_valid_fold(filtered_test, min_test_samples=config.min_test_samples)
        records.append(
            {
                "task": "year",
                "protocol": "rolling_origin_seen_sites",
                "fold": int(spec.fold),
                "train_idx": train_idx,
                "raw_test_idx": raw_test_idx,
                "filtered_test_idx": filtered_test_idx,
                "raw_test_n": int(len(raw_test)),
                "raw_test_sites": int(raw_test["SID"].nunique()) if not raw_test.empty else 0,
                "dropped_n": int(len(raw_test_idx) - len(filtered_test_idx)),
                "dropped_sites": tuple(sorted(set(raw_test["SID"]) - seen_sites)),
                "valid_flag": bool(valid_flag),
                "invalid_reason": invalid_reason,
                **_summarize_filtered_test(train, filtered_test),
            }
        )
    return records


def _protocol_records(df: pd.DataFrame, task: str, folds: tuple[YearFoldConfig, ...] | None = None) -> list[dict[str, object]]:
    if task == "sample":
        return _sample_protocol_records(df)
    if task == "site":
        return _site_protocol_records(df)
    if task == "year":
        return _year_protocol_records(df, folds=folds)
    raise ValueError(f"Unsupported task={task}")


def sample_random_splits(df: pd.DataFrame, n_splits: int | None = None, seed: int | None = None):
    for record in _sample_protocol_records(df, n_splits=n_splits, seed=seed):
        yield int(record["fold"]), record["train_idx"], record["filtered_test_idx"]


def site_extrapolation_splits(df: pd.DataFrame, n_splits: int | None = None, split_mode: str | None = None):
    if split_mode not in {None, "greedy"}:
        raise ValueError("Site extrapolation now uses a fixed greedy protocol.")
    for record in _site_protocol_records(df, n_splits=n_splits):
        yield int(record["fold"]), record["train_idx"], record["filtered_test_idx"]


def year_extrapolation_splits(df: pd.DataFrame, folds: tuple[YearFoldConfig, ...] | None = None):
    for record in _year_protocol_records(df, folds=folds):
        yield int(record["fold"]), record["train_idx"], record["filtered_test_idx"]


def collect_protocol_audit(df: pd.DataFrame, task: str, folds: tuple[YearFoldConfig, ...] | None = None) -> dict[str, pd.DataFrame]:
    records = _protocol_records(df, task, folds=folds)

    fold_rows = []
    year_rows = []
    site_rows = []
    for record in records:
        train = df.iloc[record["train_idx"]].reset_index(drop=True)
        raw_test = df.iloc[record["raw_test_idx"]].reset_index(drop=True)
        filtered_test = df.iloc[record["filtered_test_idx"]].reset_index(drop=True)
        fold_rows.append(
            {
                "task": str(record["task"]),
                "protocol": str(record["protocol"]),
                "fold": int(record["fold"]),
                "train_n": int(record["train_n"]),
                "raw_test_n": int(record["raw_test_n"]),
                "filtered_test_n": int(record["filtered_test_n"]),
                "train_sites": int(record["train_sites"]),
                "raw_test_sites": int(record["raw_test_sites"]),
                "filtered_test_sites": int(record["filtered_test_sites"]),
                "dropped_n": int(record["dropped_n"]),
                "valid_flag": bool(record["valid_flag"]),
                "invalid_reason": record["invalid_reason"],
                "train_year_min": record["train_year_min"],
                "train_year_max": record["train_year_max"],
                "filtered_test_year_min": record["filtered_test_year_min"],
                "filtered_test_year_max": record["filtered_test_year_max"],
            }
        )
        for partition, frame in [("train", train), ("raw_test", raw_test), ("filtered_test", filtered_test)]:
            if frame.empty:
                continue
            for year, count in frame.groupby("year").size().items():
                year_rows.append(
                    {
                        "task": str(record["task"]),
                        "protocol": str(record["protocol"]),
                        "fold": int(record["fold"]),
                        "partition": partition,
                        "year": int(year),
                        "n": int(count),
                    }
                )
        for partition, frame in [("raw_test", raw_test), ("filtered_test", filtered_test)]:
            if frame.empty:
                continue
            grouped = frame.groupby("SID").agg(test_n=("year", "size"), year_min=("year", "min"), year_max=("year", "max")).reset_index()
            for row in grouped.itertuples(index=False):
                site_rows.append(
                    {
                        "task": str(record["task"]),
                        "protocol": str(record["protocol"]),
                        "fold": int(record["fold"]),
                        "partition": partition,
                        "SID": int(row.SID),
                        "test_n": int(row.test_n),
                        "year_min": int(row.year_min),
                        "year_max": int(row.year_max),
                    }
                )

    fold_summary = pd.DataFrame(fold_rows)
    if fold_summary.empty:
        validity_summary = pd.DataFrame()
    else:
        validity_summary = (
            fold_summary.groupby(["task", "protocol"], as_index=False)
            .agg(
                n_folds=("fold", "nunique"),
                valid_folds=("valid_flag", "sum"),
                raw_test_n_min=("raw_test_n", "min"),
                raw_test_n_max=("raw_test_n", "max"),
                filtered_test_n_min=("filtered_test_n", "min"),
                filtered_test_n_max=("filtered_test_n", "max"),
                filtered_test_sites_min=("filtered_test_sites", "min"),
                filtered_test_sites_max=("filtered_test_sites", "max"),
                dropped_n_total=("dropped_n", "sum"),
            )
        )
        validity_summary["invalid_folds"] = validity_summary["n_folds"] - validity_summary["valid_folds"]

    return {
        "protocol_fold_audit": fold_summary,
        "protocol_year_coverage": pd.DataFrame(year_rows),
        "protocol_site_coverage": pd.DataFrame(site_rows),
        "protocol_validity_summary": validity_summary,
    }


def collect_split_audit(df: pd.DataFrame, task: str, split_mode: str | None = None) -> dict[str, pd.DataFrame]:
    del split_mode
    return collect_protocol_audit(df, task)
