from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.features.feature_factory import compute_features


CANONICAL_DIR = ROOT / "dataset/canonical/A_stock/2016-2025"
FEATURE_DIR = ROOT / "dataset/features/A_stock/2016-2025"

NORMATIVE_FEATURE_COLUMNS = {
    "liq__close_vwap_dev",
    "liq__turnover",
    "liq__turnover_z__60",
    "ret__adx__14",
    "ret__roc__20",
    "ret__roc__60",
    "ret__rsi__14",
    "rng__atr__14",
    "rng__atr_z__60",
    "rng__bb_width__20",
    "rng__close_off__20",
    "time__days_since_listed",
    "time__dow",
    "time__is_month_end",
    "time__month",
    "trn__adx_delta__1",
    "trn__bb_width_delta__1",
    "trn__rsi_delta__1",
    "trn__turnover_z_delta__1",
}


def _series_allclose(a: pd.Series, b: pd.Series, *, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    if not a.isna().equals(b.isna()):
        return False
    mask = ~a.isna()
    if mask.sum() == 0:
        return True
    return np.allclose(a[mask].to_numpy(), b[mask].to_numpy(), rtol=rtol, atol=atol)


def _pick_sample_symbol_path() -> Path:
    preferred = CANONICAL_DIR / "300422.parquet"
    if preferred.exists():
        return preferred
    files = sorted(CANONICAL_DIR.glob("*.parquet"))
    if not files:
        raise AssertionError(f"no canonical parquet files in {CANONICAL_DIR}")
    return files[0]


def _load_panel(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        raise AssertionError("no frames loaded for panel")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def gate1_deterministic() -> None:
    sample_path = _pick_sample_symbol_path()
    df_in = pd.read_parquet(sample_path)
    feat1 = compute_features(df_in)
    feat2 = compute_features(df_in)
    if feat1.shape[0] != feat2.shape[0]:
        raise AssertionError(f"row_count_mismatch: {feat1.shape[0]} vs {feat2.shape[0]}")
    if list(feat1.columns) != list(feat2.columns):
        raise AssertionError("columns_order_mismatch")
    if not feat1.isna().sum().equals(feat2.isna().sum()):
        raise AssertionError("nan_count_mismatch")
    if not feat1.isna().equals(feat2.isna()):
        raise AssertionError("nan_mask_mismatch")
    keys1 = set(zip(feat1["symbol"], feat1["date"]))
    keys2 = set(zip(feat2["symbol"], feat2["date"]))
    if keys1 != keys2:
        raise AssertionError("primary_key_set_mismatch")
    num_cols = [
        c
        for c in feat1.columns
        if pd.api.types.is_numeric_dtype(feat1[c]) and not pd.api.types.is_bool_dtype(feat1[c])
    ]
    for c in num_cols:
        s1 = feat1[c]
        s2 = feat2[c]
        mask = s1.notna() & s2.notna()
        v1 = s1[mask].to_numpy()
        v2 = s2[mask].to_numpy()
        if v1.size == 0:
            continue
        if not np.allclose(v1, v2, rtol=1e-10, atol=1e-12):
            diff = v1 - v2
            max_abs = float(np.max(np.abs(diff)))
            raise AssertionError(f"numeric_mismatch_{c}_max_abs={max_abs}")


def gate2_prefix_invariance() -> None:
    sample_path = _pick_sample_symbol_path()
    df_full = pd.read_parquet(sample_path)
    if df_full.empty:
        raise AssertionError("empty input for prefix invariance")
    df_full = df_full.sort_values("date")
    n = len(df_full)
    feat_full = compute_features(df_full)
    key_cols = ["symbol", "date"]
    feat_full = feat_full.sort_values(key_cols).reset_index(drop=True)
    input_cols = list(df_full.columns)
    feature_cols = [c for c in feat_full.columns if c not in input_cols]
    df_full_idx = feat_full.set_index(key_cols)
    if df_full_idx.index.has_duplicates:
        raise AssertionError("duplicate (symbol,date) keys detected in full feature panel")
    ts = sorted(set([max(70, n // 2), n - 10, 20, 60, 120]))
    ts = [x for x in ts if 10 < x < n]
    if not ts:
        return
    for t in ts:
        df_prefix = df_full.iloc[:t].copy()
        feat_prefix = compute_features(df_prefix)
        feat_prefix = feat_prefix.sort_values(key_cols).reset_index(drop=True)
        df_pref_idx = feat_prefix.set_index(key_cols)
        if df_pref_idx.index.has_duplicates:
            raise AssertionError("duplicate (symbol,date) keys detected in prefix feature panel")
        common_index = df_pref_idx.index
        df_full_common = df_full_idx.loc[common_index]
        for c in feature_cols:
            s1 = df_pref_idx[c]
            s2 = df_full_common[c]
            if not _series_allclose(s1, s2):
                raise AssertionError(f"prefix_invariance_mismatch_{c}_t={t}")


def gate3_nan_boundary() -> None:
    sample_path = _pick_sample_symbol_path()
    df_in = pd.read_parquet(sample_path)
    feat = compute_features(df_in)
    sym_groups = feat.groupby("symbol")
    for _, g in sym_groups:
        g = g.sort_values("date")
        if len(g) == 0:
            continue
        if not g["ret__roc__20"].head(20).isna().all():
            raise AssertionError("ret__roc__20_warmup_nan_violation")
        if not g["ret__roc__60"].head(60).isna().all():
            raise AssertionError("ret__roc__60_warmup_nan_violation")
        if not g["ret__rsi__14"].head(14).isna().all():
            raise AssertionError("ret__rsi__14_warmup_nan_violation")
        if not g["ret__adx__14"].head(14).isna().all():
            raise AssertionError("ret__adx__14_warmup_nan_violation")
        if not g["rng__atr__14"].head(14).isna().all():
            raise AssertionError("rng__atr__14_warmup_nan_violation")
        if not g["rng__atr_z__60"].head(59).isna().all():
            raise AssertionError("rng__atr_z__60_warmup_nan_violation")
        if not g["liq__turnover_z__60"].head(59).isna().all():
            raise AssertionError("liq__turnover_z__60_warmup_nan_violation")
        if not g["rng__bb_width__20"].head(19).isna().all():
            raise AssertionError("rng__bb_width__20_warmup_nan_violation")
        if not g["rng__close_off__20"].head(19).isna().all():
            raise AssertionError("rng__close_off__20_warmup_nan_violation")
        if not pd.isna(g["trn__rsi_delta__1"].iloc[0]):
            raise AssertionError("trn__rsi_delta__1_first_not_nan")
        if not pd.isna(g["trn__adx_delta__1"].iloc[0]):
            raise AssertionError("trn__adx_delta__1_first_not_nan")
        if not pd.isna(g["trn__bb_width_delta__1"].iloc[0]):
            raise AssertionError("trn__bb_width_delta__1_first_not_nan")
        if not pd.isna(g["trn__turnover_z_delta__1"].iloc[0]):
            raise AssertionError("trn__turnover_z_delta__1_first_not_nan")
    feature_cols = [
        c
        for c in feat.columns
        if c not in df_in.columns
    ]
    for c in feature_cols:
        vals = feat[c].to_numpy()
        if np.isinf(vals).any():
            raise AssertionError(f"inf_in_feature_{c}")


def gate4_cross_symbol_scale() -> None:
    files = sorted(CANONICAL_DIR.glob("*.parquet"))
    if not files:
        raise AssertionError("no canonical parquet files for Gate4")
    if len(files) <= 50:
        sampled_files = files
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(files), size=50, replace=False)
        indices = sorted(indices.tolist())
        sampled_files = [files[i] for i in indices]
    single_path = sampled_files[0]
    single_df = pd.read_parquet(single_path)
    single_feat = compute_features(single_df)
    single_cols = list(single_feat.columns)
    small_paths = sampled_files[: min(5, len(sampled_files))]
    small_panel = _load_panel(small_paths)
    small_feat = compute_features(small_panel)
    if list(small_feat.columns) != single_cols:
        raise AssertionError("schema_mismatch_small_vs_single")
    if small_feat.duplicated(subset=["symbol", "date"]).any():
        raise AssertionError("duplicate_keys_in_small_batch")
    sym = str(single_df["symbol"].iloc[0])
    single_sym = single_feat[single_feat["symbol"] == sym].sort_values("date")
    small_sym = small_feat[small_feat["symbol"] == sym].sort_values("date")
    if not single_sym["date"].equals(small_sym["date"]):
        raise AssertionError("date_mismatch_for_symbol_in_small_batch")
    for c in single_cols:
        s1 = single_sym[c].reset_index(drop=True)
        s2 = small_sym[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(s1):
            mask = s1.notna() & s2.notna()
            v1 = s1[mask].to_numpy()
            v2 = s2[mask].to_numpy()
            if v1.size == 0:
                continue
            if not np.allclose(v1, v2, rtol=1e-10, atol=1e-12):
                raise AssertionError(f"cross_symbol_mismatch_small_{c}")
        else:
            if not s1.equals(s2):
                raise AssertionError(f"cross_symbol_mismatch_small_{c}")
    full_paths = sampled_files
    full_panel = _load_panel(full_paths)
    full_feat = compute_features(full_panel)
    if list(full_feat.columns) != single_cols:
        raise AssertionError("schema_mismatch_full_vs_single")
    if full_feat.duplicated(subset=["symbol", "date"]).any():
        raise AssertionError("duplicate_keys_in_full_batch")


def gate5_schema_feature_columns() -> None:
    feature_files = sorted(FEATURE_DIR.glob("*.parquet"))
    if not feature_files:
        raise AssertionError(f"no feature parquet files in {FEATURE_DIR}")
    canonical_files = {p.name: p for p in CANONICAL_DIR.glob("*.parquet")}
    errors: list[str] = []
    for feat_path in feature_files:
        name = feat_path.name
        canon_path = canonical_files.get(name)
        if canon_path is None:
            errors.append(f"missing_canonical_for_{name}")
            continue
        feat_cols = {field.name for field in pq.read_schema(feat_path)}
        canon_cols = {field.name for field in pq.read_schema(canon_path)}
        feature_only = feat_cols - canon_cols
        canonical_missing = canon_cols - feat_cols
        if canonical_missing:
            errors.append(f"{name}:missing_canonical_columns:{sorted(canonical_missing)}")
        if feature_only != NORMATIVE_FEATURE_COLUMNS:
            extra = sorted(feature_only - NORMATIVE_FEATURE_COLUMNS)
            missing = sorted(NORMATIVE_FEATURE_COLUMNS - feature_only)
            errors.append(f"{name}:feature_schema_mismatch:extra={extra},missing={missing}")
    if errors:
        raise AssertionError(";".join(errors))


def main() -> int:
    gates = [
        ("Gate1_Deterministic", gate1_deterministic),
        ("Gate2_PrefixInvariance", gate2_prefix_invariance),
        ("Gate3_NanBoundary", gate3_nan_boundary),
        ("Gate4_CrossSymbolScale", gate4_cross_symbol_scale),
        ("Gate5_SchemaFeatureColumns", gate5_schema_feature_columns),
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in gates:
        try:
            fn()
            print(f"{name}: PASS")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"{name}: FAIL: {exc}")
    if failures:
        for name, msg in failures:
            print(f"[Result] {name} -> FAIL: {msg}")
        return 1
    print("[Result] All FeatureFactory gates PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
