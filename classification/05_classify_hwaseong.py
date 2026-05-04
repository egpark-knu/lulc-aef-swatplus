"""
Phase 2 -- Task 2+3+4: Hwaseong/Dongtan AEF Classification (Spatial Block CV)

Replicates the Sejong Phase 1 pipeline for a rapidly urbanizing area:
  - Hwaseong-si / Dongtan New Town (2017-2024 massive development)

Pipeline:
  1. AEF 2021 + WorldCover 2021 + DW 2021 sample extraction (4000 pts)
  2. 6-class mapping (same WC_TO_6CLASS as Sejong)
  3. Stable pixel mask (WC intersect DW agreement)
  4. Spatial Block CV (250m, 500m, 1km, 2km blocks) -- G1 Gate
  5. Cross-year transfer 2017-2024 (fixed points, same seed=42)
  6. Monotonic urban constraint (corrected fractions)
  7. G2 Gate (Built-up Delta, DW overlap, Urban trend)
  8. Go/No-Go verdict
  9. Save results to phase2/data/hwaseong/

Memory Safety:
  - Thread caps (OPENBLAS, OMP, MKL = 1)
  - .limit(4000) hard cap on getInfo()
  - gc.collect() + ee.Reset() + ee.Initialize() per year
  - check_memory() monitoring
  - Cache: phase2/data/hwaseong/hwaseong_2021_samples.npz
  - Transfer cache: phase2/data/hwaseong/transfer_years/*.json

Usage:
    cd /Users/eungyupark/Dropbox/myproj/dev_260402_LULC
    /Users/eungyupark/anaconda3/envs/newproj/bin/python phase2/scripts/hwaseong_t234_classification.py
"""

import os
import sys

# ── Thread Safety (MUST be before numpy/sklearn imports) ──
DEFAULT_SAFE_MODE = (
    os.environ.get("T234_SAFE_MODE", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
MAX_THREADS = int(os.environ.get("T234_MAX_THREADS", "1" if DEFAULT_SAFE_MODE else "4"))
for _var in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_var, str(MAX_THREADS))

import ee
import gc
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# ── Logging ──
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "hwaseong_classification.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("hwaseong_t234")

# ── Memory Safety ──
MEM_LIMIT_GB = 180  # 192GB system, 12GB headroom


def check_memory(label: str = "") -> bool:
    """Print current process memory usage. Warn if over limit."""
    try:
        import psutil

        proc = psutil.Process()
        mem_gb = proc.memory_info().rss / 1e9
        sys_mem = psutil.virtual_memory()
        sys_used_gb = sys_mem.used / 1e9
        log.info(
            "  [MEM] %s: process=%.1fGB, system=%.1f/%.0fGB",
            label, mem_gb, sys_used_gb, sys_mem.total / 1e9,
        )
        if sys_used_gb > MEM_LIMIT_GB:
            log.warning(
                "  System memory %.0fGB > %.0fGB limit!", sys_used_gb, MEM_LIMIT_GB
            )
            return False
        return True
    except ImportError:
        return True  # psutil not installed -- skip


# ── Helper: env flags and year spec (inline to avoid cross-package import) ──
def env_flag(raw, default=False):
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_years_spec(spec, default_years):
    if spec is None or not spec.strip():
        return list(default_years)
    years = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            years.update(range(int(start_s), int(end_s) + 1))
        else:
            years.add(int(token))
    return sorted(years)


def year_cache_path(cache_dir, year):
    return cache_dir / f"{year}.json"


def pending_years(years, cache_dir, force_rerun=False):
    if force_rerun:
        return list(years)
    return [y for y in years if not year_cache_path(cache_dir, y).exists()]


# ══════════════════════════════════════════════════════════
# CONFIGURATION -- Hwaseong / Dongtan
# ══════════════════════════════════════════════════════════
HWASEONG_BBOX = [126.85, 37.05, 127.25, 37.35]  # west, south, east, north

ee.Initialize()
HWASEONG_AOI = ee.Geometry.Rectangle(HWASEONG_BBOX)

# Sampling parameters
SAMPLE_SIZE = 4000
SAMPLE_LIMIT = 4000  # hard cap via .limit()
BLOCK_SIZES = [250, 500, 1000, 2000]  # meters
N_FOLDS = 5
YEARS = list(range(2017, 2025))  # 2017-2024
SAFE_N_JOBS = int(os.environ.get("T234_N_JOBS", "1" if DEFAULT_SAFE_MODE else "-1"))

FORCE_SAMPLE_REBUILD = env_flag(os.environ.get("T234_FORCE_SAMPLE_REBUILD"), default=False)
FORCE_RERUN_YEARS = env_flag(os.environ.get("T234_FORCE_RERUN_YEARS"), default=False)
SKIP_CROSS_YEAR = env_flag(os.environ.get("T234_SKIP_CROSS_YEAR"), default=False)
TARGET_YEARS = parse_years_spec(os.environ.get("T234_TARGET_YEARS"), YEARS)

# WorldCover -> 6-class mapping (identical to Sejong)
WC_TO_6CLASS = {
    10: 0,   # Tree cover -> Forest
    20: 3,   # Shrubland -> Grassland (merge)
    30: 3,   # Grassland -> Grassland
    40: 1,   # Cropland -> Cropland
    50: 2,   # Built-up -> Urban
    60: 5,   # Bare -> Barren
    70: -1,  # Snow -> remove
    80: 4,   # Water -> Water
    90: 4,   # Wetland -> Water (merge)
    95: -1,  # Mangrove -> remove
    100: -1, # Moss -> remove
}

CLASS_NAMES = ["Forest", "Cropland", "Urban", "Grassland", "Water", "Barren"]
SWAT_CODES = ["FRSE", "AGRR", "URLD", "PAST", "WATR", "SWRN"]

# DW label -> 6-class mapping (identical to Sejong)
DW_TO_6CLASS = {
    0: 4,   # water -> Water
    1: 0,   # trees -> Forest
    2: 3,   # grass -> Grassland
    3: 4,   # flooded_veg -> Water
    4: 1,   # crops -> Cropland
    5: 3,   # shrub_scrub -> Grassland
    6: 2,   # built -> Urban
    7: 5,   # bare -> Barren
    8: -1,  # snow_ice -> remove
}

# Output directories
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hwaseong"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSFER_DIR = DATA_DIR / "transfer_years"
TRANSFER_DIR.mkdir(exist_ok=True)
SAMPLE_CACHE = DATA_DIR / "hwaseong_2021_samples.npz"

# Coordinate conversion constants for spatial blocking
# Hwaseong center: ~37.2N, ~127.05E
# 1 deg lat ~ 111 km, 1 deg lon ~ 88.3 km at 37.2N
LAT_M = 111_000  # m per degree latitude
LON_M = 88_300   # m per degree longitude at ~37.2N
COORD_ORIGIN_LON = 126.85  # west edge of bbox
COORD_ORIGIN_LAT = 37.05   # south edge of bbox


# ══════════════════════════════════════════════════════════
# RUNTIME BANNER
# ══════════════════════════════════════════════════════════
def print_banner():
    log.info("=" * 60)
    log.info("Hwaseong/Dongtan AEF Classification Pipeline")
    log.info("=" * 60)
    log.info("  AOI: Hwaseong bbox %s", HWASEONG_BBOX)
    log.info("  safe_mode=%s, max_threads=%s, sklearn_n_jobs=%s",
             DEFAULT_SAFE_MODE, MAX_THREADS, SAFE_N_JOBS)
    log.info("  target_years=%s", TARGET_YEARS)
    log.info("  force_sample_rebuild=%s, force_rerun_years=%s",
             FORCE_SAMPLE_REBUILD, FORCE_RERUN_YEARS)
    log.info("  sample_cache=%s", SAMPLE_CACHE)
    log.info("  data_dir=%s", DATA_DIR)


# ══════════════════════════════════════════════════════════
# STEP 1: Extract AEF 2021 + WorldCover + DW samples
# ══════════════════════════════════════════════════════════
def step1_extract_samples():
    log.info("=" * 60)
    log.info("STEP 1: AEF 2021 + WorldCover + DW Sample Extraction")
    log.info("=" * 60)

    if SAMPLE_CACHE.exists() and not FORCE_SAMPLE_REBUILD:
        log.info("  Loading cache: %s", SAMPLE_CACHE)
        cached = np.load(SAMPLE_CACHE)
        X = cached["X"].astype(np.float32, copy=False)
        y = cached["y"].astype(np.int16, copy=False)
        y_dw = cached["y_dw"].astype(np.int16, copy=False)
        coords = cached["coords"].astype(np.float32, copy=False)
        stable_mask = cached["stable_mask"].astype(bool, copy=False)
        return X, y, y_dw, coords, stable_mask

    # AEF 2021
    aef_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    aef_2021 = (
        aef_col.filter(ee.Filter.calendarRange(2021, 2021, "year"))
        .filterBounds(HWASEONG_AOI)
        .mosaic()
        .clip(HWASEONG_AOI)
    )

    # WorldCover v200
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(HWASEONG_AOI)
    wc_label = wc.select("Map").rename("WC")

    # DynamicWorld 2021 mode
    dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    dw_2021_mode = (
        dw_col.filter(ee.Filter.calendarRange(2021, 2021, "year"))
        .filterBounds(HWASEONG_AOI)
        .select("label")
        .mode()
        .clip(HWASEONG_AOI)
        .rename("DW")
    )

    combined = aef_2021.addBands(wc_label).addBands(dw_2021_mode)

    log.info("  Sampling... (numPixels=%d, hard limit=%d)", SAMPLE_SIZE, SAMPLE_LIMIT)
    sample = combined.sample(
        region=HWASEONG_AOI,
        scale=10,
        numPixels=SAMPLE_SIZE,
        seed=42,
        geometries=True,
    ).limit(SAMPLE_LIMIT)

    features = sample.getInfo()["features"]
    log.info("  Extracted: %d samples", len(features))
    check_memory("after getInfo")

    # Parse features
    embeddings_list = []
    wc_labels = []
    dw_labels = []
    coords_list = []

    for feat in features:
        props = feat["properties"]
        wc_val = props.get("WC")
        dw_val = props.get("DW")
        if wc_val is None or dw_val is None:
            continue

        emb = [props.get(f"A{i:02d}", 0.0) for i in range(64)]
        embeddings_list.append(emb)
        wc_labels.append(int(wc_val))
        dw_labels.append(int(dw_val))
        coords_list.append(feat["geometry"]["coordinates"])

    X_raw = np.asarray(embeddings_list, dtype=np.float32)
    y_wc_raw = np.asarray(wc_labels, dtype=np.int16)
    y_dw_raw = np.asarray(dw_labels, dtype=np.int16)
    coords_raw = np.asarray(coords_list, dtype=np.float32)

    # Free raw GeoJSON immediately
    del features, embeddings_list, wc_labels, dw_labels, coords_list, combined, sample
    gc.collect()

    log.info("  Valid samples (pre-filter): %d", len(X_raw))

    # 6-class mapping
    y_6class = np.asarray([WC_TO_6CLASS.get(v, -1) for v in y_wc_raw], dtype=np.int16)
    y_dw_6class = np.asarray([DW_TO_6CLASS.get(v, -1) for v in y_dw_raw], dtype=np.int16)

    # Remove unmapped (-1)
    valid_mask = (y_6class >= 0) & (y_dw_6class >= 0)
    X = X_raw[valid_mask].astype(np.float32, copy=False)
    y = y_6class[valid_mask].astype(np.int16, copy=False)
    y_dw = y_dw_6class[valid_mask].astype(np.int16, copy=False)
    coords = coords_raw[valid_mask].astype(np.float32, copy=False)

    stable_mask = y == y_dw
    np.savez(
        SAMPLE_CACHE,
        X=X, y=y, y_dw=y_dw, coords=coords, stable_mask=stable_mask,
    )
    log.info("  Cache saved: %s", SAMPLE_CACHE)

    del X_raw, y_wc_raw, y_dw_raw, coords_raw, y_6class, y_dw_6class, valid_mask
    gc.collect()

    return X, y, y_dw, coords, stable_mask


def print_sample_stats(X, y, y_dw, stable_mask):
    log.info("  6-class valid samples: %d", len(X))
    log.info("")
    log.info("  Class distribution (WorldCover 6-class):")
    for i, name in enumerate(CLASS_NAMES):
        n = int((y == i).sum())
        log.info("    %d: %-12s (%s): %6d (%5.1f%%)", i, name, SWAT_CODES[i], n, n / len(y) * 100)

    n_stable = int(stable_mask.sum())
    n_change = int((~stable_mask).sum())
    log.info("")
    log.info("  Stable pixels (WC & DW agreement): %d (%.1f%%)", n_stable, n_stable / len(y) * 100)
    log.info("  Change candidates (disagreement):   %d (%.1f%%)", n_change, n_change / len(y) * 100)
    check_memory("after sample cache ready")


# ══════════════════════════════════════════════════════════
# STEP 2: Spatial Block CV (4 block sizes)
# ══════════════════════════════════════════════════════════
def step2_spatial_block_cv(X, y, coords, stable_mask):
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Spatial Block CV -- Linear Probe")
    log.info("=" * 60)

    X_stable = X[stable_mask]
    y_stable = y[stable_mask]
    coords_stable = coords[stable_mask]

    log.info("  Train/val data: %d stable pixels", len(X_stable))

    # Convert coordinates to meters (approximate)
    coords_m = np.zeros_like(coords_stable)
    coords_m[:, 0] = (coords_stable[:, 0] - COORD_ORIGIN_LON) * LON_M
    coords_m[:, 1] = (coords_stable[:, 1] - COORD_ORIGIN_LAT) * LAT_M

    block_cv_results = {}

    for block_size in BLOCK_SIZES:
      try:
        log.info("")
        log.info("  Block size: %dm", block_size)

        # Block ID assignment
        block_ids = (
            (coords_m[:, 0] // block_size).astype(int) * 10000
            + (coords_m[:, 1] // block_size).astype(int)
        )

        n_blocks = len(np.unique(block_ids))
        log.info("    Blocks: %d", n_blocks)

        if n_blocks < N_FOLDS:
            log.warning("    Block count < %d, skipping", N_FOLDS)
            continue

        gkf = GroupKFold(n_splits=N_FOLDS)

        fold_oas = []
        fold_kappas = []
        all_preds = np.full(len(y_stable), -1, dtype=int)

        for fold_i, (train_idx, test_idx) in enumerate(
            gkf.split(X_stable, y_stable, groups=block_ids)
        ):
            X_tr, X_te = X_stable[train_idx], X_stable[test_idx]
            y_tr, y_te = y_stable[train_idx], y_stable[test_idx]

            # Inner CV for C tuning: also spatial (GroupKFold)
            inner_block_ids = block_ids[train_idx]
            n_inner_blocks = len(np.unique(inner_block_ids))
            inner_n_splits = min(3, n_inner_blocks)

            if inner_n_splits >= 2:
                inner_gkf = GroupKFold(n_splits=inner_n_splits)
                inner_cv = list(inner_gkf.split(X_tr, y_tr, groups=inner_block_ids))
            else:
                inner_cv = 3  # fallback to stratified

            clf = LogisticRegressionCV(
                Cs=np.logspace(-3, 3, 10),
                cv=inner_cv,
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42,
                n_jobs=SAFE_N_JOBS,
            )
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_te)
            all_preds[test_idx] = y_pred

            oa = accuracy_score(y_te, y_pred)
            kappa = cohen_kappa_score(y_te, y_pred)
            fold_oas.append(oa)
            fold_kappas.append(kappa)
            log.info(
                "    Fold %d: OA=%.4f, Kappa=%.4f, best_C=%.4f, train=%d, test=%d",
                fold_i + 1, oa, kappa, clf.C_[0], len(X_tr), len(X_te),
            )

        mean_oa = np.mean(fold_oas)
        std_oa = np.std(fold_oas)
        mean_kappa = np.mean(fold_kappas)
        std_kappa = np.std(fold_kappas)

        log.info("    -> Mean OA: %.4f +/- %.4f", mean_oa, std_oa)
        log.info("    -> Mean Kappa: %.4f +/- %.4f", mean_kappa, std_kappa)

        # Classification report (aggregated over all folds)
        valid_pred = all_preds >= 0
        if valid_pred.sum() > 0:
            present_labels = sorted(set(y_stable[valid_pred]) | set(all_preds[valid_pred]))
            present_names = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]
            cr = classification_report(
                y_stable[valid_pred],
                all_preds[valid_pred],
                labels=present_labels,
                target_names=present_names,
                output_dict=True,
                zero_division=0,
            )
            log.info("")
            log.info("    Class-wise F1 (%dm block):", block_size)
            for cls_name in present_names:
                if cls_name in cr:
                    f1 = cr[cls_name]["f1-score"]
                    sup = cr[cls_name]["support"]
                    log.info("      %-12s: F1=%.3f (n=%d)", cls_name, f1, sup)

        block_cv_results[block_size] = {
            "mean_oa": float(mean_oa),
            "std_oa": float(std_oa),
            "mean_kappa": float(mean_kappa),
            "std_kappa": float(std_kappa),
            "fold_oas": [float(x) for x in fold_oas],
            "fold_kappas": [float(x) for x in fold_kappas],
            "n_blocks": int(n_blocks),
        }
      except Exception as e:
        log.warning("    Block size %dm FAILED: %s — skipping", block_size, e)
        continue

    return block_cv_results, X_stable, y_stable, coords_m


# ══════════════════════════════════════════════════════════
# G1 GATE: Block CV Intermediate Decision
# ══════════════════════════════════════════════════════════
def g1_gate_check(block_cv_results):
    log.info("")
    log.info("=" * 60)
    log.info("G1 GATE: Spatial Block CV Intermediate Decision")
    log.info("=" * 60)

    gate_block = 1000
    if gate_block not in block_cv_results:
        gate_block = min(block_cv_results.keys()) if block_cv_results else None

    if gate_block is None:
        log.error("  No Block CV results -- STOP")
        sys.exit(1)

    gate_oa = block_cv_results[gate_block]["mean_oa"]
    gate_kappa = block_cv_results[gate_block]["mean_kappa"]

    if gate_oa >= 0.80 and gate_kappa >= 0.60:
        g1_gate = "PASS"
        log.info("  G1 PASS: OA=%.4f (>=0.80), Kappa=%.4f (>=0.60)", gate_oa, gate_kappa)
        log.info("  -> Proceeding to T4b (Cross-year Transfer)")
    elif gate_oa >= 0.70 and gate_kappa >= 0.40:
        g1_gate = "CONDITIONAL"
        log.info("  G1 CONDITIONAL: OA=%.4f (>=0.70), Kappa=%.4f (>=0.40)", gate_oa, gate_kappa)
        log.info("  -> Proceeding with caution to Cross-year Transfer")
    else:
        g1_gate = "FAIL"
        log.info("  G1 FAIL: OA=%.4f (<0.70), Kappa=%.4f", gate_oa, gate_kappa)
        log.info("  -> No cross-year. Review AOI or reduce classes.")

        partial_results = {
            "block_cv": {str(k): v for k, v in block_cv_results.items()},
            "g1_gate": g1_gate,
            "g1_oa": float(gate_oa),
            "g1_kappa": float(gate_kappa),
            "verdict": "NO-GO (G1 Gate Fail)",
        }
        results_path = DATA_DIR / "hwaseong_classification_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(partial_results, f, indent=2, ensure_ascii=False)
        log.info("  Results saved: %s", results_path)
        sys.exit(0)

    return g1_gate, gate_oa, gate_kappa, gate_block


# ══════════════════════════════════════════════════════════
# STEP 3: Cross-year Transfer (2021 model -> 2017-2024)
# ══════════════════════════════════════════════════════════
def step3_cross_year_transfer(X_stable, y_stable, coords_m, gate_block):
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3 (T4b): Cross-year Transfer (2021 model -> 2017-2024)")
    log.info("=" * 60)

    # Train final model: find best C from spatial CV on 1km blocks
    gate_block_ids = (
        (coords_m[:, 0] // gate_block).astype(int) * 10000
        + (coords_m[:, 1] // gate_block).astype(int)
    )
    gkf_final = GroupKFold(n_splits=N_FOLDS)
    best_Cs = []

    for train_idx, test_idx in gkf_final.split(X_stable, y_stable, groups=gate_block_ids):
        inner_block_ids = gate_block_ids[train_idx]
        n_inner = len(np.unique(inner_block_ids))
        inner_splits = min(3, n_inner)

        if inner_splits >= 2:
            inner_gkf = GroupKFold(n_splits=inner_splits)
            inner_cv = list(
                inner_gkf.split(X_stable[train_idx], y_stable[train_idx], groups=inner_block_ids)
            )
        else:
            inner_cv = 3

        clf_tmp = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            cv=inner_cv,
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
            n_jobs=SAFE_N_JOBS,
        )
        clf_tmp.fit(X_stable[train_idx], y_stable[train_idx])
        best_Cs.append(clf_tmp.C_[0])

    final_C = float(np.median(best_Cs))
    clf_final = LogisticRegression(
        C=final_C,
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
        n_jobs=SAFE_N_JOBS,
    )
    clf_final.fit(X_stable, y_stable)
    log.info(
        "  Final model: C=%.4f (median of spatial CV), trained on %d samples",
        final_C, len(X_stable),
    )
    del best_Cs, clf_tmp
    gc.collect()

    # Load existing cached transfer results
    transfer_results = {}
    for cached_path in sorted(TRANSFER_DIR.glob("*.json")):
        with open(cached_path, encoding="utf-8") as f:
            cached_result = json.load(f)
        transfer_results[int(cached_result["year"])] = {
            k: v for k, v in cached_result.items() if k != "year"
        }

    years_to_run = (
        []
        if SKIP_CROSS_YEAR
        else pending_years(TARGET_YEARS, cache_dir=TRANSFER_DIR, force_rerun=FORCE_RERUN_YEARS)
    )
    log.info("  Years to run: %s", years_to_run)
    if transfer_results:
        log.info("  Cached years: %s", sorted(transfer_results))
    if SKIP_CROSS_YEAR:
        log.info("  Cross-year transfer disabled via env var")

    # Generate fixed sample points (same locations across all years)
    fixed_points = None
    if not SKIP_CROSS_YEAR and years_to_run:
        log.info("  Generating fixed sample points (seed=42, n=%d)...", SAMPLE_SIZE)
        fixed_points = (
            ee.Image.constant(1)
            .clip(HWASEONG_AOI)
            .sample(
                region=HWASEONG_AOI,
                scale=10,
                numPixels=SAMPLE_SIZE,
                seed=42,
                geometries=True,
            )
            .limit(SAMPLE_LIMIT)
        )
        log.info("  -> Fixed point count: %d", fixed_points.size().getInfo())

    for year in years_to_run:
        log.info("")
        log.info("  %d:", year)
        if not check_memory(f"before {year}"):
            log.warning("  Memory low -- stopping cross-year")
            break

        try:
            aef_year = (
                ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                .filter(ee.Filter.calendarRange(year, year, "year"))
                .filterBounds(HWASEONG_AOI)
                .mosaic()
                .clip(HWASEONG_AOI)
            )

            # DW mode for this year (proxy truth)
            dw_year_mode = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filter(ee.Filter.calendarRange(year, year, "year"))
                .filterBounds(HWASEONG_AOI)
                .select("label")
                .mode()
                .clip(HWASEONG_AOI)
                .rename("DW")
            )

            combined_year = aef_year.addBands(dw_year_mode)

            # Extract values at fixed points
            sample_year = combined_year.sampleRegions(
                collection=fixed_points,
                scale=10,
                geometries=False,
            ).limit(SAMPLE_LIMIT)

            feats_year = sample_year.getInfo()["features"]
            log.info("    Samples: %d", len(feats_year))
            check_memory(f"after {year} getInfo")

            if len(feats_year) == 0:
                continue

            X_year = []
            y_dw_year = []
            for feat in feats_year:
                props = feat["properties"]
                dw_val = props.get("DW")
                if dw_val is None:
                    continue
                emb = [props.get(f"A{i:02d}", 0.0) for i in range(64)]
                X_year.append(emb)
                y_dw_year.append(DW_TO_6CLASS.get(int(dw_val), -1))

            X_year = np.asarray(X_year, dtype=np.float32)
            y_dw_year = np.asarray(y_dw_year, dtype=np.int16)

            valid = y_dw_year >= 0
            X_year = X_year[valid]
            y_dw_year = y_dw_year[valid]

            if len(X_year) == 0:
                continue

            # Predict with 2021 model
            y_pred_year = clf_final.predict(X_year)

            # Class fractions
            fractions = {}
            for i, name in enumerate(CLASS_NAMES):
                frac = (y_pred_year == i).sum() / len(y_pred_year) * 100
                fractions[name] = float(frac)

            # DW proxy agreement
            dw_agreement = accuracy_score(y_dw_year, y_pred_year)

            # Urban-specific IoU
            urban_pred = y_pred_year == 2
            urban_dw = y_dw_year == 2
            urban_tp = int((urban_pred & urban_dw).sum())
            urban_fp = int((urban_pred & ~urban_dw).sum())
            urban_fn = int((~urban_pred & urban_dw).sum())
            urban_iou = urban_tp / max(urban_tp + urban_fp + urban_fn, 1)

            log.info(
                "    DW proxy agreement: %.3f, Urban IoU: %.3f",
                dw_agreement, urban_iou,
            )
            log.info(
                "    Class fractions: %s",
                ", ".join(f"{n}={fractions[n]:.1f}%" for n in CLASS_NAMES if fractions[n] > 1),
            )

            year_result = {
                "n_samples": int(len(X_year)),
                "dw_agreement": float(dw_agreement),
                "urban_iou": float(urban_iou),
                "class_fractions": fractions,
            }
            transfer_results[year] = year_result
            with open(year_cache_path(TRANSFER_DIR, year), "w", encoding="utf-8") as f:
                json.dump({"year": year, **year_result}, f, indent=2, ensure_ascii=False)

            # Memory cleanup per year
            del aef_year, dw_year_mode, combined_year, sample_year
            del feats_year, X_year, y_dw_year, y_pred_year, valid
            del fractions, urban_pred, urban_dw
            gc.collect()
            ee.Reset()
            ee.Initialize()

            if not check_memory(f"after {year}"):
                log.warning("  Memory limit exceeded -- skipping remaining years")
                break

        except Exception as e:
            log.error("    %d failed: %s", year, e)
            gc.collect()
            ee.Reset()
            ee.Initialize()
            break

    return transfer_results, clf_final


# ══════════════════════════════════════════════════════════
# STEP 4: G1/G2 Go/No-Go Verdict
# ══════════════════════════════════════════════════════════
def step4_verdict(block_cv_results, transfer_results, g1_gate, X_stable, X):
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4: Go/No-Go Verdict")
    log.info("=" * 60)

    # G1 check
    best_block = 1000
    if best_block not in block_cv_results:
        best_block = min(block_cv_results.keys())

    g1_oa = block_cv_results[best_block]["mean_oa"]
    g1_kappa = block_cv_results[best_block]["mean_kappa"]
    g1_pass = g1_oa >= 0.80 and g1_kappa >= 0.60

    log.info(
        "  G1 (Classification): OA=%.4f (>=0.80? %s), Kappa=%.4f (>=0.60? %s) -> %s",
        g1_oa,
        "YES" if g1_oa >= 0.80 else "NO",
        g1_kappa,
        "YES" if g1_kappa >= 0.60 else "NO",
        "PASS" if g1_pass else "FAIL",
    )

    # Monotonic Non-Decreasing Constraint (Urban)
    urban_trend_raw = []
    for yr in sorted(transfer_results.keys()):
        uf = transfer_results[yr]["class_fractions"].get("Urban", 0)
        urban_trend_raw.append((yr, uf))

    urban_trend_corrected = []
    if urban_trend_raw:
        running_max = urban_trend_raw[0][1]
        for yr, uf in urban_trend_raw:
            corrected = max(uf, running_max)
            urban_trend_corrected.append((yr, corrected))
            running_max = corrected

    if urban_trend_raw:
        log.info("")
        log.info("  Urban Fraction -- Monotonic Constraint:")
        log.info(
            "    Raw:       %s",
            ", ".join(f"{yr}:{f:.1f}%" for yr, f in urban_trend_raw),
        )
        log.info(
            "    Corrected: %s",
            ", ".join(f"{yr}:{f:.1f}%" for yr, f in urban_trend_corrected),
        )
        n_corrected = sum(
            1
            for (_, r), (_, c) in zip(urban_trend_raw, urban_trend_corrected)
            if abs(r - c) > 0.01
        )
        log.info("    Corrected years: %d", n_corrected)

    # G2: Change signal checks
    g2_checks = []

    # G2a: Built-up area delta >= 5%/decade (corrected)
    corrected_by_year = {yr: uf for yr, uf in urban_trend_corrected} if urban_trend_corrected else {}
    if 2017 in corrected_by_year and 2024 in corrected_by_year:
        urban_2017 = corrected_by_year[2017]
        urban_2024 = corrected_by_year[2024]
        delta_urban = urban_2024 - urban_2017
        g2a = delta_urban >= 5.0
        g2_checks.append(("Built-up delta>=5%", g2a, f"{delta_urban:+.1f}%"))
        log.info(
            "  G2a (Built-up delta, corrected): %.1f%% -> %.1f%% = %+.1f%% (>=5%%? %s)",
            urban_2017, urban_2024, delta_urban, "YES" if g2a else "NO",
        )

    # G2b: Urban-class IoU for 2024
    if 2024 in transfer_results:
        tr_2024 = transfer_results[2024]
        urban_iou = tr_2024.get("urban_iou", 0)
        g2b = urban_iou >= 0.60
        g2_checks.append(("DW urban overlap>=60%", g2b, f"{urban_iou:.1%}"))
        log.info(
            "  G2b (DW urban overlap): %.1f%% (>=60%%? %s)",
            urban_iou * 100, "YES" if g2b else "NO",
        )

    # G2c: Urban fraction monotonic trend
    if len(urban_trend_corrected) >= 3:
        fracs_corrected = [f for _, f in urban_trend_corrected]
        total_increase = fracs_corrected[-1] - fracs_corrected[0]
        g2c = total_increase >= 2.0
        g2_checks.append(("Urban trend delta>=2%p", g2c, f"{total_increase:+.1f}%p total"))
        log.info(
            "  G2c (Urban trend, corrected): total increase %+.1f%%p (>=2%%p? %s)",
            total_increase, "YES" if g2c else "NO",
        )

    n_g2_pass = sum(1 for _, p, _ in g2_checks if p)
    n_g2_computed = len(g2_checks)

    if n_g2_computed < 3:
        g2_pass = False
        g2_status = "INCOMPLETE"
        log.info(
            "  G2 (Change Signal): %d/%d passed (%d missing) -> INCOMPLETE",
            n_g2_pass, n_g2_computed, 3 - n_g2_computed,
        )
    else:
        g2_pass = n_g2_pass >= 2
        g2_status = "PASS" if g2_pass else "FAIL"
        log.info(
            "  G2 (Change Signal): %d/3 passed (>=2? %s) -> %s",
            n_g2_pass, "YES" if g2_pass else "NO", g2_status,
        )

    # Overall verdict
    if g2_status == "INCOMPLETE":
        verdict = "INCOMPLETE (G2 manual review needed)"
    elif g1_pass and g2_pass:
        verdict = "GO"
    elif g1_pass or g2_pass:
        verdict = "CONDITIONAL GO"
    elif g1_oa >= 0.70:
        verdict = "CONDITIONAL (RF fallback)"
    else:
        verdict = "NO-GO"

    log.info("")
    log.info("  +======================================+")
    log.info("  |  VERDICT: %-28s|", verdict)
    log.info("  +======================================+")

    # Assemble full results
    all_results = {
        "aoi": {
            "name": "Hwaseong/Dongtan",
            "mode": "bbox",
            "bbox": HWASEONG_BBOX,
            "description": "Hwaseong-si / Dongtan New Town (rapid urbanization 2017-2024)",
        },
        "block_cv": {str(k): v for k, v in block_cv_results.items()},
        "g1_gate": g1_gate,
        "transfer": {str(k): v for k, v in transfer_results.items()},
        "urban_trend": {
            "raw": {str(yr): float(uf) for yr, uf in urban_trend_raw},
            "corrected": {str(yr): float(uf) for yr, uf in urban_trend_corrected},
            "method": "monotonic_non_decreasing",
            "assumption": "urbanization is irreversible; dips are AEF temporal artifacts",
        },
        "g1": {
            "oa": float(g1_oa),
            "kappa": float(g1_kappa),
            "block_size": int(best_block),
            "pass": bool(g1_pass),
        },
        "g2": {
            "checks": [(n, bool(p), d) for n, p, d in g2_checks],
            "n_pass": int(n_g2_pass),
            "n_computed": int(n_g2_computed),
            "pass": bool(g2_pass),
            "status": g2_status,
        },
        "verdict": verdict,
        "n_stable_samples": int(len(X_stable)),
        "n_total_samples": int(len(X)),
    }

    results_path = DATA_DIR / "hwaseong_classification_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("Results saved: %s", results_path)
    return all_results


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print_banner()

    # Step 1: Extract samples
    X, y, y_dw, coords, stable_mask = step1_extract_samples()
    print_sample_stats(X, y, y_dw, stable_mask)

    # Step 2: Spatial Block CV
    block_cv_results, X_stable, y_stable, coords_m = step2_spatial_block_cv(
        X, y, coords, stable_mask
    )

    # G1 Gate
    g1_gate, g1_oa, g1_kappa, gate_block = g1_gate_check(block_cv_results)

    # Step 3: Cross-year transfer
    transfer_results, clf_final = step3_cross_year_transfer(
        X_stable, y_stable, coords_m, gate_block
    )

    # Step 4: Verdict
    all_results = step4_verdict(block_cv_results, transfer_results, g1_gate, X_stable, X)

    log.info("")
    log.info("=" * 60)
    log.info("Hwaseong/Dongtan T2-T4 Classification Complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
