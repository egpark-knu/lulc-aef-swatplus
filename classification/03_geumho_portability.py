"""
Phase 1 -- Task 5: Geumho River (금호강) Screening
3-Level Cosine Similarity Analysis for cross-basin portability assessment.

Analysis Levels:
    Level 1: Pixel-level cross-year cosine similarity
    Level 2: Class centroid cross-year drift
    Level 3: Basin-level portability (Sejong vs Geumho)

Usage:
    python phase1/t5_geumho_screening.py

Memory Safety: Same patterns as t234_sejong_classification.py
    - BLAS/OpenMP thread caps
    - getInfo() with .limit() hard cap
    - gc.collect() after each year
    - check_memory() monitoring
    - Results cached to phase1/data/t5_geumho_screening.json
"""

import os

# ── Thread caps (must be set before numpy import) ──
DEFAULT_SAFE_MODE = os.environ.get("T5_SAFE_MODE", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
MAX_THREADS = int(os.environ.get("T5_MAX_THREADS", "1" if DEFAULT_SAFE_MODE else "4"))
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
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import norm

import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR = PROJ_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "t5_geumho_screening.json"
SEJONG_SAMPLE_CACHE = DATA_DIR / "sejong_2021_samples.npz"
SEJONG_RESULTS = DATA_DIR / "sejong_classification_results.json"
TRANSFER_DIR = DATA_DIR / "transfer_years"

# ── Logging ──
log_file = LOG_DIR / f"t5_geumho_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Memory Safety ──
MEM_LIMIT_GB = 180


def check_memory(label: str = "") -> bool:
    """Print current memory usage. Warn if above threshold."""
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
            log.warning("  System memory %.0fGB > %.0fGB limit!", sys_used_gb, MEM_LIMIT_GB)
            return False
        return True
    except ImportError:
        return True


# ── GEE init ──
ee.Initialize()

# ── Configuration ──
SAMPLE_SIZE = 2000
SAMPLE_LIMIT = 2000
YEARS = list(range(2017, 2025))  # 2017-2024
N_BANDS = 64  # AEF A00-A63
BAND_NAMES = [f"A{i:02d}" for i in range(N_BANDS)]

# WorldCover -> 6-class mapping (identical to t234)
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

# ── Geumho River AOI (Daegu area) ──
# Same as poc_01_extract_embeddings.py: center [128.6014, 35.8714], 15km buffer
GEUMHO_CENTER = [128.6014, 35.8714]
GEUMHO_BUFFER_KM = 15
GEUMHO_AOI = ee.Geometry.Point(GEUMHO_CENTER).buffer(GEUMHO_BUFFER_KM * 1000)

# Sejong AOI bbox for reference (from sejong_classification_results.json)
SEJONG_BBOX = [126.85, 36.38, 127.15, 36.62]


# ════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    n_a = norm(a)
    n_b = norm(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return float(np.dot(a, b) / (n_a * n_b))


def cosine_similarity_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices of the same shape."""
    dot = np.sum(A * B, axis=1)
    nA = norm(A, axis=1)
    nB = norm(B, axis=1)
    denom = nA * nB
    denom[denom == 0] = 1e-12
    return dot / denom


def extract_aef_for_year(
    year: int,
    aoi: ee.Geometry,
    fixed_points: ee.FeatureCollection,
    with_wc: bool = False,
) -> dict | None:
    """Extract AEF embeddings for a given year at fixed sample points.

    Returns dict with 'embeddings' (N x 64), 'wc_labels' (optional), 'n_valid'.
    Returns None on failure.
    """
    try:
        aef_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        aef_year = (
            aef_col.filter(ee.Filter.calendarRange(year, year, "year"))
            .filterBounds(aoi)
            .mosaic()
            .clip(aoi)
        )

        if with_wc and year == 2021:
            wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
            wc_label = wc.select("Map").rename("WC")
            combined = aef_year.addBands(wc_label)
        else:
            combined = aef_year

        sample = combined.sampleRegions(
            collection=fixed_points,
            scale=10,
            geometries=False,
        ).limit(SAMPLE_LIMIT)

        features = sample.getInfo()["features"]
        if not features:
            return None

        embeddings = []
        wc_labels = []
        for feat in features:
            props = feat["properties"]
            emb = [props.get(f"A{i:02d}", 0.0) for i in range(N_BANDS)]
            # Skip if any band is None (no data pixel)
            if any(v is None for v in emb):
                continue
            embeddings.append(emb)
            if with_wc and year == 2021:
                wc_val = props.get("WC")
                wc_labels.append(int(wc_val) if wc_val is not None else -1)

        if not embeddings:
            return None

        result = {
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "n_valid": len(embeddings),
        }
        if with_wc and wc_labels:
            result["wc_labels_raw"] = np.asarray(wc_labels, dtype=np.int16)
        return result
    except Exception as e:
        log.error("  Failed to extract year %d: %s", year, e)
        return None


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main() -> None:
    results: dict = {}
    log.info("=" * 60)
    log.info("Task 5: Geumho River Screening")
    log.info("=" * 60)
    log.info("  AOI: Geumho (Daegu), center=%s, buffer=%dkm", GEUMHO_CENTER, GEUMHO_BUFFER_KM)
    log.info("  Samples: %d, Years: %s", SAMPLE_SIZE, YEARS)
    log.info("  safe_mode=%s, max_threads=%d", DEFAULT_SAFE_MODE, MAX_THREADS)
    log.info("  Log file: %s", log_file)
    check_memory("start")

    # ──────────────────────────────────────────────
    # STEP 0: Generate fixed sample points for Geumho
    # ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 0: Generate fixed sample points for Geumho")
    log.info("=" * 60)

    fixed_points = (
        ee.Image.constant(1)
        .clip(GEUMHO_AOI)
        .sample(
            region=GEUMHO_AOI,
            scale=10,
            numPixels=SAMPLE_SIZE,
            seed=42,
            geometries=True,
        )
        .limit(SAMPLE_LIMIT)
    )
    n_fixed = fixed_points.size().getInfo()
    log.info("  Fixed sample points: %d", n_fixed)

    # ──────────────────────────────────────────────
    # STEP 1: Extract embeddings for all years
    # ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1: Extract Geumho AEF embeddings (2017-2024)")
    log.info("=" * 60)

    geumho_embeddings: dict[int, np.ndarray] = {}
    geumho_wc_labels: np.ndarray | None = None
    n_common: int | None = None  # Number of pixels with valid data across all years

    for year in YEARS:
        log.info("  %d:", year)
        if not check_memory(f"before {year}"):
            log.warning("  Memory limit approached -- stopping extraction")
            break

        # First year (2021): also extract WorldCover labels
        with_wc = (year == 2021)
        data = extract_aef_for_year(year, GEUMHO_AOI, fixed_points, with_wc=with_wc)

        if data is None:
            log.warning("  %d: No data extracted, skipping", year)
            gc.collect()
            ee.Reset()
            ee.Initialize()
            continue

        geumho_embeddings[year] = data["embeddings"]
        log.info("    Valid pixels: %d, shape: %s", data["n_valid"], data["embeddings"].shape)

        if with_wc and "wc_labels_raw" in data:
            geumho_wc_labels = data["wc_labels_raw"]
            log.info("    WorldCover labels extracted: %d", len(geumho_wc_labels))

        # Memory cleanup
        del data
        gc.collect()
        ee.Reset()
        ee.Initialize()

    available_years = sorted(geumho_embeddings.keys())
    log.info("\n  Extracted years: %s", available_years)
    if len(available_years) < 2:
        log.error("  Need at least 2 years for cross-year analysis. Aborting.")
        sys.exit(1)

    # Find the minimum sample count across years (for consistent pairing)
    min_n = min(emb.shape[0] for emb in geumho_embeddings.values())
    log.info("  Min samples across years: %d (will use first %d for pairing)", min_n, min_n)

    # Trim all years to min_n for pixel-level pairing
    for yr in available_years:
        geumho_embeddings[yr] = geumho_embeddings[yr][:min_n]

    if geumho_wc_labels is not None and len(geumho_wc_labels) > min_n:
        geumho_wc_labels = geumho_wc_labels[:min_n]

    check_memory("after all extractions")

    # ──────────────────────────────────────────────
    # Load Sejong data for comparison
    # ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Loading Sejong reference data")
    log.info("=" * 60)

    sejong_results: dict = {}
    if SEJONG_RESULTS.exists():
        with open(SEJONG_RESULTS, encoding="utf-8") as f:
            sejong_results = json.load(f)
        log.info("  Sejong classification results loaded")

    # Load Sejong 2021 sample embeddings
    sejong_emb_2021: np.ndarray | None = None
    sejong_labels: np.ndarray | None = None
    if SEJONG_SAMPLE_CACHE.exists():
        cached = np.load(SEJONG_SAMPLE_CACHE)
        sejong_emb_2021 = cached["X"].astype(np.float32, copy=False)
        sejong_labels = cached["y"].astype(np.int16, copy=False)
        log.info("  Sejong 2021 embeddings: %s, labels: %s", sejong_emb_2021.shape, sejong_labels.shape)
    else:
        log.warning("  Sejong 2021 samples not found at %s", SEJONG_SAMPLE_CACHE)

    # Load Sejong cross-year transfer data
    sejong_transfer: dict[int, dict] = {}
    if TRANSFER_DIR.exists():
        for tf_path in sorted(TRANSFER_DIR.glob("*.json")):
            with open(tf_path, encoding="utf-8") as f:
                td = json.load(f)
            sejong_transfer[int(td["year"])] = td
        log.info("  Sejong transfer years loaded: %s", sorted(sejong_transfer.keys()))

    # ══════════════════════════════════════════════
    # LEVEL 1: Pixel-level cross-year cosine similarity
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("LEVEL 1: Pixel-level Cross-Year Cosine Similarity")
    log.info("=" * 60)

    year_pairs = list(combinations(available_years, 2))
    log.info("  Year pairs: %d combinations", len(year_pairs))

    pixel_cossim: dict[str, dict] = {}  # "2017-2018" -> {mean, std, min, max, p5, p95}
    all_cossim_values: list[float] = []

    for y1, y2 in year_pairs:
        emb1 = geumho_embeddings[y1]
        emb2 = geumho_embeddings[y2]

        sims = cosine_similarity_batch(emb1, emb2)
        pair_key = f"{y1}-{y2}"

        stats = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "p5": float(np.percentile(sims, 5)),
            "p25": float(np.percentile(sims, 25)),
            "median": float(np.median(sims)),
            "p75": float(np.percentile(sims, 75)),
            "p95": float(np.percentile(sims, 95)),
            "n_pixels": int(len(sims)),
        }
        pixel_cossim[pair_key] = stats
        all_cossim_values.extend(sims.tolist())

        log.info(
            "  %s: mean=%.4f, std=%.4f, [p5=%.4f, median=%.4f, p95=%.4f]",
            pair_key, stats["mean"], stats["std"], stats["p5"], stats["median"], stats["p95"],
        )

    # Basin-wide summary
    all_arr = np.array(all_cossim_values)
    level1_summary = {
        "basin_mean": float(np.mean(all_arr)),
        "basin_std": float(np.std(all_arr)),
        "basin_min": float(np.min(all_arr)),
        "basin_max": float(np.max(all_arr)),
        "basin_p5": float(np.percentile(all_arr, 5)),
        "basin_median": float(np.median(all_arr)),
        "basin_p95": float(np.percentile(all_arr, 95)),
        "n_total_pairs": len(all_cossim_values),
        "n_year_combinations": len(year_pairs),
        "n_pixels_per_pair": min_n,
    }

    log.info("\n  Geumho Basin-wide Summary:")
    log.info("    Mean cosine sim: %.4f +/- %.4f", level1_summary["basin_mean"], level1_summary["basin_std"])
    log.info(
        "    Range: [%.4f, %.4f], p5=%.4f, p95=%.4f",
        level1_summary["basin_min"], level1_summary["basin_max"],
        level1_summary["basin_p5"], level1_summary["basin_p95"],
    )

    results["level1_pixel_crossyear"] = {
        "year_pairs": pixel_cossim,
        "summary": level1_summary,
    }

    del all_cossim_values, all_arr
    gc.collect()
    check_memory("after Level 1")

    # ══════════════════════════════════════════════
    # STEP 2: Extract Sejong per-year embeddings (shared by Level 2 & 3)
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 2: Extract Sejong per-year embeddings (for Level 2 & 3)")
    log.info("=" * 60)

    sejong_aoi = ee.Geometry.Rectangle(SEJONG_BBOX)
    sejong_fixed_pts = (
        ee.Image.constant(1)
        .clip(sejong_aoi)
        .sample(
            region=sejong_aoi,
            scale=10,
            numPixels=SAMPLE_SIZE,
            seed=42,
            geometries=True,
        )
        .limit(SAMPLE_LIMIT)
    )
    n_sejong_pts = sejong_fixed_pts.size().getInfo()
    log.info("  Sejong fixed sample points: %d", n_sejong_pts)

    sejong_wc_labels_raw: np.ndarray | None = None
    sejong_per_year_embeddings: dict[int, np.ndarray] = {}

    for year in available_years:
        log.info("  Sejong %d:", year)
        if not check_memory(f"sejong before {year}"):
            break

        with_wc_s = (year == 2021)
        data_s = extract_aef_for_year(year, sejong_aoi, sejong_fixed_pts, with_wc=with_wc_s)

        if data_s is None:
            log.warning("    Sejong %d: no data", year)
            gc.collect()
            ee.Reset()
            ee.Initialize()
            continue

        sejong_per_year_embeddings[year] = data_s["embeddings"]
        log.info("    Valid: %d", data_s["n_valid"])

        if with_wc_s and "wc_labels_raw" in data_s:
            sejong_wc_labels_raw = data_s["wc_labels_raw"]

        del data_s
        gc.collect()
        ee.Reset()
        ee.Initialize()

    # Trim Sejong to common size across years
    if sejong_per_year_embeddings:
        min_n_s = min(e.shape[0] for e in sejong_per_year_embeddings.values())
        for yr in sejong_per_year_embeddings:
            sejong_per_year_embeddings[yr] = sejong_per_year_embeddings[yr][:min_n_s]
        if sejong_wc_labels_raw is not None and len(sejong_wc_labels_raw) > min_n_s:
            sejong_wc_labels_raw = sejong_wc_labels_raw[:min_n_s]
        log.info("  Sejong trimmed to %d pixels/year", min_n_s)

    # Pre-compute Sejong basin means (for Level 3)
    sejong_basin_mean: dict[int, np.ndarray] = {}
    for yr, emb_s in sejong_per_year_embeddings.items():
        sejong_basin_mean[yr] = emb_s.mean(axis=0)

    check_memory("after Sejong extraction")

    # ══════════════════════════════════════════════
    # LEVEL 2: Class centroid cross-year drift
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("LEVEL 2: Class Centroid Cross-Year Drift")
    log.info("=" * 60)

    anchor_year = 2021
    centroid_drift: dict[str, dict[str, float]] = {}
    sejong_centroid_drift: dict[str, dict[str, float]] = {}

    if geumho_wc_labels is None:
        log.warning("  No WorldCover labels available for Geumho. Skipping Level 2.")
        results["level2_centroid_drift"] = {"status": "skipped", "reason": "no_wc_labels"}
    else:
        # Map raw WC labels to 6-class
        geumho_6class = np.array(
            [WC_TO_6CLASS.get(int(v), -1) for v in geumho_wc_labels],
            dtype=np.int16,
        )
        valid_mask = geumho_6class >= 0
        log.info("  Valid 6-class pixels: %d / %d", valid_mask.sum(), len(geumho_6class))

        # Compute class centroids per year for Geumho
        geumho_centroids: dict[int, dict[int, np.ndarray]] = {}

        for year in available_years:
            emb = geumho_embeddings[year]
            centroids: dict[int, np.ndarray] = {}
            for cls_id in range(len(CLASS_NAMES)):
                cls_mask = valid_mask & (geumho_6class == cls_id)
                if cls_mask.sum() < 5:
                    continue
                centroids[cls_id] = emb[cls_mask].mean(axis=0)
            geumho_centroids[year] = centroids

        # Compute drift from 2021 anchor
        if anchor_year in geumho_centroids:
            anchor_centroids = geumho_centroids[anchor_year]

            for cls_id, cls_name in enumerate(CLASS_NAMES):
                if cls_id not in anchor_centroids:
                    continue
                anchor_vec = anchor_centroids[cls_id]
                drift_per_year: dict[str, float] = {}
                for year in available_years:
                    if cls_id in geumho_centroids[year]:
                        sim = cosine_similarity(anchor_vec, geumho_centroids[year][cls_id])
                        drift_per_year[str(year)] = sim

                centroid_drift[cls_name] = drift_per_year
                sims_list = list(drift_per_year.values())
                if sims_list:
                    log.info(
                        "  %s: anchor=2021, min_sim=%.4f, max_sim=%.4f, mean_sim=%.4f",
                        cls_name, min(sims_list), max(sims_list), np.mean(sims_list),
                    )
        else:
            log.warning("  2021 not in available years; cannot compute centroid drift")

        # ── Sejong centroid comparison (using already-extracted data) ──
        if sejong_wc_labels_raw is not None and sejong_per_year_embeddings:
            log.info("\n  Computing Sejong class centroid drift for comparison...")
            sejong_6class = np.array(
                [WC_TO_6CLASS.get(int(v), -1) for v in sejong_wc_labels_raw],
                dtype=np.int16,
            )
            valid_mask_s = sejong_6class >= 0

            sejong_yr_centroids: dict[int, dict[int, np.ndarray]] = {}
            for year_s in sorted(sejong_per_year_embeddings.keys()):
                emb_s = sejong_per_year_embeddings[year_s]
                ctr_s: dict[int, np.ndarray] = {}
                for cls_id in range(len(CLASS_NAMES)):
                    cls_m = valid_mask_s & (sejong_6class == cls_id)
                    if cls_m.sum() < 5:
                        continue
                    ctr_s[cls_id] = emb_s[cls_m].mean(axis=0)
                sejong_yr_centroids[year_s] = ctr_s

            if anchor_year in sejong_yr_centroids:
                s_anchor = sejong_yr_centroids[anchor_year]
                for cls_id, cls_name in enumerate(CLASS_NAMES):
                    if cls_id not in s_anchor:
                        continue
                    s_anchor_vec = s_anchor[cls_id]
                    s_drift: dict[str, float] = {}
                    for year_s in sorted(sejong_per_year_embeddings.keys()):
                        if cls_id in sejong_yr_centroids[year_s]:
                            sim_s = cosine_similarity(s_anchor_vec, sejong_yr_centroids[year_s][cls_id])
                            s_drift[str(year_s)] = sim_s
                    sejong_centroid_drift[cls_name] = s_drift

            del sejong_yr_centroids
            gc.collect()

        # Comparison table
        log.info("\n  --- Centroid Drift Comparison (cos sim to 2021 anchor) ---")
        log.info("  %-12s | %-30s | %-30s", "Class", "Geumho (mean)", "Sejong (mean)")
        log.info("  " + "-" * 78)

        centroid_comparison: dict[str, dict] = {}
        for cls_name in CLASS_NAMES:
            g_vals = list(centroid_drift.get(cls_name, {}).values())
            s_vals = list(sejong_centroid_drift.get(cls_name, {}).values())
            g_mean = float(np.mean(g_vals)) if g_vals else None
            s_mean = float(np.mean(s_vals)) if s_vals else None
            g_str = f"{g_mean:.4f}" if g_mean is not None else "N/A"
            s_str = f"{s_mean:.4f}" if s_mean is not None else "N/A"
            log.info("  %-12s | %-30s | %-30s", cls_name, g_str, s_str)
            centroid_comparison[cls_name] = {
                "geumho_mean": g_mean,
                "sejong_mean": s_mean,
            }

        results["level2_centroid_drift"] = {
            "geumho": {
                k: {str(yr): float(v) for yr, v in vals.items()}
                for k, vals in centroid_drift.items()
            },
            "sejong": {
                k: {str(yr): float(v) for yr, v in vals.items()}
                for k, vals in sejong_centroid_drift.items()
            },
            "comparison": centroid_comparison,
            "anchor_year": anchor_year,
        }

    # Free Sejong per-year embeddings (basin means already computed)
    del sejong_per_year_embeddings
    gc.collect()
    check_memory("after Level 2")

    # ══════════════════════════════════════════════
    # LEVEL 3: Basin-level portability
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("LEVEL 3: Basin-Level Portability (Sejong vs Geumho)")
    log.info("=" * 60)

    # Compute basin mean embedding per year for Geumho
    geumho_basin_mean: dict[int, np.ndarray] = {}
    for year in available_years:
        geumho_basin_mean[year] = geumho_embeddings[year].mean(axis=0)

    # Cross-basin cosine similarity per year
    cross_basin_per_year: dict[str, float] = {}
    log.info("\n  Cross-basin cosine similarity (Sejong vs Geumho):")
    for year in available_years:
        if year in geumho_basin_mean and year in sejong_basin_mean:
            sim = cosine_similarity(geumho_basin_mean[year], sejong_basin_mean[year])
            cross_basin_per_year[str(year)] = sim
            log.info("    %d: cos_sim = %.4f", year, sim)

    if cross_basin_per_year:
        vals = list(cross_basin_per_year.values())
        cross_basin_summary = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        log.info("  Summary: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
                 cross_basin_summary["mean"], cross_basin_summary["std"],
                 cross_basin_summary["min"], cross_basin_summary["max"])
    else:
        cross_basin_summary = {"mean": None, "std": None, "min": None, "max": None}

    # Within-basin temporal stability (self-similarity across years)
    geumho_within: list[float] = []
    for y1, y2 in combinations(available_years, 2):
        if y1 in geumho_basin_mean and y2 in geumho_basin_mean:
            geumho_within.append(
                cosine_similarity(geumho_basin_mean[y1], geumho_basin_mean[y2])
            )

    sejong_within: list[float] = []
    for y1, y2 in combinations(available_years, 2):
        if y1 in sejong_basin_mean and y2 in sejong_basin_mean:
            sejong_within.append(
                cosine_similarity(sejong_basin_mean[y1], sejong_basin_mean[y2])
            )

    within_basin_stability = {
        "geumho": {
            "mean": float(np.mean(geumho_within)) if geumho_within else None,
            "std": float(np.std(geumho_within)) if geumho_within else None,
            "n_pairs": len(geumho_within),
        },
        "sejong": {
            "mean": float(np.mean(sejong_within)) if sejong_within else None,
            "std": float(np.std(sejong_within)) if sejong_within else None,
            "n_pairs": len(sejong_within),
        },
    }

    log.info("\n  Within-basin temporal stability (basin-mean cos sim across year pairs):")
    for basin_name, vals_dict in within_basin_stability.items():
        if vals_dict["mean"] is not None:
            log.info("    %s: mean=%.4f, std=%.4f (n=%d pairs)",
                     basin_name, vals_dict["mean"], vals_dict["std"], vals_dict["n_pairs"])

    results["level3_basin_portability"] = {
        "cross_basin_per_year": cross_basin_per_year,
        "cross_basin_summary": cross_basin_summary,
        "within_basin_stability": within_basin_stability,
    }

    # ══════════════════════════════════════════════
    # PORTABILITY ASSESSMENT
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("PORTABILITY ASSESSMENT")
    log.info("=" * 60)

    # Thresholds for portability
    THRESHOLD_CROSS_BASIN = 0.90  # basin mean similarity
    THRESHOLD_PIXEL_MEAN = 0.85   # pixel-level mean cos sim
    THRESHOLD_CENTROID_MIN = 0.90  # min class centroid drift

    checks: list[tuple[str, bool, str]] = []

    # Check 1: Cross-basin similarity
    cb_mean = cross_basin_summary.get("mean")
    if cb_mean is not None:
        c1_pass = cb_mean >= THRESHOLD_CROSS_BASIN
        checks.append(("Cross-basin mean sim >= 0.90", c1_pass, f"{cb_mean:.4f}"))
    else:
        checks.append(("Cross-basin mean sim >= 0.90", False, "N/A"))

    # Check 2: Pixel-level temporal stability
    px_mean = level1_summary.get("basin_mean")
    if px_mean is not None:
        c2_pass = px_mean >= THRESHOLD_PIXEL_MEAN
        checks.append(("Pixel cross-year mean >= 0.85", c2_pass, f"{px_mean:.4f}"))
    else:
        checks.append(("Pixel cross-year mean >= 0.85", False, "N/A"))

    # Check 3: Class centroid minimum stability
    if centroid_drift:
        all_centroid_sims = []
        for cls_name, drift_dict in centroid_drift.items():
            all_centroid_sims.extend(drift_dict.values())
        if all_centroid_sims:
            centroid_min = float(np.min(all_centroid_sims))
            c3_pass = centroid_min >= THRESHOLD_CENTROID_MIN
            checks.append(("Centroid min drift >= 0.90", c3_pass, f"{centroid_min:.4f}"))
        else:
            checks.append(("Centroid min drift >= 0.90", False, "N/A"))
    else:
        checks.append(("Centroid min drift >= 0.90", False, "N/A"))

    n_pass = sum(1 for _, p, _ in checks if p)
    portability_verdict = "PORTABLE" if n_pass >= 2 else ("MARGINAL" if n_pass >= 1 else "NOT PORTABLE")

    log.info("\n  Portability Checks:")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        log.info("    [%s] %s: %s", status, name, detail)

    log.info("\n  Result: %d/3 checks passed", n_pass)
    log.info("  =============================================")
    log.info("  VERDICT: %s", portability_verdict)
    log.info("  =============================================")

    results["portability_assessment"] = {
        "checks": [(n, bool(p), d) for n, p, d in checks],
        "n_pass": n_pass,
        "n_total": len(checks),
        "verdict": portability_verdict,
        "thresholds": {
            "cross_basin": THRESHOLD_CROSS_BASIN,
            "pixel_mean": THRESHOLD_PIXEL_MEAN,
            "centroid_min": THRESHOLD_CENTROID_MIN,
        },
    }

    # ══════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("SUMMARY COMPARISON TABLE")
    log.info("=" * 60)
    log.info("")
    log.info("  %-30s | %-15s | %-15s", "Metric", "Geumho", "Sejong")
    log.info("  " + "-" * 66)

    # Pixel cross-year mean
    log.info("  %-30s | %-15s | %-15s",
             "Pixel cross-year cos (mean)",
             f"{level1_summary['basin_mean']:.4f}",
             "N/A (see below)")

    # Within-basin stability
    g_ws = within_basin_stability["geumho"]["mean"]
    s_ws = within_basin_stability["sejong"]["mean"]
    log.info("  %-30s | %-15s | %-15s",
             "Basin mean temporal stability",
             f"{g_ws:.4f}" if g_ws else "N/A",
             f"{s_ws:.4f}" if s_ws else "N/A")

    # Cross-basin
    log.info("  %-30s | %-15s |",
             "Cross-basin cos (mean)",
             f"{cross_basin_summary['mean']:.4f}" if cross_basin_summary["mean"] else "N/A")

    # Class centroid stability (per class)
    for cls_name in CLASS_NAMES:
        g_vals = list(centroid_drift.get(cls_name, {}).values())
        s_vals = list(sejong_centroid_drift.get(cls_name, {}).values())
        g_str = f"{np.mean(g_vals):.4f}" if g_vals else "N/A"
        s_str = f"{np.mean(s_vals):.4f}" if s_vals else "N/A"
        log.info("  %-30s | %-15s | %-15s",
                 f"  {cls_name} centroid (mean)", g_str, s_str)

    # Portability verdict
    log.info("  " + "-" * 66)
    log.info("  %-30s | %-15s |", "PORTABILITY VERDICT", portability_verdict)

    # ══════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════
    results["metadata"] = {
        "geumho_aoi": {
            "center": GEUMHO_CENTER,
            "buffer_km": GEUMHO_BUFFER_KM,
        },
        "sejong_bbox": SEJONG_BBOX,
        "sample_size": SAMPLE_SIZE,
        "years": available_years,
        "n_pixels_per_year": min_n,
    }

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("\nResults saved: %s", CACHE_FILE)
    log.info("Log saved: %s", log_file)
    log.info("\n" + "=" * 60)
    log.info("Task 5 Complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
