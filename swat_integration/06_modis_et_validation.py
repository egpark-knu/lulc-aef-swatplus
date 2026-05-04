#!/usr/bin/env python3
"""
Phase 2 -- Script 10: MODIS ET Validation against SWAT+ ET
===========================================================

Extracts MODIS MOD16A2GF (gap-filled) 8-day ET at 500 m,
computes annual totals (2017-2024) for Hwaseong and Sejong basins,
and compares with SWAT+ simulated ET.

Additionally extracts MODIS ET stratified by ESA WorldCover 2021
LULC classes (Forest, Cropland, Urban) to validate the physical
premise that Urban ET < Forest ET.

Outputs:
    phase2/data/modis_et/modis_et_annual.json
    phase2/data/modis_et/modis_et_by_lulc.json

Usage:
    python p2_10_modis_et_validation.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# -- Thread caps (before numpy) --
for _var in [
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_var, "1")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent.parent  # dev_260402_LULC/
DATA_DIR = PROJECT / "phase2" / "data" / "modis_et"
LOG_DIR = PROJECT / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "p2_10_modis_et_validation.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AOI definitions
# ---------------------------------------------------------------------------
AOIS: dict[str, dict[str, Any]] = {
    "Hwaseong": {
        "bbox": [126.85, 37.05, 127.25, 37.35],
        "swat_et": {
            "static_aa": 494.3,   # mm/yr (all-year average, from comparison.json)
            "dynamic_aa": 501.6,
        },
    },
    "Sejong": {
        "bbox": [126.85, 36.38, 127.15, 36.62],
        "swat_et": {
            "static_aa": None,  # filled from sejong_comparison.json
            "dynamic_aa": None,
        },
    },
}

YEARS = list(range(2017, 2025))  # 2017-2024

# WorldCover 2021 LULC codes of interest
LULC_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    80: "Permanent water",
}


def load_sejong_swat_et() -> None:
    """Load Sejong SWAT+ ET from comparison.json."""
    cmp_path = (
        PROJECT / "phase2" / "data" / "sejong_swat_results" / "sejong_comparison.json"
    )
    if not cmp_path.exists():
        log.warning("Sejong comparison.json not found -- SWAT+ ET will be None")
        return

    with open(cmp_path) as f:
        d = json.load(f)

    # Compute mean ET across years for static & dynamic
    for scen_key, aoi_key in [("static_years", "static_aa"), ("dynamic_years", "dynamic_aa")]:
        years_data = d.get(scen_key, [])
        if years_data:
            mean_et = sum(yr["et"] for yr in years_data) / len(years_data)
            AOIS["Sejong"]["swat_et"][aoi_key] = round(mean_et, 1)
            log.info("Sejong SWAT+ %s mean ET = %.1f mm/yr", aoi_key, mean_et)


def init_gee() -> Any:
    """Initialize Google Earth Engine."""
    import ee
    try:
        ee.Initialize()
        log.info("GEE initialized successfully")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        log.info("GEE initialized after authentication")
    return ee


def make_aoi(ee_mod, bbox: list[float]):
    """Create ee.Geometry.Rectangle from [xmin, ymin, xmax, ymax]."""
    return ee_mod.Geometry.Rectangle(bbox)


def get_modis_annual_et(
    ee_mod,
    aoi_geom,
    year: int,
) -> float | None:
    """
    Compute MODIS MOD16A2GF annual ET (mm) for a given year and AOI.

    MOD16A2GF 'ET' band: kg/m2/8day, scale factor 0.1.
    Annual ET = sum of all 8-day composites * 0.1.
    """
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    col = (
        ee_mod.ImageCollection("MODIS/061/MOD16A2GF")
        .filterDate(start, end)
        .filterBounds(aoi_geom)
        .select("ET")
    )

    # Sum all 8-day composites, then apply scale factor
    annual_sum = col.sum().multiply(0.1)

    # Basin-mean ET
    stats = annual_sum.reduceRegion(
        reducer=ee_mod.Reducer.mean(),
        geometry=aoi_geom,
        scale=500,
        maxPixels=1e8,
    ).getInfo()

    et_val = stats.get("ET")
    if et_val is not None:
        return round(float(et_val), 1)
    return None


def get_modis_et_by_lulc(
    ee_mod,
    aoi_geom,
    years: list[int],
) -> dict[str, dict[str, float | None]]:
    """
    Compute mean annual MODIS ET stratified by WorldCover 2021 LULC classes.

    Returns: {lulc_name: {year: et_mm, ...}, ...}
    """
    # WorldCover v200 is an ImageCollection (one image per tile).
    # Mosaic to get a single global image, then select "Map" band.
    worldcover = (
        ee_mod.ImageCollection("ESA/WorldCover/v200")
        .mosaic()
        .select("Map")
    )

    results: dict[str, dict[str, float | None]] = {}

    for lulc_code, lulc_name in LULC_CLASSES.items():
        results[lulc_name] = {}

        # Create mask for this LULC class
        mask = worldcover.eq(lulc_code)

        # Check if this LULC class has significant area in the AOI
        area_check = mask.reduceRegion(
            reducer=ee_mod.Reducer.sum(),
            geometry=aoi_geom,
            scale=500,
            maxPixels=1e8,
        ).getInfo()

        pixel_count = area_check.get("Map", 0)
        if pixel_count is None or pixel_count < 10:
            log.info("  %s: too few pixels (%s) -- skipping", lulc_name, pixel_count)
            for yr in years:
                results[lulc_name][str(yr)] = None
            continue

        for yr in years:
            start = f"{yr}-01-01"
            end = f"{yr + 1}-01-01"

            col = (
                ee_mod.ImageCollection("MODIS/061/MOD16A2GF")
                .filterDate(start, end)
                .filterBounds(aoi_geom)
                .select("ET")
            )

            annual_sum = col.sum().multiply(0.1)
            masked = annual_sum.updateMask(mask)

            stats = masked.reduceRegion(
                reducer=ee_mod.Reducer.mean(),
                geometry=aoi_geom,
                scale=500,
                maxPixels=1e8,
            ).getInfo()

            et_val = stats.get("ET")
            if et_val is not None:
                results[lulc_name][str(yr)] = round(float(et_val), 1)
            else:
                results[lulc_name][str(yr)] = None

            log.info("  %s %d: ET = %s mm", lulc_name, yr, results[lulc_name][str(yr)])

        gc.collect()

    return results


def print_comparison_table(annual_results: dict) -> None:
    """Print MODIS vs SWAT+ comparison table."""
    print("\n" + "=" * 80)
    print("MODIS ET vs SWAT+ ET Comparison")
    print("=" * 80)

    for site_name, site_data in annual_results.items():
        print(f"\n--- {site_name} ---")
        swat_et = AOIS[site_name]["swat_et"]

        # Header
        print(f"{'Year':<8} {'MODIS ET (mm)':<16} {'SWAT+ Static':<16} {'SWAT+ Dynamic':<16}")
        print("-" * 56)

        modis_vals = []
        for yr in YEARS:
            modis_et = site_data.get(str(yr))
            modis_str = f"{modis_et:.1f}" if modis_et is not None else "N/A"
            if modis_et is not None:
                modis_vals.append(modis_et)
            print(f"{yr:<8} {modis_str:<16} {'---':<16} {'---':<16}")

        # Averages
        if modis_vals:
            mean_modis = sum(modis_vals) / len(modis_vals)
            print("-" * 56)
            static_str = f"{swat_et['static_aa']:.1f}" if swat_et['static_aa'] else "N/A"
            dynamic_str = f"{swat_et['dynamic_aa']:.1f}" if swat_et['dynamic_aa'] else "N/A"
            print(f"{'Mean':<8} {mean_modis:<16.1f} {static_str:<16} {dynamic_str:<16}")

            if swat_et["static_aa"]:
                ratio = mean_modis / swat_et["static_aa"]
                print(f"\nMODIS/SWAT+ ratio (vs static): {ratio:.2f}")

    print()


def print_lulc_table(lulc_results: dict) -> None:
    """Print MODIS ET by LULC class."""
    print("\n" + "=" * 80)
    print("MODIS ET by LULC Class (WorldCover 2021)")
    print("=" * 80)

    for site_name, site_data in lulc_results.items():
        print(f"\n--- {site_name} ---")

        # Compute multi-year mean for each class
        class_means: dict[str, float] = {}
        for lulc_name, year_data in site_data.items():
            vals = [v for v in year_data.values() if v is not None]
            if vals:
                class_means[lulc_name] = sum(vals) / len(vals)

        # Sort by ET descending
        sorted_classes = sorted(class_means.items(), key=lambda x: x[1], reverse=True)

        print(f"{'LULC Class':<20} {'Mean ET (mm/yr)':<18} {'Min':<10} {'Max':<10}")
        print("-" * 58)
        for lulc_name, mean_et in sorted_classes:
            year_data = site_data[lulc_name]
            vals = [v for v in year_data.values() if v is not None]
            mn = min(vals) if vals else 0
            mx = max(vals) if vals else 0
            print(f"{lulc_name:<20} {mean_et:<18.1f} {mn:<10.1f} {mx:<10.1f}")

        # Key validation: Forest > Cropland > Urban?
        f_et = class_means.get("Tree cover")
        c_et = class_means.get("Cropland")
        u_et = class_means.get("Built-up")
        print()
        if f_et and u_et:
            print(f"  Forest ET:  {f_et:.1f} mm/yr")
            print(f"  Urban ET:   {u_et:.1f} mm/yr")
            print(f"  Difference: {f_et - u_et:.1f} mm/yr (Forest - Urban)")
            if f_et > u_et:
                print("  --> VALIDATED: Forest ET > Urban ET (physically consistent)")
            else:
                print("  --> WARNING: Forest ET <= Urban ET (unexpected)")
        if c_et and u_et:
            print(f"  Cropland ET: {c_et:.1f} mm/yr")
            print(f"  Cropland - Urban: {c_et - u_et:.1f} mm/yr")

    print()


def main() -> None:
    log.info("=" * 60)
    log.info("Phase 2 Script 10: MODIS ET Validation")
    log.info("=" * 60)
    t0 = time.time()

    # Load Sejong SWAT+ ET
    load_sejong_swat_et()

    # Initialize GEE
    ee = init_gee()

    # ── Part 1: Annual basin-mean MODIS ET ──
    log.info("Part 1: Extracting annual MODIS ET (MOD16A2GF) ...")
    annual_results: dict[str, dict[str, float | None]] = {}

    for site_name, site_info in AOIS.items():
        log.info("Processing %s ...", site_name)
        aoi = make_aoi(ee, site_info["bbox"])
        annual_results[site_name] = {}

        for yr in YEARS:
            et_val = get_modis_annual_et(ee, aoi, yr)
            annual_results[site_name][str(yr)] = et_val
            log.info("  %s %d: MODIS ET = %s mm", site_name, yr, et_val)
            gc.collect()

    # Save annual results
    out_annual = {
        "description": "MODIS MOD16A2GF annual ET (mm/yr), basin mean",
        "collection": "MODIS/061/MOD16A2GF",
        "band": "ET",
        "scale_factor": 0.1,
        "resolution_m": 500,
        "generated": datetime.now().isoformat(),
        "sites": {},
    }
    for site_name in annual_results:
        swat = AOIS[site_name]["swat_et"]
        modis_vals = [v for v in annual_results[site_name].values() if v is not None]
        out_annual["sites"][site_name] = {
            "bbox": AOIS[site_name]["bbox"],
            "modis_et_annual_mm": annual_results[site_name],
            "modis_et_mean_mm": round(sum(modis_vals) / len(modis_vals), 1) if modis_vals else None,
            "swat_et_static_aa_mm": swat["static_aa"],
            "swat_et_dynamic_aa_mm": swat["dynamic_aa"],
        }

    annual_path = DATA_DIR / "modis_et_annual.json"
    with open(annual_path, "w") as f:
        json.dump(out_annual, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", annual_path)

    print_comparison_table(annual_results)

    # ── Part 2: MODIS ET by LULC class ──
    log.info("Part 2: Extracting MODIS ET by LULC class ...")
    lulc_results: dict[str, dict] = {}

    for site_name, site_info in AOIS.items():
        log.info("Processing %s by LULC ...", site_name)
        aoi = make_aoi(ee, site_info["bbox"])
        lulc_results[site_name] = get_modis_et_by_lulc(ee, aoi, YEARS)
        gc.collect()

    # Save LULC results
    out_lulc = {
        "description": "MODIS MOD16A2GF annual ET by WorldCover 2021 LULC class",
        "lulc_source": "ESA/WorldCover/v200",
        "generated": datetime.now().isoformat(),
        "sites": lulc_results,
    }

    lulc_path = DATA_DIR / "modis_et_by_lulc.json"
    with open(lulc_path, "w") as f:
        json.dump(out_lulc, f, indent=2, ensure_ascii=False)
    log.info("Saved: %s", lulc_path)

    print_lulc_table(lulc_results)

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
