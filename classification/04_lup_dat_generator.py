"""
Phase 1 -- Task 8: Superset HRU + lup.dat Generation

Generates the SWAT+ land use update (lup.dat) file framework from AEF
classification results (2017-2024). Uses the Superset HRU concept where
each spatial unit contains ALL possible LULC types with time-varying fractions.

Key concepts:
  - Superset HRU: union of all LULC classes observed across 2017-2024
  - lup.dat: SWAT+ land use update file that changes HRU fractions yearly
  - Monotonic urban constraint: urban fraction never decreases
  - Area accounting: fractions sum to 1.0 per subbasin per year

Data sources:
  - phase1/data/sejong_classification_results.json  (urban_trend corrected)
  - phase1/data/transfer_years/*.json               (per-year fractions)

Usage:
    python -m phase1.t8_lup_dat_generator
    python -m phase1.t8_lup_dat_generator --subbasins 10
    python -m phase1.t8_lup_dat_generator --apply-urban-correction
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
PHASE1_DIR = PROJECT_DIR / "phase1"
DATA_DIR = PHASE1_DIR / "data"
TRANSFER_DIR = DATA_DIR / "transfer_years"
SWAT_DIR = DATA_DIR / "swat_prep"
LOG_DIR = PROJECT_DIR / "logs"

for d in [SWAT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"t8_lup_dat_{_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
RESULTS_JSON = DATA_DIR / "sejong_classification_results.json"

# 6 LULC classes in canonical order (matches t7b_watershed_prep.py)
LULC_CLASSES = ["Forest", "Cropland", "Urban", "Grassland", "Water", "Barren"]
LULC_TO_SWAT = {
    "Forest": "FRSE",
    "Cropland": "AGRR",
    "Urban": "URLD",
    "Grassland": "PAST",
    "Water": "WATR",
    "Barren": "SWRN",
}

YEARS = list(range(2017, 2025))  # 2017-2024
LUP_DAT_PATH = SWAT_DIR / "lup.dat"
SUPERSET_PATH = SWAT_DIR / "superset_hru.json"
VALIDATION_PATH = SWAT_DIR / "lup_validation.json"


# ── Data Loading ──────────────────────────────────────────────

def load_transfer_results() -> dict[int, dict[str, float]]:
    """Load per-year class fractions from transfer_years/*.json.

    Returns:
        {year: {class_name: fraction_pct, ...}, ...}
    """
    fractions: dict[int, dict[str, float]] = {}

    for year in YEARS:
        path = TRANSFER_DIR / f"{year}.json"
        if not path.exists():
            log.warning(f"  Missing transfer file: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        fractions[year] = data["class_fractions"]
        log.info(f"  Loaded {year}: {len(data['class_fractions'])} classes, "
                 f"n_samples={data['n_samples']}")

    return fractions


def load_urban_correction() -> dict[str, dict[str, float]] | None:
    """Load monotonic urban correction from classification results.

    Returns:
        {"raw": {year: pct}, "corrected": {year: pct}} or None if unavailable.
    """
    if not RESULTS_JSON.exists():
        return None
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    return data.get("urban_trend")


# ── Urban Correction ──────────────────────────────────────────

def apply_urban_correction(
    fractions: dict[int, dict[str, float]],
    urban_trend: dict[str, dict[str, float]],
) -> dict[int, dict[str, float]]:
    """Apply monotonic non-decreasing urban correction and redistribute residuals.

    When urban fraction is raised to satisfy the monotonic constraint,
    the excess is subtracted proportionally from non-urban, non-water classes
    (Forest, Cropland, Grassland, Barren). Water is kept stable since it
    represents permanent water bodies.

    Args:
        fractions: Raw per-year class fractions (%).
        urban_trend: Dict with 'raw' and 'corrected' urban fractions.

    Returns:
        Corrected fractions dictionary.
    """
    corrected_urban = urban_trend.get("corrected", {})
    corrected = {}

    for year in sorted(fractions.keys()):
        year_str = str(year)
        raw = fractions[year].copy()
        raw_urban = raw["Urban"]

        if year_str in corrected_urban:
            new_urban = corrected_urban[year_str]
        else:
            new_urban = raw_urban

        delta = new_urban - raw_urban

        if abs(delta) < 1e-6:
            corrected[year] = raw
            continue

        # Redistribute delta across non-urban, non-water classes
        adjustable = ["Forest", "Cropland", "Grassland", "Barren"]
        total_adjustable = sum(raw[c] for c in adjustable)

        if total_adjustable < 1e-6:
            log.warning(f"  {year}: No adjustable classes for redistribution")
            corrected[year] = raw
            continue

        new_fracs = raw.copy()
        new_fracs["Urban"] = new_urban

        for cls in adjustable:
            share = raw[cls] / total_adjustable
            new_fracs[cls] = raw[cls] - delta * share

        # Clamp negatives and renormalize
        for cls in LULC_CLASSES:
            new_fracs[cls] = max(0.0, new_fracs[cls])

        total = sum(new_fracs[c] for c in LULC_CLASSES)
        if abs(total - 100.0) > 0.01:
            for cls in LULC_CLASSES:
                new_fracs[cls] = new_fracs[cls] / total * 100.0

        corrected[year] = new_fracs
        log.info(f"  {year}: Urban {raw_urban:.2f}% -> {new_urban:.2f}% (delta={delta:+.2f}%)")

    return corrected


# ── Superset HRU Construction ─────────────────────────────────

def build_superset_hru(
    fractions: dict[int, dict[str, float]],
    n_subbasins: int,
) -> dict[str, Any]:
    """Build the Superset HRU structure.

    Each subbasin gets HRUs for ALL 6 LULC classes regardless of whether
    that class has nonzero area in any given year. This means the HRU
    structure is fixed across years; only the area fractions change.

    For this mock, all subbasins share the same AOI-level fractions
    (since we have no spatial disaggregation yet). In a real application,
    QSWAT+ delineation would provide subbasin-specific fractions.

    Args:
        fractions: Per-year class fractions (%).
        n_subbasins: Number of placeholder subbasins.

    Returns:
        Superset HRU metadata dict.
    """
    log.info(f"Building Superset HRU: {n_subbasins} subbasins x {len(LULC_CLASSES)} classes")

    hru_list = []
    hru_id = 0

    for sub_id in range(1, n_subbasins + 1):
        for cls in LULC_CLASSES:
            hru_id += 1
            hru_list.append({
                "hru_id": hru_id,
                "subbasin": sub_id,
                "lulc_class": cls,
                "swat_code": LULC_TO_SWAT[cls],
            })

    # Build year-by-year fraction table
    fraction_table: dict[str, list[dict[str, float]]] = {}

    for year in sorted(fractions.keys()):
        year_fracs = fractions[year]
        year_rows = []
        for sub_id in range(1, n_subbasins + 1):
            for cls in LULC_CLASSES:
                pct = year_fracs.get(cls, 0.0)
                year_rows.append({
                    "hru_id": (sub_id - 1) * len(LULC_CLASSES) + LULC_CLASSES.index(cls) + 1,
                    "subbasin": sub_id,
                    "lulc_class": cls,
                    "fraction": round(pct / 100.0, 6),
                })
        fraction_table[str(year)] = year_rows

    superset = {
        "concept": "Superset HRU",
        "description": (
            "Each subbasin contains HRUs for all 6 LULC classes. "
            "HRU structure is fixed; only area fractions change yearly via lup.dat."
        ),
        "n_subbasins": n_subbasins,
        "n_lulc_classes": len(LULC_CLASSES),
        "total_hrus": len(hru_list),
        "hrus": hru_list,
        "fraction_table": fraction_table,
        "years": sorted(fractions.keys()),
        "note": (
            "All subbasins currently share AOI-level fractions. "
            "Spatial disaggregation requires QSWAT+ delineation output."
        ),
    }

    log.info(f"  Total HRUs: {len(hru_list)}")
    return superset


# ── lup.dat Writer ────────────────────────────────────────────

def write_lup_dat(
    superset: dict[str, Any],
    fractions: dict[int, dict[str, float]],
    n_subbasins: int,
) -> str:
    """Write SWAT+ lup.dat file.

    lup.dat format (SWAT+ revision 60.5+):
        Line 1: header comment
        Line 2: number of land use updates
        For each update:
            Line: year  day_start  day_end  n_changes
            For each change:
                Line: subbasin  hru_within_sub  new_fraction  land_use_code

    Since we update once per year (Jan 1), day_start=1, day_end=365.
    """
    log.info("Writing lup.dat")

    lines: list[str] = []

    # Header
    lines.append("Land use update file for Sejong LULC change 2017-2024")
    lines.append("Generated by t8_lup_dat_generator.py")
    lines.append("")

    # Count total updates (one per year, excluding first year which is baseline)
    update_years = sorted(y for y in fractions.keys() if y > min(fractions.keys()))
    lines.append(f"{len(update_years)}    | NUM_LU_CHANGE: number of land use change events")
    lines.append("")

    for year in update_years:
        year_fracs = fractions[year]
        n_changes = n_subbasins * len(LULC_CLASSES)

        lines.append(f"{'':4s}{'YEAR':>6s}{'JDAY_S':>8s}{'JDAY_E':>8s}{'N_CHG':>8s}")
        lines.append(f"{'':4s}{year:>6d}{'1':>8s}{'365':>8s}{n_changes:>8d}")
        lines.append(f"{'':4s}{'SUB':>6s}{'HRU':>6s}{'FRAC':>10s}{'LU_CODE':>10s}")

        for sub_id in range(1, n_subbasins + 1):
            for hru_idx, cls in enumerate(LULC_CLASSES, start=1):
                frac = year_fracs.get(cls, 0.0) / 100.0
                swat_code = LULC_TO_SWAT[cls]
                lines.append(
                    f"{'':4s}{sub_id:>6d}{hru_idx:>6d}{frac:>10.6f}{'':2s}{swat_code:<8s}"
                )

        lines.append("")

    lup_text = "\n".join(lines)
    LUP_DAT_PATH.write_text(lup_text, encoding="utf-8")
    log.info(f"  Saved: {LUP_DAT_PATH}")
    log.info(f"  Updates: {len(update_years)} years, {n_subbasins} subbasins")

    return lup_text


# ── Validation ────────────────────────────────────────────────

def validate_fractions(
    fractions: dict[int, dict[str, float]],
    urban_corrected: bool,
) -> dict[str, Any]:
    """Validate the fraction data for SWAT+ consistency.

    Checks:
      1. All fractions sum to ~100% per year
      2. No negative fractions
      3. Monotonic urban constraint (if correction applied)
      4. All 6 classes present every year
    """
    log.info("Validating fractions ...")

    issues: list[str] = []
    warnings: list[str] = []
    year_sums: dict[int, float] = {}

    # Check 1: Sum to 100%
    for year in sorted(fractions.keys()):
        total = sum(fractions[year].get(c, 0.0) for c in LULC_CLASSES)
        year_sums[year] = round(total, 4)
        if abs(total - 100.0) > 0.5:
            issues.append(f"{year}: fractions sum to {total:.2f}%, expected ~100%")
        elif abs(total - 100.0) > 0.1:
            warnings.append(f"{year}: fractions sum to {total:.4f}% (minor deviation)")

    # Check 2: No negatives
    for year in sorted(fractions.keys()):
        for cls in LULC_CLASSES:
            val = fractions[year].get(cls, 0.0)
            if val < 0:
                issues.append(f"{year}/{cls}: negative fraction {val:.4f}%")

    # Check 3: Monotonic urban
    urban_values = [fractions[y].get("Urban", 0.0) for y in sorted(fractions.keys())]
    urban_monotonic = all(
        urban_values[i] <= urban_values[i + 1] + 0.01  # small tolerance
        for i in range(len(urban_values) - 1)
    )
    if not urban_monotonic and urban_corrected:
        issues.append("Urban fraction is not monotonically non-decreasing after correction")
    elif not urban_monotonic:
        warnings.append(
            "Urban fraction is not monotonically non-decreasing "
            "(expected without --apply-urban-correction)"
        )

    # Check 4: All classes present
    for year in sorted(fractions.keys()):
        missing = [c for c in LULC_CLASSES if c not in fractions[year]]
        if missing:
            issues.append(f"{year}: missing classes: {missing}")

    # Overall verdict
    if issues:
        verdict = "FAIL"
    elif warnings:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "PASS"

    log.info(f"  Verdict: {verdict}")
    if issues:
        for iss in issues:
            log.error(f"  ISSUE: {iss}")
    if warnings:
        for w in warnings:
            log.warning(f"  WARNING: {w}")

    result = {
        "verdict": verdict,
        "n_issues": len(issues),
        "n_warnings": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "year_sums": year_sums,
        "urban_monotonic": urban_monotonic,
        "urban_values": {str(y): round(fractions[y].get("Urban", 0.0), 4)
                         for y in sorted(fractions.keys())},
    }

    VALIDATION_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  Saved: {VALIDATION_PATH}")
    return result


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8: Superset HRU + lup.dat generator")
    parser.add_argument("--subbasins", type=int, default=5,
                        help="Number of placeholder subbasins (default: 5)")
    parser.add_argument("--apply-urban-correction", action="store_true",
                        help="Apply monotonic non-decreasing urban correction")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Task 8: Superset HRU + lup.dat Generation")
    log.info(f"  Subbasins: {args.subbasins}")
    log.info(f"  Urban correction: {args.apply_urban_correction}")
    log.info("=" * 60)

    t0 = time.time()

    # 1. Load transfer results
    log.info("\n--- Loading transfer results ---")
    fractions = load_transfer_results()
    if not fractions:
        log.error("No transfer results found. Aborting.")
        sys.exit(1)
    log.info(f"  Loaded {len(fractions)} years: {sorted(fractions.keys())}")

    # 2. Apply urban correction if requested
    if args.apply_urban_correction:
        log.info("\n--- Applying urban correction ---")
        urban_trend = load_urban_correction()
        if urban_trend:
            fractions = apply_urban_correction(fractions, urban_trend)
        else:
            log.warning("  Urban trend data not found; skipping correction")

    # 3. Build Superset HRU
    log.info("\n--- Building Superset HRU ---")
    superset = build_superset_hru(fractions, args.subbasins)

    SUPERSET_PATH.write_text(
        json.dumps(superset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"  Saved: {SUPERSET_PATH}")

    # 4. Write lup.dat
    log.info("\n--- Writing lup.dat ---")
    write_lup_dat(superset, fractions, args.subbasins)

    # 5. Validate
    log.info("\n--- Validation ---")
    validation = validate_fractions(fractions, args.apply_urban_correction)

    # 6. Summary
    elapsed = time.time() - t0

    summary = {
        "script": "t8_lup_dat_generator.py",
        "run_time": _ts,
        "n_subbasins": args.subbasins,
        "n_years": len(fractions),
        "years": sorted(fractions.keys()),
        "urban_correction_applied": args.apply_urban_correction,
        "total_hrus": superset["total_hrus"],
        "validation": validation["verdict"],
        "outputs": {
            "lup_dat": str(LUP_DAT_PATH),
            "superset_hru": str(SUPERSET_PATH),
            "validation": str(VALIDATION_PATH),
        },
        "elapsed_sec": round(elapsed, 1),
    }

    summary_path = SWAT_DIR / "t8_lup_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(f"\n{'=' * 60}")
    log.info(f"Task 8 complete in {elapsed:.1f}s")
    log.info(f"  lup.dat: {LUP_DAT_PATH}")
    log.info(f"  Superset HRU: {SUPERSET_PATH}")
    log.info(f"  Validation: {validation['verdict']}")
    log.info(f"  Log: {LOG_FILE}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
