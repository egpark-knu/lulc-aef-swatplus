#!/usr/bin/env python3
"""
p2_12_annual_dynamic_experiment.py
Annual-Dynamic SWAT+ sensitivity experiment for Table 6 (LULC paper).

Design:
  For each year Y in 2017-2024:
    - Set LULC fractions to year-Y corrected values (monotonic urban)
    - Run SWAT+ over full 2016-2024 period
    - Extract year-Y row from basin_wb_yr.txt (diagonal extraction)

  This gives an "Annual-Dynamic" trajectory:
    year 2017 → simulation with 2017 LULC → extract 2017 output
    year 2018 → simulation with 2018 LULC → extract 2018 output
    ...
    year 2024 → simulation with 2024 LULC → extract 2024 output

Both Hwaseong and Sejong basins (Table 6, 6 scenarios × 2 sites).

Results saved to:
  phase2/data/annual_dynamic_experiment/
    hwaseong/annual_dynamic_{year}/  (8 subdirs, each with basin_wb_yr.txt)
    sejong/annual_dynamic_{year}/
    hwaseong_annual_dynamic_trajectory.json
    sejong_annual_dynamic_trajectory.json
    comparison_table6.json
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from _climate_install import install_site_climate  # noqa: E402


def modify_time_sim(work_dir: Path, yrc_start: int, yrc_end: int) -> None:
    """Overwrite time.sim so all scenarios share the same simulation window.

    Must match p2_14.SIM_YRC_START / SIM_YRC_END, otherwise Continuous and
    Static use different weather realizations (different WGN stream or
    different warm-up spin-up years) and are not directly comparable.
    """
    content = (
        "time.sim: written by p2_12 for scenario consistency\n"
        "day_start  yrc_start   day_end   yrc_end      step  \n"
        f"       0      {yrc_start}         0      {yrc_end}         0\n"
    )
    (work_dir / "time.sim").write_text(content)


# Must stay identical to p2_14.SIM_YRC_START / SIM_YRC_END.
SIM_YRC_START = 2016
SIM_YRC_END = 2024

# ───────────────────────────────────────────────────────────────
# Add p2_09 to path so we can reuse its functions
# ───────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))

# Patch config constants before importing p2_09
import importlib.util

# We'll import needed functions via direct import of the module
# (copy key standalone functions here to avoid config conflicts)

import collections
import re

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
PROJECT = Path("/Users/eungyupark/Dropbox/myproj/dev_260402_LULC")
PHASE2 = PROJECT / "phase2"
PHASE1 = PROJECT / "phase1"

SEOUL_TXTINOUT = Path(
    "/Users/eungyupark/Dropbox/Manuscripts/0_swat_mcp/swatplus-model/TxtInOut"
)
SWATPLUS_EXE = Path(
    "/Users/eungyupark/Dropbox/myproj/dev_260222/bin/swatplus-intel"
)
# ARM64 binaries fail with "bad CPU type"; use Intel binary via Rosetta 2
SWATPLUS_CMD = ["arch", "-x86_64", str(SWATPLUS_EXE)]
CLIMATE_DIR = PHASE2 / "data" / "climate_swat"
RESULTS_DIR = PHASE2 / "data" / "annual_dynamic_experiment"
LOG_DIR = PHASE2 / "logs"

HWASEONG_AEF = PHASE2 / "data" / "hwaseong" / "hwaseong_classification_results.json"
SEJONG_AEF = PHASE1 / "data" / "sejong_classification_results.json"

YEARS = list(range(2017, 2025))  # 2017-2024

for _d in [RESULTS_DIR, RESULTS_DIR / "hwaseong", RESULTS_DIR / "sejong", LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# LUM mapping (copied from p2_09)
# ───────────────────────────────────────────────────────────────
LUM_TO_SIXCLASS = {
    "frse_lum": "Forest", "frst_lum": "Forest",
    "urml_lum": "Urban", "urhd_lum": "Urban",
    "ucom_lum": "Urban", "utrn_lum": "Urban",
    "uidu_lum": "Urban", "urld_lum": "Urban",
    "past_lum": "Grassland", "watr_lum": "Water",
}
URBAN_SUBTYPE_WEIGHTS = {
    "urml_lum": 0.45, "urhd_lum": 0.20,
    "ucom_lum": 0.20, "utrn_lum": 0.10,
    "uidu_lum": 0.03, "urld_lum": 0.02,
}

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"p2_12_annual_dynamic_{_ts}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# HRU analysis (copied from p2_09)
# ───────────────────────────────────────────────────────────────
def parse_hru_data(txtinout: Path) -> list[dict]:
    fpath = txtinout / "hru-data.hru"
    lines = fpath.read_text().splitlines()
    headers = lines[1].split()
    hrus = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        row = {h: parts[i] for i, h in enumerate(headers) if i < len(parts)}
        hrus.append(row)
    return hrus


def parse_hru_con(txtinout: Path) -> list[dict]:
    fpath = txtinout / "hru.con"
    lines = fpath.read_text().splitlines()
    headers = lines[1].split()
    hrus = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        row = {}
        for i, h in enumerate(headers):
            if i < len(parts):
                try:
                    row[h] = float(parts[i])
                except ValueError:
                    row[h] = parts[i]
        hrus.append(row)
    return hrus


def analyse_hru_structure(txtinout: Path) -> dict:
    hru_data = parse_hru_data(txtinout)
    hru_con = parse_hru_con(txtinout)
    area_by_id = {str(int(hc.get("id", 0))): float(hc.get("area", 0.0)) for hc in hru_con}
    hru_list = []
    lu_areas: dict[str, float] = {}
    for hd in hru_data:
        hid = str(hd.get("id", "0"))
        lu = hd.get("lu_mgt", "unknown")
        area = area_by_id.get(hid, 0.0)
        lu_areas[lu] = lu_areas.get(lu, 0.0) + area
        hru_list.append({"id": int(hid), "name": hd.get("name", ""),
                         "lu_mgt": lu, "soil": hd.get("soil", ""), "area_ha": area})
    total_area = sum(lu_areas.values())
    return {"hru_list": hru_list, "n_hrus": len(hru_list),
            "total_area": total_area, "lu_areas": lu_areas}


def compute_4class_targets(aef_cf: dict) -> dict:
    """Convert raw class_fractions → 4-class SWAT+ targets."""
    forest = aef_cf.get("Forest", 0)
    cropland = aef_cf.get("Cropland", 0)
    urban = aef_cf.get("Urban", 0)
    # Barren → Grassland (consistent with p2_09; neither crop nor open water)
    grass = aef_cf.get("Grassland", 0) + cropland + aef_cf.get("Barren", 0)
    water = aef_cf.get("Water", 0)
    total = forest + urban + grass + water
    if total < 1e-6:
        total = 100.0
    return {
        "Urban": urban / total * 100,
        "Forest": forest / total * 100,
        "Grassland": grass / total * 100,
        "Water": water / total * 100,
    }


def compute_corrected_4class(raw_cf: dict, corrected_urban: float) -> dict:
    """Apply monotonic urban correction, adjust Forest proportionally."""
    raw_urban = raw_cf.get("Urban", 0)
    forest = raw_cf.get("Forest", 0)
    cropland = raw_cf.get("Cropland", 0)
    # Barren → Grassland (consistent with p2_09)
    grass = raw_cf.get("Grassland", 0) + cropland + raw_cf.get("Barren", 0)
    water = raw_cf.get("Water", 0)
    # If urban increases, reduce forest proportionally
    delta_urban = corrected_urban - raw_urban
    corrected_forest = max(0, forest - delta_urban)
    total = corrected_urban + corrected_forest + grass + water
    if total < 1e-6:
        total = 100.0
    return {
        "Urban": corrected_urban / total * 100,
        "Forest": corrected_forest / total * 100,
        "Grassland": grass / total * 100,
        "Water": water / total * 100,
    }


def redistribute_hrus(hru_analysis: dict, target_fractions: dict, scenario_name: str) -> list[dict]:
    hru_list = hru_analysis["hru_list"]
    total_area = hru_analysis["total_area"]
    n_hrus = hru_analysis["n_hrus"]
    water_hrus = [h for h in hru_list if h["lu_mgt"] == "watr_lum"]
    non_water_hrus = [h for h in hru_list if h["lu_mgt"] != "watr_lum"]
    water_area = sum(h["area_ha"] for h in water_hrus)
    non_water_area = total_area - water_area
    water_frac_actual = water_area / total_area * 100
    target_urban_raw = target_fractions.get("Urban", 40)
    target_forest_raw = target_fractions.get("Forest", 40)
    target_grass_raw = target_fractions.get("Grassland", 10)
    sum_3 = target_urban_raw + target_forest_raw + target_grass_raw
    if sum_3 < 1e-6:
        sum_3 = 100
    target_areas = {
        "Urban": target_urban_raw / sum_3 * non_water_area,
        "Forest": target_forest_raw / sum_3 * non_water_area,
        "Grassland": target_grass_raw / sum_3 * non_water_area,
    }
    sorted_hrus = sorted(non_water_hrus, key=lambda h: h["id"])
    filled_areas = {"Urban": 0.0, "Forest": 0.0, "Grassland": 0.0}
    assignments: list[dict] = []
    for h in sorted_hrus:
        area = h["area_ha"]
        deficits = {cls: target_areas[cls] - filled_areas[cls] for cls in ["Urban", "Forest", "Grassland"]}
        best_cls = max(deficits, key=lambda c: deficits[c])
        filled_areas[best_cls] += area
        assignments.append({"id": h["id"], "name": h["name"], "original_lu": h["lu_mgt"],
                             "soil": h["soil"], "area_ha": area, "assigned_class": best_cls})
    # Assign lu_mgt types
    urban_hrus = [a for a in assignments if a["assigned_class"] == "Urban"]
    urban_total_area = sum(h["area_ha"] for h in urban_hrus)
    urban_subtypes = list(URBAN_SUBTYPE_WEIGHTS.keys())
    urban_weights = list(URBAN_SUBTYPE_WEIGHTS.values())
    cum_weights = []
    s = 0
    for w in urban_weights:
        s += w
        cum_weights.append(s)
    urban_filled = 0.0
    for uh in urban_hrus:
        ratio = urban_filled / urban_total_area if urban_total_area > 0 else 0
        chosen = urban_subtypes[0]
        for i, cw in enumerate(cum_weights):
            if ratio <= cw:
                chosen = urban_subtypes[i]
                break
        uh["new_lu_mgt"] = chosen
        urban_filled += uh["area_ha"]
    for a in assignments:
        if a["assigned_class"] == "Forest":
            a["new_lu_mgt"] = "frse_lum"
        elif a["assigned_class"] == "Grassland":
            a["new_lu_mgt"] = "past_lum"
    for wh in water_hrus:
        assignments.append({"id": wh["id"], "name": wh["name"], "original_lu": wh["lu_mgt"],
                             "soil": wh["soil"], "area_ha": wh["area_ha"],
                             "assigned_class": "Water", "new_lu_mgt": "watr_lum"})
    assignments.sort(key=lambda a: a["id"])
    return assignments


def write_modified_hru_data(work_dir: Path, assignments: list[dict], original_txtinout: Path) -> None:
    orig_lines = (original_txtinout / "hru-data.hru").read_text().splitlines()
    headers = orig_lines[1].split()
    lu_mgt_idx = headers.index("lu_mgt") if "lu_mgt" in headers else None
    id_idx = headers.index("id") if "id" in headers else 0
    lu_map = {a["id"]: a["new_lu_mgt"] for a in assignments}
    new_lines = [orig_lines[0], orig_lines[1]]
    for line in orig_lines[2:]:
        parts = line.split()
        if not parts or lu_mgt_idx is None:
            new_lines.append(line)
            continue
        try:
            hid = int(parts[id_idx])
        except (ValueError, IndexError):
            new_lines.append(line)
            continue
        if hid in lu_map and lu_mgt_idx < len(parts):
            parts[lu_mgt_idx] = lu_map[hid]
        new_lines.append("   ".join(parts))
    (work_dir / "hru-data.hru").write_text("\n".join(new_lines) + "\n")


def copy_txtinout(src: Path, label: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"swat_annual_{label}_"))
    for fpath in src.iterdir():
        if fpath.is_file():
            shutil.copy2(fpath, tmp_dir / fpath.name)
    return tmp_dir


def install_sejong_climate(work_dir: Path) -> None:
    for fname in ["sejong.pcp", "sejong.tmp", "sejong.slr", "sejong.hmd", "sejong.wnd"]:
        src = CLIMATE_DIR / fname
        if src.exists():
            shutil.copy2(src, work_dir / fname)
    wst_content = (
        "weather-sta.cli: Hwaseong experiment (Sejong KMA climate)\n"
        "name                           wgn                        pcp                        "
        "tmp                        slr                        hmd                        "
        "wnd                        pet          atmo_dep  \n"
        "sejong_sta                    seoul108.wgn               sejong.pcp                 "
        "sejong.tmp                 sejong.slr                 sejong.hmd                 "
        "sejong.wnd                 hargreaves   no\n"
    )
    (work_dir / "weather-sta.cli").write_text(wst_content)


def run_swatplus(work_dir: Path, timeout: int = 300) -> dict:
    try:
        result = subprocess.run(
            SWATPLUS_CMD, cwd=str(work_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "elapsed_sec": 0,
            "stdout_tail": result.stdout[-1000:] if result.stdout else "",
            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "exit_code": -1, "elapsed_sec": timeout,
                "stdout_tail": "", "stderr_tail": "TIMEOUT"}
    except OSError as exc:
        return {"success": False, "exit_code": -2, "elapsed_sec": 0,
                "stdout_tail": "", "stderr_tail": str(exc)}


def parse_wb_file(fpath: Path) -> list[dict]:
    if not fpath.exists():
        return []
    lines = fpath.read_text().splitlines()
    if len(lines) < 4:
        return []
    headers = lines[1].split()
    rows = []
    for line in lines[3:]:
        parts = line.split()
        if not parts:
            continue
        try:
            float(parts[0])
        except ValueError:
            continue
        row = {}
        for i, h in enumerate(headers):
            if i < len(parts):
                try:
                    row[h] = float(parts[i])
                except ValueError:
                    row[h] = parts[i]
        rows.append(row)
    return rows


def extract_year_row(wb_yr: list[dict], target_year: int) -> dict | None:
    """Extract the row for a specific year from basin_wb_yr output."""
    for row in wb_yr:
        if int(row.get("yr", 0)) == target_year:
            return row
    return None


# ───────────────────────────────────────────────────────────────
# Annual AEF fractions loader
# ───────────────────────────────────────────────────────────────
def load_annual_fractions(aef_path: Path, use_corrected_urban: bool = True) -> dict:
    """Load annual class fractions from classification results JSON.

    Returns: {year_str: {class: fraction%}}
    """
    with open(aef_path) as f:
        data = json.load(f)
    transfer = data.get("transfer", {})
    urban_corrected = data.get("urban_trend", {}).get("corrected", {})
    annual_fracs = {}
    for yr_str, yr_data in transfer.items():
        if yr_str not in [str(y) for y in YEARS]:
            continue
        cf = yr_data.get("class_fractions", {})
        if use_corrected_urban and yr_str in urban_corrected:
            corrected_urban = float(urban_corrected[yr_str])
            targets = compute_corrected_4class(cf, corrected_urban)
        else:
            targets = compute_4class_targets(cf)
        annual_fracs[yr_str] = targets
    return annual_fracs


# ───────────────────────────────────────────────────────────────
# Main Annual-Dynamic runner
# ───────────────────────────────────────────────────────────────
def run_annual_dynamic(site_name: str, aef_path: Path, site_results_dir: Path) -> dict:
    """Run 8 SWAT+ simulations (one per year) and extract diagonal values."""
    log.info(f"\n{'='*70}")
    log.info(f"  ANNUAL-DYNAMIC EXPERIMENT: {site_name.upper()}")
    log.info(f"{'='*70}")

    # Analyse HRU structure once
    hru_analysis = analyse_hru_structure(SEOUL_TXTINOUT)
    log.info(f"  HRU count: {hru_analysis['n_hrus']}")

    # Load annual fractions
    annual_fracs = load_annual_fractions(aef_path, use_corrected_urban=True)
    log.info(f"  Annual fractions loaded: {sorted(annual_fracs.keys())}")

    trajectory = {}  # year → water balance dict

    for yr in YEARS:
        yr_str = str(yr)
        if yr_str not in annual_fracs:
            log.warning(f"  No fractions for year {yr}, skipping")
            continue

        target_fractions = annual_fracs[yr_str]
        log.info(f"\n  --- Year {yr}: U={target_fractions['Urban']:.1f}% F={target_fractions['Forest']:.1f}%"
                 f" G={target_fractions['Grassland']:.1f}% W={target_fractions['Water']:.1f}% ---")

        # Check if already computed
        saved_dir = site_results_dir / f"annual_dynamic_{yr}"
        saved_wb = saved_dir / "basin_wb_yr.txt"
        if saved_wb.exists():
            log.info(f"  Cached: {saved_wb}")
            wb_yr = parse_wb_file(saved_wb)
        else:
            # Run simulation
            work_dir = copy_txtinout(SEOUL_TXTINOUT, f"{site_name}_{yr}")
            try:
                assignments = redistribute_hrus(hru_analysis, target_fractions, f"{site_name}_{yr}")
                write_modified_hru_data(work_dir, assignments, SEOUL_TXTINOUT)

                start_t = datetime.now()
                sim = run_swatplus(work_dir, timeout=600)
                elapsed = (datetime.now() - start_t).total_seconds()
                log.info(f"  SWAT+ {'OK' if sim['success'] else 'FAILED'} in {elapsed:.1f}s")

                if sim["success"]:
                    # Save outputs
                    saved_dir.mkdir(parents=True, exist_ok=True)
                    for fname in ["basin_wb_aa.txt", "basin_wb_yr.txt", "checker.out", "simulation.out"]:
                        src = work_dir / fname
                        if src.exists():
                            shutil.copy2(src, saved_dir / fname)
                    # Also save hru-data.hru for reproducibility
                    shutil.copy2(work_dir / "hru-data.hru", saved_dir / "hru-data.hru")
                    wb_yr = parse_wb_file(saved_dir / "basin_wb_yr.txt")
                else:
                    log.error(f"  FAILED: {sim['stderr_tail'][:200]}")
                    wb_yr = []
            finally:
                try:
                    shutil.rmtree(work_dir)
                except OSError:
                    pass

        # Extract diagonal: year Y from simulation with year Y's LULC
        row = extract_year_row(wb_yr, yr)
        if row:
            trajectory[yr_str] = {
                "target_fractions": target_fractions,
                "wb": {k: round(float(v), 3) for k, v in row.items()
                       if k in ["precip", "surq_gen", "latq", "wateryld", "perc", "et", "pet", "sw_ave"]},
            }
            wb = trajectory[yr_str]["wb"]
            log.info(f"  Year {yr}: P={wb.get('precip', 0):.0f} SQ={wb.get('surq_gen', 0):.1f}"
                     f" LQ={wb.get('latq', 0):.1f} WY={wb.get('wateryld', 0):.1f}"
                     f" ET={wb.get('et', 0):.1f} mm")
        else:
            log.warning(f"  Year {yr}: no output row found in basin_wb_yr.txt")

    return {
        "site": site_name,
        "generated": datetime.now().isoformat(),
        "method": "diagonal_extraction_corrected_urban",
        "years": YEARS,
        "trajectory": trajectory,
    }


# ───────────────────────────────────────────────────────────────
# Static/Dynamic re-reader (for comparison)
# ───────────────────────────────────────────────────────────────
def read_static_dynamic_trajectories(site_name: str) -> dict:
    """Read existing static/dynamic basin_wb_yr.txt files."""
    if site_name == "hwaseong":
        exp_dir = PHASE2 / "data" / "hwaseong_experiment"
        static_dir = exp_dir / "static"
        dynamic_dir = exp_dir / "dynamic"
    else:
        exp_dir = PHASE2 / "data" / "sejong_swat_results"
        static_dir = exp_dir / "sejong_static"
        dynamic_dir = exp_dir / "sejong_dynamic"

    result = {}
    for scenario, src_dir in [("Static-2017", static_dir), ("Static-2024", dynamic_dir)]:
        wb_yr_path = src_dir / "basin_wb_yr.txt"
        if not wb_yr_path.exists():
            log.warning(f"  Missing: {wb_yr_path}")
            continue
        wb_yr = parse_wb_file(wb_yr_path)
        traj = {}
        for row in wb_yr:
            yr = int(row.get("yr", 0))
            if 2017 <= yr <= 2024:
                traj[str(yr)] = {k: round(float(v), 3) for k, v in row.items()
                                  if k in ["precip", "surq_gen", "latq", "wateryld", "perc", "et", "sw_ave"]}
        result[scenario] = traj
    return result


# ───────────────────────────────────────────────────────────────
# Sejong endpoint re-runs (HRU redistribution, consistent method)
# ───────────────────────────────────────────────────────────────
def run_endpoint_scenario(site_name: str, year: int, aef_path: Path,
                           out_dir: Path, hru_analysis: dict,
                           install_climate: bool = True) -> dict:
    """Run one static scenario (endpoint year) and return all-year WB trajectory.

    install_climate: if True (default), installs ERA5-Land climate for the site
        (hwaseong.* or sejong.*). Set False only to keep the template's embedded
        climate (e.g., Seoul108pcp.pcp) for legacy reproduction.
    """
    annual_fracs = load_annual_fractions(aef_path, use_corrected_urban=True)
    yr_str = str(year)
    if yr_str not in annual_fracs:
        log.warning(f"  No fractions for endpoint year {year}")
        return {}
    target_fractions = annual_fracs[yr_str]
    label = f"{site_name}_static{year}"
    saved_dir = out_dir / f"static_{year}"
    saved_wb = saved_dir / "basin_wb_yr.txt"
    if saved_wb.exists():
        log.info(f"  Cached: {label}")
        wb_yr = parse_wb_file(saved_wb)
    else:
        log.info(f"  Running {label}: U={target_fractions['Urban']:.1f}%"
                 f" F={target_fractions['Forest']:.1f}%")
        work_dir = copy_txtinout(SEOUL_TXTINOUT, label)
        try:
            assignments = redistribute_hrus(hru_analysis, target_fractions, label)
            write_modified_hru_data(work_dir, assignments, SEOUL_TXTINOUT)
            if install_climate:
                install_site_climate(site_name, work_dir)
                log.info(f"    Climate: {site_name} ERA5-Land installed")
            modify_time_sim(work_dir, SIM_YRC_START, SIM_YRC_END)
            start_t = datetime.now()
            sim = run_swatplus(work_dir, timeout=600)
            elapsed = (datetime.now() - start_t).total_seconds()
            log.info(f"  SWAT+ {'OK' if sim['success'] else 'FAILED'} in {elapsed:.1f}s")
            if sim["success"]:
                saved_dir.mkdir(parents=True, exist_ok=True)
                for fname in ["basin_wb_aa.txt", "basin_wb_yr.txt"]:
                    src = work_dir / fname
                    if src.exists():
                        shutil.copy2(src, saved_dir / fname)
                wb_yr = parse_wb_file(saved_dir / "basin_wb_yr.txt")
            else:
                log.error(f"  FAILED: {sim['stderr_tail'][:200]}")
                wb_yr = []
        finally:
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass
    traj = {}
    for row in wb_yr:
        yr = int(row.get("yr", 0))
        if 2017 <= yr <= 2024:
            traj[str(yr)] = {k: round(float(v), 3) for k, v in row.items()
                              if k in ["precip", "surq_gen", "latq", "wateryld", "perc", "et", "sw_ave"]}
    return traj


# ───────────────────────────────────────────────────────────────
# Table 6 comparison builder
# ───────────────────────────────────────────────────────────────
def build_table6(hw_dyn: dict, sj_dyn: dict,
                 sejong_static_endpoints: dict | None = None,
                 hwaseong_static_endpoints: dict | None = None) -> dict:
    """Build Table 6 comparison: annual average + per-scenario stats."""
    def avg_wb(trajectory: dict) -> dict:
        keys = ["precip", "surq_gen", "latq", "wateryld", "perc", "et"]
        sums = {k: 0.0 for k in keys}
        n = 0
        for yr_data in trajectory.values():
            wb = yr_data if "precip" in yr_data else yr_data.get("wb", {})
            for k in keys:
                sums[k] += float(wb.get(k, 0))
            n += 1
        return {k: round(v / n, 2) if n > 0 else 0 for k, v in sums.items()}

    table = {"generated": datetime.now().isoformat(), "sites": {}}

    for site_name, annual_dyn_result, aef_path in [
        ("hwaseong", hw_dyn, HWASEONG_AEF),
        ("sejong", sj_dyn, SEJONG_AEF),
    ]:
        # Prefer freshly-run HRU-redistribution endpoints over legacy ESCO-based runs
        if site_name == "hwaseong" and hwaseong_static_endpoints is not None:
            static_dynamic = hwaseong_static_endpoints
        elif site_name == "sejong" and sejong_static_endpoints is not None:
            static_dynamic = sejong_static_endpoints
        else:
            static_dynamic = read_static_dynamic_trajectories(site_name)

        # Annual-Dynamic trajectory
        ad_traj = {yr: v["wb"] for yr, v in annual_dyn_result["trajectory"].items()}

        table["sites"][site_name] = {
            "Static-2017": {
                "annual_avg": avg_wb(static_dynamic.get("Static-2017", {})),
                "yearly": static_dynamic.get("Static-2017", {}),
            },
            "Annual-Dynamic": {
                "annual_avg": avg_wb(ad_traj),
                "yearly": ad_traj,
                "lulc_fractions": {yr: v["target_fractions"]
                                    for yr, v in annual_dyn_result["trajectory"].items()},
            },
            "Static-2024": {
                "annual_avg": avg_wb(static_dynamic.get("Static-2024", {})),
                "yearly": static_dynamic.get("Static-2024", {}),
            },
        }

        # Compute deltas: Annual-Dynamic vs Static-2017
        s17 = table["sites"][site_name]["Static-2017"]["annual_avg"]
        ad = table["sites"][site_name]["Annual-Dynamic"]["annual_avg"]
        s24 = table["sites"][site_name]["Static-2024"]["annual_avg"]

        for key in ["surq_gen", "latq", "wateryld", "perc", "et"]:
            s17v = s17.get(key, 0) or 1e-6
            s24v = s24.get(key, 0) or 1e-6
            adv = ad.get(key, 0)
            log.info(f"  {site_name} {key}: S17={s17v:.1f} AD={adv:.1f} S24={s24v:.1f}"
                     f"  [AD-S17={adv-s17v:+.1f}, {(adv-s17v)/s17v*100:+.1f}%]")

    return table


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  p2_12: Annual-Dynamic SWAT+ Experiment (Table 6)")
    log.info("=" * 70)

    # Run Hwaseong Annual-Dynamic
    hw_result = run_annual_dynamic(
        "hwaseong", HWASEONG_AEF, RESULTS_DIR / "hwaseong"
    )
    with open(RESULTS_DIR / "hwaseong_annual_dynamic.json", "w") as f:
        json.dump(hw_result, f, indent=2)
    log.info(f"\n  Hwaseong Annual-Dynamic saved.")

    # Run Sejong Annual-Dynamic
    sj_result = run_annual_dynamic(
        "sejong", SEJONG_AEF, RESULTS_DIR / "sejong"
    )
    with open(RESULTS_DIR / "sejong_annual_dynamic.json", "w") as f:
        json.dump(sj_result, f, indent=2)
    log.info(f"\n  Sejong Annual-Dynamic saved.")

    # Re-run static endpoints with HRU redistribution (consistent with Annual-Dynamic)
    log.info("\n  Running static endpoint re-runs (HRU redistribution)...")
    hru_analysis = analyse_hru_structure(SEOUL_TXTINOUT)

    # Hwaseong endpoints
    hw_endpoints_dir = RESULTS_DIR / "hwaseong"
    hw_s17_traj = run_endpoint_scenario("hwaseong", 2017, HWASEONG_AEF,
                                         hw_endpoints_dir, hru_analysis)
    hw_s24_traj = run_endpoint_scenario("hwaseong", 2024, HWASEONG_AEF,
                                         hw_endpoints_dir, hru_analysis)
    hw_endpoints = {"Static-2017": hw_s17_traj, "Static-2024": hw_s24_traj}
    with open(RESULTS_DIR / "hwaseong_static_endpoints.json", "w") as f:
        json.dump(hw_endpoints, f, indent=2)
    log.info(f"  Hwaseong endpoint re-runs saved.")

    # Sejong endpoints
    sj_endpoints_dir = RESULTS_DIR / "sejong"
    sj_s17_traj = run_endpoint_scenario("sejong", 2017, SEJONG_AEF,
                                         sj_endpoints_dir, hru_analysis)
    sj_s24_traj = run_endpoint_scenario("sejong", 2024, SEJONG_AEF,
                                         sj_endpoints_dir, hru_analysis)
    sj_endpoints = {"Static-2017": sj_s17_traj, "Static-2024": sj_s24_traj}
    with open(RESULTS_DIR / "sejong_static_endpoints.json", "w") as f:
        json.dump(sj_endpoints, f, indent=2)
    log.info(f"  Sejong endpoint re-runs saved.")

    # Build comparison table with freshly-run endpoints for both sites
    log.info("\n  Building Table 6 comparison...")
    table6 = build_table6(hw_result, sj_result,
                           sejong_static_endpoints=sj_endpoints,
                           hwaseong_static_endpoints=hw_endpoints)
    with open(RESULTS_DIR / "comparison_table6.json", "w") as f:
        json.dump(table6, f, indent=2)

    log.info(f"\n{'='*70}")
    log.info(f"  COMPLETE — results in {RESULTS_DIR}")
    log.info(f"{'='*70}")

    # Print summary
    print("\n=== TABLE 6 ANNUAL AVERAGES (mm/yr) ===")
    for site in ["hwaseong", "sejong"]:
        print(f"\n{site.upper()}")
        for scen in ["Static-2017", "Annual-Dynamic", "Static-2024"]:
            aa = table6["sites"][site][scen]["annual_avg"]
            print(f"  {scen:20s}: SQ={aa.get('surq_gen',0):6.1f}  LQ={aa.get('latq',0):6.1f}"
                  f"  WY={aa.get('wateryld',0):6.1f}  ET={aa.get('et',0):6.1f}"
                  f"  Perc={aa.get('perc',0):6.1f}")


if __name__ == "__main__":
    main()
