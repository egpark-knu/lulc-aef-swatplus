#!/usr/bin/env python3
"""
p2_14e_era5_unified_rerun.py
Rerun ALL LULC scenarios (orig + patched, continuous + static, hwaseong + sejong)
with a unified ERA5-Land climate per site, addressing the KMA-obs vs ERA5
confound raised by the user ("全部用 ERA5").

Scenarios per site (6 total per template, 12 total):
  Continuous-Native  (p2_14.run_continuous_native)
  Static-2017        (p2_12.run_endpoint_scenario)
  Static-2024        (p2_12.run_endpoint_scenario)

Templates:
  TxtInOut                        -> output/continuous_native_experiment_era5/
                                     output/annual_dynamic_experiment_era5/
  TxtInOut_patched_urbanveg       -> output/continuous_native_experiment_urbanveg/

Climate installer: _climate_install.install_site_climate (ERA5-Land for both sites).

Outputs:
  compare_orig_vs_patched_era5.json   — matches the old compare_orig_vs_patched.json
                                         layout but with unified ERA5 climate
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

PROJECT = Path("/Users/eungyupark/Dropbox/myproj/dev_260402_LULC")
PHASE2 = PROJECT / "phase2"

ORIG_TXTINOUT = Path(
    "/Users/eungyupark/Dropbox/Manuscripts/0_swat_mcp/swatplus-model/TxtInOut"
)
PATCHED_TXTINOUT = Path(
    "/Users/eungyupark/Dropbox/Manuscripts/0_swat_mcp/swatplus-model/TxtInOut_patched_urbanveg"
)
for p in (ORIG_TXTINOUT, PATCHED_TXTINOUT):
    assert p.exists(), f"Template not found: {p}"

ORIG_OUT = PHASE2 / "data" / "continuous_native_experiment_era5"
PATCHED_OUT = PHASE2 / "data" / "continuous_native_experiment_urbanveg"
ORIG_STATIC_OUT = PHASE2 / "data" / "annual_dynamic_experiment_era5"
for d in (ORIG_OUT, PATCHED_OUT, ORIG_STATIC_OUT):
    d.mkdir(parents=True, exist_ok=True)

LOG_DIR = PHASE2 / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"p2_14e_era5_unified_{_ts}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

import p2_12_annual_dynamic_experiment as p2_12
import p2_13_continuous_sequential as p2_13
import p2_14_continuous_lulc_update as p2_14


WB_VARS = ["precip", "surq_gen", "latq", "wateryld", "perc",
           "et", "ecanopy", "eplant", "esoil", "pet", "sw_ave"]


def parse_wb_yr(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 4:
        return []
    header = lines[1].split()
    rows = []
    for ln in lines[3:]:
        toks = ln.split()
        if len(toks) < 20:
            continue
        rows.append(dict(zip(header, toks)))
    return rows


def avg_over_years(rows: list[dict], years: list[int]) -> dict:
    out = {}
    for col in WB_VARS:
        vals = []
        for r in rows:
            try:
                if int(r.get("yr", 0)) in years:
                    vals.append(float(r[col]))
            except (ValueError, KeyError):
                continue
        out[col] = round(sum(vals) / len(vals), 3) if vals else float("nan")
    return out


def run_scenarios_for_template(
    tag: str, template: Path, cont_out: Path, static_out: Path,
) -> dict:
    log.info(f"\n{'#'*72}\n# TEMPLATE = {tag}  ({template.name})\n{'#'*72}")

    # Monkey-patch SEOUL_TXTINOUT in all three modules
    for mod in (p2_12, p2_13, p2_14):
        mod.SEOUL_TXTINOUT = template

    p2_14.RESULTS_DIR = cont_out
    p2_12.RESULTS_DIR = static_out

    results = {"continuous": {}, "static": {}}

    # Continuous-Native (install ERA5 per site)
    for site, aef in [("hwaseong", p2_14.HWASEONG_AEF), ("sejong", p2_14.SEJONG_AEF)]:
        log.info(f"\n--- {tag}: continuous-native {site} ---")
        site_dir = cont_out / site
        # Clear prior continuous outputs so parse is fresh
        for fname in ("basin_wb_yr.txt", "basin_wb_aa.txt"):
            fp = site_dir / fname
            if fp.exists():
                fp.unlink()
        r = p2_14.run_continuous_native(site, aef, site_dir, install_climate=True)
        results["continuous"][site] = r
        with open(cont_out / f"{site}_continuous_native.json", "w") as f:
            json.dump(r, f, indent=2)

    # Static endpoints (install ERA5 per site)
    for site, aef in [("hwaseong", p2_14.HWASEONG_AEF), ("sejong", p2_14.SEJONG_AEF)]:
        log.info(f"\n--- {tag}: static endpoints {site} ---")
        hru_analysis = p2_12.analyse_hru_structure(template)
        site_out = static_out / site
        # Clear cached static results
        for yr in (2017, 2024):
            d = site_out / f"static_{yr}"
            if d.exists():
                shutil.rmtree(d)
        traj = {}
        for yr in (2017, 2024):
            t = p2_12.run_endpoint_scenario(
                site, yr, aef, site_out, hru_analysis, install_climate=True
            )
            traj[str(yr)] = t
        results["static"][site] = traj
        with open(static_out / f"{site}_static_endpoints.json", "w") as f:
            json.dump(traj, f, indent=2)

    return results


def build_compare_json(orig: dict, patched: dict) -> dict:
    out = {"generated": datetime.now().isoformat(),
           "climate": "unified ERA5-Land (hwaseong.* + sejong.*)",
           "templates": {"orig": "TxtInOut",
                         "patched": "TxtInOut_patched_urbanveg"}}
    years = list(range(2017, 2025))
    out["sites"] = {}
    for site in ("hwaseong", "sejong"):
        site_block = {}
        for scen_tag, key in [("Continuous-Native", "continuous"),
                              ("Static-2017",       "static_2017"),
                              ("Static-2024",       "static_2024")]:
            if scen_tag == "Continuous-Native":
                orig_rows = parse_wb_yr(ORIG_OUT / site / "basin_wb_yr.txt")
                patch_rows = parse_wb_yr(PATCHED_OUT / site / "basin_wb_yr.txt")
            else:
                yr = 2017 if scen_tag.endswith("2017") else 2024
                orig_rows = parse_wb_yr(ORIG_STATIC_OUT / site / f"static_{yr}" / "basin_wb_yr.txt")
                patch_rows = parse_wb_yr(PATCHED_OUT / site / f"static_{yr}" / "basin_wb_yr.txt")
            orig_avg = avg_over_years(orig_rows, years)
            patch_avg = avg_over_years(patch_rows, years)
            delta = {}
            for k in WB_VARS:
                try:
                    delta[k] = round(patch_avg[k] - orig_avg[k], 3)
                except TypeError:
                    delta[k] = float("nan")
            site_block[scen_tag] = {
                "orig": orig_avg,
                "patched": patch_avg,
                "delta_patched_minus_orig": delta,
            }
        out["sites"][site] = site_block
    return out


def main():
    # 1. Patched template (all ERA5)
    patched = run_scenarios_for_template(
        "PATCHED", PATCHED_TXTINOUT, PATCHED_OUT, PATCHED_OUT,
    )
    # 2. Orig template (all ERA5)
    orig = run_scenarios_for_template(
        "ORIG", ORIG_TXTINOUT, ORIG_OUT, ORIG_STATIC_OUT,
    )
    # 3. Compare
    cmp_json = build_compare_json(orig, patched)
    out_path = PATCHED_OUT / "compare_orig_vs_patched_era5.json"
    with open(out_path, "w") as f:
        json.dump(cmp_json, f, indent=2)
    log.info(f"\nWrote {out_path}")

    # 4. Quick summary to stdout
    for site, block in cmp_json["sites"].items():
        log.info(f"\n===== {site.upper()} (ERA5, 8-yr avg) =====")
        for scen, payload in block.items():
            o = payload["orig"]; p = payload["patched"]; d = payload["delta_patched_minus_orig"]
            log.info(f"  {scen:20s}  P={o.get('precip'):7.2f} | "
                     f"ET orig={o.get('et'):7.2f}  patched={p.get('et'):7.2f}  Δ={d.get('et'):+7.2f}")
            log.info(f"                         eplant Δ={d.get('eplant'):+7.2f}  "
                     f"esoil Δ={d.get('esoil'):+7.2f}  "
                     f"LQ Δ={d.get('latq'):+7.2f}")


if __name__ == "__main__":
    main()
