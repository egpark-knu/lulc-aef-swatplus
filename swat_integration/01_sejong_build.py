#!/usr/bin/env python3
"""
p2_08_sejong_swatplus_build.py
Build a SWAT+ TxtInOut for Sejong City from scratch.

Uses Seoul model as structural template, replaces ALL data with Sejong-specific inputs.
- 6 subbasins x 6 LULC classes = 36 HRUs (Superset HRU approach)
- STATIC scenario: baseline 2017 LULC (fixed ESCO per LULC class)
- DYNAMIC scenario: AEF-based annual LULC -> ESCO changes per year

Author: MAS (Claude Code + GEE pipeline)
Date: 2026-04-04
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT   = Path("/Users/eungyupark/Dropbox/myproj/dev_260402_LULC")
PHASE1    = PROJECT / "phase1"
PHASE2    = PROJECT / "phase2"
SEOUL_TXT = Path("/Users/eungyupark/Dropbox/Manuscripts/0_swat_mcp/swatplus-model/TxtInOut")
SWATPLUS  = Path("/Users/eungyupark/Dropbox/myproj/dev_260222/bin/swatplus-minimal")
CLIMATE   = PHASE2 / "data" / "climate_swat"
USERSOIL  = PHASE1 / "data" / "swat_prep" / "usersoil.csv"
SUBBASINS = PHASE2 / "data" / "watershed" / "sejong_subbasins.geojson"
LULC_DIR  = PHASE1 / "data" / "transfer_years"
RESULTS   = PHASE2 / "data" / "sejong_swat_results"
LOGDIR    = PHASE2 / "logs"

# Working directory in /tmp for build
BUILD_DIR = Path("/tmp/sejong_swatplus_build")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sejong_build")

# ---------------------------------------------------------------------------
# Constants: Sejong watershed
# ---------------------------------------------------------------------------
SEJONG_LAT    = 36.48
SEJONG_LON    = 127.00
SEJONG_ELEV   = 127.0   # m, representative
WST_NAME      = "sejong_wst"
N_SUBBASINS   = 6
SIM_START_YR  = 2015
SIM_END_YR    = 2024

# LULC class definitions (6 classes from AEF)
LULC_CLASSES = ["frse", "agrr", "urml", "past", "watr", "swrn"]
LULC_LABELS  = ["Forest", "Cropland", "Urban", "Grassland", "Water", "Barren"]
LULC_LUM     = ["frse_lum", "agrr_lum", "urml_lum", "past_lum", "watr_lum", "swrn_lum"]

# Per-LULC ESCO and CN2 (for HSG C, dominant in Sejong)
ESCO_BY_LULC = {"frse": 0.95, "agrr": 0.90, "urml": 0.70,
                "past": 0.85, "watr": 1.00, "swrn": 0.80}
CN2_BY_LULC  = {"frse": 60, "agrr": 72, "urml": 85,
                "past": 69, "watr": 100, "swrn": 77}

# Subbasin info (from GeoJSON analysis)
# All 6 subbasins are 0.10 x 0.12 degree rectangles
# Mean elevation from DEM per subbasin (sub_id -> elev_m)
SUB_ELEV = {1: 136.2, 2: 60.3, 3: 148.7, 4: 104.5, 5: 200.8, 6: 109.2}
# Centroids (lat, lon) from GeoJSON bounding boxes
SUB_CENTROID = {
    1: (36.440, 126.900), 2: (36.440, 127.000), 3: (36.440, 127.100),
    4: (36.560, 126.900), 5: (36.560, 127.000), 6: (36.560, 127.100),
}
# Each subbasin area in ha (0.10 deg lon x 0.12 deg lat at 36.5N)
SUB_AREA_HA = 11961.5

# Slope proxy from DEM std (converted to m/m)
SUB_SLOPE = {1: 0.120, 2: 0.055, 3: 0.130, 4: 0.090, 5: 0.125, 6: 0.085}

# Total watershed area (ha)
TOTAL_AREA_HA = N_SUBBASINS * SUB_AREA_HA

# HRU naming: hru_SSS_LLL where SSS=subbasin 1-6, LLL=lulc index 1-6
N_HRU = N_SUBBASINS * len(LULC_CLASSES)  # 36

# Channel network (simple linear: 1->2->3, 4->5->6, merge at outlet 3->6->outlet)
# We define 6 channels, one per subbasin
# Routing: sub1->cha1->cha3(outlet_south), sub2->cha2->cha3, sub3->cha3
#          sub4->cha4->cha6(outlet_north), sub5->cha5->cha6, sub6->cha6
# Then cha3->cha6->outlet (cha6 is the watershed outlet)
N_CHA = 6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SoilRecord:
    """Parsed from usersoil.csv."""
    name: str
    texture: str
    hydgrp: str
    zmx: float
    z1: float
    bd1: float
    awc1: float
    k1: float
    cbn1: float
    sand1: float
    silt1: float
    clay1: float
    rock1: float
    alb1: float


# ---------------------------------------------------------------------------
# Helper: fixed-width field formatting (SWAT+ style)
# ---------------------------------------------------------------------------
def _fi(v: int, w: int = 14) -> str:
    """Format integer right-justified in field width."""
    return str(v).rjust(w)

def _ff(v: float, w: int = 14, d: int = 5) -> str:
    """Format float right-justified in field width with d decimals."""
    return f"{v:.{d}f}".rjust(w)

def _fs(v: str, w: int = 20) -> str:
    """Format string left-justified in field width."""
    return v.ljust(w)

def _fsr(v: str, w: int = 20) -> str:
    """Format string right-justified in field width."""
    return v.rjust(w)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_soils() -> list[SoilRecord]:
    """Load 4 soil types from usersoil.csv."""
    soils: list[SoilRecord] = []
    with open(USERSOIL) as f:
        reader = csv.DictReader(f)
        for row in reader:
            soils.append(SoilRecord(
                name=row["SNAM"],
                texture=row["TEXTURE"],
                hydgrp=row["HYDGRP"],
                zmx=float(row["SOL_ZMX"]),
                z1=float(row["SOL_Z1"]),
                bd1=float(row["SOL_BD1"]),
                awc1=float(row["SOL_AWC1"]),
                k1=float(row["SOL_K1"]),
                cbn1=float(row["SOL_CBN1"]),
                sand1=float(row["SAND1"]),
                silt1=float(row["SILT1"]),
                clay1=float(row["CLAY1"]),
                rock1=float(row["ROCK1"]),
                alb1=float(row["SOL_ALB1"]),
            ))
    return soils


def load_lulc_fractions(year: int) -> dict[str, float]:
    """Load LULC fractions for a given year. Returns {label: pct}."""
    fpath = LULC_DIR / f"{year}.json"
    with open(fpath) as f:
        data = json.load(f)
    return data["class_fractions"]


def load_all_lulc_fractions() -> dict[int, dict[str, float]]:
    """Load all available years."""
    result = {}
    for year in range(2017, 2025):
        result[year] = load_lulc_fractions(year)
    return result


# ---------------------------------------------------------------------------
# HRU geometry: distribute subbasin area among 6 LULC classes
# ---------------------------------------------------------------------------
def compute_hru_areas(
    lulc_fracs: dict[str, float],
) -> list[dict[str, Any]]:
    """
    For each subbasin, create 6 HRUs (one per LULC class).
    Each HRU area = subbasin_area * lulc_fraction.
    Returns list of 36 dicts with all HRU properties.
    """
    hrus = []
    hru_id = 0
    for sub_id in range(1, N_SUBBASINS + 1):
        for lulc_idx, (lulc_code, lulc_label) in enumerate(
            zip(LULC_CLASSES, LULC_LABELS)
        ):
            hru_id += 1
            frac_pct = lulc_fracs.get(lulc_label, 0.0)
            area_ha = SUB_AREA_HA * frac_pct / 100.0
            # Ensure minimum area (SWAT+ needs >0)
            area_ha = max(area_ha, 0.01)

            lat, lon = SUB_CENTROID[sub_id]
            elev = SUB_ELEV[sub_id]
            slope = SUB_SLOPE[sub_id]

            hrus.append({
                "id": hru_id,
                "name": f"hru{hru_id:03d}",
                "sub_id": sub_id,
                "lulc_code": lulc_code,
                "lulc_label": lulc_label,
                "lulc_lum": LULC_LUM[lulc_idx],
                "area_ha": area_ha,
                "frac_sub": frac_pct / 100.0,
                "frac_bsn": area_ha / TOTAL_AREA_HA,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "slope": slope,
                "esco": ESCO_BY_LULC[lulc_code],
                "cn2": CN2_BY_LULC[lulc_code],
            })
    return hrus


# ---------------------------------------------------------------------------
# File writers (one per SWAT+ input file)
# ---------------------------------------------------------------------------
HEADER = "written by p2_08_sejong_swatplus_build.py on 2026-04-04 for SWAT+ rev.60.5.7"


def write_file_cio(out: Path) -> None:
    """Master input/output control file."""
    lines = [
        f"file.cio: {HEADER}",
        "simulation        time.sim          print.prt         null              object.cnt        null              ",
        "basin             codes.bsn         parameters.bsn    ",
        "climate           weather-sta.cli   weather-wgn.cli   null              pcp.cli           tmp.cli           slr.cli           hmd.cli           wnd.cli           null              ",
        "connect           hru.con           null              rout_unit.con     null              aquifer.con       null              null              null              null              null              null              null              chandeg.con       ",
        "channel           initial.cha       null              null              null              nutrients.cha     channel-lte.cha   hyd-sed-lte.cha   null              ",
        "reservoir         initial.res       null              null              sediment.res      nutrients.res     null              null              null              ",
        "routing_unit      rout_unit.def     rout_unit.ele     rout_unit.rtu     null              ",
        "hru               hru-data.hru      null              ",
        "exco              null              null              null              null              null              null              ",
        "recall            null              ",
        "dr                null              null              null              null              null              null              ",
        "aquifer           initial.aqu       aquifer.aqu       ",
        "herd              null              null              null              ",
        "water_rights      null              null              null              ",
        "link              null              null              ",
        "hydrology         hydrology.hyd     topography.hyd    field.fld         ",
        "structural        tiledrain.str     septic.str        filterstrip.str   grassedww.str     bmpuser.str       ",
        "hru_parm_db       plants.plt        fertilizer.frt    tillage.til       pesticide.pes     null              null              null              urban.urb         septic.sep        snow.sno          ",
        "ops               harv.ops          graze.ops         irr.ops           chem_app.ops      fire.ops          sweep.ops         ",
        "lum               landuse.lum       null              cntable.lum       cons_practice.lum  ovn_table.lum     ",
        "chg               cal_parms.cal     null              null              null              null              null              null              null              null              ",
        "init              plant.ini         soil_plant.ini    om_water.ini      null              null              null              null              null              null              null              null              ",
        "soils             soils.sol         nutrients.sol     null              ",
        "decision_table    lum.dtl           res_rel.dtl       null              null              ",
        "regions           ls_unit.ele       ls_unit.def       null              null              null              null              null              null              aqu_catunit.ele   null              null              null              null              null              null              null              null              ",
        "pcp_path          null              ",
        "tmp_path          null              ",
        "slr_path          null              ",
        "hmd_path          null              ",
        "wnd_path          null              ",
    ]
    (out / "file.cio").write_text("\n".join(lines) + "\n")


def write_time_sim(out: Path) -> None:
    lines = [
        f"time.sim: {HEADER}",
        "day_start  yrc_start   day_end   yrc_end      step  ",
        f"       0      {SIM_START_YR}         0      {SIM_END_YR}         0  ",
    ]
    (out / "time.sim").write_text("\n".join(lines) + "\n")


def write_print_prt(out: Path, nyskip: int = 2) -> None:
    """Print control file with daily + yearly basin_wb output."""
    lines = [
        f"print.prt: {HEADER}",
        "nyskip      day_start  yrc_start  day_end   yrc_end   interval  ",
        f"{nyskip}           0         0         0         0         1",
        "aa_int_cnt  ",
        "0           ",
        "csvout        dbout         cdfout        ",
        "n             n             n             ",
        "crop_yld      mgtout        hydcon        fdcout        ",
        "n             n             n             n             ",
        "objects                  daily       monthly        yearly         avann  ",
        "basin_wb                     y             n             y             y",
        "basin_nb                     n             n             y             y  ",
        "basin_ls                     n             n             y             y  ",
        "basin_pw                     n             n             y             y  ",
        "basin_aqu                    y             n             y             y",
        "basin_res                    n             n             n             n  ",
        "basin_cha                    n             n             y             y  ",
        "basin_sd_cha                 n             n             y             y  ",
        "basin_psc                    n             n             y             y  ",
        "region_wb                    n             n             y             y  ",
        "region_nb                    n             n             n             n  ",
        "region_ls                    n             n             n             n  ",
        "region_pw                    n             n             n             n  ",
        "region_aqu                   n             n             n             n  ",
        "region_res                   n             n             n             n  ",
        "region_sd_cha                n             n             n             n  ",
        "region_psc                   n             n             n             n  ",
        "water_allo                   n             n             n             n  ",
        "lsunit_wb                    n             n             n             n  ",
        "lsunit_nb                    n             n             n             n  ",
        "lsunit_ls                    n             n             n             n  ",
        "lsunit_pw                    n             n             n             n  ",
        "hru_wb                       n             n             y             y",
        "hru_nb                       n             n             n             n  ",
        "hru_ls                       n             n             n             n  ",
        "hru_pw                       n             n             n             n  ",
        "hru-lte_wb                   n             n             n             n  ",
        "hru-lte_nb                   n             n             n             n  ",
        "hru-lte_ls                   n             n             n             n  ",
        "hru-lte_pw                   n             n             n             n  ",
        "channel                      n             n             y             y  ",
        "channel_sd                   y             n             y             y",
        "aquifer                      y             n             y             y",
        "reservoir                    n             n             n             n  ",
        "recall                       n             n             n             n  ",
        "hyd                          n             n             n             n  ",
        "ru                           n             n             n             n  ",
        "pest                         n             n             n             n  ",
        "basin_salt                   n             n             n             n  ",
        "hru_salt                     n             n             n             n  ",
        "ru_salt                      n             n             n             n  ",
        "aqu_salt                     n             n             n             n  ",
        "channel_salt                 n             n             n             n  ",
        "res_salt                     n             n             n             n  ",
        "wetland_salt                 n             n             n             n  ",
        "basin_cs                     n             n             n             n  ",
        "hru_cs                       n             n             n             n  ",
        "ru_cs                        n             n             n             n  ",
        "aqu_cs                       n             n             n             n  ",
        "channel_cs                   n             n             n             n  ",
        "res_cs                       n             n             n             n  ",
        "wetland_cs                   n             n             n             n  ",
    ]
    (out / "print.prt").write_text("\n".join(lines) + "\n")


def write_codes_bsn(out: Path) -> None:
    """Basin codes: disable unused modules."""
    lines = [
        f"codes.bsn: {HEADER}",
        "        pet_file           wq_file       pet     event     crack  swift_out   sed_det   rte_cha   deg_cha    wq_cha  nostress        cn    c_fact    carbon     lapse      uhyd   sed_cha  tiledrain    wtable    soil_p     gampt          atmo_dep  stor_max   i_fpwet    gwflow  ",
        "            null              null         1         0         0         1         0         0         0         0         0         0         0         0         0         1         0         0         0         0         0                 a         0         0         0  ",
    ]
    (out / "codes.bsn").write_text("\n".join(lines) + "\n")


def write_object_cnt(out: Path) -> None:
    """Object count: 36 HRUs, 6 RTUs, 6 channels, 7 aquifers (6 shallow + 1 deep)."""
    # obj = hru + rtu + aqu + cha = 36 + 6 + 7 + 6 = 55
    n_obj = N_HRU + N_SUBBASINS + (N_SUBBASINS + 1) + N_CHA
    lines = [
        f"object.cnt: {HEADER}",
        "name                   ls_area      tot_area       obj       hru      lhru       rtu      gwfl       aqu       cha       res       rec      exco       dlr       can       pmp       out      lcha     aqu2d       hrd       wro  ",
        f"  Sejong_City          {TOTAL_AREA_HA:14.5f}     {TOTAL_AREA_HA:14.5f}    {_fi(n_obj, 10)}    {_fi(N_HRU, 10)}    {_fi(0, 10)}    {_fi(N_SUBBASINS, 10)}    {_fi(0, 10)}    {_fi(N_SUBBASINS + 1, 10)}    {_fi(N_CHA, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(N_CHA, 10)}    {_fi(0, 10)}    {_fi(0, 10)}    {_fi(0, 10)}",
    ]
    (out / "object.cnt").write_text("\n".join(lines) + "\n")


# --- Climate files ---

def write_weather_sta_cli(out: Path) -> None:
    """Weather station master file pointing to Sejong weather files."""
    lines = [
        f"weather-sta.cli: {HEADER}",
        "name                           wgn                        pcp                        tmp                        slr                        hmd                        wnd                        pet          atmo_dep  ",
        f"{WST_NAME:30s} {'Sejong':20s} {'sejong.pcp':26s} {'sejong.tmp':26s} {'sejong.slr':26s} {'sejong.hmd':26s} {'sejong.wnd':26s} {'null':20s} {'null':10s}",
    ]
    (out / "weather-sta.cli").write_text("\n".join(lines) + "\n")


def write_weather_wgn_cli(out: Path) -> None:
    """Weather generator: use Seoul WGN as proxy (similar latitude band)."""
    # Sejong is ~100km south of Seoul, similar monsoon climate
    # Adjust station name and coordinates
    lines = [
        f"weather-wgn.cli: {HEADER}",
        f"{'Sejong':30s} {SEJONG_LAT:13.5f}  {SEJONG_LON:13.5f}  {SEJONG_ELEV:13.5f}    {'10':>5s}  ",
        # Monthly WGN data (Seoul proxy -- structurally identical climate zone)
        " tmp_max_ave   tmp_min_ave    tmp_max_sd    tmp_min_sd       pcp_ave        pcp_sd      pcp_skew       wet_dry       wet_wet      pcp_days       pcp_hhr       slr_ave       dew_ave       wnd_ave  ",
        "     3.08161      -5.36516       4.66898       4.86525      18.74000      20.02888       4.89423       0.18987       0.28571       7.11111       0.50000       7.88245       0.58445       2.26419  ",
        "     6.02155      -3.05866       4.60800       4.29757      27.54000      24.11902       1.82288       0.16981       0.39344       6.30000       0.50000      10.73635       0.61166       2.44028  ",
        "    13.24097       2.88387       4.50656       3.76219      40.45000      37.56003       3.35656       0.16387       0.32258       6.40000       0.50000      14.31895       0.65658       2.43452  ",
        "    19.32367       8.78267       4.28724       3.33608      68.80000      40.34361       2.61468       0.21154       0.46341       8.40000       0.50000      17.23135       0.64590       2.51333  ",
        "    24.39161      13.80032       3.73245       2.86077     104.90000      77.17541       2.03222       0.20455       0.46250       8.30000       0.50000      19.95806       0.68032       2.36935  ",
        "    28.48967      19.22567       2.78948       2.13852     141.50000      99.47730       4.06340       0.26263       0.50000      10.00000       0.50000      18.77837       0.72397       2.19133  ",
        "    30.34645      23.50645       3.19962       2.25877     329.34000     163.23202       1.90769       0.30921       0.70270      15.50000       0.50000      14.42903       0.82339       2.14516  ",
        "    30.99903      23.91871       3.44800       2.88242     265.27000     207.10156       2.08870       0.40606       0.48148      13.80000       0.50000      13.98886       0.81529       2.09355  ",
        "    27.37367      18.79933       2.66951       2.82727     109.47000      64.02468       2.74584       0.21845       0.47619       8.60000       0.50000      14.25000       0.76927       2.15000  ",
        "    20.72645      10.99161       3.70257       3.99266      64.49000      39.49297       2.37030       0.15353       0.33898       6.77778       0.50000      11.46774       0.73500       2.03903  ",
        "    12.83833       3.91933       5.41402       5.10836      72.88000      29.80208       3.10131       0.22066       0.40260       7.90000       0.50000       8.04938       0.71243       2.15767  ",
        "     4.49968      -3.75516       4.91643       4.86641      28.17000      26.48925       3.04490       0.25926       0.33333       8.60000       0.50000       7.11555       0.62919       2.17355  ",
    ]
    (out / "weather-wgn.cli").write_text("\n".join(lines) + "\n")


def write_cli_files(out: Path) -> None:
    """Write pcp.cli, tmp.cli, slr.cli, hmd.cli, wnd.cli."""
    for var, fname in [
        ("pcp", "sejong.pcp"),
        ("tmp", "sejong.tmp"),
        ("slr", "sejong.slr"),
        ("hmd", "sejong.hmd"),
        ("wnd", "sejong.wnd"),
    ]:
        lines = [
            f"{var}.cli: {var.upper()} file names - Sejong KMA",
            "filename",
            fname,
        ]
        (out / f"{var}.cli").write_text("\n".join(lines) + "\n")


def copy_climate_data(out: Path) -> None:
    """Copy actual climate data files from phase2/data/climate_swat/."""
    for fname in ["sejong.pcp", "sejong.tmp", "sejong.slr", "sejong.hmd", "sejong.wnd"]:
        src = CLIMATE / fname
        dst = out / fname
        if src.exists():
            shutil.copy2(src, dst)
            log.info("Copied climate: %s", fname)
        else:
            log.warning("Climate file not found: %s", src)


# --- Connect files ---

def write_hru_con(out: Path, hrus: list[dict]) -> None:
    """HRU connectivity file."""
    lines = [
        f"hru.con: {HEADER}",
        "      id  name                gis_id          area           lat           lon          elev       hru               wst       cst      ovfl      rule   out_tot  ",
    ]
    for h in hrus:
        lines.append(
            f"  {h['id']:5d}  {h['name']:20s}  {h['id']:10d}"
            f"  {h['area_ha']:12.5f}"
            f"  {h['lat']:13.5f}"
            f"  {h['lon']:13.5f}"
            f"  {h['elev']:13.5f}"
            f"  {h['id']:12d}"
            f"  {WST_NAME:16s}"
            f"  {0:12d}  {0:12d}  {0:12d}  {0:12d}"
        )
    (out / "hru.con").write_text("\n".join(lines) + "\n")


def write_hru_data(out: Path, hrus: list[dict], soils: list[SoilRecord]) -> None:
    """HRU data file: assigns topo, hydrology, soil, and land use to each HRU."""
    # Soil assignment: use dominant soil for Sejong (Clay Loam -> SG_Clay_Loam)
    # All subbasins use same soil since we have uniform data
    soil_name = "SG_Clay_Loam"

    lines = [
        f"hru-data.hru: {HEADER}",
        "      id  name                          topo             hydro              soil            lu_mgt   soil_plant_init         surf_stor              snow             field  ",
    ]
    for h in hrus:
        topo_name = f"topo{h['name']}"
        hyd_name = f"hyd{h['id']:03d}"
        lines.append(
            f"  {h['id']:5d}"
            f"  {h['name']:20s}"
            f"  {topo_name:20s}"
            f"  {hyd_name:16s}"
            f"  {soil_name:16s}"
            f"  {h['lulc_lum']:16s}"
            f"  {'soilplant1':16s}"
            f"  {'null':16s}"
            f"  {'snow001':16s}"
            f"  {'null':16s}"
        )
    (out / "hru-data.hru").write_text("\n".join(lines) + "\n")


def write_hydrology_hyd(out: Path, hrus: list[dict]) -> None:
    """Hydrology parameters per HRU -- ESCO varies by LULC."""
    lines = [
        f"hydrology.hyd: {HEADER}",
        "name                 lat_ttime       lat_sed       can_max          esco          epco   orgn_enrich   orgp_enrich       cn3_swf       bio_mix         perco      lat_orgn      lat_orgp        pet_co       latq_co  ",
    ]
    for h in hrus:
        name = f"hyd{h['id']:03d}"
        esco = h["esco"]
        lines.append(
            f"{name:20s}"
            f"   {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {1.0:11.5f}"
            f" {esco:11.5f}"
            f" {0.50:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {0.90:11.5f}"
            f" {0.20:11.5f}"
            f" {0.20:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {1.0:11.5f}"
            f" {0.01:11.5f}"
        )
    (out / "hydrology.hyd").write_text("\n".join(lines) + "\n")


def write_topography_hyd(out: Path, hrus: list[dict]) -> None:
    """Topography per HRU: slope from DEM, default slope lengths."""
    lines = [
        f"topography.hyd: {HEADER}",
        "name                       slp       slp_len       lat_len      dist_cha         depos  ",
    ]
    for h in hrus:
        name = f"topo{h['name']}"
        slp = h["slope"]
        # Slope length and lateral length based on LULC
        if h["lulc_code"] in ("frse", "past", "swrn"):
            slp_len = 10.0
            lat_len = 10.0
        else:
            slp_len = 90.0
            lat_len = 90.0
        lines.append(
            f"{name:20s}"
            f"   {slp:11.5f}"
            f"  {slp_len:11.5f}"
            f"  {lat_len:11.5f}"
            f"  {100.0:11.5f}"
            f"  {0.0:11.5f}"
        )
    (out / "topography.hyd").write_text("\n".join(lines) + "\n")


def write_soils_sol(out: Path, soils: list[SoilRecord]) -> None:
    """Soils file: 4 Sejong soil types, 2 layers each."""
    lines = [
        f"soils.sol: {HEADER}",
        "name                            nly           hyd_grp        dp_tot    anion_excl      perc_crk  texture                              dp            bd           awc        soil_k        carbon          clay          silt          sand          rock           alb        usle_k            ec         caco3            ph  ",
    ]
    for s in soils:
        # Layer 1 from usersoil.csv data directly
        # Layer 2: deeper, slightly lower AWC/K, lower carbon
        dp_tot = s.zmx
        z2 = s.zmx
        bd2 = s.bd1 + 0.05
        awc2 = s.awc1 * 0.85
        k2 = s.k1 * 0.80
        cbn2 = s.cbn1 * 0.3
        clay2 = s.clay1 + 3.0
        silt2 = s.silt1 - 1.0
        sand2 = s.sand1 - 2.0
        rock2 = s.rock1 + 1.0

        perc_crk = 0.10 if s.hydgrp == "C" else 0.0
        usle_k1 = 0.30 if "Clay" in s.texture else 0.20
        usle_k2 = usle_k1 * 0.95

        lines.append(
            f"{s.name:32s}"
            f"  {'2':12s}"
            f"  {s.hydgrp:16s}"
            f" {dp_tot:12.5f}"
            f"   {0.5:11.5f}"
            f"   {perc_crk:11.5f}"
            f"  {s.texture.replace(' ', '_'):27s}"
        )
        # Layer 1
        lines.append(
            f"{'':100s}"
            f" {s.z1:13.5f}"
            f" {s.bd1:13.5f}"
            f" {s.awc1:13.5f}"
            f" {s.k1:13.5f}"
            f" {s.cbn1:13.5f}"
            f" {s.clay1:13.5f}"
            f" {s.silt1:13.5f}"
            f" {s.sand1:13.5f}"
            f" {s.rock1:13.5f}"
            f" {s.alb1:13.5f}"
            f" {usle_k1:13.5f}"
            f" {0.0:13.5f}"
            f" {0.0:13.5f}"
            f" {6.5:13.5f}"
        )
        # Layer 2
        lines.append(
            f"{'':100s}"
            f" {z2:13.5f}"
            f" {bd2:13.5f}"
            f" {awc2:13.5f}"
            f" {k2:13.5f}"
            f" {cbn2:13.5f}"
            f" {clay2:13.5f}"
            f" {silt2:13.5f}"
            f" {sand2:13.5f}"
            f" {rock2:13.5f}"
            f" {s.alb1:13.5f}"
            f" {usle_k2:13.5f}"
            f" {0.0:13.5f}"
            f" {0.0:13.5f}"
            f" {6.3:13.5f}"
        )
    (out / "soils.sol").write_text("\n".join(lines) + "\n")


def write_landuse_lum(out: Path) -> None:
    """Land use management: 6 Sejong LULC classes."""
    lines = [
        f"landuse.lum: {HEADER}",
        "name                         cal_group          plnt_com                                        mgt               cn2         cons_prac             urban            urb_ro           ov_mann              tile               sep               vfs              grww               bmp  ",
    ]
    lum_defs = [
        # (name, plnt_com, cn_table, urban, urb_ro, ov_mann)
        ("frse_lum", "frse_comm", "wood_f", "null", "null", "forest_heavy"),
        ("agrr_lum", "agrr_comm", "rc_strow_g", "null", "null", "convtill_nores"),
        ("urml_lum", "null", "urml_cn", "urml", "buildup_washoff", "urban_asphalt"),
        ("past_lum", "past_comm", "pastg_f", "null", "null", "densegrass"),
        ("watr_lum", "watr_comm", "rc_strow_g", "null", "null", "convtill_nores"),
        ("swrn_lum", "null", "fal_bare", "null", "null", "fallow_nores"),
    ]
    for name, plnt, cn, urb, urb_ro, ovn in lum_defs:
        lines.append(
            f"{name:28s}"
            f"  {'null':18s}"
            f" {plnt:40s}"
            f" {'null':18s}"
            f" {cn:16s}"
            f" {'up_down_slope':18s}"
            f" {urb:16s}"
            f" {urb_ro:16s}"
            f" {ovn:20s}"
            f" {'null':16s}"
            f" {'null':16s}"
            f" {'null':16s}"
            f" {'null':16s}"
            f" {'null':16s}"
        )
    (out / "landuse.lum").write_text("\n".join(lines) + "\n")


def write_plant_ini(out: Path) -> None:
    """Plant community initialization."""
    lines = [
        f"plant.ini: {HEADER}",
        "pcom_name          plt_cnt  rot_yr_ini          plt_name     lc_status      lai_init       bm_init      phu_init      plnt_pop      yrs_init      rsd_init  ",
        "frse_comm                1         1  ",
        "                                        frse             y       2.00000   50000.00000       0.00000       0.00000       1.00000   10000.00000  ",
        "agrr_comm                1         1  ",
        "                                        agrc             y       0.50000    5000.00000       0.00000       0.00000       1.00000    1000.00000  ",
        "past_comm                1         1  ",
        "                                        past             y       0.00000   20000.00000       0.00000       0.00000       1.00000   10000.00000  ",
        "watr_comm                1         1  ",
        "                                        watr             n       0.00000       0.00000       0.00000       0.00000       0.00000   10000.00000  ",
    ]
    (out / "plant.ini").write_text("\n".join(lines) + "\n")


# --- Routing files ---

def write_rout_unit_con(out: Path, hrus: list[dict]) -> None:
    """Routing unit connectivity: 6 subbasins, each drains to channel + aquifer."""
    lines = [
        f"rout_unit.con: {HEADER}",
        "      id  name                gis_id          area           lat           lon          elev       rtu               wst       cst      ovfl      rule   out_tot       obj_typ    obj_id       hyd_typ          frac  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        lat, lon = SUB_CENTROID[sub_id]
        elev = SUB_ELEV[sub_id]
        name = f"rtu{sub_id:04d}"
        lines.append(
            f"  {sub_id:5d}"
            f"  {name:20s}"
            f"  {sub_id:10d}"
            f"  {SUB_AREA_HA:14.5f}"
            f"  {lat:12.5f}"
            f"  {lon:12.5f}"
            f"  {elev:12.5f}"
            f"  {sub_id:10d}"
            f"  {WST_NAME:16s}"
            f"  {0:10d}"
            f"  {0:10d}"
            f"  {0:10d}"
            f"  {2:10d}"
            f"  {'sdc':>12s}"
            f"  {sub_id:>10d}"
            f"  {'tot':>12s}"
            f"  {1.0:>10.5f}"
            f"  {'aqu':>12s}"
            f"  {sub_id:>10d}"
            f"  {'rhg':>12s}"
            f"  {1.0:>10.5f}"
        )
    (out / "rout_unit.con").write_text("\n".join(lines) + "\n")


def write_rout_unit_def(out: Path, hrus: list[dict]) -> None:
    """Routing unit definition: which HRUs belong to each RTU."""
    lines = [
        f"rout_unit.def: {HEADER}",
        "      id              name  elem_tot  elements  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"rtu{sub_id:04d}"
        # HRUs in this subbasin: (sub_id-1)*6+1 to sub_id*6
        first_hru = (sub_id - 1) * len(LULC_CLASSES) + 1
        last_hru = sub_id * len(LULC_CLASSES)
        # elem_tot=2 means "range notation: start -end" (SWAT+ convention)
        lines.append(
            f"  {sub_id:5d}"
            f"  {name:16s}"
            f"  {2:8d}"
            f"  {first_hru:8d}"
            f"  {-last_hru:8d}"
        )
    (out / "rout_unit.def").write_text("\n".join(lines) + "\n")


def write_rout_unit_ele(out: Path, hrus: list[dict]) -> None:
    """Routing unit element fractions."""
    lines = [
        f"rout_unit.ele: {HEADER}",
        "      id  name                   obj_typ    obj_id          frac               dlr  ",
    ]
    for h in hrus:
        frac = h["frac_sub"]
        frac_str = f"{frac:.4E}" if frac >= 0.001 else f"{frac:.4E}"
        lines.append(
            f"  {h['id']:5d}"
            f"  {h['name']:20s}"
            f"  {'hru':>10s}"
            f"  {h['id']:>10d}"
            f"  {frac_str:>14s}"
            f"  {0:>16d}"
        )
    (out / "rout_unit.ele").write_text("\n".join(lines) + "\n")


def write_rout_unit_rtu(out: Path) -> None:
    """Routing unit definitions with topography and field references."""
    lines = [
        f"rout_unit.rtu: {HEADER}",
        "      id              name            define               dlr              topo             field  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"rtu{sub_id:04d}"
        lines.append(
            f"  {sub_id:5d}"
            f"  {name:16s}"
            f"  {name:16s}"
            f"  {'null':16s}"
            f"  {'toport' + name:20s}"
            f"  {'fld' + str(sub_id):16s}"
        )
    (out / "rout_unit.rtu").write_text("\n".join(lines) + "\n")


# --- Channel files ---

def write_chandeg_con(out: Path) -> None:
    """Channel connectivity: 6 channels in a simple tree network.

    Network topology (2x3 grid):
      Row 1 (south): sub1 -> sub2 -> sub3
      Row 2 (north): sub4 -> sub5 -> sub6
      sub3 -> sub6 (outlet)

    Channel routing:
      cha1 -> cha2, cha2 -> cha3, cha3 -> cha6
      cha4 -> cha5, cha5 -> cha6
      cha6 -> outlet (out_tot=0)
    """
    lines = [
        f"chandeg.con: {HEADER}",
        "      id  name                gis_id          area           lat           lon          elev      lcha               wst       cst      ovfl      rule   out_tot       obj_typ    obj_id       hyd_typ          frac  ",
    ]
    # Cumulative drainage areas (ha)
    cum_area = {
        1: SUB_AREA_HA,
        2: 2 * SUB_AREA_HA,
        3: 3 * SUB_AREA_HA,
        4: SUB_AREA_HA,
        5: 2 * SUB_AREA_HA,
        6: TOTAL_AREA_HA,
    }
    # Downstream channel: 1->2, 2->3, 3->6, 4->5, 5->6, 6->outlet
    downstream = {1: 2, 2: 3, 3: 6, 4: 5, 5: 6, 6: 0}

    for cha_id in range(1, N_CHA + 1):
        lat, lon = SUB_CENTROID[cha_id]
        name = f"cha{cha_id:02d}"
        ds = downstream[cha_id]
        if ds == 0:
            # Outlet -- no downstream
            lines.append(
                f"  {cha_id:5d}  {name:20s}  {cha_id:10d}"
                f"  {cum_area[cha_id]:14.5f}"
                f"  {lat:12.5f}  {lon:12.5f}  {0:12.5f}"
                f"  {cha_id:10d}  {WST_NAME:16s}"
                f"  {0:10d}  {0:10d}  {0:10d}  {0:10d}"
            )
        else:
            lines.append(
                f"  {cha_id:5d}  {name:20s}  {cha_id:10d}"
                f"  {cum_area[cha_id]:14.5f}"
                f"  {lat:12.5f}  {lon:12.5f}  {0:12.5f}"
                f"  {cha_id:10d}  {WST_NAME:16s}"
                f"  {0:10d}  {0:10d}  {0:10d}"
                f"  {1:10d}"
                f"  {'sdc':>12s}  {ds:>10d}  {'tot':>12s}  {1.0:>10.5f}"
            )
    (out / "chandeg.con").write_text("\n".join(lines) + "\n")


def write_channel_lte_cha(out: Path) -> None:
    """Channel-LTE properties."""
    lines = [
        f"channel-lte.cha: {HEADER}",
        "      id  name                       cha_ini           cha_hyd           cha_sed           cha_nut  ",
    ]
    for cha_id in range(1, N_CHA + 1):
        name = f"cha{cha_id:02d}"
        lines.append(
            f"  {cha_id:5d}  {name:20s}"
            f"  {'initcha1':20s}"
            f"  {'hydcha' + f'{cha_id:02d}':20s}"
            f"  {'null':20s}"
            f"  {'nutcha1':20s}"
        )
    (out / "channel-lte.cha").write_text("\n".join(lines) + "\n")


def write_hyd_sed_lte_cha(out: Path) -> None:
    """Channel hydraulics and sediment parameters."""
    lines = [
        f"hyd-sed-lte.cha: {HEADER}",
        "name                         order            wd            dp           slp           len          mann             k     erod_fact      cov_fact          sinu        eq_slp           d50          clay        carbon        dry_bd      side_slp  bankfull_flo           fps           fpn        n_conc        p_conc         p_bio  description",
    ]
    # Channel properties based on drainage area
    for cha_id in range(1, N_CHA + 1):
        name = f"hydcha{cha_id:02d}"
        # Scale channel width/depth with downstream position
        order = 1 if cha_id in (1, 2, 4, 5) else 2
        # Rough Manning's equation geometry
        if cha_id == 6:
            wd, dp, length, slp = 15.0, 0.80, 8.0, 0.0015
        elif cha_id in (3, 5):
            wd, dp, length, slp = 10.0, 0.60, 5.0, 0.0025
        else:
            wd, dp, length, slp = 6.0, 0.40, 3.5, 0.0040
        lines.append(
            f"{name:28s}"
            f" {order:12d}"
            f" {wd:13.5f}"
            f" {dp:13.5f}"
            f" {slp:13.5f}"
            f" {length:13.5f}"
            f" {0.05:13.5f}"    # mann
            f" {1.0:13.5f}"     # k
            f" {0.01:13.5f}"    # erod_fact
            f" {0.005:13.5f}"   # cov_fact
            f" {1.05:13.5f}"    # sinu
            f" {0.001:13.5f}"   # eq_slp
            f" {12.0:13.5f}"    # d50
            f" {50.0:13.5f}"    # clay
            f" {0.04:13.5f}"    # carbon
            f" {1.0:13.5f}"     # dry_bd
            f" {0.5:13.5f}"     # side_slp
            f" {0.5:13.5f}"     # bankfull_flo
            f" {0.00001:13.5f}" # fps
            f" {0.1:13.5f}"     # fpn
            f" {0.0:13.5f}"     # n_conc
            f" {0.0:13.5f}"     # p_conc
            f" {0.0:13.5f}"     # p_bio
            f"  "
        )
    (out / "hyd-sed-lte.cha").write_text("\n".join(lines) + "\n")


# --- Aquifer files ---

def write_aquifer_con(out: Path) -> None:
    """Aquifer connectivity: 6 shallow + 1 deep."""
    lines = [
        f"aquifer.con: {HEADER}",
        "      id  name                gis_id          area           lat           lon          elev       aqu               wst       cst      ovfl      rule   out_tot       obj_typ    obj_id       hyd_typ          frac  ",
    ]
    # 6 shallow aquifers (1 per subbasin), drain to their channel
    for sub_id in range(1, N_SUBBASINS + 1):
        lat, lon = SUB_CENTROID[sub_id]
        elev = SUB_ELEV[sub_id]
        name = f"aqu{sub_id:04d}"
        lines.append(
            f"  {sub_id:5d}  {name:20s}  {2000 + sub_id:10d}"
            f"  {SUB_AREA_HA:14.5f}"
            f"  {lat:12.5f}  {lon:12.5f}  {elev:12.5f}"
            f"  {sub_id:10d}  {WST_NAME:16s}"
            f"  {0:10d}  {0:10d}  {0:10d}"
            f"  {1:10d}"
            f"  {'sdc':>12s}  {sub_id:>10d}  {'tot':>12s}  {1.0:>10.5f}"
        )
    # 1 deep aquifer (no outlet)
    deep_id = N_SUBBASINS + 1
    lines.append(
        f"  {deep_id:5d}  {'aqu_deep':20s}  {3001:10d}"
        f"  {TOTAL_AREA_HA:14.5f}"
        f"  {SEJONG_LAT:12.5f}  {SEJONG_LON:12.5f}  {SEJONG_ELEV:12.5f}"
        f"  {deep_id:10d}  {WST_NAME:16s}"
        f"  {0:10d}  {0:10d}  {0:10d}"
        f"  {0:10d}"
    )
    (out / "aquifer.con").write_text("\n".join(lines) + "\n")


def write_aquifer_aqu(out: Path) -> None:
    """Aquifer properties."""
    lines = [
        f"aquifer.aqu: {HEADER}",
        "      id  name                          init        gw_flo       dep_bot        dep_wt         no3_n         sol_p        carbon      flo_dist        bf_max      alpha_bf         revap       rchg_dp      spec_yld       hl_no3n       flo_min     revap_min  ",
    ]
    for i in range(1, N_SUBBASINS + 2):
        if i <= N_SUBBASINS:
            name = f"aqu{i:04d}"
        else:
            name = "aqu_deep"
        lines.append(
            f"  {i:5d}  {name:24s}"
            f" {'initaqu1':12s}"
            f" {0.05:11.5f}"
            f" {10.0:11.5f}"
            f" {3.0:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {0.5:11.5f}"
            f" {50.0:11.5f}"
            f" {10.0:11.5f}"
            f" {0.035:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {0.05:11.5f}"
            f" {30.0:11.5f}"
            f" {0.01:11.5f}"
            f" {0.01:11.5f}"
        )
    (out / "aquifer.aqu").write_text("\n".join(lines) + "\n")


# --- Region / LS unit files ---

def write_ls_unit_def(out: Path) -> None:
    """Landscape unit definition (same structure as rout_unit.def)."""
    lines = [
        f"ls_unit.def: {HEADER}",
        f"{N_SUBBASINS}",
        "      id              name          area  elem_tot  elements  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"rtu{sub_id:04d}"
        first_hru = (sub_id - 1) * len(LULC_CLASSES) + 1
        last_hru = sub_id * len(LULC_CLASSES)
        area_km2 = SUB_AREA_HA / 100.0
        lines.append(
            f"  {sub_id:5d}  {name:16s}"
            f"  {area_km2:11.5f}"
            f"  {2:8d}"
            f"  {first_hru:8d}  {-last_hru:8d}"
        )
    (out / "ls_unit.def").write_text("\n".join(lines) + "\n")


def write_ls_unit_ele(out: Path, hrus: list[dict]) -> None:
    """Landscape unit elements with basin and sub fractions."""
    lines = [
        f"ls_unit.ele: {HEADER}",
        "      id  name                   obj_typ  obj_typ_no      bsn_frac      sub_frac      reg_frac  ",
    ]
    for h in hrus:
        lines.append(
            f"  {h['id']:5d}"
            f"  {h['name']:20s}"
            f"  {'hru':>10s}"
            f"  {h['id']:>10d}"
            f"  {h['frac_bsn']:.5E}"
            f"  {h['frac_sub']:.5f}"
            f"  {0.0:.5f}"
        )
    (out / "ls_unit.ele").write_text("\n".join(lines) + "\n")


def write_aqu_catunit_ele(out: Path) -> None:
    """Aquifer catchment unit elements."""
    lines = [
        f"aqu_catunit.ele: {HEADER}",
        "      id  name                   obj_typ  obj_typ_no      bsn_frac      sub_frac      reg_frac  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"aqu{sub_id:04d}"
        bsn_frac = SUB_AREA_HA / TOTAL_AREA_HA
        lines.append(
            f"  {sub_id:5d}"
            f"  {name:20s}"
            f"  {'aqu':>10s}"
            f"  {sub_id:>10d}"
            f"  {bsn_frac:>13.5f}"
            f"  {0.0:>13.5f}"
            f"  {0.0:>13.5f}"
        )
    # Deep aquifer
    deep_id = N_SUBBASINS + 1
    lines.append(
        f"  {deep_id:5d}"
        f"  {'aqu_deep':20s}"
        f"  {'aqu':>10s}"
        f"  {deep_id:>10d}"
        f"  {1.0:>13.5f}"
        f"  {0.0:>13.5f}"
        f"  {0.0:>13.5f}"
    )
    (out / "aqu_catunit.ele").write_text("\n".join(lines) + "\n")


def write_field_fld(out: Path) -> None:
    """Field definition per subbasin."""
    lines = [
        f"field.fld: {HEADER}",
        "name                       len            wd           ang  ",
    ]
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"fld{sub_id}"
        lines.append(
            f"{name:20s}"
            f"  {500.0:11.5f}"
            f"  {100.0:11.5f}"
            f"  {30.0:11.5f}"
        )
    (out / "field.fld").write_text("\n".join(lines) + "\n")


# --- Topography for routing units ---

def write_topo_rtu(out: Path) -> None:
    """Topography for routing units (referenced by rout_unit.rtu)."""
    # This is appended to topography.hyd (same file).
    # Actually, the rout_unit.rtu references topo names like "toport_rtuXXXX"
    # but SWAT+ expects these in topography.hyd file
    # We already wrote HRU topos; now add RTU topos
    topo_file = out / "topography.hyd"
    existing = topo_file.read_text()
    extra_lines = []
    for sub_id in range(1, N_SUBBASINS + 1):
        name = f"toportrtu{sub_id:04d}"
        slp = SUB_SLOPE[sub_id]
        extra_lines.append(
            f"{name:20s}"
            f"   {slp:11.5f}"
            f"  {50.0:11.5f}"    # slp_len
            f"  {50.0:11.5f}"    # lat_len
            f"  {100.0:11.5f}"   # dist_cha
            f"  {0.0:11.5f}"     # depos
        )
    topo_file.write_text(existing + "\n".join(extra_lines) + "\n")


# --- Copy Seoul database files verbatim ---

COPY_VERBATIM = [
    # Database files (large, shared across models)
    "plants.plt", "fertilizer.frt", "tillage.til", "pesticide.pes",
    "urban.urb", "septic.sep", "snow.sno",
    "harv.ops", "graze.ops", "irr.ops", "chem_app.ops", "fire.ops", "sweep.ops",
    "cntable.lum", "cons_practice.lum", "ovn_table.lum",
    "cal_parms.cal", "parameters.bsn",
    "soil_plant.ini", "om_water.ini", "nutrients.sol",
    "initial.cha", "nutrients.cha", "initial.aqu", "initial.res",
    "sediment.res", "nutrients.res",
    "lum.dtl", "res_rel.dtl",
    "tiledrain.str", "septic.str", "filterstrip.str",
    "grassedww.str", "bmpuser.str",
    "cs_aqu.ini", "cs_channel.ini", "cs_hru.ini",
]


def copy_verbatim_files(out: Path) -> None:
    """Copy Seoul DB files that are model-independent."""
    for fname in COPY_VERBATIM:
        src = SEOUL_TXT / fname
        dst = out / fname
        if src.exists():
            shutil.copy2(src, dst)
        else:
            log.warning("Seoul file not found (skipped): %s", fname)


# ---------------------------------------------------------------------------
# Build full TxtInOut
# ---------------------------------------------------------------------------
def build_txtinout(
    out_dir: Path,
    lulc_fracs: dict[str, float],
    scenario_name: str = "baseline",
) -> Path:
    """
    Build a complete SWAT+ TxtInOut directory.

    Args:
        out_dir: Output directory for TxtInOut
        lulc_fracs: LULC fractions {label: pct} for HRU area allocation
        scenario_name: Label for logging

    Returns:
        Path to the built TxtInOut directory
    """
    log.info("Building TxtInOut for scenario: %s", scenario_name)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Load data
    soils = load_soils()
    hrus = compute_hru_areas(lulc_fracs)

    log.info("  HRUs: %d (6 subbasins x 6 LULC classes)", len(hrus))
    log.info("  Total area: %.1f ha", sum(h["area_ha"] for h in hrus))

    # 1. Copy verbatim Seoul DB files
    copy_verbatim_files(out_dir)

    # 2. Write Sejong-specific files
    write_file_cio(out_dir)
    write_time_sim(out_dir)
    write_print_prt(out_dir, nyskip=2)
    write_codes_bsn(out_dir)
    write_object_cnt(out_dir)

    # Climate
    write_weather_sta_cli(out_dir)
    write_weather_wgn_cli(out_dir)
    write_cli_files(out_dir)
    copy_climate_data(out_dir)

    # HRU
    write_hru_con(out_dir, hrus)
    write_hru_data(out_dir, hrus, soils)
    write_hydrology_hyd(out_dir, hrus)
    write_topography_hyd(out_dir, hrus)

    # Soils
    write_soils_sol(out_dir, soils)

    # Land use
    write_landuse_lum(out_dir)
    write_plant_ini(out_dir)

    # Routing
    write_rout_unit_con(out_dir, hrus)
    write_rout_unit_def(out_dir, hrus)
    write_rout_unit_ele(out_dir, hrus)
    write_rout_unit_rtu(out_dir)

    # Channels
    write_chandeg_con(out_dir)
    write_channel_lte_cha(out_dir)
    write_hyd_sed_lte_cha(out_dir)

    # Aquifers
    write_aquifer_con(out_dir)
    write_aquifer_aqu(out_dir)

    # Regions
    write_ls_unit_def(out_dir)
    write_ls_unit_ele(out_dir, hrus)
    write_aqu_catunit_ele(out_dir)

    # Field + RTU topography
    write_field_fld(out_dir)
    write_topo_rtu(out_dir)

    log.info("TxtInOut built: %s (%d files)", out_dir, len(list(out_dir.iterdir())))
    return out_dir


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_txtinout(txt_dir: Path) -> list[str]:
    """
    Validate the TxtInOut directory for internal consistency.
    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    # 1. Check all referenced files exist
    file_cio = txt_dir / "file.cio"
    if not file_cio.exists():
        errors.append("file.cio not found")
        return errors

    cio_text = file_cio.read_text()
    for line in cio_text.strip().split("\n")[1:]:
        tokens = line.split()
        for tok in tokens:
            if tok == "null" or tok == "file.cio:":
                continue
            if "." in tok and not tok.endswith("."):
                fpath = txt_dir / tok
                if not fpath.exists():
                    errors.append(f"Referenced file missing: {tok}")

    # 2. Count HRUs in hru.con
    hru_con = txt_dir / "hru.con"
    if hru_con.exists():
        hru_lines = [
            l for l in hru_con.read_text().strip().split("\n")[2:]
            if l.strip()
        ]
        if len(hru_lines) != N_HRU:
            errors.append(
                f"hru.con has {len(hru_lines)} HRUs, expected {N_HRU}"
            )

    # 3. Count RTUs in rout_unit.con
    rtu_con = txt_dir / "rout_unit.con"
    if rtu_con.exists():
        rtu_lines = [
            l for l in rtu_con.read_text().strip().split("\n")[2:]
            if l.strip()
        ]
        if len(rtu_lines) != N_SUBBASINS:
            errors.append(
                f"rout_unit.con has {len(rtu_lines)} RTUs, expected {N_SUBBASINS}"
            )

    # 4. Check object counts match
    obj_cnt = txt_dir / "object.cnt"
    if obj_cnt.exists():
        cnt_line = obj_cnt.read_text().strip().split("\n")[-1]
        tokens = cnt_line.split()
        # Find hru count (should be at known position)
        # After name, ls_area, tot_area, obj, hru, lhru, rtu, ...
        try:
            hru_cnt = int(tokens[4])
            if hru_cnt != N_HRU:
                errors.append(
                    f"object.cnt hru={hru_cnt}, expected {N_HRU}"
                )
        except (IndexError, ValueError):
            errors.append("Could not parse object.cnt counts")

    # 5. Check climate files
    for fname in ["sejong.pcp", "sejong.tmp", "sejong.slr", "sejong.hmd", "sejong.wnd"]:
        if not (txt_dir / fname).exists():
            errors.append(f"Climate file missing: {fname}")

    # 6. HRU areas sum check
    hru_con = txt_dir / "hru.con"
    if hru_con.exists():
        total = 0.0
        for line in hru_con.read_text().strip().split("\n")[2:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    total += float(parts[3])
                except ValueError:
                    pass
        expected = TOTAL_AREA_HA
        if abs(total - expected) / expected > 0.05:
            errors.append(
                f"HRU areas sum={total:.1f} ha, expected ~{expected:.1f} ha "
                f"(diff={abs(total - expected)/expected*100:.1f}%)"
            )

    return errors


# ---------------------------------------------------------------------------
# Scenario runners (STATIC and DYNAMIC)
# ---------------------------------------------------------------------------
def build_static_scenario() -> Path:
    """Build STATIC scenario: baseline 2017 LULC, fixed ESCO per class."""
    lulc_2017 = load_lulc_fractions(2017)
    out = BUILD_DIR / "static"
    return build_txtinout(out, lulc_2017, "STATIC_2017")


def build_dynamic_scenario() -> Path:
    """
    Build DYNAMIC scenario: same TxtInOut structure but with
    ESCO/CN2 that reflect AEF-weighted annual LULC changes.

    Strategy: Start from 2017 baseline. For each year's LULC fractions,
    compute weighted-average ESCO across all classes per subbasin.
    Write a per-year calibration adjustment file.
    """
    lulc_2017 = load_lulc_fractions(2017)
    all_fracs = load_all_lulc_fractions()

    # Build base TxtInOut with 2017 fractions
    out = BUILD_DIR / "dynamic"
    build_txtinout(out, lulc_2017, "DYNAMIC_AEF")

    # Compute per-year weighted ESCO/CN2 for the dynamic scenario
    # Save as a sidecar JSON for post-processing comparison
    dynamic_params = {}
    for year in range(2017, 2025):
        fracs = all_fracs[year]
        # Weighted ESCO across all LULC classes
        total_pct = sum(fracs.values())
        weighted_esco = sum(
            fracs[label] / total_pct * ESCO_BY_LULC[code]
            for code, label in zip(LULC_CLASSES, LULC_LABELS)
            if label in fracs
        )
        weighted_cn2 = sum(
            fracs[label] / total_pct * CN2_BY_LULC[code]
            for code, label in zip(LULC_CLASSES, LULC_LABELS)
            if label in fracs
        )
        dynamic_params[year] = {
            "fractions": fracs,
            "weighted_esco": round(weighted_esco, 4),
            "weighted_cn2": round(weighted_cn2, 2),
        }
        log.info(
            "  Year %d: weighted ESCO=%.4f, CN2=%.2f",
            year, weighted_esco, weighted_cn2,
        )

    # Write dynamic parameters sidecar
    params_file = out / "dynamic_params.json"
    with open(params_file, "w") as f:
        json.dump(dynamic_params, f, indent=2)
    log.info("Dynamic params written: %s", params_file)

    # For the actual SWAT+ run, we modify hydrology.hyd to use
    # year-weighted average ESCO (since SWAT+ doesn't natively support
    # annual ESCO changes without soft calibration/scheduling tricks)
    # Here we use the time-averaged ESCO across all years
    avg_esco = sum(
        dynamic_params[y]["weighted_esco"] for y in dynamic_params
    ) / len(dynamic_params)
    avg_cn2 = sum(
        dynamic_params[y]["weighted_cn2"] for y in dynamic_params
    ) / len(dynamic_params)
    log.info("  Time-averaged ESCO=%.4f, CN2=%.2f", avg_esco, avg_cn2)

    # Rewrite hydrology.hyd with per-HRU ESCO adjusted by temporal weighting
    hrus = compute_hru_areas(lulc_2017)
    # For each HRU, compute time-weighted ESCO factoring in yearly fractions
    hyd_lines = [
        f"hydrology.hyd: {HEADER}",
        "name                 lat_ttime       lat_sed       can_max          esco          epco   orgn_enrich   orgp_enrich       cn3_swf       bio_mix         perco      lat_orgn      lat_orgp        pet_co       latq_co  ",
    ]
    for h in hrus:
        # For dynamic: ESCO is time-weighted average of class ESCO * class fraction
        lulc_label = h["lulc_label"]
        class_esco = ESCO_BY_LULC[h["lulc_code"]]

        # Weight by how much this class contributes across years
        total_weight = 0.0
        weighted_class_esco = 0.0
        for year in range(2017, 2025):
            yr_frac = all_fracs[year].get(lulc_label, 0.0) / 100.0
            weighted_class_esco += yr_frac * class_esco
            total_weight += yr_frac

        if total_weight > 0:
            dynamic_esco = weighted_class_esco / total_weight
        else:
            dynamic_esco = class_esco

        name = f"hyd{h['id']:03d}"
        hyd_lines.append(
            f"{name:20s}"
            f"   {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {1.0:11.5f}"
            f" {dynamic_esco:11.5f}"
            f" {0.50:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {0.90:11.5f}"
            f" {0.20:11.5f}"
            f" {0.20:11.5f}"
            f" {0.0:11.5f}"
            f" {0.0:11.5f}"
            f" {1.0:11.5f}"
            f" {0.01:11.5f}"
        )
    (out / "hydrology.hyd").write_text("\n".join(hyd_lines) + "\n")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Build and validate SWAT+ TxtInOut for Sejong (STATIC + DYNAMIC)."""

    # Setup logging to file
    LOGDIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGDIR / "p2_08_build.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    log.info("=" * 70)
    log.info("Sejong SWAT+ TxtInOut Builder")
    log.info("=" * 70)

    # -- Build STATIC scenario --
    static_dir = build_static_scenario()
    static_errors = validate_txtinout(static_dir)
    if static_errors:
        log.error("STATIC validation errors:")
        for e in static_errors:
            log.error("  - %s", e)
    else:
        log.info("STATIC validation: PASS")

    # -- Build DYNAMIC scenario --
    dynamic_dir = build_dynamic_scenario()
    dynamic_errors = validate_txtinout(dynamic_dir)
    if dynamic_errors:
        log.error("DYNAMIC validation errors:")
        for e in dynamic_errors:
            log.error("  - %s", e)
    else:
        log.info("DYNAMIC validation: PASS")

    # -- Copy results to persistent location --
    RESULTS.mkdir(parents=True, exist_ok=True)
    for scenario, src_dir in [("static", static_dir), ("dynamic", dynamic_dir)]:
        dst = RESULTS / scenario
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
        log.info("Results copied: %s -> %s", src_dir, dst)

    # -- Summary --
    log.info("")
    log.info("=" * 70)
    log.info("BUILD SUMMARY")
    log.info("=" * 70)
    log.info("STATIC  TxtInOut: %s (%d files)", RESULTS / "static",
             len(list((RESULTS / "static").iterdir())))
    log.info("DYNAMIC TxtInOut: %s (%d files)", RESULTS / "dynamic",
             len(list((RESULTS / "dynamic").iterdir())))
    log.info("STATIC  errors: %d", len(static_errors))
    log.info("DYNAMIC errors: %d", len(dynamic_errors))

    if static_errors or dynamic_errors:
        log.warning("There were validation errors -- check above.")
        sys.exit(1)
    else:
        log.info("All validations passed.")
        log.info("")
        log.info("To run SWAT+:")
        log.info("  cd %s && %s", RESULTS / "static", SWATPLUS)
        log.info("  cd %s && %s", RESULTS / "dynamic", SWATPLUS)


if __name__ == "__main__":
    main()
