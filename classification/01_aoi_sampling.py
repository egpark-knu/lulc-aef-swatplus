"""
Phase 1 — Task 1: 세종시 AOI 정의 + AEF 가용성 검증

1. GEE에서 세종시 행정경계 기반 AOI 정의
2. AEF 2017-2024 가용성 확인 (타일 수, 커버리지)
3. Quick sample로 임베딩 품질 사전 점검
4. WorldCover 2021 + Dynamic World 커버리지 확인

Usage:
    python phase1/t1_sejong_aoi.py
"""

import ee
import numpy as np
import json
import os
from pathlib import Path
from phase1.aoi_config import build_aoi, resolve_aoi_settings

ee.Initialize()

# ── 세종시 AOI 정의 ──
SEJONG_CENTER = [127.00, 36.48]

print("=" * 60)
print("Task 1: 세종시 AOI 정의")
print("=" * 60)

aoi_settings = resolve_aoi_settings(
    aoi_asset=os.environ.get("SEJONG_AOI_ASSET"),
    bbox_text=os.environ.get("SEJONG_AOI_BBOX"),
)
sejong_aoi = build_aoi(ee, aoi_settings)
aoi_source = aoi_settings["description"]
print(f"  AOI mode: {aoi_settings['mode']}")
if "bbox" in aoi_settings:
    print(f"  Reference bbox: {aoi_settings['bbox']}")
if aoi_settings.get("asset_id"):
    print(f"  Asset: {aoi_settings['asset_id']}")

# AOI 면적 계산
area_km2 = sejong_aoi.area().divide(1e6).getInfo()
print(f"  AOI 면적: {area_km2:.1f} km²")
print(f"  AOI 출처: {aoi_source}")

# ── AEF 가용성 확인 ──
print(f"\n{'=' * 60}")
print("AEF Annual 가용성 확인 (2017-2024)")
print("=" * 60)

aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
YEARS = list(range(2017, 2025))  # 2017-2024

aef_stats = {}
for year in YEARS:
    yearly = aef.filter(ee.Filter.calendarRange(year, year, 'year'))
    filtered = yearly.filterBounds(sejong_aoi)
    count = filtered.size().getInfo()

    if count > 0:
        mosaic = filtered.mosaic().clip(sejong_aoi)
        bands = mosaic.bandNames().getInfo()
        n_bands = len(bands)
    else:
        n_bands = 0

    aef_stats[year] = {'tiles': count, 'bands': n_bands}
    status = "OK" if count > 0 else "MISSING"
    print(f"  {year}: {status} (tiles={count}, bands={n_bands})")

# ── Quick Sample (2021) — 임베딩 품질 사전 점검 ──
print(f"\n{'=' * 60}")
print("Quick Sample: AEF 2021 임베딩 (세종)")
print("=" * 60)

aef_2021 = (aef.filter(ee.Filter.calendarRange(2021, 2021, 'year'))
            .filterBounds(sejong_aoi)
            .mosaic()
            .clip(sejong_aoi))

QUICK_SAMPLE_SIZE = 500
sample = aef_2021.sample(
    region=sejong_aoi,
    scale=10,
    numPixels=QUICK_SAMPLE_SIZE,
    seed=42,
    geometries=True,
)

features = sample.getInfo()['features']
print(f"  추출 샘플: {len(features)}")

if len(features) > 0:
    embeddings = []
    coords = []
    for feat in features:
        props = feat['properties']
        emb = [props.get(f'A{i:02d}', 0.0) for i in range(64)]
        embeddings.append(emb)
        coords.append(feat['geometry']['coordinates'])

    embeddings = np.array(embeddings)
    coords = np.array(coords)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  임베딩 shape: {embeddings.shape}")
    print(f"  Norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")
    print(f"  좌표 범위: lon=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}], "
          f"lat=[{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]")

    # NaN/zero 체크
    n_nan = np.isnan(embeddings).any(axis=1).sum()
    n_zero = (norms == 0).sum()
    print(f"  NaN 샘플: {n_nan}, Zero-norm 샘플: {n_zero}")

# ── WorldCover 2021 커버리지 확인 ──
print(f"\n{'=' * 60}")
print("WorldCover 2021 커버리지 확인")
print("=" * 60)

wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(sejong_aoi)

# 클래스 분포 히스토그램
wc_hist = wc.select('Map').reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=sejong_aoi,
    scale=10,
    maxPixels=1e9,
    bestEffort=True,
)

hist_data = wc_hist.getInfo()['Map']
print(f"  클래스 분포 (픽셀 수):")

LULC_MAP = {
    '10': 'Tree cover', '20': 'Shrubland', '30': 'Grassland',
    '40': 'Cropland', '50': 'Built-up', '60': 'Bare/sparse',
    '70': 'Snow/Ice', '80': 'Water', '90': 'Wetland',
    '95': 'Mangrove', '100': 'Moss/lichen'
}

total_pixels = sum(hist_data.values())
for cls_code in sorted(hist_data.keys(), key=lambda x: -hist_data[x]):
    count = hist_data[cls_code]
    pct = count / total_pixels * 100
    name = LULC_MAP.get(cls_code, f'Unknown({cls_code})')
    print(f"    {cls_code:>3s} ({name:15s}): {count:>10,} ({pct:5.1f}%)")

print(f"  Total pixels: {total_pixels:,}")
print(f"  Approx area at 10m: {total_pixels * 100 / 1e6:.1f} km²")

# ── Dynamic World 2021 커버리지 확인 ──
print(f"\n{'=' * 60}")
print("Dynamic World 2021 커버리지 확인")
print("=" * 60)

dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
dw_2021 = (dw.filter(ee.Filter.calendarRange(2021, 2021, 'year'))
           .filterBounds(sejong_aoi)
           .select('label'))

n_dw = dw_2021.size().getInfo()
print(f"  2021 DW 이미지 수: {n_dw}")

if n_dw > 0:
    dw_mode = dw_2021.mode().clip(sejong_aoi)
    dw_hist = dw_mode.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=sejong_aoi,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    )

    DW_CLASSES = {
        '0': 'water', '1': 'trees', '2': 'grass', '3': 'flooded_veg',
        '4': 'crops', '5': 'shrub_scrub', '6': 'built', '7': 'bare', '8': 'snow_ice'
    }

    dw_data = dw_hist.getInfo()['label']
    total_dw = sum(dw_data.values())
    print(f"  클래스 분포 (연간 모드):")
    for cls in sorted(dw_data.keys(), key=lambda x: -dw_data[x]):
        count = dw_data[cls]
        pct = count / total_dw * 100
        name = DW_CLASSES.get(cls, f'Unknown({cls})')
        print(f"    {cls}: {name:15s}: {count:>10,} ({pct:5.1f}%)")

# ── 결과 저장 ──
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

result = {
    'aoi': {
        'source': aoi_source,
        'mode': aoi_settings['mode'],
        'bbox': aoi_settings.get('bbox'),
        'asset_id': aoi_settings.get('asset_id'),
        'area_km2': area_km2,
        'center': SEJONG_CENTER,
    },
    'aef_availability': {str(k): v for k, v in aef_stats.items()},
    'worldcover_histogram': hist_data,
    'quick_sample': {
        'n_samples': len(features),
        'mean_norm': float(norms.mean()) if len(features) > 0 else None,
    },
}

with open(OUT_DIR / 't1_sejong_aoi_result.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print("T1 완료 — 결과 저장됨")
print("=" * 60)
print(f"  AOI: {aoi_source}, {area_km2:.1f} km²")
print(f"  AEF: {sum(1 for v in aef_stats.values() if v['tiles'] > 0)}/8 연도 가용")
print(f"  결과: {OUT_DIR / 't1_sejong_aoi_result.json'}")
