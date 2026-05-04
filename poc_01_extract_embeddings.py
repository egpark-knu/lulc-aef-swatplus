"""
Phase 0 PoC — Step 1: AEF 임베딩 추출 및 PCA 시각화
AlphaEarth Foundations 연간 임베딩을 금호강 유역에서 추출하고,
PCA로 시계열 변화 패턴을 확인한다.

Usage:
    python poc_01_extract_embeddings.py
"""

import ee
import numpy as np
import json
from pathlib import Path

# ── GEE 초기화 ──
ee.Initialize()

# ── 금호강 유역 (대구) 대략적 바운딩 박스 ──
# 실제 유역 shapefile 적용 전 테스트용
DAEGU_CENTER = [128.6014, 35.8714]
BUFFER_KM = 15  # 15km 반경

daegu = ee.Geometry.Point(DAEGU_CENTER).buffer(BUFFER_KM * 1000)

# ── AEF 임베딩 컬렉션 ──
aef = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
print(f"AEF 전체 이미지 수: {aef.size().getInfo()}")

# ── 연도별 임베딩 추출 (2017-2025) ──
YEARS = list(range(2017, 2026))
SAMPLE_SIZE = 500  # 유역 내 500개 랜덤 포인트

results = {}

for year in YEARS:
    print(f"\n{'='*50}")
    print(f"처리 중: {year}")

    # 해당 연도 필터링
    yearly = aef.filter(ee.Filter.calendarRange(year, year, 'year'))
    count = yearly.size().getInfo()
    print(f"  타일 수: {count}")

    if count == 0:
        print(f"  ⚠️ {year}년 데이터 없음, 건너뜀")
        continue

    # 유역 영역으로 모자이크
    mosaic = yearly.filterBounds(daegu).mosaic().clip(daegu)

    # 밴드 정보 확인 (첫 해만)
    if year == YEARS[0]:
        bands = mosaic.bandNames().getInfo()
        print(f"  밴드 수: {len(bands)}")
        print(f"  밴드 이름: {bands[:5]}...{bands[-3:]}")

    # 랜덤 포인트에서 임베딩 샘플링
    sample = mosaic.sample(
        region=daegu,
        scale=10,  # 10m 해상도
        numPixels=SAMPLE_SIZE,
        seed=42,
        geometries=True,
    )

    features = sample.getInfo()['features']
    n_samples = len(features)
    print(f"  샘플 수: {n_samples}")

    if n_samples == 0:
        print(f"  ⚠️ {year}년 샘플 없음, 건너뜀")
        continue

    # 임베딩 벡터 추출 (64차원)
    embeddings = []
    coords = []
    for feat in features:
        props = feat['properties']
        # A00 ~ A63 밴드
        emb = [props.get(f'A{i:02d}', 0.0) for i in range(64)]
        embeddings.append(emb)

        geo = feat['geometry']['coordinates']
        coords.append(geo)

    embeddings = np.array(embeddings)
    coords = np.array(coords)

    results[year] = {
        'embeddings': embeddings,
        'coords': coords,
        'n_samples': n_samples,
    }

    print(f"  임베딩 shape: {embeddings.shape}")
    print(f"  평균 norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"  임베딩 샘플: {embeddings[0, :5].tolist()}")

# ── 결과 저장 (numpy) ──
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

for year, data in results.items():
    np.savez(
        OUT_DIR / f"aef_daegu_{year}.npz",
        embeddings=data['embeddings'],
        coords=data['coords'],
    )

print(f"\n✅ 저장 완료: {OUT_DIR}")
print(f"   연도: {sorted(results.keys())}")
print(f"   총 샘플: {sum(d['n_samples'] for d in results.values())}")

# ── 요약 통계 ──
print(f"\n{'='*50}")
print("연도별 요약:")
for year in sorted(results.keys()):
    emb = results[year]['embeddings']
    print(f"  {year}: n={emb.shape[0]}, "
          f"mean_norm={np.linalg.norm(emb, axis=1).mean():.4f}, "
          f"std_norm={np.linalg.norm(emb, axis=1).std():.4f}")
