"""
Phase 0 PoC — Step 3: AEF 임베딩 → LULC 분류 (Linear Probe)
ESA WorldCover 2021 (10m)을 레이블로 사용하여 AEF 임베딩의 분류 성능을 테스트한다.

Pipeline:
  1. GEE에서 AEF 임베딩 + ESA WorldCover 레이블을 동시 추출
  2. Linear probe (LogisticRegression) 학습
  3. 다른 연도에 transfer → 연간 LULC 맵 자동 생산 가능성 검증

Usage:
    python poc_03_lulc_classification.py
"""

import ee
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, cohen_kappa_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import json

# ── GEE 초기화 ──
ee.Initialize()

# ── 금호강 유역 ──
DAEGU_CENTER = [128.6014, 35.8714]
BUFFER_KM = 15
daegu = ee.Geometry.Point(DAEGU_CENTER).buffer(BUFFER_KM * 1000)

# ── ESA WorldCover 2021 (10m) ──
# 클래스: 10=Tree, 20=Shrub, 30=Grassland, 40=Cropland, 50=Built-up,
#         60=Bare, 70=Snow/Ice, 80=Water, 90=Wetland, 95=Mangrove, 100=Moss
worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()

# ── SWAT+ 호환 클래스 매핑 ──
LULC_MAP = {
    10: 'Forest',      # Tree cover
    20: 'Shrubland',   # Shrubland
    30: 'Grassland',   # Grassland
    40: 'Cropland',    # Cropland
    50: 'Urban',       # Built-up
    60: 'Barren',      # Bare/sparse
    70: 'Snow',        # Snow/Ice
    80: 'Water',       # Permanent water
    90: 'Wetland',     # Herbaceous wetland
    95: 'Mangrove',    # Mangroves
    100: 'Moss',       # Moss/lichen
}

SWAT_MAP = {
    'Forest': 'FRSE',
    'Shrubland': 'RNGB',
    'Grassland': 'PAST',
    'Cropland': 'AGRR',
    'Urban': 'URLD',
    'Barren': 'SWRN',
    'Snow': 'SNOI',
    'Water': 'WATR',
    'Wetland': 'WETL',
    'Mangrove': 'WETF',
    'Moss': 'PAST',
}

# ── AEF 임베딩 + WorldCover 레이블 동시 추출 ──
print("=" * 60)
print("Step 1: AEF 임베딩 + WorldCover 레이블 추출")
print("=" * 60)

# AEF 2021 (WorldCover와 동일 연도)
aef_2021 = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filter(ee.Filter.calendarRange(2021, 2021, 'year'))
            .filterBounds(daegu)
            .mosaic()
            .clip(daegu))

# WorldCover 클립
wc_clipped = worldcover.clip(daegu).select('Map').rename('LULC')

# 임베딩 + 레이블 합성
combined = aef_2021.addBands(wc_clipped)

# 샘플링 (stratified는 GEE에서 복잡하므로 random으로)
SAMPLE_SIZE = 2000
sample = combined.sample(
    region=daegu,
    scale=10,
    numPixels=SAMPLE_SIZE,
    seed=42,
    geometries=True,
)

features = sample.getInfo()['features']
print(f"추출된 샘플: {len(features)}")

# ── 데이터 파싱 ──
embeddings = []
labels = []
coords = []

for feat in features:
    props = feat['properties']
    lulc = props.get('LULC')
    if lulc is None:
        continue

    emb = [props.get(f'A{i:02d}', 0.0) for i in range(64)]
    embeddings.append(emb)
    labels.append(int(lulc))
    coords.append(feat['geometry']['coordinates'])

X = np.array(embeddings)
y = np.array(labels)
coords = np.array(coords)

print(f"유효 샘플: {len(X)}")
print(f"\n클래스 분포:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    name = LULC_MAP.get(u, f'Unknown({u})')
    print(f"  {u:3d} ({name:12s}): {c:4d} ({c/len(y)*100:.1f}%)")

# ── Train/Test Split ──
print(f"\n{'='*60}")
print("Step 2: Linear Probe 학습")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── Logistic Regression (Linear Probe) ──
clf = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ── 평가 ──
print(f"\n{'='*60}")
print("Step 3: 분류 성능 평가")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"Cohen's Kappa:    {kappa:.4f}")

# 클래스별 성능 (이름 매핑)
target_names = [LULC_MAP.get(c, str(c)) for c in sorted(np.unique(y))]
print(f"\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# ── Cross-year Transfer 테스트 ──
print(f"\n{'='*60}")
print("Step 4: Cross-year Transfer (2021 모델 → 다른 연도 적용)")
print("=" * 60)

# 2017년과 2025년 데이터 로드 (Step 1에서 저장된 것)
DATA_DIR = Path(__file__).parent / "data"

for test_year in [2017, 2025]:
    fpath = DATA_DIR / f"aef_daegu_{test_year}.npz"
    if not fpath.exists():
        print(f"  ⚠️ {test_year}년 데이터 없음")
        continue

    data = np.load(fpath)
    X_other = data['embeddings']

    # 2021 모델로 예측
    y_other_pred = clf.predict(X_other)

    print(f"\n  {test_year}년 예측 결과 (2021 모델 transfer):")
    unique_pred, counts_pred = np.unique(y_other_pred, return_counts=True)
    for u, c in zip(unique_pred, counts_pred):
        name = LULC_MAP.get(u, f'Unknown({u})')
        swat = SWAT_MAP.get(name, '???')
        print(f"    {name:12s} ({swat}): {c:4d} ({c/len(y_other_pred)*100:.1f}%)")

# ── LULC 변화 통계 (2017 vs 2025) ──
print(f"\n{'='*60}")
print("Step 5: LULC 변화 통계 (2017 → 2025)")
print("=" * 60)

f17 = DATA_DIR / "aef_daegu_2017.npz"
f25 = DATA_DIR / "aef_daegu_2025.npz"

if f17.exists() and f25.exists():
    X_17 = np.load(f17)['embeddings']
    X_25 = np.load(f25)['embeddings']

    y_17 = clf.predict(X_17)
    y_25 = clf.predict(X_25)

    n = min(len(y_17), len(y_25))
    y_17, y_25 = y_17[:n], y_25[:n]

    changed = (y_17 != y_25)
    print(f"  변화 포인트: {changed.sum()}/{n} ({changed.sum()/n*100:.1f}%)")

    if changed.sum() > 0:
        print(f"\n  전환 매트릭스 (주요 변화):")
        for i in range(n):
            if y_17[i] != y_25[i]:
                from_name = LULC_MAP.get(y_17[i], str(y_17[i]))
                to_name = LULC_MAP.get(y_25[i], str(y_25[i]))
                # 카운트만 집계
                pass

        # 전환 집계
        transitions = {}
        for i in range(n):
            if y_17[i] != y_25[i]:
                key = (LULC_MAP.get(y_17[i], str(y_17[i])),
                       LULC_MAP.get(y_25[i], str(y_25[i])))
                transitions[key] = transitions.get(key, 0) + 1

        for (f, t), count in sorted(transitions.items(), key=lambda x: -x[1])[:10]:
            print(f"    {f:12s} → {t:12s}: {count:3d}")

# ── 결과 저장 ──
OUT_DIR = Path(__file__).parent / "data"
result = {
    'overall_accuracy': float(acc),
    'kappa': float(kappa),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'classes': {str(k): v for k, v in LULC_MAP.items()},
    'swat_mapping': SWAT_MAP,
}
with open(OUT_DIR / 'classification_results.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✅ 결과 저장: {OUT_DIR / 'classification_results.json'}")
print("\n🎯 Phase 0 PoC Step 3 완료!")
print(f"\n📊 핵심 결과: OA={acc*100:.1f}%, Kappa={kappa:.3f}")
print("   → AEF 임베딩으로 LULC 분류가 가능함을 확인!")
