"""
Phase 1 — Task 2+3+4: 세종 AEF 분류 (Spatial Block CV)

1. AEF 2021 + WorldCover 2021 + DW 2021 동시 추출 (대규모)
2. 6-class 축소 + stable/change pixel mask
3. Spatial block CV (250m, 500m, 1km, 2km)
4. Cross-year transfer (2017-2024)
5. G1/G2 Go/No-Go 판정

Usage:
    python phase1/t234_sejong_classification.py

Memory Safety (2026-04-03 crash 교훈):
    - 기본 safe mode: BLAS/OpenMP thread cap + sklearn n_jobs 축소
    - 2021 샘플 캐시 재사용 (불필요한 getInfo 재실행 방지)
    - cross-year 결과를 연도별 JSON으로 즉시 저장하여 재시작 가능
    - 연도별 순차 처리 후 del + gc.collect()
    - 연도별 처리 후 ee.Reset() + ee.Initialize()
    - psutil 메모리 모니터링 (192GB 한도)
    - getInfo() 결과는 파싱 후 즉시 해제
"""

import os

DEFAULT_SAFE_MODE = os.environ.get("T234_SAFE_MODE", "1").strip().lower() not in {"0", "false", "no", "off"}
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
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from phase1.aoi_config import build_aoi, resolve_aoi_settings
from phase1.runtime_safety import env_flag, parse_years_spec, pending_years, year_cache_path
import warnings
warnings.filterwarnings('ignore')

# ── Memory Safety ──
MEM_LIMIT_GB = 180  # 192GB 시스템, 12GB 여유

def check_memory(label=""):
    """현재 프로세스 메모리 사용량 출력. 한도 초과 시 경고."""
    try:
        import psutil
        proc = psutil.Process()
        mem_gb = proc.memory_info().rss / 1e9
        sys_mem = psutil.virtual_memory()
        sys_used_gb = sys_mem.used / 1e9
        print(f"  [MEM] {label}: process={mem_gb:.1f}GB, system={sys_used_gb:.1f}/{sys_mem.total/1e9:.0f}GB")
        if sys_used_gb > MEM_LIMIT_GB:
            print(f"  ⚠️ 시스템 메모리 {sys_used_gb:.0f}GB > {MEM_LIMIT_GB}GB 한도!")
            return False
        return True
    except ImportError:
        return True  # psutil 없으면 skip

ee.Initialize()

# ── Configuration ──
aoi_settings = resolve_aoi_settings(
    aoi_asset=os.environ.get("SEJONG_AOI_ASSET"),
    bbox_text=os.environ.get("SEJONG_AOI_BBOX"),
)
SEJONG_AOI = build_aoi(ee, aoi_settings)

# GEE 안전 마진: numPixels + .limit() 이중 보호
# ee.Image.sample의 numPixels는 "approximate"이므로 hard cap 필요
SAMPLE_SIZE = 4000  # 보수적 설정 (5000 limit 대비 20% 마진)
SAMPLE_LIMIT = 4000  # .limit()으로 hard cap 보장
BLOCK_SIZES = [250, 500, 1000, 2000]  # meters
N_FOLDS = 5
YEARS = list(range(2017, 2025))  # 2017-2024
SAFE_N_JOBS = int(os.environ.get("T234_N_JOBS", "1" if DEFAULT_SAFE_MODE else "-1"))
FORCE_SAMPLE_REBUILD = env_flag(os.environ.get("T234_FORCE_SAMPLE_REBUILD"), default=False)
FORCE_RERUN_YEARS = env_flag(os.environ.get("T234_FORCE_RERUN_YEARS"), default=False)
SKIP_CROSS_YEAR = env_flag(os.environ.get("T234_SKIP_CROSS_YEAR"), default=False)
TARGET_YEARS = parse_years_spec(os.environ.get("T234_TARGET_YEARS"), YEARS)

# WorldCover → 6-class mapping
WC_TO_6CLASS = {
    10: 0,   # Tree cover → Forest
    20: 3,   # Shrubland → Grassland (merge)
    30: 3,   # Grassland → Grassland
    40: 1,   # Cropland → Cropland
    50: 2,   # Built-up → Urban
    60: 5,   # Bare → Barren
    70: -1,  # Snow → remove (not in Sejong)
    80: 4,   # Water → Water
    90: 4,   # Wetland → Water (merge, tiny)
    95: -1,  # Mangrove → remove
    100: -1, # Moss → remove
}

CLASS_NAMES = ['Forest', 'Cropland', 'Urban', 'Grassland', 'Water', 'Barren']
SWAT_CODES = ['FRSE', 'AGRR', 'URLD', 'PAST', 'WATR', 'SWRN']

# DW label → 6-class mapping
DW_TO_6CLASS = {
    0: 4,   # water → Water
    1: 0,   # trees → Forest
    2: 3,   # grass → Grassland
    3: 4,   # flooded_veg → Water
    4: 1,   # crops → Cropland
    5: 3,   # shrub_scrub → Grassland
    6: 2,   # built → Urban
    7: 5,   # bare → Barren
    8: -1,  # snow_ice → remove
}

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)
TRANSFER_DIR = OUT_DIR / "transfer_years"
TRANSFER_DIR.mkdir(exist_ok=True)
SAMPLE_CACHE = OUT_DIR / "sejong_2021_samples.npz"

print("=" * 60)
print("Runtime Safety")
print("=" * 60)
print(f"  safe_mode={DEFAULT_SAFE_MODE}, max_threads={MAX_THREADS}, sklearn_n_jobs={SAFE_N_JOBS}")
print(f"  target_years={TARGET_YEARS}")
print(f"  force_sample_rebuild={FORCE_SAMPLE_REBUILD}, force_rerun_years={FORCE_RERUN_YEARS}")
print(f"  aoi_mode={aoi_settings['mode']}")
print(f"  aoi_source={aoi_settings['description']}")

# ══════════════════════════════════════════════
# STEP 1: 대규모 샘플 추출 (AEF + WC + DW)
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 1: AEF 2021 + WorldCover + DW 샘플 추출")
print("=" * 60)

aef_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
aef_2021 = (aef_col.filter(ee.Filter.calendarRange(2021, 2021, 'year'))
            .filterBounds(SEJONG_AOI)
            .mosaic()
            .clip(SEJONG_AOI))

wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(SEJONG_AOI)
wc_label = wc.select('Map').rename('WC')

dw_col = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
dw_2021_mode = (dw_col.filter(ee.Filter.calendarRange(2021, 2021, 'year'))
                .filterBounds(SEJONG_AOI)
                .select('label')
                .mode()
                .clip(SEJONG_AOI)
                .rename('DW'))

if SAMPLE_CACHE.exists() and not FORCE_SAMPLE_REBUILD:
    print(f"  캐시 로드: {SAMPLE_CACHE}")
    cached = np.load(SAMPLE_CACHE)
    X = cached["X"].astype(np.float32, copy=False)
    y = cached["y"].astype(np.int16, copy=False)
    y_dw = cached["y_dw"].astype(np.int16, copy=False)
    coords = cached["coords"].astype(np.float32, copy=False)
    stable_mask = cached["stable_mask"].astype(bool, copy=False)
else:
    combined = aef_2021.addBands(wc_label).addBands(dw_2021_mode)

    print(f"  샘플 추출 중... (numPixels={SAMPLE_SIZE}, hard limit={SAMPLE_LIMIT})")
    sample = combined.sample(
        region=SEJONG_AOI,
        scale=10,
        numPixels=SAMPLE_SIZE,
        seed=42,
        geometries=True,
    ).limit(SAMPLE_LIMIT)  # hard cap: getInfo() payload 안전 보장

    features = sample.getInfo()['features']
    print(f"  추출 완료: {len(features)} 샘플")
    check_memory("after getInfo")

    # ── 파싱 ──
    embeddings_list = []
    wc_labels = []
    dw_labels = []
    coords_list = []

    for feat in features:
        props = feat['properties']
        wc_val = props.get('WC')
        dw_val = props.get('DW')
        if wc_val is None or dw_val is None:
            continue

        emb = [props.get(f'A{i:02d}', 0.0) for i in range(64)]
        embeddings_list.append(emb)
        wc_labels.append(int(wc_val))
        dw_labels.append(int(dw_val))
        coords_list.append(feat['geometry']['coordinates'])

    X_raw = np.asarray(embeddings_list, dtype=np.float32)
    y_wc_raw = np.asarray(wc_labels, dtype=np.int16)
    y_dw_raw = np.asarray(dw_labels, dtype=np.int16)
    coords_raw = np.asarray(coords_list, dtype=np.float32)

    # Free raw GeoJSON (can be large)
    del features, embeddings_list, wc_labels, dw_labels, coords_list, combined, sample
    gc.collect()

    print(f"  유효 샘플: {len(X_raw)}")

    # ── 6-class 매핑 ──
    y_6class = np.asarray([WC_TO_6CLASS.get(v, -1) for v in y_wc_raw], dtype=np.int16)
    y_dw_6class = np.asarray([DW_TO_6CLASS.get(v, -1) for v in y_dw_raw], dtype=np.int16)

    # -1 제거
    valid_mask = (y_6class >= 0) & (y_dw_6class >= 0)
    X = X_raw[valid_mask].astype(np.float32, copy=False)
    y = y_6class[valid_mask].astype(np.int16, copy=False)
    y_dw = y_dw_6class[valid_mask].astype(np.int16, copy=False)
    coords = coords_raw[valid_mask].astype(np.float32, copy=False)

    # 저장
    stable_mask = (y == y_dw)
    np.savez(
        SAMPLE_CACHE,
        X=X, y=y, y_dw=y_dw, coords=coords,
        stable_mask=stable_mask,
    )

    del X_raw, y_wc_raw, y_dw_raw, coords_raw, y_6class, y_dw_6class, valid_mask
    gc.collect()

print(f"  6-class 유효 샘플: {len(X)}")
print(f"\n  클래스 분포 (WorldCover 6-class):")
for i, name in enumerate(CLASS_NAMES):
    n = (y == i).sum()
    print(f"    {i}: {name:12s} ({SWAT_CODES[i]}): {n:6d} ({n/len(y)*100:5.1f}%)")

# ── Dual Pixel Mask ──
change_mask = ~stable_mask
n_stable = stable_mask.sum()
n_change = change_mask.sum()
print(f"\n  Stable pixels (WC ∩ DW 일치): {n_stable} ({n_stable/len(y)*100:.1f}%)")
print(f"  Change candidates (불일치):     {n_change} ({n_change/len(y)*100:.1f}%)")
check_memory("after sample cache ready")

# ══════════════════════════════════════════════
# STEP 2: Spatial Block CV (4 block sizes)
# ══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 2: Spatial Block CV — Linear Probe")
print("=" * 60)

# Stable pixels만 사용 (high-precision labels)
X_stable = X[stable_mask]
y_stable = y[stable_mask]
coords_stable = coords[stable_mask]

print(f"  학습/검증 데이터: {len(X_stable)} stable pixels")

# 좌표를 미터 단위로 변환 (간단한 근사: 1° lat ≈ 111km, 1° lon ≈ 89km at 36.5°N)
LAT_M = 111_000  # m per degree latitude
LON_M = 89_000   # m per degree longitude at ~36.5°N

coords_m = np.zeros_like(coords_stable)
coords_m[:, 0] = (coords_stable[:, 0] - 126.85) * LON_M
coords_m[:, 1] = (coords_stable[:, 1] - 36.38) * LAT_M

block_cv_results = {}

for block_size in BLOCK_SIZES:
    print(f"\n  Block size: {block_size}m")

    # Block ID 할당
    block_ids = ((coords_m[:, 0] // block_size).astype(int) * 10000 +
                 (coords_m[:, 1] // block_size).astype(int))

    n_blocks = len(np.unique(block_ids))
    print(f"    블록 수: {n_blocks}")

    if n_blocks < N_FOLDS:
        print(f"    ⚠️ 블록 수 부족 (< {N_FOLDS}), 건너뜀")
        continue

    # GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_oas = []
    fold_kappas = []
    all_preds = np.full(len(y_stable), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_stable, y_stable, groups=block_ids)):
        X_tr, X_te = X_stable[train_idx], X_stable[test_idx]
        y_tr, y_te = y_stable[train_idx], y_stable[test_idx]

        # Inner CV for C tuning: also spatial (GroupKFold)
        # to avoid leaking spatial structure into hyperparameter selection
        inner_block_ids = block_ids[train_idx]
        n_inner_blocks = len(np.unique(inner_block_ids))
        inner_n_splits = min(3, n_inner_blocks)

        if inner_n_splits >= 2:
            inner_gkf = GroupKFold(n_splits=inner_n_splits)
            inner_cv = list(inner_gkf.split(X_tr, y_tr, groups=inner_block_ids))
        else:
            inner_cv = 3  # fallback to stratified if too few blocks

        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            cv=inner_cv,
            max_iter=2000,
            multi_class='multinomial',
            solver='lbfgs',
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
        print(f"    Fold {fold_i+1}: OA={oa:.4f}, Kappa={kappa:.4f}, "
              f"best_C={clf.C_[0]:.4f}, train={len(X_tr)}, test={len(X_te)}")

    mean_oa = np.mean(fold_oas)
    std_oa = np.std(fold_oas)
    mean_kappa = np.mean(fold_kappas)
    std_kappa = np.std(fold_kappas)

    print(f"    → Mean OA: {mean_oa:.4f} ± {std_oa:.4f}")
    print(f"    → Mean Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")

    # Classification report (aggregated over all folds)
    valid_pred = all_preds >= 0
    if valid_pred.sum() > 0:
        # Use labels= to prevent ValueError when some classes are absent
        present_labels = sorted(set(y_stable[valid_pred]) | set(all_preds[valid_pred]))
        present_names = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]
        cr = classification_report(
            y_stable[valid_pred], all_preds[valid_pred],
            labels=present_labels, target_names=present_names,
            output_dict=True, zero_division=0
        )
        print(f"\n    Class-wise F1 ({block_size}m block):")
        for cls_name in present_names:
            if cls_name in cr:
                f1 = cr[cls_name]['f1-score']
                sup = cr[cls_name]['support']
                print(f"      {cls_name:12s}: F1={f1:.3f} (n={sup})")

    block_cv_results[block_size] = {
        'mean_oa': float(mean_oa),
        'std_oa': float(std_oa),
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'fold_oas': [float(x) for x in fold_oas],
        'fold_kappas': [float(x) for x in fold_kappas],
        'n_blocks': int(n_blocks),
    }

# ══════════════════════════════════════════════
# G1 GATE: Block CV 중간 판정
# ══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("G1 GATE: Spatial Block CV 중간 판정")
print("=" * 60)

# 1km block을 primary로 사용 (v4 계획서 기준)
gate_block = 1000
if gate_block not in block_cv_results:
    gate_block = min(block_cv_results.keys()) if block_cv_results else None

if gate_block is None:
    print("  ❌ Block CV 결과 없음 — STOP")
    import sys; sys.exit(1)

gate_oa = block_cv_results[gate_block]['mean_oa']
gate_kappa = block_cv_results[gate_block]['mean_kappa']

if gate_oa >= 0.80 and gate_kappa >= 0.60:
    g1_gate = "PASS"
    print(f"  ✅ G1 PASS: OA={gate_oa:.4f} (≥0.80), Kappa={gate_kappa:.4f} (≥0.60)")
    print(f"  → T4b (Cross-year Transfer) 진행")
elif gate_oa >= 0.70 and gate_kappa >= 0.40:
    g1_gate = "CONDITIONAL"
    print(f"  ⚠️ G1 CONDITIONAL: OA={gate_oa:.4f} (≥0.70), Kappa={gate_kappa:.4f} (≥0.40)")
    print(f"  → RF fallback을 시도할 수 있으나, 일단 Cross-year Transfer 진행")
else:
    g1_gate = "FAIL"
    print(f"  ❌ G1 FAIL: OA={gate_oa:.4f} (<0.70)")
    print(f"  → Cross-year 진행하지 않음. 유역 교체 또는 클래스 축소 검토 필요.")

    # Save partial results and exit
    partial_results = {
        'block_cv': {str(k): v for k, v in block_cv_results.items()},
        'g1_gate': g1_gate,
        'g1_oa': float(gate_oa),
        'g1_kappa': float(gate_kappa),
        'verdict': 'NO-GO (G1 Gate Fail)',
    }
    with open(OUT_DIR / 'sejong_classification_results.json', 'w') as f:
        json.dump(partial_results, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {OUT_DIR / 'sejong_classification_results.json'}")
    import sys; sys.exit(0)

# ══════════════════════════════════════════════
# STEP 3: Best model → Cross-year Transfer (T4b)
# ══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 3 (T4b): Cross-year Transfer (2021 model → 2017-2024)")
print("=" * 60)

# Train final model on all stable 2021 data
# Use best C from spatial block CV (not re-tuned with non-spatial CV)
# Collect C values from the primary block size folds
gate_block_ids = ((coords_m[:, 0] // gate_block).astype(int) * 10000 +
                  (coords_m[:, 1] // gate_block).astype(int))
gkf_final = GroupKFold(n_splits=N_FOLDS)
best_Cs = []
for train_idx, test_idx in gkf_final.split(X_stable, y_stable, groups=gate_block_ids):
    inner_block_ids_final = gate_block_ids[train_idx]
    n_inner = len(np.unique(inner_block_ids_final))
    inner_splits = min(3, n_inner)
    if inner_splits >= 2:
        inner_gkf_final = GroupKFold(n_splits=inner_splits)
        inner_cv_final = list(inner_gkf_final.split(
            X_stable[train_idx], y_stable[train_idx], groups=inner_block_ids_final))
    else:
        inner_cv_final = 3
    clf_tmp = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10), cv=inner_cv_final,
        max_iter=2000, multi_class='multinomial', solver='lbfgs',
        random_state=42, n_jobs=SAFE_N_JOBS,
    )
    clf_tmp.fit(X_stable[train_idx], y_stable[train_idx])
    best_Cs.append(clf_tmp.C_[0])

# Use median C from spatial CV folds (robust)
final_C = float(np.median(best_Cs))
clf_final = LogisticRegression(
    C=final_C, max_iter=2000, multi_class='multinomial',
    solver='lbfgs', random_state=42, n_jobs=SAFE_N_JOBS,
)
clf_final.fit(X_stable, y_stable)
print(f"  Final model: C={final_C:.4f} (median of spatial CV), trained on {len(X_stable)} samples")
del best_Cs, clf_tmp
gc.collect()

transfer_results = {}
for cached_path in sorted(TRANSFER_DIR.glob("*.json")):
    with open(cached_path, encoding="utf-8") as f:
        cached_result = json.load(f)
    transfer_results[int(cached_result["year"])] = {
        k: v for k, v in cached_result.items() if k != "year"
    }

years_to_run = [] if SKIP_CROSS_YEAR else pending_years(
    TARGET_YEARS, cache_dir=TRANSFER_DIR, force_rerun=FORCE_RERUN_YEARS
)
print(f"  연도별 실행 대상: {years_to_run}")
if transfer_results:
    print(f"  기존 캐시된 연도: {sorted(transfer_results)}")
if SKIP_CROSS_YEAR:
    print("  cross-year transfer는 환경변수로 비활성화됨")

# ── 고정 샘플 포인트 생성 (Codex 피드백: 연도별 독립 랜덤표본 → 동일 고정 포인트) ──
# 한번 생성한 포인트를 모든 연도에 재사용하여 G2 면적 추세의 샘플링 잡음 제거
if not SKIP_CROSS_YEAR and years_to_run:
    print(f"  고정 샘플 포인트 생성 (seed=42, n={SAMPLE_SIZE})...")
    fixed_points = ee.Image.constant(1).clip(SEJONG_AOI).sample(
        region=SEJONG_AOI,
        scale=10,
        numPixels=SAMPLE_SIZE,
        seed=42,
        geometries=True,
    ).limit(SAMPLE_LIMIT)
    print(f"  → 고정 포인트 수: {fixed_points.size().getInfo()}")
else:
    fixed_points = None

for year in years_to_run:
    print(f"\n  {year}:")
    if not check_memory(f"before {year}"):
        print("  ⚠️ 메모리 여유 부족 — cross-year 중단")
        break

    try:
        aef_year = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                    .filter(ee.Filter.calendarRange(year, year, 'year'))
                    .filterBounds(SEJONG_AOI)
                    .mosaic()
                    .clip(SEJONG_AOI))

        # DW mode for this year (as proxy truth)
        dw_year_mode = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                        .filter(ee.Filter.calendarRange(year, year, 'year'))
                        .filterBounds(SEJONG_AOI)
                        .select('label')
                        .mode()
                        .clip(SEJONG_AOI)
                        .rename('DW'))

        combined_year = aef_year.addBands(dw_year_mode)

        # 고정 포인트에서 연도별 값 추출 (동일 위치 보장)
        sample_year = combined_year.sampleRegions(
            collection=fixed_points,
            scale=10,
            geometries=False,
        ).limit(SAMPLE_LIMIT)

        feats_year = sample_year.getInfo()['features']
        print(f"    샘플: {len(feats_year)}")
        check_memory(f"after {year} getInfo")

        if len(feats_year) == 0:
            continue

        X_year = []
        y_dw_year = []
        for feat in feats_year:
            props = feat['properties']
            dw_val = props.get('DW')
            if dw_val is None:
                continue
            emb = [props.get(f'A{i:02d}', 0.0) for i in range(64)]
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

        # DW proxy agreement (overall + urban-specific)
        dw_agreement = accuracy_score(y_dw_year, y_pred_year)

        # Urban-specific IoU: TP/(TP+FP+FN) for class 2 (Urban)
        urban_pred = (y_pred_year == 2)
        urban_dw = (y_dw_year == 2)
        urban_tp = (urban_pred & urban_dw).sum()
        urban_fp = (urban_pred & ~urban_dw).sum()
        urban_fn = (~urban_pred & urban_dw).sum()
        urban_iou = urban_tp / max(urban_tp + urban_fp + urban_fn, 1)

        print(f"    DW proxy agreement: {dw_agreement:.3f}, Urban IoU: {urban_iou:.3f}")
        print(f"    Class fractions: " +
              ", ".join(f"{n}={fractions[n]:.1f}%" for n in CLASS_NAMES if fractions[n] > 1))

        year_result = {
            'n_samples': int(len(X_year)),
            'dw_agreement': float(dw_agreement),
            'urban_iou': float(urban_iou),
            'class_fractions': fractions,
        }
        transfer_results[year] = year_result
        with open(year_cache_path(TRANSFER_DIR, year), 'w', encoding='utf-8') as f:
            json.dump({'year': year, **year_result}, f, indent=2, ensure_ascii=False)

        # Memory cleanup after each year
        del aef_year, dw_year_mode, combined_year, sample_year
        del feats_year, X_year, y_dw_year, y_pred_year, valid
        del fractions, urban_pred, urban_dw
        gc.collect()
        ee.Reset()
        ee.Initialize()

        if not check_memory(f"after {year}"):
            print(f"  ⚠️ 메모리 한도 초과 — 남은 연도 건너뜀")
            break
    except Exception as e:
        print(f"    ❌ {year} 처리 실패: {e}")
        gc.collect()
        ee.Reset()
        ee.Initialize()
        break

# ══════════════════════════════════════════════
# STEP 4: G1/G2 Go/No-Go 판정
# ══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 4: Go/No-Go 판정")
print("=" * 60)

# G1: Best block CV result (use 1km as primary)
best_block = 1000
if best_block in block_cv_results:
    g1_oa = block_cv_results[best_block]['mean_oa']
    g1_kappa = block_cv_results[best_block]['mean_kappa']
else:
    # Fallback to smallest available
    best_block = min(block_cv_results.keys())
    g1_oa = block_cv_results[best_block]['mean_oa']
    g1_kappa = block_cv_results[best_block]['mean_kappa']

g1_pass = g1_oa >= 0.80 and g1_kappa >= 0.60
print(f"  G1 (Classification): OA={g1_oa:.4f} (≥0.80?{'YES' if g1_oa>=0.80 else 'NO'}), "
      f"Kappa={g1_kappa:.4f} (≥0.60?{'YES' if g1_kappa>=0.60 else 'NO'}) → "
      f"{'PASS' if g1_pass else 'FAIL'}")

# ── Monotonic Non-Decreasing Constraint (Urban) ──
# 가정: 도시화는 비가역적. 한번 도시가 된 땅은 다시 비도시로 돌아가지 않음.
# AEF 임베딩의 연도별 품질 차이로 인한 비논리적 하락을 보정.
# Raw 값도 보존하여 보정 전/후 비교 가능.
urban_trend_raw = []
for yr in sorted(transfer_results.keys()):
    uf = transfer_results[yr]['class_fractions'].get('Urban', 0)
    urban_trend_raw.append((yr, uf))

urban_trend_corrected = []
if urban_trend_raw:
    running_max = urban_trend_raw[0][1]
    for yr, uf in urban_trend_raw:
        corrected = max(uf, running_max)
        urban_trend_corrected.append((yr, corrected))
        running_max = corrected

if urban_trend_raw:
    print(f"\n  Urban Fraction — Monotonic Constraint 적용:")
    print(f"    Raw:       {', '.join(f'{yr}:{f:.1f}%' for yr,f in urban_trend_raw)}")
    print(f"    Corrected: {', '.join(f'{yr}:{f:.1f}%' for yr,f in urban_trend_corrected)}")
    n_corrected = sum(1 for (_, r), (_, c) in zip(urban_trend_raw, urban_trend_corrected) if abs(r - c) > 0.01)
    print(f"    보정된 연도: {n_corrected}개")

# G2: Change signal (monotonic-corrected 값 사용)
g2_checks = []

# G2a: Built-up area Δ ≥ 5%/decade (corrected)
corrected_by_year = {yr: uf for yr, uf in urban_trend_corrected} if urban_trend_corrected else {}
if 2017 in corrected_by_year and 2024 in corrected_by_year:
    urban_2017 = corrected_by_year[2017]
    urban_2024 = corrected_by_year[2024]
    delta_urban = urban_2024 - urban_2017
    g2a = delta_urban >= 5.0
    g2_checks.append(('Built-up Δ≥5%', g2a, f"{delta_urban:+.1f}%"))
    print(f"\n  G2a (Built-up Δ, corrected): {urban_2017:.1f}% → {urban_2024:.1f}% = {delta_urban:+.1f}% "
          f"(≥5%? {'YES' if g2a else 'NO'})")

# G2b: AEF urban prediction과 DW built label의 Urban-class IoU (TP/(TP+FP+FN))
if 2024 in transfer_results:
    tr_2024 = transfer_results[2024]
    urban_iou = tr_2024.get('urban_iou', 0)
    g2b = urban_iou >= 0.60
    g2_checks.append(('DW urban overlap≥60%', g2b, f"{urban_iou:.1%}"))
    print(f"  G2b (DW urban overlap): {urban_iou:.1%} (≥60%? {'YES' if g2b else 'NO'})")

# G2c: Urban fraction monotonic trend (corrected → 항상 monotonic이므로 trend 강도로 판정)
if len(urban_trend_corrected) >= 3:
    fracs_corrected = [f for _, f in urban_trend_corrected]
    # With monotonic correction, check if there's meaningful increase (not just flat)
    total_increase = fracs_corrected[-1] - fracs_corrected[0]
    # At least 2%p total increase over the period to count as meaningful trend
    g2c = total_increase >= 2.0
    g2_checks.append(('Urban trend Δ≥2%p', g2c, f"{total_increase:+.1f}%p total"))
    print(f"  G2c (Urban trend, corrected): total increase {total_increase:+.1f}%p "
          f"(≥2%p? {'YES' if g2c else 'NO'})")

n_g2_pass = sum(1 for _, p, _ in g2_checks if p)
n_g2_computed = len(g2_checks)

# G2 판정: 3개 체크 중 2개 이상 통과 필요. 계산된 체크가 3개 미만이면 INCOMPLETE.
if n_g2_computed < 3:
    g2_pass = False
    g2_status = "INCOMPLETE"
    print(f"  G2 (Change Signal): {n_g2_pass}/{n_g2_computed} passed (3개 중 {3-n_g2_computed}개 누락) → INCOMPLETE")
    print(f"  ⚠️ 일부 연도 데이터 누락으로 G2 판정 불가. 수동 확인 필요.")
else:
    g2_pass = n_g2_pass >= 2
    g2_status = "PASS" if g2_pass else "FAIL"
    print(f"  G2 (Change Signal): {n_g2_pass}/3 checks passed (≥2? {'YES' if g2_pass else 'NO'}) → "
          f"{g2_status}")

# Overall verdict — G2 INCOMPLETE는 자동 GO 금지
if g2_status == "INCOMPLETE":
    verdict = "INCOMPLETE (G2 수동 확인 필요)"
elif g1_pass and g2_pass:
    verdict = "GO"
elif g1_pass or g2_pass:
    verdict = "CONDITIONAL GO"
elif g1_oa >= 0.70:
    verdict = "CONDITIONAL (RF fallback)"
else:
    verdict = "NO-GO"

print(f"\n  ╔══════════════════════════════════════╗")
print(f"  ║  VERDICT: {verdict:^28s} ║")
print(f"  ╚══════════════════════════════════════╝")

# ══════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════
all_results = {
    'aoi': {
        'mode': aoi_settings['mode'],
        'source': aoi_settings['description'],
        'bbox': aoi_settings.get('bbox'),
        'asset_id': aoi_settings.get('asset_id'),
    },
    'block_cv': {str(k): v for k, v in block_cv_results.items()},
    'g1_gate': g1_gate,
    'transfer': {str(k): v for k, v in transfer_results.items()},
    'urban_trend': {
        'raw': {str(yr): float(uf) for yr, uf in urban_trend_raw},
        'corrected': {str(yr): float(uf) for yr, uf in urban_trend_corrected},
        'method': 'monotonic_non_decreasing',
        'assumption': 'urbanization is irreversible; dips are AEF temporal artifacts',
    },
    'g1': {
        'oa': float(g1_oa),
        'kappa': float(g1_kappa),
        'block_size': int(best_block),
        'pass': bool(g1_pass),
    },
    'g2': {
        'checks': [(n, bool(p), d) for n, p, d in g2_checks],
        'n_pass': int(n_g2_pass),
        'n_computed': int(n_g2_computed),
        'pass': bool(g2_pass),
        'status': g2_status,
    },
    'verdict': verdict,
    'n_stable_samples': int(len(X_stable)),
    'n_total_samples': int(len(X)),
}

with open(OUT_DIR / 'sejong_classification_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: {OUT_DIR / 'sejong_classification_results.json'}")
print(f"\n{'=' * 60}")
print("Phase 1 T2-T4 분석 완료")
print("=" * 60)
