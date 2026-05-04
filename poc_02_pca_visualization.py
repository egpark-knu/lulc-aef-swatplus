"""
Phase 0 PoC — Step 2: PCA 시각화 + 시계열 변화 분석
추출된 AEF 임베딩을 PCA로 차원 축소하여 연도별 변화 패턴을 시각화한다.

Usage:
    python poc_02_pca_visualization.py
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── 데이터 로드 ──
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2017, 2026))
all_embeddings = []
all_years = []
all_coords = []

for year in YEARS:
    fpath = DATA_DIR / f"aef_daegu_{year}.npz"
    if not fpath.exists():
        print(f"⚠️ {year}년 데이터 없음, 건너뜀")
        continue
    data = np.load(fpath)
    emb = data['embeddings']
    coords = data['coords']

    all_embeddings.append(emb)
    all_years.extend([year] * len(emb))
    all_coords.append(coords)
    print(f"  {year}: {len(emb)} samples loaded")

X = np.vstack(all_embeddings)
years_arr = np.array(all_years)
coords_arr = np.vstack(all_coords)

print(f"\n전체 데이터: {X.shape[0]} samples x {X.shape[1]} dims")

# ── PCA (2D) ──
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA 설명 분산: PC1={pca.explained_variance_ratio_[0]:.3f}, "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}")

# ── PCA 시각화: 연도별 색상 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# (a) 전체 scatter
ax = axes[0]
cmap = cm.get_cmap('viridis', len(YEARS))
for i, year in enumerate(sorted(set(all_years))):
    mask = years_arr == year
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[cmap(i)], s=5, alpha=0.4, label=str(year))
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('AEF Embeddings PCA — Daegu (금호강)')
ax.legend(fontsize=8, markerscale=3)

# (b) 연도별 centroid 궤적
ax = axes[1]
centroids = []
for year in sorted(set(all_years)):
    mask = years_arr == year
    centroid = X_pca[mask].mean(axis=0)
    centroids.append((year, centroid))

centroids_arr = np.array([c[1] for c in centroids])
years_list = [c[0] for c in centroids]

ax.plot(centroids_arr[:, 0], centroids_arr[:, 1], 'k-', alpha=0.5, linewidth=1)
for i, (year, cent) in enumerate(centroids):
    ax.scatter(cent[0], cent[1], c=[cmap(i)], s=100, zorder=5, edgecolors='black')
    ax.annotate(str(year), (cent[0], cent[1]),
                textcoords='offset points', xytext=(8, 5), fontsize=9)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Centroid Trajectory (2017→2025)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'pca_yearly_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ PCA 그림 저장: {OUT_DIR / 'pca_yearly_comparison.png'}")

# ── t-SNE (선택) ──
print("\nt-SNE 계산 중 (시간 소요)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(10, 8))
for i, year in enumerate(sorted(set(all_years))):
    mask = years_arr == year
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=[cmap(i)], s=5, alpha=0.4, label=str(year))
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('AEF Embeddings t-SNE — Daegu (금호강)')
ax.legend(fontsize=8, markerscale=3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'tsne_yearly_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ t-SNE 그림 저장: {OUT_DIR / 'tsne_yearly_comparison.png'}")

# ── 연도 간 임베딩 거리 (변화 감지 proxy) ──
print(f"\n{'='*50}")
print("연도 간 평균 임베딩 거리 (centroid-based):")
for i in range(len(centroids) - 1):
    y1, c1 = centroids[i]
    y2, c2 = centroids[i + 1]
    dist = np.linalg.norm(c2 - c1)
    print(f"  {y1}→{y2}: {dist:.4f}")

# ── 공간 변화 맵 (첫해 vs 마지막해) ──
first_year = sorted(set(all_years))[0]
last_year = sorted(set(all_years))[-1]

mask_first = years_arr == first_year
mask_last = years_arr == last_year

# 동일 좌표 포인트끼리 비교 (seed=42이므로 같은 위치)
n_compare = min(mask_first.sum(), mask_last.sum())
emb_first = X[mask_first][:n_compare]
emb_last = X[mask_last][:n_compare]
coords_first = coords_arr[mask_first][:n_compare]

# 포인트별 임베딩 변화량
change_magnitude = np.linalg.norm(emb_last - emb_first, axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(coords_first[:, 0], coords_first[:, 1],
                c=change_magnitude, cmap='hot_r', s=15, alpha=0.7)
plt.colorbar(sc, label='Embedding Distance')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'LULC Change Magnitude ({first_year}→{last_year})')
plt.tight_layout()
plt.savefig(OUT_DIR / f'change_map_{first_year}_{last_year}.png',
            dpi=150, bbox_inches='tight')
print(f"\n✅ 변화 맵 저장: {OUT_DIR / f'change_map_{first_year}_{last_year}.png'}")

print("\n🎯 Phase 0 PoC Step 2 완료!")
