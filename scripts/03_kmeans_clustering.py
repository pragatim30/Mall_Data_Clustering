import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.dpi':150, 'font.size':10})
df = pd.read_csv('data/mall_customers_with_gendercode.csv')
scaled = np.load('data/scaled_features.npy')

K = 5
km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = km.fit_predict(scaled)
centroids_scaled = km.cluster_centers_

# inverse-transform centroids back to original scale for interpretability
scaler = StandardScaler()
scaler.fit(df[['AnnualIncome','SpendingScore']])
centroids_original = scaler.inverse_transform(centroids_scaled)

sil = silhouette_score(scaled, df['Cluster'])
print(f"K={K} silhouette score: {sil:.3f}")
print("\nCluster sizes:\n", df['Cluster'].value_counts().sort_index())
print("\nCentroids (original scale - Income, Spending):\n", centroids_original)

# ---- Main scatter plot with annotated clusters + centroids ----
fig, ax = plt.subplots(figsize=(8.5,6))
palette = sns.color_palette('Set1', K)
for c in range(K):
    sub = df[df['Cluster']==c]
    ax.scatter(sub['AnnualIncome'], sub['SpendingScore'], s=45, color=palette[c],
               label=f'Cluster {c} (n={len(sub)})', alpha=0.75, edgecolor='k', linewidth=0.3)
ax.scatter(centroids_original[:,0], centroids_original[:,1], s=300, marker='X',
           color='black', label='Centroids', edgecolor='white', linewidth=1.5, zorder=5)
for i, (x,y) in enumerate(centroids_original):
    ax.annotate(f'C{i}', (x,y), textcoords="offset points", xytext=(10,10), fontsize=11, fontweight='bold')
ax.set_title(f'K-Means Clustering of Mall Customers (k={K})', fontsize=14, fontweight='bold')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig('figures/03_kmeans_clusters_scatter.png', bbox_inches='tight')
plt.close()

# ---- Silhouette plot (per-sample) ----
sample_sil = silhouette_samples(scaled, df['Cluster'])
fig, ax = plt.subplots(figsize=(7.5,5.5))
y_lower = 10
for c in range(K):
    vals = np.sort(sample_sil[df['Cluster']==c])
    size = len(vals)
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=palette[c], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5*size, f'C{c}', fontsize=10, fontweight='bold')
    y_lower = y_upper + 10
ax.axvline(sil, color='red', linestyle='--', linewidth=1.5, label=f'Average = {sil:.3f}')
ax.set_title('Silhouette Plot per Cluster (k=5)', fontsize=13, fontweight='bold')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Samples (grouped by cluster)')
ax.set_yticks([])
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/04_silhouette_plot.png', bbox_inches='tight')
plt.close()

df.to_csv('data/mall_customers_clustered.csv', index=False)
print("\nSaved clustered dataset.")
