import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':150, 'font.size':10})
df = pd.read_csv('data/mall_customers_clustered.csv')
scaled = np.load('data/scaled_features.npy')

# Ward linkage - minimizes within-cluster variance, comparable objective to k-means
Z = linkage(scaled, method='ward')

fig, ax = plt.subplots(figsize=(10,5))
dendrogram(Z, ax=ax, color_threshold=7, truncate_mode=None)
ax.axhline(y=7, color='red', linestyle='--', linewidth=1.5, label='Cut at distance=7 -> 5 clusters')
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
ax.set_xlabel('Customer Index')
ax.set_ylabel('Ward Distance')
ax.legend()
plt.tight_layout()
plt.savefig('figures/05_dendrogram.png', bbox_inches='tight')
plt.close()

hier_labels = fcluster(Z, t=5, criterion='maxclust') - 1  # zero-index to match kmeans style
df['Cluster_Hierarchical'] = hier_labels

sil_hier = silhouette_score(scaled, hier_labels)
ari = adjusted_rand_score(df['Cluster'], hier_labels)
print(f"Hierarchical (5 clusters) silhouette: {sil_hier:.3f}")
print(f"K-Means silhouette (from before): 0.555")
print(f"Adjusted Rand Index between K-Means and Hierarchical labels: {ari:.3f}")
print("\nHierarchical cluster sizes:\n", pd.Series(hier_labels).value_counts().sort_index())

# side-by-side comparison plot
import seaborn as sns
fig, axes = plt.subplots(1,2, figsize=(13,5.5))
palette = sns.color_palette('Set1', 5)
for c in range(5):
    sub = df[df['Cluster']==c]
    axes[0].scatter(sub['AnnualIncome'], sub['SpendingScore'], s=40, color=palette[c], alpha=0.75, edgecolor='k', linewidth=0.3)
axes[0].set_title(f'K-Means (silhouette={0.555:.3f})', fontweight='bold')
axes[0].set_xlabel('Annual Income (k$)'); axes[0].set_ylabel('Spending Score (1-100)')

for c in sorted(df['Cluster_Hierarchical'].unique()):
    sub = df[df['Cluster_Hierarchical']==c]
    axes[1].scatter(sub['AnnualIncome'], sub['SpendingScore'], s=40, color=palette[c % 5], alpha=0.75, edgecolor='k', linewidth=0.3)
axes[1].set_title(f'Hierarchical / Ward (silhouette={sil_hier:.3f})', fontweight='bold')
axes[1].set_xlabel('Annual Income (k$)'); axes[1].set_ylabel('Spending Score (1-100)')
plt.suptitle('K-Means vs Hierarchical Clustering — Side-by-Side Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_kmeans_vs_hierarchical.png', bbox_inches='tight')
plt.close()

df.to_csv('data/mall_customers_clustered.csv', index=False)
print("\nSaved comparison.")
