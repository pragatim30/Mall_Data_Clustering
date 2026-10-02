import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':150, 'font.size':10})
df = pd.read_csv('data/mall_customers_clean.csv')

# Encode gender for completeness (not used as primary clustering feature, but retained for profiling)
df['GenderCode'] = df['Gender'].map({'Male':0, 'Female':1})

# Features used for clustering: Annual Income and Spending Score
# Age is examined separately as a profiling variable (see report rationale)
features = df[['AnnualIncome','SpendingScore']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome_scaled','SpendingScore_scaled'])

print("Feature means before scaling:\n", features.mean())
print("Feature means after scaling (should be ~0):\n", df_scaled.mean())
print("Feature std after scaling (should be ~1):\n", df_scaled.std())

# ---- Elbow method (inertia) ----
inertias = []
sil_scores = []
db_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled, labels))
    db_scores.append(davies_bouldin_score(scaled, labels))

for k, i, s, d in zip(K_range, inertias, sil_scores, db_scores):
    print(f"k={k}: inertia={i:.1f}, silhouette={s:.3f}, davies-bouldin={d:.3f}")

# Elbow plot
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(list(K_range), inertias, marker='o', color='#4C72B0', linewidth=2)
ax.set_title('Elbow Method for Optimal k', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax.annotate('Elbow around k=5', xy=(5, inertias[3]), xytext=(6.5, inertias[3]+60),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8), fontsize=10, color='red')
ax.set_xticks(list(K_range))
plt.tight_layout()
plt.savefig('figures/01_elbow_method.png', bbox_inches='tight')
plt.close()

# Silhouette score plot
fig, ax = plt.subplots(figsize=(7,4.5))
bars = ax.bar(list(K_range), sil_scores, color='#55A868')
best_k = list(K_range)[np.argmax(sil_scores)]
ax.set_title('Silhouette Score by Number of Clusters', fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Average Silhouette Score')
ax.set_xticks(list(K_range))
for b, s in zip(bars, sil_scores):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{s:.3f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/02_silhouette_scores.png', bbox_inches='tight')
plt.close()

print(f"\nBest k by silhouette score: {best_k} (score={max(sil_scores):.3f})")

np.save('data/scaled_features.npy', scaled)
df.to_csv('data/mall_customers_with_gendercode.csv', index=False)
