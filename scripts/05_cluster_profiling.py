import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.dpi':150, 'font.size':10})
df = pd.read_csv('data/mall_customers_clustered.csv')

profile = df.groupby('Cluster').agg(
    n=('CustomerID','count'),
    avg_age=('Age','mean'),
    avg_income=('AnnualIncome','mean'),
    avg_spending=('SpendingScore','mean'),
    pct_female=('GenderCode', lambda x: x.mean()*100)
).round(1)
print(profile)
profile.to_csv('data/cluster_profile_summary.csv')

# 1. Grouped bar: avg age, income, spending per cluster (normalized to compare shapes isn't ideal;
#    instead show three separate small panels since units differ)
fig, axes = plt.subplots(1, 3, figsize=(14,4.3))
palette = sns.color_palette('Set1', 5)
metrics = [('avg_age','Average Age (years)'), ('avg_income','Average Annual Income (k$)'), ('avg_spending','Average Spending Score')]
for ax, (col, title) in zip(axes, metrics):
    bars = ax.bar(profile.index.astype(str), profile[col], color=palette)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Cluster')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}", ha='center', fontsize=9)
plt.suptitle('Cluster Profile Comparison Across Key Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_cluster_profile_bars.png', bbox_inches='tight')
plt.close()

# 2. Cluster size + gender composition
fig, axes = plt.subplots(1,2, figsize=(11,4.3))
bars = axes[0].bar(profile.index.astype(str), profile['n'], color=palette)
axes[0].set_title('Cluster Size (Number of Customers)', fontweight='bold')
axes[0].set_xlabel('Cluster'); axes[0].set_ylabel('Count')
for b in bars:
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{int(b.get_height())}", ha='center', fontsize=9)

bars2 = axes[1].bar(profile.index.astype(str), profile['pct_female'], color=palette)
axes[1].axhline(56, color='gray', linestyle='--', linewidth=1, label='Overall avg (56%)')
axes[1].set_title('% Female Customers per Cluster', fontweight='bold')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('% Female')
axes[1].legend(fontsize=8)
for b in bars2:
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('figures/08_cluster_size_gender.png', bbox_inches='tight')
plt.close()

# 3. Age distribution per cluster (boxplot)
fig, ax = plt.subplots(figsize=(8,4.5))
sns.boxplot(data=df, x='Cluster', y='Age', hue='Cluster', palette=palette, ax=ax, legend=False)
ax.set_title('Age Distribution Within Each Cluster', fontsize=13, fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Age (years)')
plt.tight_layout()
plt.savefig('figures/09_age_boxplot_by_cluster.png', bbox_inches='tight')
plt.close()

print("\nSaved profiling charts.")
