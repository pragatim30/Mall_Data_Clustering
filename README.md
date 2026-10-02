# Customer Segmentation — Unsupervised Learning & Clustering (Week 3)

Segments the Mall Customer Segmentation dataset (200 customers) into 5 groups
using K-Means, cross-validated against Hierarchical (Ward) clustering, with
the number of clusters chosen via elbow method + silhouette score + Davies-Bouldin
index together.

## Structure

```
data/
  Mall_Customers.csv              # raw input
  mall_customers_clustered.csv    # output with Cluster + Cluster_Hierarchical labels
  cluster_profile_summary.csv     # per-cluster summary stats
scripts/
  01_explore.py                        # initial exploration
  02_preprocessing_and_k_selection.py  # scaling + elbow/silhouette/DB index for choosing k
  03_kmeans_clustering.py              # K-Means (k=5) + annotated scatter + silhouette plot
  04_hierarchical_clustering.py        # Ward dendrogram + comparison to K-Means
  05_cluster_profiling.py              # age/income/spending/gender profile per cluster
figures/        # all 9 charts used in the report
report/
  Clustering_Analysis_Report.docx
```

## Running it

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
python scripts/01_explore.py
python scripts/02_preprocessing_and_k_selection.py
python scripts/03_kmeans_clustering.py
python scripts/04_hierarchical_clustering.py
python scripts/05_cluster_profiling.py
```

## Key results

- k=5 selected via elbow method, silhouette score (0.555, best of k=2-10), and
  Davies-Bouldin index (0.572, best of k=2-10) — all three agree.
- K-Means and Hierarchical (Ward) clustering agree closely: Adjusted Rand Index
  = 0.942 between the two sets of labels.
- Five clusters identified: average majority, high-income/high-spending,
  low-income/high-spending, high-income/low-spending, low-income/low-spending —
  each profiled by age and gender and discussed with business implications.
- Report includes an explicit limitations/bias section and reflective commentary.
